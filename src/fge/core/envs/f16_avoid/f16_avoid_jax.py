import functools as ft
import itertools
from dataclasses import dataclass
from typing import NamedTuple

import einops as ei
import ipdb
import jax
import jax.debug as jd
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import numpy as np
from attrs import define
from chex import PRNGKey
from diffrax import ODETerm, Tsit5, diffeqsolve
from gymnasium import spaces
from jax_f16.f16 import F16
from jax_f16.f16_types import FullState
from jax_f16.lowlevel.low_level_controller import LowLevelController
from jax_f16.utils.jax_types import FloatScalar
from jaxtyping import ArrayLike, Float
from loguru import logger
from og.dyn_types import BObs
from og.jax_types import BFloat, IntScalar
from og.jax_utils import merge01, stack_broadcast
from og.tree_utils import tree_where
from protof16.f16_utils import (
    angle_between_vectors_signed,
    ftps_to_knots,
    get_R_ned_wind,
    get_v_enu,
    knots_to_ftps,
)
from protof16.guidance_utils import get_vel_vec, get_vel_vec_neu_np
from protof16.turn_controller_fixedthrottle import TurnControllerFixedThrottle
from protof16.vpp import get_pos_ned
from scipy.stats import qmc

from fge.core.bits.state_reset_id import Source
from fge.core.envs.jax_task import (
    EvalRegionInfo,
    EvalStateInfo,
    JaxTask,
    State_,
    StepOutput,
    TimedState,
)

# from fge.core.envs.f16_avoid.f16_avoid import TaskCfg


@dataclass
class TaskCfg:
    dt0: float = 5e-2
    dt: float = 5e-1

    reset_thresh_h: float = 0.0

    radius_ft: float = 4925.0
    throttle_lead: float = 0.28
    alt: float = 1e4

    max_steps: int = 1000


class IC:
    CD = 0
    ASPECT = 1
    CONEPHI = 2
    VT = 3


class F16AvoidJax(JaxTask):
    """

    range         :  [500, 1000]
    aspect (angle):  [30, 45]
    cone_phi      :  [-pi, pi]
    vt_ftps       :  [-100 knot, +5 ftps]

    We define n_vt * n_cd regions. Each region will contain the entire aspect x cone_phi space.

    coldist is the x-axis, vt is the y-axis on the big plot.

    Use a sobol sequence to cover the aspect x cone_phi space.

    """

    Cfg = TaskCfg

    @jdc.pytree_dataclass
    class State(TimedState):
        ic: jnp.ndarray
        """cone_dist, cone_angle, """

        reset_region: IntScalar

        a_x: Float[ArrayLike, "nf16 nx"]
        lead_guidance: TurnControllerFixedThrottle.GuidanceState
        chase_control: Float[ArrayLike, "nu"]

        @property
        def lead_state(self):
            return self.bandit_state

        @property
        def chase_state(self):
            return self.ego_state

        @property
        def bandit_state(self):
            return self.a_x[..., 0, :]

        @property
        def ego_state(self):
            return self.a_x[..., 1, :]

    def __init__(self, cfg: TaskCfg):
        self.f16 = F16()

        self.cfg = cfg
        self.dt0 = cfg.dt0
        self.dt = cfg.dt
        self.int_max_steps = int(2 * self.dt / self.dt0)

        lead_controller_params = TurnControllerFixedThrottle.Params(
            radius_ft=cfg.radius_ft, throttle=cfg.throttle_lead, alt=cfg.alt
        )
        self.lead_controller = TurnControllerFixedThrottle(lead_controller_params)

        self.x0_bandit, _, self.guidance_bandit, self.radius_bandit = (
            self.lead_controller.get_steady_state()
        )
        # fmt: on

        vel_vec_neu = get_vel_vec_neu_np(self.x0_bandit)
        psi_vel = np.arctan2(vel_vec_neu[1], vel_vec_neu[0])
        self.x0_bandit[F16.PSI] = -psi_vel

        thrtl_equi = LowLevelController.old_uequil[0]  # 0.1395

        #                           NZ, PS, NYR, THRTL. No afterburners.
        self.stick_min = np.array([-1.0, -4.0, -0.1, 0.0 - thrtl_equi])
        self.stick_max = np.array([6.0, 4.0, 0.1, 0.7 - thrtl_equi])  # 0.5605

        # Throttle for chase is fixed.
        # self.throttle = 0.123 # This was for the initial run.
        # self.throttle = 0.160 # This was for v6
        # self.throttle = 0.526  # For v7.
        self.throttle = 0.4  # For v9.

        #                           NZ, PS
        self.control_lo = np.array([-0.2, -1.5])
        self.control_hi = np.array([5.0, 1.5])
        self.control_mid = 0.5 * (self.control_lo + self.control_hi)
        self.control_half = 0.5 * (self.control_hi - self.control_lo)

        #                              NZ, PS
        # self.dcontrol_hi = np.array([0.15, 0.2])
        self.dcontrol_hi = np.array([0.4, 0.4])
        self.dcontrol_lo = -self.dcontrol_hi

        self.idx_bandit = 0
        self.idx_ego = 1

        # ---------------------------------------
        self.range_lo, self.range_hi = 500, 1_000
        # self.range_lo, self.range_hi = 1_000, 3_000
        # self.range_margin = 500.0
        self.range_margin = 0.5 * (self.range_hi - self.range_lo) / 2

        self.aspect_lo, self.aspect_hi = 30, 45
        self.aspect_margin_outside = 5
        self.aspect_margin_inside = 2.5

        self.alt = self.x0_bandit[F16.H]
        # The altitude when we are at 45 aspect at the maximum range.
        self.alt_hi = np.tan(np.deg2rad(45.0)) * self.range_hi
        # self.alt_margin = 500.0

        self.vt = self.vt_bandit = self.x0_bandit[F16.VT]
        # We can drop 100 knots
        self.vt_min = knots_to_ftps(ftps_to_knots(self.vt) - 100.0)
        self.vt_margin = 10.0

        self.cmd_vt_sample_lo = self.vt_min
        self.cmd_vt_sample_hi = self.x0_bandit[F16.VT] + 5

        self.beta_max_deg = 8.0
        self.beta_margin = 2.0

        self.alpha_deg_max = 45.0
        self.alpha_deg_min = -5.0
        self.alpha_deg_margin = 5.0

        # self.energy_diff_max = 6e5
        # self.energy_diff_margin = 1e5
        self.energy_diff_max = 6e4
        self.energy_diff_margin = 2e4

        # We also want to prevent the energy from getting too large.
        # This should be a redundant constraint, but hopefully it'll make it easier to learn to go back inside.
        self.energy_diff_min = -3e5

        self.h_margin = 0.1

        self.cone_phi_eps = np.pi

        # Set action space
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )
        super().__init__(action_space)

        # Eval info
        self._eval_rollout_T = self.cfg.max_steps

        # -------- Eval Regions -----------------
        self.n_vt_regions = 4
        self.n_cd_regions = 4
        self.n_slices = self.n_vt_regions * self.n_cd_regions

        # Use a Sobol sequence to cover the aspect x cone_phi space.
        sampler = qmc.Sobol(d=2, scramble=False)
        # 16 * 64 = 1024. 64 = 2^6.
        # [0, 1)
        b_sample2d = sampler.random_base2(6)
        # [-1, 1). (n_samples, 2). []
        b_sample2d = 2 * b_sample2d - 1

        self.n_eval_pts_per_slice = len(b_sample2d)
        self.n_eval_pts_total = (
            self.n_eval_pts_per_slice * self.n_cd_regions * self.n_vt_regions
        )

        lim_lo, lim_hi = self.get_ic_lims()
        cd_lo, cd_hi = lim_lo[0], lim_hi[0]
        aspect_lo, aspect_hi = lim_lo[1], lim_hi[1]
        conephi_lo, conephi_hi = lim_lo[2], lim_hi[2]
        vt_lo, vt_hi = lim_lo[3], lim_hi[3]
        self.b_vt, self.b_vt_edge = get_bins(vt_lo, vt_hi, self.n_vt_regions)
        self.b_cd, self.b_cd_edge = get_bins(cd_lo, cd_hi, self.n_cd_regions)

        self.s_aspect = (
            0.5 * (b_sample2d[:, 0] + 1) * (aspect_hi - aspect_lo) + aspect_lo
        )
        self.s_aspect_deg = np.rad2deg(self.s_aspect)
        self.s_conephi = (
            0.5 * (b_sample2d[:, 1] + 1) * (conephi_hi - conephi_lo) + conephi_lo
        )

    @property
    def cmd_vt(self):
        """vt for lead."""
        return self.vt_bandit

    @property
    def cmd_alt(self):
        return self.lead_controller.params.alt

    @property
    def nu(self):
        # [ Nz, Ps ]
        return 2

    @property
    def u_labels(self):
        return ["Nz", "Ps"]

    @property
    def ic_dim_names(self):
        return ["conedist", "aspect", "conephi", "vt_ftps"]

    def action_to_dcontrol(self, action: jnp.ndarray):
        """Map from action: [-1, 1] to control."""
        assert action.shape == (self.nu,)
        # [-1, 1] -> [0, 1]
        frac = (action + 1.0) / 2.0
        # return self.control_lo + frac * (self.control_hi - self.control_lo)
        return self.dcontrol_lo + frac * (self.dcontrol_hi - self.dcontrol_lo)

    def step(self, state: State, action: jnp.ndarray) -> StepOutput:
        assert action.shape == (self.nu,)
        dcontrol = self.action_to_dcontrol(action)
        control = state.chase_control
        control_new = control + dcontrol
        control_new = jnp.clip(control_new, self.control_lo, self.control_hi)
        stick_ego = jnp.array([*control_new, 0.0, self.throttle])
        state_new, info = self.integrate(state, stick_ego)

        state_new = jdc.replace(state_new, chase_control=control_new)

        h_dict, h_info = self.h_components(state)
        info = info | h_info

        # Freeze the state if the constraints are violated at the current state.
        is_frozen = self.should_reset(state)
        state_new = tree_where(is_frozen, state, state_new)

        info["h_vector"] = self.h_vector(state)
        info["is_frozen"] = is_frozen

        term = is_frozen
        trunc = self._is_finished(state)

        rew = jnp.where(is_frozen, -1.0, 0.0)

        obs = self.get_obs(state_new)

        return StepOutput(state_new, obs, rew, term, trunc, info)

    def integrate(self, state0: State, stick_ego: jnp.ndarray):
        def vf(t, b_x, args):
            b_control = jnp.stack([stick_bandit, stick_ego], axis=0)
            xdot = jax.vmap(self.f16.xdot)(b_x, b_control)
            return xdot

        state_bandit = state0.a_x[self.idx_bandit]
        stick_bandit, lead_guidance_new, _ = self.lead_controller.compute_command(
            state_bandit, state0.lead_guidance, self.dt
        )

        assert state0.a_x.shape == (2, F16.NX)
        term = ODETerm(vf)
        solver = Tsit5()
        # jd.print("> diffeqsolve", ordered=True)
        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=self.dt,
            dt0=self.dt0,
            y0=state0.a_x,
            max_steps=self.int_max_steps,
        )

        stats, result = sol.stats, sol.result

        # (1, nf16, nx)
        ys = sol.ys
        assert ys.shape == (1, 2, F16.NX)
        b_state = ys[0]

        info = {
            "odeint/stats": stats,
            "odeint/result": result,
            "stick/bandit": stick_bandit,
            "stick/ego": stick_ego,
        }

        state_new = jdc.replace(
            state0, step=state0.step + 1, a_x=b_state, lead_guidance=lead_guidance_new
        )
        return state_new, info

    @property
    def h_limits(self):
        return {
            "energy_diff": [self.energy_diff_min, self.energy_diff_max],
            "d_ft": [self.range_lo, self.range_hi],
            "theta_deg": [self.aspect_lo, self.aspect_hi],
            "vT": [self.vt_min, None],
            "alpha_deg": [self.alpha_deg_min, self.alpha_deg_max],
            "beta_deg": [-self.beta_max_deg, self.beta_max_deg],
        }

    @property
    def h_term_limits(self):
        return {
            "energy_diff": [
                self.energy_diff_min - self.energy_diff_margin,
                self.energy_diff_max + self.energy_diff_margin,
            ],
            "d_ft": [
                self.range_lo - self.range_margin,
                self.range_hi + self.range_margin,
            ],
            "theta_deg": [
                self.aspect_lo - self.aspect_margin_outside,
                self.aspect_hi + self.aspect_margin_outside,
            ],
            "vT": [self.vt_min - self.vt_margin, None],
            "alpha_deg": [
                self.alpha_deg_min - self.alpha_deg_margin,
                self.alpha_deg_max + self.alpha_deg_margin,
            ],
            "beta_deg": [
                -self.beta_max_deg - self.beta_margin,
                self.beta_max_deg + self.beta_margin,
            ],
        }

    def h_components(self, state: State) -> tuple[dict[str, FloatScalar], dict]:
        """Negative is safe."""
        state_lead = state.a_x[self.idx_bandit]
        state_chase = state.a_x[self.idx_ego]
        p_lead_chase_ned = get_pos_ned(state_chase) - get_pos_ned(state_lead)

        R_ned_leadwind = get_R_ned_wind(state_lead)
        R_leadwind_ned = R_ned_leadwind.T
        p_lead_chase_leadwind = R_leadwind_ned @ p_lead_chase_ned
        d_ft = -p_lead_chase_leadwind[0]

        theta_rad = jnp.atan2(jnp.linalg.norm(p_lead_chase_leadwind[1:]), d_ft)
        theta_deg = jnp.rad2deg(theta_rad)

        # FRD
        p_lead_chase_leadwind_up = -p_lead_chase_leadwind[2]
        p_lead_chase_leadwind_right = p_lead_chase_leadwind[1]

        conephi = jnp.arctan2(p_lead_chase_leadwind_up, p_lead_chase_leadwind_right)
        conephi_noroll = conephi - state_lead[F16.PHI]

        # 1: Cone Distance constraint. [-1, 1]
        range_cone = d_ft
        range_lo, range_hi = self.range_lo, self.range_hi
        h_range = (
            jnp.maximum(-(range_cone - range_lo), range_cone - range_hi)
            / self.range_margin
        )
        h_range = jnp.clip(h_range, -1.0, 1.0)

        # 2: Cone Angle constraint. [-1, 1].
        #    Make the slope steeper on the inside.
        aspect_deg = theta_deg
        aspect_deg_abs = jnp.abs(aspect_deg)
        aspect_lo, aspect_hi = self.aspect_lo, self.aspect_hi
        aspect_inside = (aspect_lo <= aspect_deg_abs) & (aspect_deg_abs <= aspect_hi)
        aspect_margin = jnp.where(
            aspect_inside, self.aspect_margin_inside, self.aspect_margin_outside
        )
        h_aspect = (
            jnp.maximum(-(aspect_deg_abs - aspect_lo), aspect_deg_abs - aspect_hi)
            / aspect_margin
        )
        h_aspect = jnp.clip(h_aspect, -1.0, 1.0)

        # 4: vT constraint. Prevent vT from dropping too low.
        vT = state.a_x[self.idx_ego, F16.VT]
        h_vT = -(vT - self.vt_min) / self.vt_margin
        h_vT = jnp.clip(h_vT, -1.0, 1.0)

        # 5: Keep alpha small. Within morelli bounds.
        alpha = state.a_x[self.idx_ego, F16.ALPHA]
        alpha_deg = jnp.abs(jnp.rad2deg(alpha))
        h_alpha = (
            jnp.maximum(
                -(alpha_deg - self.alpha_deg_min), alpha_deg - self.alpha_deg_max
            )
            / self.alpha_deg_margin
        )
        h_alpha = jnp.clip(h_alpha, -1.0, 1.0)

        # 6: Keep beta small. Within [-5, 5]
        beta = state.a_x[self.idx_ego, F16.BETA]
        beta_deg = jnp.abs(jnp.rad2deg(beta))
        h_beta = (beta_deg - self.beta_max_deg) / self.beta_margin
        h_beta = jnp.clip(h_beta, -1.0, 1.0)

        # 7: Make sure the chase aircraft doesn't have too little energy. Also make sure it doesn't have too much.
        # Both KE and PE are proportional to mass, so we can ignore it.
        _GRAVITY = 32.174
        energy = (
            0.5 * state.a_x[self.idx_ego, F16.VT] ** 2
            + _GRAVITY * state.a_x[self.idx_ego, F16.H]
        )
        energy_lead = (
            0.5 * state.a_x[self.idx_bandit, F16.VT] ** 2
            + _GRAVITY * state.a_x[self.idx_bandit, F16.H]
        )
        # We want energy_diff < self.energy_diff_max.
        energy_diff = energy_lead - energy
        # h_energy_diff = (energy_diff - self.energy_diff_max) / self.energy_diff_margin
        h_energy_diff = (
            jnp.maximum(
                -(energy_diff - self.energy_diff_min),
                (energy_diff - self.energy_diff_max),
            )
            / self.energy_diff_margin
        )
        h_energy_diff = jnp.clip(h_energy_diff, -1.0, 1.0)

        info = {
            "p_lead_chase_leadwind": p_lead_chase_leadwind,
            "d_ft": d_ft,
            "theta_deg": theta_deg,
            "conephi": conephi,
            "conephi_noroll": conephi_noroll,
            "vT": vT,
            "alpha_deg": alpha_deg,
            "beta_deg": beta_deg,
            "energy": energy,
            "energy_lead": energy_lead,
            "energy_diff": energy_diff,
            "alt_above": state.a_x[self.idx_ego, F16.H]
            - state.a_x[self.idx_bandit, F16.H],
        }
        h_dict = {
            "range": h_range,
            "aspect": h_aspect,
            "vT": h_vT,
            "alpha": h_alpha,
            "beta": h_beta,
            "energy_diff": h_energy_diff,
        }

        h_dict = {k: self.add_margin(v, self.h_margin) for k, v in h_dict.items()}

        return h_dict, info

    @staticmethod
    def add_margin(h: float, margin: float):
        """Map [-1, 0] to [-1, -margin] and [0, 1] to [margin, 1]."""
        h_squash = (1 - margin) * h
        return h_squash + jnp.where(h < 0, -margin, +margin)

    @property
    def h_labels(self):
        return ["range", "aspect", "vT", "alpha", "beta", "energy_diff"]

    @property
    def nh(self):
        return 6

    def h_vector(self, state: State) -> jnp.ndarray:
        h_dict, _ = self.h_components(state)
        h_h = jnp.array([h for h in h_dict.values()])
        return h_h

    def should_reset(self, state: State):
        """Reset if any of the constraints are maximally violated."""
        h_h = self.h_vector(state)
        return jnp.any(h_h >= self.cfg.reset_thresh_h)

    def get_obs(self, state: State):
        state_lead = state.a_x[self.idx_bandit]
        state_chase = state.a_x[self.idx_ego]
        p_lead_chase_ned = get_pos_ned(state_chase) - get_pos_ned(state_lead)
        R_ned_leadwind = get_R_ned_wind(state_lead)
        R_leadwind_ned = R_ned_leadwind.T
        p_lead_chase_leadwind = R_leadwind_ned @ p_lead_chase_ned

        d_ft = -p_lead_chase_leadwind[0]

        theta_rad = jnp.atan2(jnp.linalg.norm(p_lead_chase_leadwind[1:]), d_ft)
        theta_deg = jnp.rad2deg(theta_rad)

        normal = jnp.array([0.0, 0.0, 1.0])
        heading_crossing_angle = angle_between_vectors_signed(
            get_v_enu(state_lead), get_v_enu(state_chase), normal
        )

        _GRAVITY = 32.174
        energy = (
            0.5 * state.a_x[self.idx_ego, F16.VT] ** 2
            + _GRAVITY * state.a_x[self.idx_ego, F16.H]
        )
        energy_lead = (
            0.5 * state.a_x[self.idx_bandit, F16.VT] ** 2
            + _GRAVITY * state.a_x[self.idx_bandit, F16.H]
        )
        energy_diff = energy_lead - energy
        obs_task = jnp.array(
            [*p_lead_chase_leadwind, theta_deg, heading_crossing_angle, energy_diff]
        )
        # obs_task = jnp.array([*p_lead_chase_leadwind, theta_deg, heading_crossing_angle])

        obs_control = state.chase_control

        dyn_state_idxs = np.array(
            [
                F16.VT,
                F16.ALPHA,
                F16.BETA,
                F16.PHI,
                F16.THETA,
                F16.P,
                F16.Q,
                F16.R,
                F16.H,
                F16.NZINT,
                F16.PS,
                F16.NYRINT,
            ]
        )
        dyn_state_chase = state_chase[dyn_state_idxs]

        obs = jnp.concatenate([obs_task, obs_control, dyn_state_chase])
        obs_mean, obs_std = self.get_obs_meanstd()
        obs = (obs - obs_mean) / obs_std
        return obs

    def get_obs_meanstd(self):
        obs_mean_task = np.array([-750.0, 0.0, 0.0, 37.5, 0.0, 0.0])
        obs_std_task = np.array([100.0, 5e2, 5e2, 4.0, 0.2, 6e5])
        # obs_mean_task = np.array([750.0, 0.0, 0.0, 37.5, 0.0, 0.0])
        # obs_std_task = np.array([250.0, 5e2, 5e2, 5.0, 0.2, 6e5])
        # obs_mean_task = np.array([2e3, 0.0, 0.0, 37.5, 0.0])
        # obs_std_task = np.array([8e2, 1.5e3, 1.5e3, 5.0, 0.2])

        obs_mean_control = self.control_mid
        obs_std_control = 0.25 * self.control_half

        # fmt: off
        obs_mean_dyn_ego = np.array(
            #   [   VT, ALPHA, BETA, PHI, THETA, P,   Q,   R,          H,   NZINT, PS, NYRINT ]
            [5.32e2, 0.1, 0.02, 0.5, 0.0, 0.0, 0.0, 0.0, self.cmd_alt, -0.35, 0.08, 0.03])
        #                         [   VT, ALPHA, BETA, PHI, THETA, P,   Q,   R,     H, NZINT, PS, NYRINT ]
        obs_std_dyn_ego = np.array([10.0, 0.1, 0.02, 1.5, 0.5, 1.0, 0.5, 0.5, 500.0, 0.5, 0.5, 0.5])
        # fmt: on

        obs_mean = jnp.concatenate([obs_mean_task, obs_mean_control, obs_mean_dyn_ego])
        obs_std = jnp.concatenate([obs_std_task, obs_std_control, obs_std_dyn_ego])
        return obs_mean, obs_std

    def get_ic_lims(self):
        """The limits for the initial conditions."""

        dist_frac_lo, dist_frac_hi = 0.0, 1.0
        angle_frac_lo, angle_frac_hi = 0.0, 1.0
        cone_phi_lo, cone_phi_hi = -self.cone_phi_eps, +self.cone_phi_eps

        dist_range = self.range_hi - self.range_lo
        dist_lo = self.range_lo + dist_frac_lo * dist_range
        dist_hi = self.range_lo + dist_frac_hi * dist_range

        angle_range = self.aspect_hi - self.aspect_lo
        angle_lo = self.aspect_lo + angle_frac_lo * angle_range
        angle_hi = self.aspect_lo + angle_frac_hi * angle_range

        angle_lo, angle_hi = np.deg2rad(angle_lo), np.deg2rad(angle_hi)

        vt_ftps_lo = self.cmd_vt_sample_lo
        vt_ftps_hi = self.cmd_vt_sample_hi

        lim_lo = np.array([dist_lo, angle_lo, cone_phi_lo, vt_ftps_lo])
        lim_hi = np.array([dist_hi, angle_hi, cone_phi_hi, vt_ftps_hi])

        return lim_lo, lim_hi

    def reset_from_box(self, uniform: jnp.ndarray) -> State:
        lim_lo, lim_hi = self.get_ic_lims()
        assert lim_lo.shape == lim_hi.shape == uniform.shape

        # Rescale from [-1, 1] to [lim_lo, lim_hi]
        vec01 = 0.5 * (uniform + 1.0)
        ic_vec = vec01 * (lim_hi - lim_lo) + lim_lo
        cone_dist, cone_angle, cone_phi, vt_ftps = ic_vec

        return self.ic_to_state(cone_dist, cone_angle, cone_phi, vt_ftps)

    def box_from_reset(self, state: State) -> jnp.ndarray:
        cone_dist, cone_angle, cone_phi, vt_ftps = state.ic
        ic_vec = jnp.array([cone_dist, cone_angle, cone_phi, vt_ftps])
        lim_lo, lim_hi = self.get_ic_lims()
        vec01 = (ic_vec - lim_lo) / (lim_hi - lim_lo)
        uniform = 2.0 * vec01 - 1.0
        return uniform

    @property
    def x0_unif_shape(self) -> tuple[int, ...]:
        return (4,)

    def ic_to_state(
        self, cone_dist: float, cone_angle: float, cone_phi: float, vt_ftps: float
    ) -> State:
        ic = jnp.array([cone_dist, cone_angle, cone_phi, vt_ftps])
        bandit_state0 = self.x0_bandit
        ego_state0, u0_ego = F16AvoidJax.get_ego_state(
            bandit_state0,
            self.radius_bandit,
            self.throttle,
            vt_ftps,
            cone_dist,
            cone_angle,
            cone_phi,
        )
        u0_ego = jnp.clip(u0_ego, self.control_lo, self.control_hi)

        # # Perturb psi.
        # lim_hi = np.array([np.deg2rad(0.2)])
        # (dpsi,) = jr.uniform(key_pert, (1,), minval=-lim_hi, maxval=lim_hi)
        # ego_state0 = ego_state0.at[F16.PSI].set(ego_state0[F16.PSI] + dpsi)

        guidance_state = self.guidance_bandit
        a_state = jnp.stack([bandit_state0, ego_state0], axis=0)
        reset_region = self.ic_to_reset_region(ic)
        state = F16AvoidJax.State(
            0, Source.BASE, ic, reset_region, a_state, guidance_state, u0_ego
        )
        return state

    def ic_to_reset_region(self, ic) -> IntScalar:
        return 0

    def sample_inside(self, key: PRNGKey) -> State:
        lim_lo, lim_hi = self.get_ic_lims()
        ic_vec = jr.uniform(key, (4,), minval=lim_lo, maxval=lim_hi)
        cone_dist, cone_angle, cone_phi, vt_ftps = ic_vec
        return self.ic_to_state(cone_dist, cone_angle, cone_phi, vt_ftps)

    def reset(self, key: PRNGKey) -> State:
        return self.sample_inside(key)

    def reset_papereval(self, b_key: PRNGKey, num: int) -> State_:
        """Reset uniformly to remove any randomness."""
        del b_key

        def f(ic):
            assert ic.shape == (4,)
            state = self.ic_to_state(*ic)
            state = jdc.replace(state, source=Source.BASE)
            return state

        key = jr.PRNGKey(54321)
        lim_lo, lim_hi = self.get_ic_lims()
        b_ic = jr.uniform(key, (num, 4), minval=lim_lo, maxval=lim_hi)
        return jax.vmap(f)(b_ic)

    @staticmethod
    def get_ego_pos(
        state_lead: FullState, cone_dist: float, cone_angle: float, phi: float
    ):
        """
        :param cone_dist: Distance from the ego to the bandit in the bandit's wind frame.
        :param cone_angle: Angle from the ego to the bandit's velocity vector.
        :param phi: Angle of the ego relative to the horizon. phi=0 or pi is when the ego is at the same altitude.
        :return:
        """
        # Compensate for the lead aircrat's roll. We want phi=0 to be at the horizon, not parallel to the lead's wings.
        phi = phi - state_lead[F16.PHI]

        circ_radius = jnp.tan(cone_angle) * cone_dist
        p_lead_chase_leadwind = jnp.array(
            [-cone_dist, circ_radius * jnp.cos(phi), circ_radius * jnp.sin(phi)]
        )

        R_ned_leadwind = get_R_ned_wind(state_lead)
        p_lead_chase_ned = R_ned_leadwind @ p_lead_chase_leadwind

        p_W_lead_ned = get_pos_ned(state_lead)
        p_W_chase_ned = p_W_lead_ned + p_lead_chase_ned

        return p_W_chase_ned

    @staticmethod
    def get_ego_state(
        state_lead: FullState,
        radius_bandit: float,
        throttle: float,
        vt_ftps: float,
        cone_dist: float,
        cone_angle: float,
        cone_phi: float,
    ):
        pos_ego_ned = F16AvoidJax.get_ego_pos(
            state_lead, cone_dist, cone_angle, cone_phi
        )

        pos_ego_en = pos_ego_ned[np.array([1, 0])]
        center_bandit = state_lead[F16.POS2D] + np.array([radius_bandit, 0.0])
        radius_chase = jnp.linalg.norm(pos_ego_en - center_bandit)

        # Use cone_phi to linear interp for the nom [ vt, alpha, beta, psi, phi, P, Q, R, POW, NZINT, PSINT, NYRINT ]
        # radius_chase_delta = 45
        # radius_chase = radius_chase + radius_chase_delta

        # Compute the angle that chase should be pointed at.
        chase_angle = jnp.arctan2(
            pos_ego_en[1] - center_bandit[1], pos_ego_en[0] - center_bandit[0]
        )
        # Convert to the angle measured from North going clockwise.
        chase_psi = np.pi / 2 - chase_angle
        # Add 90 degrees to make it the tangent.
        chase_psi = chase_psi + np.pi / 2

        chase_controller_params = TurnControllerFixedThrottle.Params(
            radius_ft=radius_chase, throttle=throttle
        )
        chase_controller = TurnControllerFixedThrottle(chase_controller_params)
        # x0_ego, u0_ego = chase_controller.get_transient_state_jax(vt_ftps)
        x0_ego, u0_ego = chase_controller.get_transient_state2_jax(vt_ftps)
        u0_ego = u0_ego[:2]
        #
        # if print:
        #     jd.print("radius_chase: {}, throttle: {}, vt_ftps: {}", radius_chase, throttle, vt_ftps, ordered=True)

        # Set the position.
        x0_ego = x0_ego.at[F16.VT].set(vt_ftps)
        x0_ego = x0_ego.at[F16.PE].set(pos_ego_ned[1])
        x0_ego = x0_ego.at[F16.PN].set(pos_ego_ned[0])
        x0_ego = x0_ego.at[F16.H].set(-pos_ego_ned[2])
        x0_ego = x0_ego.at[F16.PSI].set(0.0)

        #     Take beta!=0 into account.
        vel_vec_neu = get_vel_vec(x0_ego)
        psi_vel = jnp.arctan2(vel_vec_neu[1], vel_vec_neu[0])
        x0_ego = x0_ego.at[F16.PSI].set(chase_psi - psi_vel)

        return x0_ego, u0_ego

    @property
    def eval_rollout_T(self) -> int:
        return self.cfg.max_steps

    @eval_rollout_T.setter
    def eval_rollout_T(self, value: int):
        self._eval_rollout_T = value

    def _is_finished(self, state: State):
        # Define it as a time-based thing.
        return state.step >= self.eval_rollout_T

    @property
    def region_names(self):
        lim_lo, lim_hi = self.get_ic_lims()
        # cone_dist, cone_angle, cone_phi, vt_ftps
        assert lim_lo.shape == (4,)
        assert lim_hi.shape == (4,)

        # x-axis is cone dist, y-axis is vt
        bb_cd, bb_vt = np.meshgrid(self.b_cd, self.b_vt)
        b_cd_flat, b_vt_flat = bb_cd.flatten(), bb_vt.flatten()

        region_names = []
        for ii, (cd, vt) in enumerate(zip(b_cd_flat, b_vt_flat)):
            region_name = "cd{:.0f}_vt{:.0f}".format(cd, vt)
            region_names.append(region_name)
        return region_names

    def get_eval_states(self) -> EvalStateInfo:
        # Get the region names.
        region_names = self.region_names

        # x-axis is cone dist, y-axis is vt
        bb_cd, bb_vt = np.meshgrid(self.b_cd, self.b_vt)
        b_cd_flat, b_vt_flat = bb_cd.flatten(), bb_vt.flatten()

        # (b, s, 4)
        bs_ic = stack_broadcast(
            [b_cd_flat[:, None], self.s_aspect, self.s_conephi, b_vt_flat[:, None]],
            axis=-1,
            which=jnp,
        )
        assert bs_ic.shape == (
            self.n_cd_regions * self.n_vt_regions,
            self.n_eval_pts_per_slice,
            4,
        )

        # (b, s, 4) -> (b * s, 4)
        b_ic = merge01(bs_ic)
        b_state = jax.vmap(self.ic_to_state)(
            b_ic[:, 0], b_ic[:, 1], b_ic[:, 2], b_ic[:, 3]
        )
        regions = [
            EvalRegionInfo(name, self.n_eval_pts_per_slice) for name in region_names
        ]

        return EvalStateInfo(b_state, self.n_eval_pts_total, regions)

    def eval_ics(self, n_per_region: int = 1):
        raise NotImplementedError("")

    def to_icval(self, state: State) -> np.ndarray:
        return state.ic

    def get_eval_contour(self) -> tuple[BFloat, BObs, State]:
        b_state, _, _ = self.get_eval_states()
        b_ic = self.to_icval(b_state)
        b_obs = jax.vmap(self.get_obs)(b_state)

        return b_ic, b_obs, b_state


def get_bins(lo, hi, n_bins: int):
    """Get bin centers for a given range and number of bins."""
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, bin_edges
