import functools as ft
import pathlib
from dataclasses import dataclass
from typing import Any, NamedTuple

import cv2
import einops as ei
import ipdb
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from jax import random as jr
from jax2d.engine import PhysicsEngine
from jax2d.sim_state import CollisionManifold
from jaxtyping import ArrayLike, Float, UInt8
from kinetix.environment import ContinuousActions, EnvParams, EnvState, PixelObservations, StaticEnvParams
from kinetix.render.renderer_pixels import PixelsObservation, make_render_pixels, make_render_pixels_rl
from kinetix.util import load_from_json_file
from loguru import logger
from og.dyn_types import BObs
from og.jax_types import BFloat, BoolScalar, FloatScalar
from og.rng import PRNGKey
from og.tree_utils import tree_cat, tree_where
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import qmc

from fge.core.bits.collector import RolloutOutput
from fge.core.bits.state_reset_id import Source
from fge.core.envs.jax_task import EvalRegionInfo, EvalStateInfo, JaxTask, State_, StepOutput, TimedState
from fge.core.envs.kinetix.uint8_renderer import make_render_pixels_uint8
from fge.core.utils.kinetix_utils import update_thruster_global_pos


@dataclass
class TaskCfgBase:
    n_frame_stack: int = 3
    """How many consecutive frames to stack."""

    frame_skip: int = 2
    max_fuel: float = 40.0
    max_steps: int = 150


@dataclass
class TaskCfg(TaskCfgBase):
    pass


@dataclass
class TaskCfgHard(TaskCfgBase):
    pass


@dataclass
class TaskCfgHardState(TaskCfgBase):
    pass


FloatImage = Float[ArrayLike, "h w c"]
UInt8Image = UInt8[ArrayLike, "h w c"]

StackedImage = Float[ArrayLike, "h w c*n"]
StackedUInt8Image = UInt8[ArrayLike, "h w c*n"]


class LanderObs(NamedTuple):
    img: StackedImage
    aux: jnp.ndarray
    """Auxiliary information to be added after the CNN."""


class MyPixelObservations:
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        self.render_function = make_render_pixels_uint8(env_params, static_env_params)

    def get_obs(self, state: EnvState):
        im = self.render_function(state)
        return PixelsObservation(image=im, global_info=None)


class BaseLanderJax(JaxTask):
    Cfg = TaskCfg

    @jdc.pytree_dataclass
    class State(TimedState):
        # Just the params. [-1, 1]. We can reconstruct the state from this.
        ic_box: jnp.ndarray

        state_kinetix: EnvState
        # How much fuel we have left.
        fuel_left: FloatScalar

    def __init__(self, cfg: TaskCfg, json_path: pathlib.Path):
        self.cfg = cfg

        # Load env
        json_path = str(json_path.absolute())
        self.level, self.static_env_params, self.env_params = load_from_json_file(json_path)
        self.len_m = float(self.static_env_params.screen_dim[0] / self.env_params.pixels_per_unit)

        self.action_type = ContinuousActions(self.env_params, self.static_env_params)
        self.render = MyPixelObservations(self.env_params, self.static_env_params)
        self.physics_engine = PhysicsEngine(self.static_env_params)

        self._eval_rollout_T = self.cfg.max_steps

        self.radius_circle = 0.2

        self.goal_pidx = 0
        self.goal_gidx = 0

        # Index within just the circles.
        self.agent_pidx = 1
        self.agent_gidx = self.agent_pidx

        # action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)

        # Discretize to 3 levels per thruster, so total of 3^2==9 actions.
        self.n_levels = 3
        action_space = spaces.Discrete(self.n_levels**2)

        super().__init__(action_space)

    @property
    def nu(self):
        # [ left_thruster, right_thruster ]
        return 2

    @property
    def u_labels(self):
        return ["L", "R"]

    def has_collided(self, state: EnvState, mfolds: tuple[CollisionManifold, CollisionManifold, CollisionManifold]):
        def get_active(manifold: CollisionManifold) -> jnp.ndarray:
            return manifold.active

        mfold_pp, mfold_cp, mfold_cc = mfolds

        rs = state.polygon_shape_roles * state.polygon.active
        cs = state.circle_shape_roles * state.circle.active

        # Only need to check circle polygon collisions.
        c1 = cs[self.physics_engine.circle_poly_pairs[:, 0]]
        r2 = rs[self.physics_engine.circle_poly_pairs[:, 1]]

        # player: role=1, obstacle: role=3
        is_valid = (r2 == 1) & (c1 == 3)
        has_collided = (is_valid * get_active(mfold_cp)).sum() > 0

        return has_collided

    def convert_action(self, action: jnp.ndarray) -> jnp.ndarray:
        assert action.shape == tuple() and action.dtype == jnp.dtype(jnp.int32)

        thruster_outputs = np.linspace(0.0, 1.0, num=self.n_levels, dtype=np.float32)
        # Cartesian product.
        X, Y = np.meshgrid(thruster_outputs, thruster_outputs)
        action_table = np.stack([X.flatten(), Y.flatten()], axis=-1)  # (n_levels**2, 2)

        return jnp.array(action_table)[action]

    def is_unsafe(self, state: State) -> BoolScalar:
        """Unsafe if out of bounds."""
        # pos = state.state_kinetix.circle.position[self.agent_cidx]
        pos = state.state_kinetix.polygon.position[self.agent_pidx]
        assert pos.shape == (2,)
        px, py = pos

        in_bounds_x = (0.0 <= px) & (px <= self.len_m)
        in_bounds_y = (0.0 <= py) & (py <= self.len_m)
        in_bounds = in_bounds_x & in_bounds_y

        # Unsafe if has no fuel AND height is above threshold.
        # no fuel => not too_high
        # <=> fuel or not too high
        has_fuel = state.fuel_left > 0.0
        # not_too_high = py <= 0.2 * self.len_m
        # safe_fuel = has_fuel | not_too_high
        safe_fuel = has_fuel

        is_safe = in_bounds & safe_fuel

        is_unsafe = ~is_safe
        return is_unsafe

    def is_unsafe_check_col(self, state: State) -> BoolScalar:
        """Also check collision."""
        mfolds = self.physics_engine.calculate_collision_manifolds(state.state_kinetix)
        has_collided = self.has_collided(state.state_kinetix, mfolds)
        is_unsafe_basic = self.is_unsafe(state)
        is_unsafe = has_collided | is_unsafe_basic
        return is_unsafe

    def is_unsafe_custom(self, state: State):
        return self.is_unsafe_check_col(state)

    def get_pos_and_verts(self, goal_width_frac: float, goal_pos_frac: float):
        len_m = self.len_m
        # Fraction of the full width that the goal takes up. 1 = full width.
        goal_width = len_m * goal_width_frac

        # 0 = left edge at left of screen, 1 = right edge at right of screen. Dependent on goal_width.
        goal_width_available = len_m - goal_width
        goal_left_edge = goal_pos_frac * goal_width_available
        goal_center = goal_left_edge + (goal_width / 2)

        # Have it take up 1 pixel in height. Take downscaling into account.
        goal_offset = (1 * self.static_env_params.downscale) / self.env_params.pixels_per_unit
        position = jnp.array([goal_center, -len_m])
        vertices = jnp.array(
            [
                [goal_width / 2, len_m + goal_offset],
                [goal_width / 2, -len_m - goal_offset],
                [-goal_width / 2, -len_m - goal_offset],
                [-goal_width / 2, len_m + goal_offset],
            ]
        )
        return position, vertices

    def box_from_reset(self, state: State):
        return state.ic_box

    def get_ic_lims(self):
        raise NotImplementedError("")

    @property
    def x0_unif_shape(self) -> tuple[int, ...]:
        lim_lo, lim_hi = self.get_ic_lims()
        return lim_lo.shape

    def reset(self, key: jr.PRNGKey) -> State:
        box = jr.uniform(key, shape=self.x0_unif_shape, minval=-1.0, maxval=1.0)
        return self.reset_from_box(box)

    @property
    def eval_rollout_T(self) -> int:
        return self._eval_rollout_T

    @eval_rollout_T.setter
    def eval_rollout_T(self, value: int):
        self._eval_rollout_T = value

    @property
    def region_names(self):
        return []

    def eval_ics(self, n_per_region: int = 1):
        raise NotImplementedError("")

    def to_icval(self, state: State) -> np.ndarray:
        return state.ic_box

    def get_eval_contour(self) -> tuple[BFloat, BObs, State]:
        raise NotImplementedError("")

    def setup_plot(self, ax: plt.Axes):

        def world_screen_size(static_params: StaticEnvParams, env_params: EnvParams):
            ppu = env_params.pixels_per_unit
            width_units = static_params.screen_dim[0] / ppu
            height_units = static_params.screen_dim[1] / ppu
            return width_units, height_units

        static_params = self.static_env_params
        state = self.level

        num_circles = int(static_params.num_circles)
        circ_positions = np.array(state.circle.position)
        circ_radius = np.array(state.circle.radius)
        circ_active = np.array(state.circle.active)

        ax.set_aspect("equal")
        w_units, h_units = world_screen_size(self.static_env_params, self.env_params)
        screen_rect = plt.Rectangle(
            (0.0, 0.0), w_units, h_units, fill=False, edgecolor="black", linewidth=2.0, zorder=1, label="Screen"
        )
        ax.add_patch(screen_rect)

        for ii in range(num_circles):
            if not circ_active[ii]:
                continue

            center = circ_positions[ii]
            rad = float(circ_radius[ii])
            color = "C0"

            cpatch = plt.Circle(
                center,
                rad,
                fill=True,
                facecolor=color,
                linewidth=0.0,
                alpha=0.3,
            )
            ax.add_patch(cpatch)

        eps = 0.005
        ax.set(xlim=(-eps * w_units, (1 + eps) * w_units), ylim=(-eps * h_units, (1 + eps) * h_units))


class BaseLanderJaxImage(BaseLanderJax):
    """Base class for lander with image observations."""

    @jdc.pytree_dataclass
    class State(BaseLanderJax.State):
        # list is for frame stacking.
        imgs: list[UInt8Image]

    @jdc.pytree_dataclass
    class CompressedState(TimedState):
        # Just the params. [-1, 1]. We can recontsruct the state from this.
        ic_box: jnp.ndarray
        state_kinetix: EnvState
        # How much fuel we have left.
        fuel_left: FloatScalar

    @classmethod
    def get_obs_treedef(cls):
        if not hasattr(cls, "obs_treedef"):
            tmp = LanderJax(LanderJax.Cfg())
            return tmp._obs_treedef()

        return cls.obs_treedef

    @classmethod
    def get_state_treedef(cls):
        if not hasattr(cls, "state_treedef"):
            tmp = LanderJax(LanderJax.Cfg())
            return tmp._state_treedef()

        return cls.state_treedef

    def get_dummy_nsf_obs(self):
        # Instead of images, return the IC directly.
        return np.zeros(self.x0_unif_shape)

    @property
    def use_ic_obs(self) -> bool:
        return True

    def compress_state(self, state: State) -> CompressedState:
        """Remove the images."""
        return LanderJax.CompressedState(
            step=state.step,
            source=state.source,
            ic_box=state.ic_box,
            state_kinetix=state.state_kinetix,
            fuel_left=state.fuel_left,
        )

    def step(self, state: State, action: jnp.ndarray) -> StepOutput:
        # Discrete -> Continuous.
        action = self.convert_action(action)

        # Need to artificially add in actions for the unused joint stuff.
        p = self.static_env_params
        action = jnp.concatenate([jnp.zeros([p.num_motor_bindings]), action])
        assert action.shape == (p.num_motor_bindings + p.num_thruster_bindings,)

        # If we are out of fuel, then force action=0.
        action = jnp.where(state.fuel_left <= 0.0, 0.0, action)

        # From just the bound actions to all motor (we don't have) + thruster actions.
        action_to_perform = self.action_type.process_action(action, state.state_kinetix, self.static_env_params)
        assert action_to_perform.shape == (3,)

        def _single_step(state_kin: EnvState, _):
            state_kin_next, mfolds = self.physics_engine.step(state_kin, self.env_params, action_to_perform)
            # Check for collisions.
            has_collided = self.has_collided(state_kin_next, mfolds)
            return state_kin_next, has_collided

        state_kin_final, has_collideds = jax.lax.scan(
            _single_step, state.state_kinetix, None, length=self.cfg.frame_skip
        )
        state_kin_final = state_kin_final.replace(timestep=state_kin_final.timestep + self.cfg.frame_skip)

        has_collided = jnp.any(has_collideds)

        fuel_used_now = jnp.abs(action).sum()
        # fuel_used = state.fuel_used + fuel_used_now
        fuel_left = state.fuel_left - fuel_used_now
        # Make sure we don't use more fuel than we have.
        fuel_left = jnp.maximum(fuel_left, 0.0)

        img_new = self.render.get_obs(state_kin_final).image

        with jdc.copy_and_mutate(state) as state_new:
            state_new.state_kinetix = state_kin_final
            state_new.fuel_left = fuel_left
            state_new.step = state.step + 1
            state_new.imgs.pop(0)
            state_new.imgs.append(img_new)

        is_unsafe = has_collided | self.is_unsafe(state_new)

        term = is_unsafe
        trunc = state_new.step >= self.cfg.max_steps
        rew = jnp.where(is_unsafe, -1.0, 0.0)

        info = {
            "has_collided": has_collided,
            "is_unsafe": is_unsafe,
            "action_to_perform": action_to_perform,
            "ic": state.ic_box,
        }

        # Check for termination.
        obs = self.get_obs(state_new)

        return StepOutput(state_new, obs, rew, term, trunc, info)

    def get_obs(self, state: State) -> LanderObs:
        assert len(state.imgs) == self.cfg.n_frame_stack
        # Stack along channel dim
        stacked_im = jnp.concatenate(state.imgs, axis=-1)
        assert stacked_im.shape == (64, 64, 3 * self.cfg.n_frame_stack)
        aux_info = self.get_aux_obs(state)
        return LanderObs(stacked_im, aux_info)

    def get_aux_obs(self, state: State) -> jnp.ndarray:
        fuel_left = state.fuel_left
        # Normalize to [-1, 1].
        fuel_left = jnp.clip(fuel_left / self.cfg.max_fuel, 0.0, 1.0) * 2.0 - 1.0
        return jnp.array([fuel_left])


class BaseLanderJaxState(BaseLanderJax):
    """Base class for lander with state observations."""

    @jdc.pytree_dataclass
    class State(BaseLanderJax.State):
        pass

    @classmethod
    def get_obs_treedef(cls):
        if not hasattr(cls, "obs_treedef"):
            tmp = LanderJax(LanderJax.Cfg())
            return tmp._obs_treedef()

        return cls.obs_treedef

    @classmethod
    def get_state_treedef(cls):
        if not hasattr(cls, "state_treedef"):
            tmp = LanderJax(LanderJax.Cfg())
            return tmp._state_treedef()

        return cls.state_treedef

    def step(self, state: State, action: jnp.ndarray) -> StepOutput:
        # Discrete -> Continuous.
        action = self.convert_action(action)

        # Need to artificially add in actions for the unused joint stuff.
        p = self.static_env_params
        action = jnp.concatenate([jnp.zeros([p.num_motor_bindings]), action])
        assert action.shape == (p.num_motor_bindings + p.num_thruster_bindings,)

        # If we are out of fuel, then force action=0.
        action = jnp.where(state.fuel_left <= 0.0, 0.0, action)

        # From just the bound actions to all motor (we don't have) + thruster actions.
        action_to_perform = self.action_type.process_action(action, state.state_kinetix, self.static_env_params)
        assert action_to_perform.shape == (3,)

        def _single_step(state_kin: EnvState, _):
            state_kin_next, mfolds = self.physics_engine.step(state_kin, self.env_params, action_to_perform)
            # Check for collisions.
            has_collided = self.has_collided(state_kin_next, mfolds)
            return state_kin_next, has_collided

        state_kin_final, has_collideds = jax.lax.scan(
            _single_step, state.state_kinetix, None, length=self.cfg.frame_skip
        )
        state_kin_final = state_kin_final.replace(timestep=state_kin_final.timestep + self.cfg.frame_skip)

        has_collided = jnp.any(has_collideds)

        fuel_used_now = jnp.abs(action).sum()
        # fuel_used = state.fuel_used + fuel_used_now
        fuel_left = state.fuel_left - fuel_used_now
        # Make sure we don't use more fuel than we have.
        fuel_left = jnp.maximum(fuel_left, 0.0)

        with jdc.copy_and_mutate(state) as state_new:
            state_new.state_kinetix = state_kin_final
            state_new.fuel_left = fuel_left
            state_new.step = state.step + 1

        is_unsafe = has_collided | self.is_unsafe(state_new)

        term = is_unsafe
        trunc = state_new.step >= self.cfg.max_steps
        rew = jnp.where(is_unsafe, -1.0, 0.0)

        info = {
            "has_collided": has_collided,
            "is_unsafe": is_unsafe,
            "action_to_perform": action_to_perform,
            "ic": state.ic_box,
        }

        obs = self.get_obs(state_new)

        return StepOutput(state_new, obs, rew, term, trunc, info)


class LanderJax(BaseLanderJaxImage):
    """2D Param space controlling initial position."""

    Cfg = TaskCfg

    State = BaseLanderJaxImage.State
    CompressedState = BaseLanderJaxImage.CompressedState

    def __init__(self, cfg: TaskCfg):
        json_path = pathlib.Path(__file__).parent / "lander_env.json"
        super().__init__(cfg, json_path)

    def get_ic_lims(self):
        px_lo, px_hi = 0.0, self.len_m
        py_lo, py_hi = 0.15 * self.len_m, self.len_m

        lim_lo = np.array([px_lo, py_lo])
        lim_hi = np.array([px_hi, py_hi])
        return lim_lo, lim_hi

    def pos_from_ic(self, uniform: np.ndarray) -> np.ndarray:
        # uniform: (..., 2)
        assert uniform.shape[-1:] == self.x0_unif_shape

        lim_lo, lim_hi = self.get_ic_lims()

        # Rescale from [-1, 1] to [lim_lo, lim_hi]
        vec01 = 0.5 * (uniform + 1.0)
        ic_vec = vec01 * (lim_hi - lim_lo) + lim_lo

        px, py = ic_vec[..., 0], ic_vec[..., 1]
        return np.stack([px, py], axis=-1)

    def reset_from_box(self, uniform: jnp.ndarray) -> State:
        # [goal_width_frac, goal_pos_frac, px, py, theta]
        # uniform is [-1, 1].
        lim_lo, lim_hi = self.get_ic_lims()
        assert lim_lo.shape == lim_hi.shape == uniform.shape

        # Rescale from [-1, 1] to [lim_lo, lim_hi]
        vec01 = 0.5 * (uniform + 1.0)
        ic_vec = vec01 * (lim_hi - lim_lo) + lim_lo

        # goal_width_frac, goal_pos_frac, px, py, theta = ic_vec
        # px = ic_vec[0]
        px, py = ic_vec

        goal_width_frac = 0.3  # Fixed
        goal_pos_frac = 0.05
        # py = 0.9 * self.len_m
        theta = 0.0

        level = self.level

        goal_width = self.len_m * goal_width_frac
        pos, vert = self.get_pos_and_verts(goal_width_frac, goal_pos_frac)

        agent_pos = jnp.array([px, py])
        poly_position = level.polygon.position.at[self.goal_pidx].set(pos).at[self.agent_pidx].set(agent_pos)
        poly_rotation = level.polygon.rotation.at[self.agent_pidx].set(theta)
        poly_vertices = level.polygon.vertices.at[self.goal_pidx].set(vert)
        polygon = level.polygon.replace(position=poly_position, rotation=poly_rotation, vertices=poly_vertices)

        # poly_position = level.polygon.position.at[self.goal_pidx].set(pos)
        # poly_vertices = level.polygon.vertices.at[self.goal_pidx].set(vert)
        # polygon = level.polygon.replace(position=poly_position, vertices=poly_vertices)
        #
        # circ_position = level.circle.position.at[self.agent_cidx].set(jnp.array([px, py]))
        # circ_rotation = level.circle.rotation.at[self.agent_cidx].set(theta)
        # circle = level.circle.replace(position=circ_position, rotation=circ_rotation)

        # Set the last circle to the right of the goal rectangle.
        radius_small_obs = 0.05
        small_obs_pos = jnp.array([pos[0] + goal_width / 2, radius_small_obs])
        circle_position = level.circle.position.at[-1].set(small_obs_pos)
        circle = level.circle.replace(position=circle_position)

        state_kin: EnvState = level.replace(polygon=polygon, circle=circle)
        # state_kin: EnvState = level.replace(polygon=polygon)

        # Update the (cached) global position of the thruster.
        update_fn = ft.partial(update_thruster_global_pos, state_kin, self.static_env_params)
        thruster_new = jax.vmap(update_fn)(state_kin.thruster)
        state_kin = state_kin.replace(thruster=thruster_new)

        # Render initial image
        pixel_obs = self.render.get_obs(state_kin)

        # Make sure each frame is a different buffer so when we donate it doesn't donate the same buffer multiple times
        imgs = [pixel_obs.image.copy() for _ in range(self.cfg.n_frame_stack)]
        fuel_left = self.cfg.max_fuel
        return LanderJax.State(0, Source.BASE, uniform, state_kin, fuel_left, imgs=imgs)

    def get_eval_states(self) -> EvalStateInfo:
        # A upside-down U shaped region. Start at the bottom left, then top left, then top right, then bottom right.
        num_per_side = 25
        frac = 2 * self.radius_circle / self.len_m
        b_px = np.linspace(-1, 1, num=num_per_side)
        b_py = np.linspace(-1 + frac, 1 - frac, num=num_per_side)

        b_ones = np.ones_like(b_py)

        left_side = np.stack([b_px[0] * b_ones, b_py], axis=-1)
        top_side = np.stack([b_px, b_py[-1] * b_ones], axis=-1)
        right_side = np.stack([b_px[-1] * b_ones, b_py[::-1]], axis=-1)
        b_ic = np.concatenate([left_side[:-1], top_side, right_side[1:]], axis=0)

        b_state = jax.jit(jax.vmap(self.reset_from_box))(b_ic)

        n_ics = len(b_ic)

        region_info = [
            EvalRegionInfo("left", num_per_side - 1),
            EvalRegionInfo("top", num_per_side),
            EvalRegionInfo("right", num_per_side - 1),
        ]

        return EvalStateInfo(b_state, n_ics, region_info)

    def reset_papereval(self, b_key: PRNGKey, num: int) -> State:
        n_ic = self.x0_unif_shape[0]
        sampler = qmc.Sobol(d=n_ic, scramble=False)
        base2_exp = int(np.ceil(np.log2(num)))
        assert num == 2**base2_exp

        # [0, 1)
        b_sample2d = sampler.random_base2(base2_exp)

        # [-1, 1). (n_samples, 2). []
        b_sample2d = 2 * b_sample2d - 1

        b_state = jax.jit(jax.vmap(self.reset_from_box))(b_sample2d)
        return b_state


class LanderJaxHard(BaseLanderJaxImage):
    """Very big param space controlling everything."""

    Cfg = TaskCfgHard

    State = BaseLanderJaxImage.State
    CompressedState = BaseLanderJaxImage.CompressedState

    def __init__(self, cfg: TaskCfg):
        json_path = pathlib.Path(__file__).parent / "lander_env_bigparam.json"
        super().__init__(cfg, json_path)

    def get_ic_lims(self):
        goal_width_lo, goal_width_hi = 0.1, 0.9
        goal_pos_lo, goal_pos_hi = 0.0, 1.0
        px_lo, px_hi = 0.0, self.len_m
        py_lo, py_hi = 0.15 * self.len_m, self.len_m
        theta_lo, theta_hi = -np.pi, np.pi

        # Position of the two obstacles.
        margin = 0.02
        obs_x_lo, obs_x_hi = margin * self.len_m, (1.0 - margin) * self.len_m
        obs_y_lo, obs_y_hi = margin * self.len_m, (1.0 - margin) * self.len_m

        lim_lo = np.array([px_lo, py_lo, theta_lo, goal_width_lo, goal_pos_lo, obs_x_lo, obs_y_lo, obs_x_lo, obs_y_lo])
        lim_hi = np.array([px_hi, py_hi, theta_hi, goal_width_hi, goal_pos_hi, obs_x_hi, obs_y_hi, obs_x_hi, obs_y_hi])
        return lim_lo, lim_hi

    def pos_from_ic(self, uniform: np.ndarray) -> np.ndarray:
        # uniform: (..., 2)
        assert uniform.shape[-1:] == self.x0_unif_shape

        lim_lo, lim_hi = self.get_ic_lims()

        # Rescale from [-1, 1] to [lim_lo, lim_hi]
        vec01 = 0.5 * (uniform + 1.0)
        ic_vec = vec01 * (lim_hi - lim_lo) + lim_lo

        px, py = ic_vec[..., 0], ic_vec[..., 1]
        return np.stack([px, py], axis=-1)

    def reset_from_box(self, uniform: jnp.ndarray) -> State:
        # uniform is [-1, 1].
        lim_lo, lim_hi = self.get_ic_lims()
        assert lim_lo.shape == lim_hi.shape == uniform.shape

        # Rescale from [-1, 1] to [lim_lo, lim_hi]
        vec01 = 0.5 * (uniform + 1.0)
        ic_vec = vec01 * (lim_hi - lim_lo) + lim_lo

        px, py, theta, goal_width_frac, goal_pos_frac, obs1_x, obs1_y, obs2_x, obs2_y = ic_vec

        goal_width = self.len_m * goal_width_frac
        pos, vert = self.get_pos_and_verts(goal_width_frac, goal_pos_frac)

        level = self.level

        agent_pos = jnp.array([px, py])
        poly_position = level.polygon.position.at[self.goal_pidx].set(pos).at[self.agent_pidx].set(agent_pos)
        poly_rotation = level.polygon.rotation.at[self.agent_pidx].set(theta)
        poly_vertices = level.polygon.vertices.at[self.goal_pidx].set(vert)
        polygon = level.polygon.replace(position=poly_position, rotation=poly_rotation, vertices=poly_vertices)

        obs1_pos = jnp.array([obs1_x, obs1_y])
        obs2_pos = jnp.array([obs2_x, obs2_y])
        circle_position = level.circle.position.at[0].set(obs1_pos).at[1].set(obs2_pos)
        circle = level.circle.replace(position=circle_position)
        state_kin: EnvState = level.replace(polygon=polygon, circle=circle)

        # Update the (cached) global position of the thruster.
        update_fn = ft.partial(update_thruster_global_pos, state_kin, self.static_env_params)
        thruster_new = jax.vmap(update_fn)(state_kin.thruster)
        state_kin = state_kin.replace(thruster=thruster_new)

        # Render initial image
        pixel_obs = self.render.get_obs(state_kin)

        # Make sure each frame is a different buffer so when we donate it doesn't donate the same buffer multiple times
        imgs = [pixel_obs.image.copy() for _ in range(self.cfg.n_frame_stack)]
        fuel_left = self.cfg.max_fuel
        return LanderJax.State(0, Source.BASE, uniform, state_kin, fuel_left, imgs=imgs)

    def get_eval_states(self) -> EvalStateInfo:
        n_total = 128
        key = jr.PRNGKey(12345)
        b_state = self.reset_papereval(key, n_total)
        region_info = [EvalRegionInfo("all", n_total)]
        return EvalStateInfo(b_state, n_total, region_info)

    def reset_papereval(self, b_key: PRNGKey, num: int) -> State:
        n_ic = self.x0_unif_shape[0]
        sampler = qmc.Sobol(d=n_ic, scramble=False)
        base2_exp = int(np.ceil(np.log2(num)))
        assert num == 2**base2_exp

        # [0, 1)
        b_sample2d = sampler.random_base2(base2_exp)

        # [-1, 1). (n_samples, 2). []
        b_sample2d = 2 * b_sample2d - 1

        b_state = jax.jit(jax.vmap(self.reset_from_box))(b_sample2d)
        b_is_unsafe = jax.jit(jax.vmap(self.is_unsafe_check_col))(b_state)
        n_unsafe = int(jnp.sum(b_is_unsafe))
        b_state_safe = jtu.tree_map(lambda x: x[~b_is_unsafe], b_state)

        # Sample n_unsafe more SAFE states to replace the unsafe ones.
        n_to_sample = 4 * num
        s_keys = jr.split(jr.PRNGKey(123456), n_to_sample)
        s_new_state = jax.jit(jax.vmap(self.reset))(s_keys)
        s_new_is_unsafe = jax.jit(jax.vmap(self.is_unsafe_check_col))(s_new_state)

        # Sort such that the safe ones come first.
        sort_idx = jnp.argsort(s_new_is_unsafe)
        s_new_state_sorted = jtu.tree_map(lambda x: x[sort_idx], s_new_state)

        s_state_safe_new = jtu.tree_map(lambda x: x[:n_unsafe], s_new_state_sorted)
        b_state = tree_cat([b_state_safe, s_state_safe_new], axis=0)

        b_is_unsafe_new = jax.jit(jax.vmap(self.is_unsafe_check_col))(b_state)
        n_unsafe_new = int(jnp.sum(b_is_unsafe_new))
        logger.info(
            "n_unsafe: {} ({:.4f}) -> {} ({:.4f})".format(n_unsafe, n_unsafe / num, n_unsafe_new, n_unsafe_new / num)
        )
        #
        # ipdb.set_trace()

        return b_state


class LanderJaxHardStateObs(BaseLanderJaxState):
    """Very big param space controlling everything. Uses state as observation instead of images."""

    Cfg = TaskCfgHard

    State = BaseLanderJaxState.State

    def __init__(self, cfg: TaskCfg):
        json_path = pathlib.Path(__file__).parent / "lander_env_bigparam.json"
        super().__init__(cfg, json_path)

    def get_obs(self, state: State):
        # static ICs in [-1, 1].
        _, _, _, goal_width, goal_pos, obs1_x, obs1_y, obs2_x, obs2_y = state.ic_box

        # agent info.
        state_kin = state.state_kinetix
        agent_pos = state_kin.polygon.position[self.agent_pidx]
        agent_rot = state_kin.polygon.rotation[self.agent_pidx]
        agent_vel = state_kin.polygon.velocity[self.agent_pidx]
        agent_angvel = state_kin.polygon.angular_velocity[self.agent_pidx]

        # fuel.
        fuel_left = state.fuel_left
        obs_fuel_left = jnp.clip(fuel_left / self.cfg.max_fuel, 0.0, 1.0) * 2.0 - 1.0
        obs_fuel_left = jnp.array([obs_fuel_left])

        obs_static_ic = jnp.array([goal_width, goal_pos, obs1_x, obs1_y, obs2_x, obs2_y])
        obs_agent = jnp.array(
            [
                agent_pos[0],
                agent_pos[1],
                jnp.cos(agent_rot),
                jnp.sin(agent_rot),
                agent_vel[0],
                agent_vel[1],
                agent_angvel,
            ]
        )

        vely_mean = -0.2
        velx_std = 0.4
        vely_std = 0.8

        angvelstd = 2.4

        # Normalize:
        obs_agent_mean = jnp.array([self.len_m / 2, self.len_m / 2, 0.0, 0.0, 0.0, vely_mean, 0.0])
        obs_agent_std = jnp.array([self.len_m / 2, self.len_m / 2, 1.0, 1.0, vely_std, velx_std, angvelstd])
        obs_agent = (obs_agent - obs_agent_mean) / obs_agent_std

        obs = jnp.concatenate([obs_static_ic, obs_agent, obs_fuel_left], axis=0)
        return obs

    def get_ic_lims(self):
        goal_width_lo, goal_width_hi = 0.1, 0.9
        goal_pos_lo, goal_pos_hi = 0.0, 1.0
        px_lo, px_hi = 0.0, self.len_m
        py_lo, py_hi = 0.15 * self.len_m, self.len_m
        theta_lo, theta_hi = -np.pi, np.pi

        # Position of the two obstacles.
        margin = 0.02
        obs_x_lo, obs_x_hi = margin * self.len_m, (1.0 - margin) * self.len_m
        obs_y_lo, obs_y_hi = margin * self.len_m, (1.0 - margin) * self.len_m

        lim_lo = np.array([px_lo, py_lo, theta_lo, goal_width_lo, goal_pos_lo, obs_x_lo, obs_y_lo, obs_x_lo, obs_y_lo])
        lim_hi = np.array([px_hi, py_hi, theta_hi, goal_width_hi, goal_pos_hi, obs_x_hi, obs_y_hi, obs_x_hi, obs_y_hi])
        return lim_lo, lim_hi

    def pos_from_ic(self, uniform: np.ndarray) -> np.ndarray:
        # uniform: (..., 2)
        assert uniform.shape[-1:] == self.x0_unif_shape

        lim_lo, lim_hi = self.get_ic_lims()

        # Rescale from [-1, 1] to [lim_lo, lim_hi]
        vec01 = 0.5 * (uniform + 1.0)
        ic_vec = vec01 * (lim_hi - lim_lo) + lim_lo

        px, py = ic_vec[..., 0], ic_vec[..., 1]
        return np.stack([px, py], axis=-1)

    def reset_from_box(self, uniform: jnp.ndarray) -> State:
        # uniform is [-1, 1].
        lim_lo, lim_hi = self.get_ic_lims()
        assert lim_lo.shape == lim_hi.shape == uniform.shape

        # Rescale from [-1, 1] to [lim_lo, lim_hi]
        vec01 = 0.5 * (uniform + 1.0)
        ic_vec = vec01 * (lim_hi - lim_lo) + lim_lo

        px, py, theta, goal_width_frac, goal_pos_frac, obs1_x, obs1_y, obs2_x, obs2_y = ic_vec

        goal_width = self.len_m * goal_width_frac
        pos, vert = self.get_pos_and_verts(goal_width_frac, goal_pos_frac)

        level = self.level

        agent_pos = jnp.array([px, py])
        poly_position = level.polygon.position.at[self.goal_pidx].set(pos).at[self.agent_pidx].set(agent_pos)
        poly_rotation = level.polygon.rotation.at[self.agent_pidx].set(theta)
        poly_vertices = level.polygon.vertices.at[self.goal_pidx].set(vert)
        polygon = level.polygon.replace(position=poly_position, rotation=poly_rotation, vertices=poly_vertices)

        obs1_pos = jnp.array([obs1_x, obs1_y])
        obs2_pos = jnp.array([obs2_x, obs2_y])
        circle_position = level.circle.position.at[0].set(obs1_pos).at[1].set(obs2_pos)
        circle = level.circle.replace(position=circle_position)
        state_kin: EnvState = level.replace(polygon=polygon, circle=circle)

        # Update the (cached) global position of the thruster.
        update_fn = ft.partial(update_thruster_global_pos, state_kin, self.static_env_params)
        thruster_new = jax.vmap(update_fn)(state_kin.thruster)
        state_kin = state_kin.replace(thruster=thruster_new)

        # Render initial image
        pixel_obs = self.render.get_obs(state_kin)

        fuel_left = self.cfg.max_fuel
        return LanderJaxHardStateObs.State(0, Source.BASE, uniform, state_kin, fuel_left)

    def get_eval_states(self) -> EvalStateInfo:
        n_total = 128
        key = jr.PRNGKey(12345)
        b_state = self.reset_papereval(key, n_total)
        region_info = [EvalRegionInfo("all", n_total)]
        return EvalStateInfo(b_state, n_total, region_info)

    def reset_papereval(self, b_key: PRNGKey, num: int) -> State:
        n_ic = self.x0_unif_shape[0]
        sampler = qmc.Sobol(d=n_ic, scramble=False)
        base2_exp = int(np.ceil(np.log2(num)))
        assert num == 2**base2_exp

        # [0, 1)
        b_sample2d = sampler.random_base2(base2_exp)

        # [-1, 1). (n_samples, 2). []
        b_sample2d = 2 * b_sample2d - 1

        b_state = jax.jit(jax.vmap(self.reset_from_box))(b_sample2d)
        return b_state


def render_traj(rollout: RolloutOutput, path: pathlib.Path):
    T_obs_now: LanderObs = rollout.T_obs_now

    if not isinstance(T_obs_now, LanderObs):
        treedef = LanderJax.get_obs_treedef()
        T_obs_now: LanderObs = jtu.tree_unflatten(treedef, T_obs_now)

    assert isinstance(T_obs_now, LanderObs)

    T_stackimg_now = T_obs_now.img  # (W, RH, n_stack * 3)
    # (Get the latest image only)
    T_img_now = np.array(T_stackimg_now[..., -3:])  # (T, W, RH, 3)

    # Convert to H, W, C. Convert origin in bottom left to top left.
    T_img_now = ei.rearrange(T_img_now, "T w rh c -> T rh w c")[:, ::-1, :, :]

    font = ImageFont.load_default(size=32)

    def process_(kk: np.ndarray, im: np.ndarray):
        if im.dtype == np.float32:
            im = np.asarray(im * 255).astype(np.uint8)

        assert im.dtype == np.uint8

        # nearest neighbor.
        im = ei.repeat(im, "h w c -> (h dh) (w dw) c", dh=4, dw=4)

        assert im.shape[0] == im.shape[1]

        # Draw the step number in the top left corner.
        im = Image.fromarray(im)
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), "k={:03}".format(kk), font=font, fill=(255, 0, 0))
        return np.array(im)

    writer = None
    fps = 30

    for kk, im in enumerate(T_img_now):
        frame_rgb = process_(kk, im)  # RGB, HxWx3 uint8

        if writer is None:
            h, w, _ = frame_rgb.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "avc1"/"H264" depending on your setup
            writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

        # OpenCV expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    if writer is not None:
        writer.release()

    logger.success("Saved video to {}".format(path))


def render_trajs_sidebyside(
    rollouts: list[RolloutOutput],
    path: pathlib.Path,
    border_px: int = 4,
):
    """Stack the frames side by side. If lengths differ, copy the last frame.
    Inserts a vertical black border of width `border_px` between frames.
    """
    if not rollouts:
        raise ValueError("render_trajs_sidebyside requires at least one rollout.")

    # ---------------------------
    # Convert rollouts → image sequences
    # ---------------------------
    def rollout_to_imgs(rollout: RolloutOutput) -> np.ndarray:
        T_obs_now = rollout.T_obs_now

        if not isinstance(T_obs_now, LanderObs):
            treedef = LanderJax.get_obs_treedef()
            T_obs_now = jtu.tree_unflatten(treedef, T_obs_now)

        T_stackimg_now = T_obs_now.img
        T_img_now = np.array(T_stackimg_now[..., -3:])  # latest RGB only
        T_img_now = ei.rearrange(T_img_now, "T w rh c -> T rh w c")[:, ::-1, :, :]
        return T_img_now  # (T, H, W, 3)

    all_T_imgs = [rollout_to_imgs(r) for r in rollouts]
    max_T = max(imgs.shape[0] for imgs in all_T_imgs)

    def get_fuel_left(rollout: RolloutOutput) -> np.ndarray:
        T_state_now = rollout.T_state_now
        if not isinstance(T_state_now, LanderJax.State):
            state_treedef = LanderJax.get_state_treedef()
            T_state_now = jtu.tree_unflatten(state_treedef, T_state_now)

        return T_state_now.fuel_left

    fuel_lefts = [get_fuel_left(r) for r in rollouts]

    def get_actions(rollout: RolloutOutput) -> np.ndarray:
        T_action_to_perform = rollout.T_info["action_to_perform"]
        return T_action_to_perform[:, 1:]

    T_actions = [get_actions(r) for r in rollouts]

    # ---------------------------
    # Frame processing
    # ---------------------------
    font = ImageFont.load_default(size=16)

    def process_single(kk: int, fuel_left: float, action: np.ndarray, is_dead: bool, im: np.ndarray) -> np.ndarray:
        if im.dtype == np.float32:
            im = (im * 255).astype(np.uint8)

        # nearest-neighbor “4×” enlargement
        im = ei.repeat(im, "h w c -> (h dh) (w dw) c", dh=4, dw=4)

        H, W, C = im.shape

        text = f"k={kk:03} | F={fuel_left:.1f} | A=[{action[0]:.1f}, {action[1]:.1f}]"

        # Use a tiny dummy image just to measure text size
        tmp_img = Image.new("RGB", (1, 1))
        tmp_draw = ImageDraw.Draw(tmp_img)
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]

        # Padding above the image: text height + some margin
        margin_top = 5
        margin_bottom = 10
        pad_top = text_h + margin_top + margin_bottom

        # Create a new canvas with extra space on top (black background)
        canvas = np.zeros((H + pad_top, W, C), dtype=np.uint8)
        # Paste original image at the bottom
        canvas[pad_top:, :, :] = im

        # Draw text in the padded area
        pil_im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_im)

        text_x = 10  # left margin
        # center text vertically in the padding region
        text_y = (pad_top - text_h) // 2

        if is_dead:
            color = (255, 30, 30)
        else:
            color = (255, 255, 255)  # white text

        draw.text((text_x, text_y), text, font=font, fill=color)

        return np.array(pil_im)

    writer = None
    fps = 30

    # Precompute the black border (will adjust height after first frame)
    border = None

    # ---------------------------
    # Combine + write frames
    # ---------------------------
    for kk in range(max_T):
        processed_frames = []
        for rollout_idx, imgs in enumerate(all_T_imgs):
            idx = min(kk, imgs.shape[0] - 1)
            fuel_left = float(fuel_lefts[rollout_idx][idx])
            action = T_actions[rollout_idx][idx]
            is_dead = kk >= imgs.shape[0] - 1
            processed_frames.append(process_single(idx, fuel_left, action, is_dead, imgs[idx]))

        # Height must match; verified by processing
        H = processed_frames[0].shape[0]
        C = processed_frames[0].shape[2]

        # Create black vertical border once we know H
        if border is None:
            border = np.zeros((H, border_px, C), dtype=np.uint8)

        # Build: F0 | border | F1 | border | … | FN
        frame_list = []
        for i, fr in enumerate(processed_frames):
            frame_list.append(fr)
            if i < len(processed_frames) - 1:
                frame_list.append(border)

        frame_rgb = np.concatenate(frame_list, axis=1)

        # Initialize writer
        if writer is None:
            h, w, _ = frame_rgb.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()

    logger.success(f"Saved side-by-side video to {path}")
