from typing import Iterable

import gymnasium as gym
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.stats import truncnorm
from shapely.geometry import Point

# @jdc.pytree_dataclass
# class TaskCfg:
#     max_timesteps: int = 1000  # Maximum number of timesteps per episode
#
#     num_lanes: int = 2  # Number of lanes
#     num_vehicles: int = 2  # Number of other vehicles
#     ego_min_vel: float = 1.0  # Ego vehicle minimum velocity
#     ego_max_vel: float = 4.0  # Ego vehicle maximum velocity
#     # All other velocities are angular velocities
#     other_min_ang_vel: float = np.pi / 32  # Vehicle minimum velocity
#     other_max_ang_vel: float = 11 * np.pi / 40  # Vehicle maximum velocity
#
#     vehicle_radius: float = 0.5  # Radius of the vehicle
#     track_radius: float = 5.0  # Radius of the inner track
#     lane_offset: float = 0.25  # Offset between lanes and vehicles
#     dt: float = 0.1  # Time step for the simulation
#     render_mode: str = 'rgb_array'
#
#     # For Debugging purposes
#     fix_other_vel: str = 'normal'  # "normal", "min", "max", "mid"
#
#     verbose: int = 0


@jdc.pytree_dataclass
class TaskCfg:
    max_timesteps: int = 1_000  # Maximum number of timesteps per episode
    num_lanes: int = 2  # Number of lanes
    num_vehicles: int = 2  # Number of other vehicles
    ego_min_vel: float = 1.0  # Ego vehicle minimum velocity
    ego_max_vel: float = 4.0  # Ego vehicle maximum velocity
    # All other velocities are angular velocities
    other_min_ang_vel: float = np.pi / 32  # Vehicle minimum velocity
    other_max_ang_vel: float = 11 * np.pi / 40  # Vehicle maximum velocity
    rng_seed: int = 12345  # Random seed for reproducibility

    vehicle_radius: float = 0.5  # Radius of the vehicle
    o_vehicle_len_range: tuple[float, float] = (0.6, 2.0)  # Length range of other vehicles

    track_radius: float = 5.0  # Radius of the inner track
    lane_offset: float = 0.25  # Offset between lanes and vehicles
    dt: float = 0.1  # Time step for the simulation
    render_mode: str = "rgb_array"

    # For Debugging purposes
    fix_other_vel: str = "normal"  # "normal", "min", "max", "mid"

    pv0: Iterable[float] = None  # px0.shape[0] == num_vehicles

    verbose: int = 0

    @property
    def track_center(self):
        return np.array([0.0, 0.0])

    @property
    def lane_width(self):
        return 2 * self.vehicle_radius + 2 * self.lane_offset

    @property
    def track_inner_radius(self):
        return self.track_radius

    @property
    def track_outer_radius(self):
        return self.track_radius + self.lane_width * self.num_lanes


class DubinsCircularTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, task_cfg: TaskCfg):
        super().__init__()
        self.task_cfg = task_cfg

        # Debug
        if self.task_cfg.fix_other_vel != "normal":
            logger.critical(
                f"Fixing other vehicle velocity to {self.task_cfg.fix_other_vel} for debugging purposes. Are you sure?"
            )

        # Define observation and action spaces

        obs_dim = 4 * (1 + task_cfg.num_vehicles)
        high = np.array([np.inf] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(obs_dim,), dtype=np.float32
        )
        # Action space is continuous with 2 dimensions: steering and acceleration
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.lane_width = 2 * task_cfg.vehicle_radius + 2 * self.task_cfg.lane_offset
        self.track_center = np.array([0.0, 0.0])
        self.track_inner_radius = self.task_cfg.track_radius
        self.track_outer_radius = self.task_cfg.track_radius + self.lane_width * (
            self.task_cfg.num_lanes
        )

        self.track_boundaries = [
            Point(0, 0).buffer(self.track_inner_radius).boundary,
            Point(0, 0).buffer(self.track_outer_radius).boundary,
        ]

        self.lane_boundaries = []
        for i in range(1, self.task_cfg.num_lanes):
            lane_radius = self.task_cfg.track_radius + i * self.lane_width
            lane = Point(0, 0).buffer(lane_radius)
            self.lane_boundaries.append(lane.boundary)

        # Compute the lane radiuses for resetting
        self.lane_radii = [
            self.track_inner_radius + 0.5 * self.lane_width + i * self.lane_width
            for i in range(self.task_cfg.num_lanes)
        ]

        self.state = None  # Ego state: [x, y, theta, v]
        self.other_cars = None

        self.timestep = 0

        # # Set up resetting logic
        # self.other_min_ang_vel = self.task_cfg.other_min_ang_vel
        # self.other_max_ang_vel = self.task_cfg.other_max_ang_vel
        # self.vel_mu = (self.task_cfg.other_min_ang_vel + self.task_cfg.other_max_ang_vel) / 2
        # self.vel_sigma = (self.task_cfg.other_max_ang_vel - self.task_cfg.other_min_ang_vel) / 6
        # a = (self.task_cfg.other_min_ang_vel - self.vel_mu) / self.vel_sigma
        # b = (self.task_cfg.other_max_ang_vel - self.vel_mu) / self.vel_sigma
        # self.other_vel_dist = truncnorm(a, b, loc=self.vel_mu, scale=self.vel_sigma)
        # self.other_vel_dist_rng = np.random.default_rng(self.task_cfg.rng_seed)

        # Rendering
        self.fig, self.ax = None, None
        self.canvas = None
        self._background = None
        self._blit_initialized = False
        self._text_annotation = None
        self._car_patches = []

        self.reset()

    def reg_to_ang_v(self, v, r):
        r = (self.track_inner_radius + self.track_outer_radius) / 2
        return v / r
        # return v / r

    def ang_to_reg_v(self, omega, r):
        r = (self.track_inner_radius + self.track_outer_radius) / 2
        return omega * r
        # return omega * r

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = 0
        self.state = self._ego_s0()
        self.other_cars = self._o_s0()

        return self._get_obs(), self._get_info()

    def _ego_s0(self):
        lane = 0  # Ego starts in the inner lane
        angle = 0  # Ego starts on right side
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        x = self.lane_radii[lane] * np.cos(angle)
        y = self.lane_radii[lane] * np.sin(angle)
        theta = (angle + np.pi / 2) % (2 * np.pi)
        v = (self.task_cfg.ego_min_vel + self.task_cfg.ego_max_vel) / 2  # Midpoint
        return np.array([x, y, theta, v], dtype=np.float32)

    def _o_s0(self, pv0: Iterable[float] = None):
        cars = []
        for i in range(self.task_cfg.num_vehicles):
            # angle = np.pi + i * (np.pi / 8)
            angle = np.pi + i * (np.pi / 4)

            angle = (angle + 2 * np.pi) % (2 * np.pi)
            lane = i % self.task_cfg.num_lanes  # Each additional car is one lane over
            x = self.lane_radii[lane] * np.cos(angle)
            y = self.lane_radii[lane] * np.sin(angle)
            theta = (angle + np.pi / 2) % (2 * np.pi)
            if pv0 is not None:
                v = pv0[i]
            else:
                match self.task_cfg.fix_other_vel:
                    case "normal":
                        # v = self.other_vel_dist.rvs(size=1, random_state=self.other_vel_dist_rng)[0]
                        v = self.np_random.uniform(
                            self.task_cfg.other_min_ang_vel,
                            self.task_cfg.other_max_ang_vel,
                        )
                    case "min":
                        v = self.task_cfg.other_min_ang_vel
                    case "max":
                        v = self.task_cfg.other_max_ang_vel
                    case "mid":
                        v = (
                            self.task_cfg.other_min_ang_vel
                            + self.task_cfg.other_max_ang_vel
                        ) / 2
                    case _:
                        v = self.np_random.uniform(
                            self.task_cfg.other_min_ang_vel,
                            self.task_cfg.other_max_ang_vel / 2,
                        )
            cars.append([x, y, theta, v])
        return np.array(cars, dtype=np.float32)

    def _get_obs(self):
        obs = []
        x, y, theta, v = self.state

        r = np.sqrt(x**2 + y**2)
        # Get r in [0,1] where 0 is the inner track and 1 is the outer track
        norm_r = (r - self.track_inner_radius) / (
            self.track_outer_radius - self.track_inner_radius
        )

        # Compute ego's global angle on track
        angle = np.arctan2(y, x)
        angle = (angle + 2 * np.pi) % (2 * np.pi)

        # Normalize the velocity between [0, 1] where 1 is the max velocity and 0 is the min velocity
        norm_v = (v - self.task_cfg.ego_min_vel) / (
            self.task_cfg.ego_max_vel - self.task_cfg.ego_min_vel
        )

        # Get the theta relative to the tangent of the track
        tangent_vector = np.array([-y, x])
        norm_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        tangent_angle = np.arctan2(norm_tangent_vector[1], norm_tangent_vector[0])
        tangent_angle = (tangent_angle + 2 * np.pi) % (2 * np.pi)

        # Get relative theta based on cos and sin
        rel_cos_theta = np.cos(tangent_angle - theta)
        rel_sin_theta = np.sin(tangent_angle - theta)

        obs.append(rel_cos_theta)
        obs.append(rel_sin_theta)
        obs.append(norm_r)
        obs.append(norm_v)

        for oc in self.other_cars:
            x_o, y_o, theta_o, v_o = oc
            r_o = np.sqrt(x_o**2 + y_o**2)
            norm_r_o = (r_o - self.track_inner_radius) / (
                self.track_outer_radius - self.track_inner_radius
            )

            angle_o = np.arctan2(y_o, x_o)
            angle_o = (angle_o + 2 * np.pi) % (2 * np.pi)

            rel_angle = angle_o - angle
            cos_o = np.cos(rel_angle)
            sin_o = np.sin(rel_angle)

            norm_v_o = (v_o - self.task_cfg.other_min_ang_vel) / (
                self.task_cfg.other_max_ang_vel - self.task_cfg.other_min_ang_vel
            )

            rel_norm_r_o = norm_r_o - norm_r
            rel_norm_v = norm_v_o - norm_v

            obs.append(cos_o)
            obs.append(sin_o)
            obs.append(rel_norm_r_o)
            obs.append(rel_norm_v)

        return np.array(obs)

    def _get_info(self):
        return {
            "cost": int(self._oob() or self._check_collide()),
        }

    def step(self, action):
        assert self.action_space.contains(action), f"{action} not in action space"
        x, y, theta, v = self.state
        steering, acc = action

        self.timestep += 1

        # Ego agent update using regular velocity
        turning_radius = (self.track_inner_radius + self.track_outer_radius) / 2
        max_turn_rate = v / turning_radius
        max_turn_rate *= 2  # Otherwise too small
        omega = steering * max_turn_rate
        v = np.clip(
            v + acc * self.task_cfg.dt,
            self.task_cfg.ego_min_vel,
            self.task_cfg.ego_max_vel,
        )
        theta += omega * self.task_cfg.dt
        theta = (theta + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2π)
        x += v * np.cos(theta) * self.task_cfg.dt
        y += v * np.sin(theta) * self.task_cfg.dt
        self.state = np.array([x, y, theta, v], dtype=np.float32)

        # Other cars update
        for i, (x_o, y_o, theta_o, v_o) in enumerate(self.other_cars):

            # Angular velocity
            r_o = np.linalg.norm([x_o, y_o])
            angle_o = np.arctan2(y_o, x_o)
            angle_o = (angle_o + 2 * np.pi) % (2 * np.pi)
            angle_o = (angle_o + v_o * self.task_cfg.dt) % (2 * np.pi)
            x_o = r_o * np.cos(angle_o)
            y_o = r_o * np.sin(angle_o)
            theta_o = (angle_o + np.pi / 2) % (2 * np.pi)
            self.other_cars[i] = [x_o, y_o, theta_o, v_o]

        term = self._oob() or self._check_collide()
        trunc = self._timeout()
        reward = -int(term)

        return self._get_obs(), reward, term, trunc, self._get_info()

    def setup_trajplot(self, ax, term=None):
        # Initialize blitting if needed
        fig = ax.figure
        # canvas = FigureCanvas(fig)
        ax.set_aspect("equal")
        lim = self.track_outer_radius + self.lane_width
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        self._draw_track(ax)
        self._init_text_annotation(ax)

        # Update texts
        if term is None:
            cost = self._get_info()["cost"]
        else:
            cost = int(term)
        text_color = "red" if cost > 0 else "green"
        text_str = f"Timestep: {self.timestep} \n Cost: {cost}"
        self._text_annotation.set_text(text_str)
        self._text_annotation.set_color(text_color)

        ax.draw_artist(self._text_annotation)

        for body, arrow in self._car_patches:
            ax.draw_artist(body)
            ax.draw_artist(arrow)

        canvas = fig.canvas
        canvas.draw()
        return np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
            canvas.get_width_height()[::-1] + (4,)
        )[:, :, :3]

    def _timeout(self):
        return self.timestep >= self.task_cfg.max_timesteps

    def _oob(self):
        """
        Check out of bounds
        """
        ego_geom = Point(self.state[0], self.state[1]).buffer(
            self.task_cfg.vehicle_radius
        )
        for boundary in self.track_boundaries:
            if ego_geom.intersects(boundary):
                return True
        return False

    def _check_collide(self):
        """
        Check collision with other cars
        """
        if self.task_cfg.num_vehicles == 0:
            return False

        rad = self.task_cfg.vehicle_radius
        dists = np.linalg.norm(self.state[:2] - self.other_cars[:, :2], ord=2, axis=-1)
        if dists.min() < 2 * rad:
            return True

        return False

    def _init_text_annotation(self, ax=None):
        if ax is None:
            ax = self.ax
        cost = self._get_info()["cost"]
        text_color = "red" if cost > 0 else "black"
        text_str = f"Timestep: {self.timestep} \n Cost: {cost}"
        self._text_annotation = ax.text(
            0.5,
            1.01,
            text_str,
            color=text_color,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=12,
            weight="bold",
        )

    def render(self):
        if self.task_cfg.render_mode not in ["human", "rgb_array"]:
            return

        # Initialize blitting if needed
        if not self._blit_initialized:
            self.fig, self.ax = plt.subplots()
            self.canvas = FigureCanvas(self.fig)
            self.ax.set_aspect("equal")
            lim = self.track_outer_radius + self.lane_width
            self.ax.set_xlim(-lim, lim)
            self.ax.set_ylim(-lim, lim)
            self._draw_track()
            self._init_car_patches()
            self._init_text_annotation()
            self.fig.tight_layout()
            self.canvas.draw()
            self._background = self.canvas.copy_from_bbox(self.ax.bbox)
            self._blit_initialized = True

        self.canvas.restore_region(self._background)
        self._update_car_patches()

        # Update texts
        cost = self._get_info()["cost"]
        text_color = "red" if cost > 0 else "black"
        text_str = f"Timestep: {self.timestep} \n Cost: {cost}"
        self._text_annotation.set_text(text_str)
        self._text_annotation.set_color(text_color)

        self.ax.draw_artist(self._text_annotation)

        for body, arrow in self._car_patches:
            self.ax.draw_artist(body)
            self.ax.draw_artist(arrow)
        self.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()

        if self.task_cfg.render_mode == "human":
            plt.pause(0.01)
        elif self.task_cfg.render_mode == "rgb_array":
            self.canvas.draw()
            return np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape(
                self.canvas.get_width_height()[::-1] + (4,)
            )[:, :, :3]

    def _draw_track(self, ax=None):
        if ax is None:
            ax = self.ax
        inner = plt.Circle(
            self.track_center,
            self.track_inner_radius,
            color="black",
            fill=False,
            linewidth=2,
            zorder=1,
        )
        outer = plt.Circle(
            self.track_center,
            self.track_outer_radius,
            color="black",
            fill=False,
            linewidth=2,
            zorder=1,
        )
        ax.add_patch(inner)
        ax.add_patch(outer)

        for lane_boundary in self.lane_boundaries:
            ax.plot(
                *lane_boundary.xy, color="gray", linestyle="--", linewidth=1, zorder=1
            )

    def _init_car_patches(self):
        self._car_patches = []
        all_cars = [self.state] + list(self.other_cars)
        colors = ["blue"] + ["gray"] * self.task_cfg.num_vehicles
        for color in colors:
            body = Circle(
                (0, 0),
                self.task_cfg.vehicle_radius,
                color=color,
                fill=True,
                zorder=2,
                alpha=0.5,
            )
            arrow = FancyArrowPatch(
                (0, 0),
                (0, 0),
                arrowstyle="->",
                mutation_scale=10,
                color="black",
                zorder=3,
            )
            self.ax.add_patch(body)
            self.ax.add_patch(arrow)
            self._car_patches.append((body, arrow))

    def _update_car_patches(self):
        all_cars = [self.state] + list(self.other_cars)
        for i, ((body, arrow), car) in enumerate(zip(self._car_patches, all_cars)):
            x, y, theta, v = car
            if i == 0:
                # Ego agent uses regular velocity
                dx = 0.4 * v * np.cos(theta)
                dy = 0.4 * v * np.sin(theta)
            else:
                # Other cars have angular velocity
                reg_v = v * (self.track_inner_radius + self.track_outer_radius) / 2
                dx = 0.4 * reg_v * np.cos(theta)
                dy = 0.4 * reg_v * np.sin(theta)
            body.center = (x, y)
            arrow.set_positions((x, y), (x + dx, y + dy))

        collide, oob = self._check_collide(), self._oob()
        if collide or oob:
            self._car_patches[0][0].set_color("purple")
            self._car_patches[0][0].set_alpha(0.8)
            if collide:
                # Set the body of the two colliding cars to red
                dists = np.linalg.norm(
                    self.state[:2] - self.other_cars[:, :2], ord=2, axis=-1
                )
                rad = self.task_cfg.vehicle_radius
                # Get all dists that are smaller than 2 * rad
                for i, dist in enumerate(dists):
                    if dist < 2 * rad:
                        self._car_patches[i + 1][0].set_color("maroon")
                        self._car_patches[i + 1][0].set_alpha(0.8)
        else:
            self._car_patches[0][0].set_color("blue")
            self._car_patches[0][0].set_alpha(0.5)
            for i in range(1, len(self._car_patches)):
                self._car_patches[i][0].set_color("gray")
                self._car_patches[i][0].set_alpha(0.5)

    def close(self):
        if self.fig:
            plt.close(self.fig)

    def label_ic(self, ax: plt.Axes):
        ax.set_xlabel("px0")

        o_minv, o_maxv = (
            self.task_cfg.other_min_ang_vel,
            self.task_cfg.other_max_ang_vel,
        )

        # Draw a rectangle around the area.
        rect = plt.Rectangle(
            (o_minv, o_minv),
            o_maxv - o_minv,
            o_maxv - o_minv,
            linewidth=1,
            edgecolor="C3",
            facecolor="none",
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(rect)

        ax.set_aspect("equal")

        # # Label the sections.
        # for x in self.boundaries:
        #     ax.axvline(x, color="C3", linestyle="--", alpha=0.5)
