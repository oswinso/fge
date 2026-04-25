import gymnasium as gym
import jax_dataclasses as jdc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box, Discrete
from loguru import logger
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon


@jdc.pytree_dataclass
class TaskCfg:
    # On both x and y axes
    env_xlb: int = 0
    env_xub: int = 50
    env_ylb: int = 0
    env_yub: int = 50

    # Define the level walls
    easy_xlb: int = 0
    easy_xub: int = 35
    hard_xlb: int = 35
    hard_xub: int = 45
    impossible_xlb: int = 45
    impossible_xub: int = 50

    # Define level reset distributions
    # Normal
    easy_prob: float = 0.90
    hard_prob: float = 0.0001
    impossible_prob: float = 0.0999

    # Define goal region
    goal_x_lb: int = 0
    goal_x_ub: int = 50
    goal_y_lb: int = 0
    goal_y_ub: int = 1

    # Define disturb region
    disturb_x_lb: int = 35
    disturb_x_ub: int = 45
    disturb_y_lb: int = 5
    disturb_y_ub: int = 15

    # Define avoid set as a triangle. The ylb and ylb is given from the disturb region
    # And x_lb gives the left point of the triangle
    avoid_x_lb: int = 38
    add_cost_to_reward: bool = False

    # Define an obstacle for the impossible region
    obstacle_x_lb: int = 45
    obstacle_x_ub: int = 50
    obstacle_y_lb: int = 1
    obstacle_y_ub: int = 2.5

    agent_radius: float = 0.25
    # agent_drop_rate: float = 0.1
    agent_drop_rate: float = 0.25

    # max_steps: int = 100
    max_steps: int = 300

    continuous: bool = False


class ToyLevels(gym.Env):
    """
    Test exploration of different exploration algorithms.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, task_cfg: TaskCfg, paths, render_mode=None, **kwargs):
        # matplotlib.use("qtagg")
        self.task_cfg = task_cfg
        self.paths = paths
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        assert task_cfg.easy_prob + task_cfg.hard_prob + task_cfg.impossible_prob == 1

        self.observation_space = Box(
            low=np.array([self.task_cfg.env_xlb, self.task_cfg.env_ylb]),
            high=np.array([self.task_cfg.env_xub, self.task_cfg.env_yub]),
            shape=(2,),
            dtype=np.float32,
        )

        self.agent = None
        # Step size is the slope of the avoid triangle
        x1 = self.task_cfg.disturb_x_ub - self.task_cfg.avoid_x_lb
        y2 = self.task_cfg.agent_drop_rate
        y1 = self.task_cfg.disturb_y_ub - self.task_cfg.disturb_y_lb
        self.agent_stepsize = (x1 * y2) / y1
        logger.info(f"Agent stepsize: {self.agent_stepsize}")

        if self.task_cfg.continuous:
            # Only move left or right
            self.action_space = Box(
                low=np.array([-self.agent_stepsize]),
                high=np.array([self.agent_stepsize]),
                shape=(1,),
                dtype=np.float64,
            )
            self.continuous_action_space = True
        else:
            self.action_space = Discrete(
                n=3,  # left, right, nothing
            )
            self.continuous_action_space = False

        self.goal_box = Polygon(
            [
                (self.task_cfg.goal_x_lb, self.task_cfg.goal_y_lb),
                (self.task_cfg.goal_x_ub, self.task_cfg.goal_y_lb),
                (self.task_cfg.goal_x_ub, self.task_cfg.goal_y_ub),
                (self.task_cfg.goal_x_lb, self.task_cfg.goal_y_ub),
            ]
        )
        self.disturb_box = Polygon(
            [
                (self.task_cfg.disturb_x_lb - 1, self.task_cfg.disturb_y_lb),
                (self.task_cfg.disturb_x_ub + 1, self.task_cfg.disturb_y_lb),
                (self.task_cfg.disturb_x_ub + 1, self.task_cfg.disturb_y_ub),
                (self.task_cfg.disturb_x_lb - 1, self.task_cfg.disturb_y_ub),
            ]
        )
        self.wall1 = LineString(
            [
                (self.task_cfg.easy_xub, self.task_cfg.goal_y_ub),
                (self.task_cfg.easy_xub, self.task_cfg.env_yub),
            ]
        )
        self.wall2 = LineString(
            [
                (self.task_cfg.hard_xub, self.task_cfg.goal_y_ub),
                (self.task_cfg.hard_xub, self.task_cfg.env_yub),
            ]
        )
        self.imp_obs = Polygon(
            [
                (self.task_cfg.obstacle_x_lb, self.task_cfg.obstacle_y_lb),
                (self.task_cfg.obstacle_x_ub, self.task_cfg.obstacle_y_lb),
                (self.task_cfg.obstacle_x_ub, self.task_cfg.obstacle_y_ub),
                (self.task_cfg.obstacle_x_lb, self.task_cfg.obstacle_y_ub),
            ]
        )

        # Define the reset regions for easy, hard, and impossible
        self.easy_reset_region = (
            self.task_cfg.easy_xlb
            + 0.1 * (self.task_cfg.easy_xub - self.task_cfg.easy_xlb),
            self.task_cfg.easy_xub
            - 0.1 * (self.task_cfg.easy_xub - self.task_cfg.easy_xlb),
        )
        self.hard_reset_region = (
            self.task_cfg.hard_xlb
            + 0.1 * (self.task_cfg.hard_xub - self.task_cfg.hard_xlb),
            self.task_cfg.hard_xub
            - 0.1 * (self.task_cfg.hard_xub - self.task_cfg.hard_xlb),
        )
        self.impossible_reset_region = (
            self.task_cfg.impossible_xlb
            + 0.1 * (self.task_cfg.impossible_xub - self.task_cfg.impossible_xlb),
            self.task_cfg.impossible_xub
            - 0.1 * (self.task_cfg.impossible_xub - self.task_cfg.impossible_xlb),
        )

        self.timestep = 0
        self.old_dist = None

        # Logging
        self.reset_region = None

        # Rendering
        if self.render_mode == "human":
            matplotlib.use("qtagg")
            plt.show()
        self._fig: matplotlib.figure.Figure = None
        self._ax: matplotlib.axes.Axes = None
        self._background = None
        self._agent_circle = None
        self._timestep_text = None
        self._accum_rew_text = None
        self._accum_cost_text = None
        self._accumulated_rew = None
        self._accumulated_cost = None
        self._collided_text = None

    def _get_obs(self):
        return np.array(self.agent.centroid.coords[0])

    def _get_info(self):
        return {
            "timestep": self.timestep,
            "agent_pos": self.agent.centroid.coords[0],
            "cost": 1 if self._get_collided() else 0,
            "reset_region": self.reset_region,
        }

    def _get_collided(self):
        # See if agent is out of bounds
        if (
            not self.task_cfg.env_xlb <= self.agent.centroid.x <= self.task_cfg.env_xub
        ) or (
            not self.task_cfg.env_ylb <= self.agent.centroid.y <= self.task_cfg.env_yub
        ):
            return True

        # See if the agent is in any wall
        if self.agent.intersects(self.wall1) or self.agent.intersects(self.wall2):
            return True

        # See if agent went into impossible obstacle
        if self.agent.intersects(self.imp_obs):
            return True

        # See if the agent has changed regions
        if self.reset_region != self.which_reset_region(
            self.agent.centroid.x, self.agent.centroid.y
        ):
            return True

        # if not self.task_cfg.env_low <= self.agent.centroid.x <= self.task_cfg.env_high:
        return False

    def _convert_action(self, a):
        match a:
            case 0:
                # return (-self.agent_stepsize, 0)  # left
                return (0, 0)  # nothing
            case 1:
                # return (0, 0)  # nothing
                return (-self.agent_stepsize, 0)  # left
            case 2:
                return (self.agent_stepsize, 0)  # right

    def which_reset_region(self, x, y):
        if self.task_cfg.easy_xlb <= x <= self.task_cfg.easy_xub:
            return "easy"
        elif self.task_cfg.hard_xlb <= x <= self.task_cfg.hard_xub:
            return "hard"
        elif self.task_cfg.impossible_xlb <= x <= self.task_cfg.impossible_xub:
            return "impossible"
        else:
            raise ValueError(f"Invalid x value {x}")

    def reset(self, seed: int = None, options: dict = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        reset_idx = np.random.choice(
            [0, 1, 2],
            p=[
                self.task_cfg.easy_prob,
                self.task_cfg.hard_prob,
                self.task_cfg.impossible_prob,
            ],
        )
        reset_options = [
            self.easy_reset_region,
            self.hard_reset_region,
            self.impossible_reset_region,
        ]
        reset_option_names = ["easy", "hard", "impossible"]
        reset_xlb, reset_xub = reset_options[reset_idx]
        reset_x = np.random.uniform(reset_xlb, reset_xub)
        reset_y = self.task_cfg.env_yub - 2 * self.task_cfg.agent_radius
        self.agent = Point([reset_x, reset_y]).buffer(self.task_cfg.agent_radius)

        # Logging
        self.reset_region = reset_option_names[reset_idx]

        self.timestep = 0
        self.old_dist = self.goal_box.distance(self.agent)

        # Render stuff
        self._accumulated_rew = 0
        self._accumulated_cost = 0

        self._fig, self._ax = None, None

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.timestep += 1
        if not self.continuous_action_space:
            action = self._convert_action(action)
        else:
            # Append 0 to the action to make it 2D
            action = np.array([action[0], 0])
        new_coord = np.array(
            [
                self.agent.centroid.x + action[0],
                self.agent.centroid.y + action[1],
            ]
        )

        # If agent is in disturb region, push it left by a fixed amount
        if self.agent.intersects(self.disturb_box):
            new_coord[0] += 2 * self.agent_stepsize

        # Agent always goes downward
        new_coord[1] -= self.task_cfg.agent_drop_rate

        self.agent = Point(new_coord.tolist()).buffer(self.task_cfg.agent_radius)
        new_dist = self.goal_box.distance(self.agent)

        obs = self._get_obs()
        trunc = self.timestep >= self.task_cfg.max_steps
        term = self.agent.intersects(self.goal_box)
        info = self._get_info()

        reward = 0
        # Living penalty
        if self.task_cfg.add_cost_to_reward:
            reward -= info["cost"]

        self.old_dist = new_dist

        term = term or self._get_collided()

        # Render stuff
        self._accumulated_rew += reward
        self._accumulated_cost += info["cost"]

        return obs, reward, term, trunc, info

    def close(self):
        plt.close()

    def render(self):
        match self.render_mode:
            case "rgb_array":
                return self._render_frame()
            case "human":
                self._render_frame()
            case _:
                raise NotImplementedError(
                    f"Render mode {self.render_mode} not implemented"
                )

    def _get_fig_and_ax(self, draw_agent: bool = True, ax: plt.Axes = None):
        if self._fig is None:
            plt.close()
            if self.render_mode == "human":
                plt.show()

            if ax is None:
                self._fig, self._ax = plt.subplots(figsize=(8, 8), layout="constrained")
            else:
                self._ax = ax
                self._fig = ax.get_figure()
            self._ax.set_aspect("equal")
            fig, ax = self._fig, self._ax
            ax.set_facecolor("white")

            # Set limits
            ax.set_xlim([self.task_cfg.env_xlb - 1, self.task_cfg.env_xub + 1])
            ax.set_ylim([self.task_cfg.env_ylb, self.task_cfg.env_yub + 1])

            # Draw lines along boundary
            ax.plot(
                [self.task_cfg.env_xlb, self.task_cfg.env_xub],
                [self.task_cfg.env_ylb, self.task_cfg.env_ylb],
                color="black",
                alpha=0.5,
            )
            ax.plot(
                [self.task_cfg.env_xlb, self.task_cfg.env_xub],
                [self.task_cfg.env_yub, self.task_cfg.env_yub],
                color="black",
                alpha=0.5,
            )
            ax.plot(
                [self.task_cfg.env_xlb, self.task_cfg.env_xlb],
                [self.task_cfg.env_ylb, self.task_cfg.env_yub],
                color="black",
                alpha=0.5,
            )
            ax.plot(
                [self.task_cfg.env_xub, self.task_cfg.env_xub],
                [self.task_cfg.env_ylb, self.task_cfg.env_yub],
                color="black",
                alpha=0.5,
            )

            # Draw goal box
            goal_box = matplotlib.patches.Polygon(
                self.goal_box.exterior.coords,
                facecolor="green",
                edgecolor="black",
            )
            ax.add_patch(goal_box)

            # Agent visualization: A circle representing the agent
            if draw_agent:
                self._agent_circle = matplotlib.patches.Circle(
                    self.agent.centroid.coords[0],
                    radius=self.task_cfg.agent_radius,
                    facecolor="blue",
                    edgecolor="black",
                )
                ax.add_patch(self._agent_circle)

                self._timestep_text = self._ax.text(
                    0.1,
                    1.01,
                    f"Step: {self.timestep}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=20,
                )

                self._accum_rew_text = self._ax.text(
                    0.5,
                    1.01,
                    f"Rew: {0}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=20,
                )

                self._accum_cost_text = self._ax.text(
                    0.9,
                    1.01,
                    f"Cost: {0}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=20,
                )

            # Set easy, hard, and impossible text
            easy_x = (self.task_cfg.easy_xub / 2) / self.task_cfg.env_xub
            hard_x = (
                (self.task_cfg.hard_xlb + self.task_cfg.hard_xub) / 2
            ) / self.task_cfg.env_xub
            imp_x = (
                (self.task_cfg.impossible_xlb + self.task_cfg.impossible_xub) / 2
            ) / self.task_cfg.env_xub

            ax.text(
                easy_x,
                0.94,
                "Easy",
                transform=ax.transAxes,
                ha="center",
                fontsize=20,
                alpha=0.6,
            )
            ax.text(
                hard_x,
                0.94,
                "Hard",
                transform=ax.transAxes,
                ha="center",
                fontsize=20,
                alpha=0.6,
            )
            ax.text(
                imp_x - 0.01,
                0.94,
                "Im",
                transform=ax.transAxes,
                ha="center",
                fontsize=20,
                alpha=0.6,
            )

            # If collided, show red text for "Collided". Else, show "Safe".
            if draw_agent:
                if self._get_collided():
                    self._collided_text = ax.text(
                        0.5,
                        -0.1,
                        "Collided",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=20,
                        color="red",
                    )
                else:
                    self._collided_text = ax.text(
                        0.5,
                        -0.1,
                        "Safe",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=20,
                        color="green",
                    )

            # Draw disturb region as a yellow rectangle
            disturb_box = matplotlib.patches.Rectangle(
                (self.task_cfg.disturb_x_lb, self.task_cfg.disturb_y_lb),
                self.task_cfg.disturb_x_ub - self.task_cfg.disturb_x_lb,
                self.task_cfg.disturb_y_ub - self.task_cfg.disturb_y_lb,
                facecolor="yellow",
                edgecolor="black",
                alpha=0.5,
            )
            ax.add_patch(disturb_box)

            # Draw avoid triangle region
            avoid_triangle_top = matplotlib.patches.Polygon(
                [
                    (self.task_cfg.avoid_x_lb, self.task_cfg.disturb_y_ub),
                    (self.task_cfg.disturb_x_ub, self.task_cfg.disturb_y_ub),
                    (
                        self.task_cfg.disturb_x_ub,
                        self.task_cfg.disturb_y_ub
                        + (self.task_cfg.disturb_y_ub - self.task_cfg.disturb_y_lb),
                    ),
                ],
                facecolor="red",
                edgecolor="black",
                alpha=0.5,
            )
            ax.add_patch(avoid_triangle_top)

            avoid_triangle_bot = matplotlib.patches.Polygon(
                [
                    (self.task_cfg.avoid_x_lb, self.task_cfg.disturb_y_ub),
                    (self.task_cfg.disturb_x_ub, self.task_cfg.disturb_y_lb),
                    (self.task_cfg.disturb_x_ub, self.task_cfg.disturb_y_ub),
                ],
                facecolor="red",
                edgecolor="black",
                alpha=0.5,
            )
            ax.add_patch(avoid_triangle_bot)

            # Draw obstacle in the impossible region
            obstacle = matplotlib.patches.Polygon(
                self.imp_obs.exterior.coords,
                facecolor="blue",
                edgecolor="blue",
            )
            ax.add_patch(obstacle)

            # Draw the walls
            wall1 = matplotlib.lines.Line2D(
                [self.task_cfg.easy_xub, self.task_cfg.easy_xub],
                [self.task_cfg.goal_y_ub, self.task_cfg.env_yub],
                color="black",
                alpha=0.5,
            )
            ax.add_line(wall1)

            wall2 = matplotlib.lines.Line2D(
                [self.task_cfg.hard_xub, self.task_cfg.hard_xub],
                [self.task_cfg.goal_y_ub, self.task_cfg.env_yub],
                color="black",
                alpha=0.5,
            )
            ax.add_line(wall2)

            # Save background
            fig.canvas.draw()
            self._background = fig.canvas.copy_from_bbox(ax.bbox)

        return self._fig, self._ax

    def _render_frame(self):
        fig, ax = self._get_fig_and_ax()

        # Restore background
        # fig.canvas.restore_region(self._background)

        # Move agent
        self._agent_circle.set_center(self.agent.centroid.coords[0])
        ax.draw_artist(self._agent_circle)

        # Update timestep
        self._timestep_text.set_text(f"Step: {self.timestep}")

        # Update reward
        self._accum_rew_text.set_text(f"Rew: {self._accumulated_rew:.1f}")

        # Update cost
        self._accum_cost_text.set_text(f"Cost: {self._accumulated_cost:.1f}")

        # Update collided text
        if self._get_collided():
            self._collided_text.set_text("Collided")
            self._collided_text.set_color("red")
            plt.pause(0.5)
        else:
            self._collided_text.set_text("Safe")
            self._collided_text.set_color("green")

        fig.canvas.blit(ax.bbox)
        if self.render_mode == "human":
            plt.pause(0.03)

    def label_ic(self, ax: plt.Axes):
        # Draw the walls as axvlines.
        wall_opts = dict()
        ax.axvline(self.task_cfg.env_xlb, **wall_opts)
        ax.axvline(self.task_cfg.env_ylb, **wall_opts)
        ax.axvline(self.task_cfg.easy_xub, **wall_opts)
        ax.axvline(self.task_cfg.hard_xub, **wall_opts)

        # Add a label.
        easy_x = (self.task_cfg.easy_xub / 2) / self.task_cfg.env_xub
        hard_x = (
            (self.task_cfg.hard_xlb + self.task_cfg.hard_xub) / 2
        ) / self.task_cfg.env_xub
        imp_x = (
            (self.task_cfg.impossible_xlb + self.task_cfg.impossible_xub) / 2
        ) / self.task_cfg.env_xub

        ax.text(
            easy_x,
            0.94,
            "Easy",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
            alpha=0.6,
        )
        ax.text(
            hard_x,
            0.94,
            "Hard",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
            alpha=0.6,
        )
        ax.text(
            imp_x - 0.01,
            0.94,
            "Im",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
            alpha=0.6,
        )
