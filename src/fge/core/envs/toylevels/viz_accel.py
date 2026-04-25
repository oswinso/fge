from typing import Self

import ipdb
import jax
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from loguru import logger
from matplotlib.colors import Normalize
from og.jax_utils import jax2np, jax_vmap
from og.register_sns_cmaps import register_cmaps
from og.tree_utils import tree_stack

from fge.core.algos.buf_custom import BufCustom
from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.algos.plr_sampler import PLRSampler
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import RolloutOutput
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.level_sampler import LevelSampler, Sampler
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.rnd import RND
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.envs.toylevels.viz_value import compute_rollout_metrics
from fge.core.utils.jax_util import myjit


@struct.dataclass(frozen=False)
class VizAccel:
    task: HopperJax = struct.field(pytree_node=False)

    @classmethod
    def create(cls, task: ToyLevelsJax) -> Self:
        register_cmaps()
        return VizAccel(task)

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        if "sampler" not in props.extra:
            return

        task = self.task

        sampler: Sampler = props.extra["sampler"]
        level_sampler: LevelSampler = props.extra["level_sampler"]

        # Visualize the points in sampler by their current score.
        size = int(sampler["size"])
        b_scores = np.array(sampler["scores"])[:size]
        b_leafs = jtu.tree_map(lambda x: np.array(x)[:size], sampler["levels"])

        b_state = task.leaf_to_state(b_leafs)
        b_px0 = task.to_icval(b_state)

        nrow = 1
        figsize = np.array([8, 3])
        fig, ax = plt.subplots(nrow, 1, figsize=figsize, layout="constrained")
        task.task_cpu.label_ic(ax)

        # Jitter the y's a bit.
        rng = np.random.default_rng(seed=12345)
        b_y = rng.normal(size=(size,))

        # Scatter the x0s, color it depending on the feasibility.
        sc = ax.scatter(b_px0, b_y, c=b_scores, alpha=0.5, zorder=7)
        cbar = fig.colorbar(sc, ax=ax)

        plot_dir = run_cfg.paths.eval_plots / "accel"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"accel_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
