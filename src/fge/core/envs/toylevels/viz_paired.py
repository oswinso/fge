import functools as ft
from typing import Self

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from loguru import logger
from matplotlib.colors import Normalize
from og.dyn_types import BObs
from og.jax_utils import jax2np, jax_vmap
from og.register_sns_cmaps import register_cmaps
from og.rng import PRNGKey
from og.tree_utils import tree_stack

from fge.core.algos.buf_custom import BufCustom
from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import RolloutOutput
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.rnd import RND
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.jax_util import myjit


@struct.dataclass(frozen=False)
class ToyLevelsVizPAIRED:
    task: HopperJax = struct.field(pytree_node=False)

    @classmethod
    def create(cls, task: ToyLevelsJax) -> Self:
        return ToyLevelsVizPAIRED(task)

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        task = props.task
        ppo_adv_x0 = props.extra["ppo_adv_x0"]
        ppo_pro = props.extra["ppo_pro"]
        ppo_ant = props.extra["ppo_ant"]
        trajsaver_pro = props.extra["trajsaver_pro"]
        trajsaver_ant = props.extra["trajsaver_ant"]
        rollout_eval_det_pro = props.extra["rollout_eval_det_pro"]
        rollout_eval_det_ant = props.extra["rollout_eval_det_ant"]
        regret = props.extra["regret"]
        max_U_A = props.extra["max_U_A"]
        E_U_P = props.extra["E_U_P"]

        import pdb

        pdb.set_trace()
        nrow = 9
        figsize = np.array([8.0, 2.5 * nrow])
        fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")

        # ------------------------------------------
        name = "paired"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
