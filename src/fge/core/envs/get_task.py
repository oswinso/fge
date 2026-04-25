from typing import Type

from fge.core.envs.dub_circ import dub_circ_jax
from fge.core.envs.dub_circ.dub_circ_jax import DubinsJax
from fge.core.envs.f16_avoid import f16_avoid
from fge.core.envs.f16_avoid.f16_avoid_jax import F16AvoidJax
from fge.core.envs.jax_task import JaxTask
from fge.core.envs.kinetix import lander
from fge.core.envs.mujoco.cheetah import cheetah
from fge.core.envs.mujoco.cheetah.cheetah_jax import CheetahJax
from fge.core.envs.mujoco.hopper import hopper
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels import toylevels
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax


def isinstance2(cfg1, cls) -> bool:
    if isinstance(cfg1, cls):
        return True

    if not isinstance(cls, tuple):
        cls = (cls,)

    for c in cls:
        cfg1_cls = cfg1.__class__
        cfg1_module = ".".join(cfg1_cls.__module__.split(".")[1:])
        cls_module = ".".join(c.__module__.split(".")[1:])
        if cfg1_cls.__name__ == c.__name__ and cfg1_module == cls_module:
            return True

    return False


def get_task_cls(task_cfg) -> Type[JaxTask]:
    if isinstance2(task_cfg, toylevels.TaskCfg):
        return ToyLevelsJax
    if isinstance2(task_cfg, hopper.TaskCfg):
        return HopperJax
    if isinstance2(task_cfg, f16_avoid.TaskCfg):
        return F16AvoidJax
    if isinstance2(task_cfg, dub_circ_jax.TaskCfg):
        return DubinsJax
    if isinstance2(task_cfg, cheetah.TaskCfg):
        return CheetahJax
    if isinstance2(task_cfg, lander.TaskCfg):
        return lander.LanderJax
    if isinstance2(task_cfg, lander.TaskCfgHard):
        return lander.LanderJaxHard
    if isinstance2(task_cfg, lander.TaskCfgHardState):
        return lander.LanderJaxHardStateObs
    raise ValueError(f"Unknown task_cfg: {task_cfg}")


def make_task(task_cfg) -> JaxTask:
    return get_task_cls(task_cfg)(task_cfg)
