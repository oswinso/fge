import pathlib
from typing import Callable, Protocol

import jax.random as jr
from jax.random import PRNGKey
from og.ckpt_utils import load_cfg_from_ckpt, load_from_ckpt
from og.dyn_types import Control, Obs
from og.none import get_or
from og.tfp import tfd

_Alg = TypeVar("_Alg", bound=Alg)


class AlgCfg(Protocol):
    def make(self, key: PRNGKey, task: ConstrCostTask) -> _Alg:
        alg_cls = self.cls()
        return alg_cls.create(key, task, self)

    def cls(self) -> type[_Alg]: ...

    @classmethod
    def fromdict(cls, d: dict, use_converter: bool = True): ...

    def asdict(self) -> dict: ...


class Alg(Protocol):
    @property
    def disc_gamma(self) -> float:
        raise NotImplementedError("")

    @staticmethod
    def name() -> str: ...

    def dummy_batch(self):
        """Return a batch of data for initializing the RB."""
        ...

    def init_rb(self): ...

    def add_to_rb(self, data, rb): ...

    def select_action(self, obs: Obs) -> Control: ...

    def select_action_jit(self, obs: Obs) -> Control: ...

    def get_mode_dist(self) -> Callable[[Obs], tfd.Distribution]: ...

    def get_td_dist(self) -> Callable[[Obs], tfd.Distribution]:
        """Distribution of actions used for computing the TD error."""
        ...


def load_algcfg_from_ckpt(ckpt_path: pathlib.Path) -> AlgCfg:
    cfg = load_cfg_from_ckpt(ckpt_path, "alg_cfg")
    return cfg


def load_alg_from_ckpt(
    task: ConstrCostTask, ckpt_path: pathlib.Path, key: PRNGKey | None = None
) -> Alg:
    key = get_or(key, jr.PRNGKey(0))
    ckpt_path = ckpt_path.absolute()
    alg_cfg = load_algcfg_from_ckpt(ckpt_path)
    alg = alg_cfg.make(key, task)
    return load_from_ckpt(ckpt_path, alg, "alg")
