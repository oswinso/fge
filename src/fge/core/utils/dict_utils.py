from typing import Any, Dict

import attrs
import jax_dataclasses as jdc


def any_to_dict(x):
    """Convert any dict-like cfg thing to a real dict"""
    if isinstance(x, dict):
        return x
    if attrs.has(x):
        return attrs.asdict(x)
    if jdc.is_dataclass(x):
        return jdc.asdict(x)
    if hasattr(x, "_asdict"):
        return x._asdict()

    raise TypeError("Cannot convert to dict: ", type(x), x)
