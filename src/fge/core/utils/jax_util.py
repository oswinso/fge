import inspect
import types
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import ipdb
import jax
from jax._src import sharding_impls
from jax._src.lib import xla_client as xc

P = ParamSpec("P")
R = TypeVar("R")


def myjit(
    fun: Callable[P, R],
    /,
    *,
    in_shardings: Any = sharding_impls.UNSPECIFIED,
    out_shardings: Any = sharding_impls.UNSPECIFIED,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: xc.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> Callable[P, R]:
    jit_fn = jax.jit(
        fun,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
        compiler_options=compiler_options,
    )

    @wraps(fun)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jit_fn(*args, **kwargs)

    # If fun is a class method (e.g., A.class_method), then set the name to A.class_method.
    # Otherwise, set the name to fun.__name__.
    if hasattr(fun, "__qualname__"):
        name = fun.__qualname__
    else:
        name = fun.__name__

    # Replace the filename in the code object with the filename of the original function.
    co_filename = inspect.getsourcefile(fun)
    co_firstlineno = inspect.getsourcelines(fun)[1]

    wrapper.__code__ = wrapper.__code__.replace(co_name=name, co_filename=co_filename, co_firstlineno=co_firstlineno)
    return wrapper


def wlabel(func: Callable[P, R], label: str) -> Callable[P, R]:
    """
    Return a new function that's byte-for-byte identical to `func`,
    except that its code object (and __name__/__qualname__) is renamed to `new_name`.
    """
    # 1. Copy the code object, swapping in the new name
    new_name = "{} ({})".format(func.__code__.co_name, label)
    new_code = func.__code__.replace(co_name=new_name)

    # 2. Rebuild the FunctionType
    new_func = types.FunctionType(
        new_code,
        func.__globals__,
        name=new_name,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )

    # 3. Copy over the “metadata” that FunctionType doesn’t pick up by itself
    new_func.__kwdefaults__ = func.__kwdefaults__
    new_func.__annotations__ = (func.__annotations__ or {}).copy()
    new_func.__doc__ = func.__doc__
    new_func.__module__ = func.__module__
    new_func.__dict__.update(func.__dict__)  # decorators may have set attrs
    new_func.__qualname__ = new_name

    return new_func


def get_leading_dim_fast(tree) -> int:
    leaf = jax.tree_util.tree_leaves(tree)[0]
    return leaf.shape[0]
