import os

import jax
from loguru import logger

from fge.core.utils.path_utils import get_root_dir


def enable_compilation_cache():
    jax.config.update("jax_default_matmul_precision", "highest")

    # Don't enable cache if NOCACHE environment variable is equal to 1
    if os.getenv("NOCACHE") == "1":
        logger.warning("Not enabling jax compilation cache because NOCACHE=1")
        return

    root_dir = get_root_dir()
    cache_dir = root_dir / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(cache_dir.absolute()))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
