import functools as ft

import jax
import jax.numpy as jnp
from jax2d.engine import select_shape
from jax2d.maths import rmat
from jax2d.scene import add_thruster_to_scene
from jax2d.sim_state import SimState, StaticSimParams, Thruster
from kinetix.environment import EnvState, StaticEnvParams


@ft.partial(jax.jit, static_argnums=(1,))
def simstate_to_envstate(sim_state: SimState, static_env_params: StaticEnvParams) -> EnvState:
    return EnvState(
        timestep=0,
        last_distance=-1.0,
        thruster_bindings=jnp.zeros(static_env_params.num_thrusters, dtype=jnp.int32),
        motor_bindings=jnp.zeros(static_env_params.num_joints, dtype=jnp.int32),
        motor_auto=jnp.zeros(static_env_params.num_joints, dtype=bool),
        polygon_shape_roles=jnp.zeros(static_env_params.num_polygons, dtype=jnp.int32),
        circle_shape_roles=jnp.zeros(static_env_params.num_circles, dtype=jnp.int32),
        polygon_highlighted=jnp.zeros(static_env_params.num_polygons, dtype=bool),
        circle_highlighted=jnp.zeros(static_env_params.num_circles, dtype=bool),
        polygon_densities=jnp.ones(static_env_params.num_polygons, dtype=jnp.float32),
        circle_densities=jnp.ones(static_env_params.num_circles, dtype=jnp.float32),
        **sim_state.__dict__,
    )


def update_thruster_global_pos(state: SimState, static_sim_params: StaticSimParams, thruster: Thruster) -> Thruster:
    parent_shape = select_shape(state, thruster.object_index, static_sim_params)
    return thruster.replace(
        global_position=parent_shape.position + jnp.matmul(rmat(parent_shape.rotation), thruster.relative_position)
    )


def add_thruster_to_scene2(
    sim_state: SimState,
    static_sim_params: StaticSimParams,
    object_index: int,
    relative_position: jnp.ndarray,
    rotation: float,
    power=1.0,
) -> tuple[SimState, int]:
    sim_state, thruster_idx = add_thruster_to_scene(sim_state, object_index, relative_position, rotation, power=power)
    thruster_new = jax.vmap(ft.partial(update_thruster_global_pos, sim_state, static_sim_params))(sim_state.thruster)
    sim_state = sim_state.replace(thruster=thruster_new)
    return sim_state, thruster_idx
