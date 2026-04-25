import jax.numpy as jnp
import numpy as np


def rot2d(theta: jnp.ndarray | np.ndarray | float) -> jnp.ndarray | np.ndarray:
    if isinstance(theta, jnp.ndarray):
        c, s = jnp.cos(theta), jnp.sin(theta)
        return jnp.array([[c, -s], [s, c]])
    else:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])


def transform2d(X_A_B: np.ndarray, X_B_C: np.ndarray):
    """
    X_A_B: [ px, pz, theta ]
    pos_B_C: [ px, py, pz ]
    """
    assert X_A_B.shape == (3,)
    assert X_B_C.shape == (3,)

    p_A_B = X_A_B[:2]
    theta_A_B = X_A_B[2]

    p_B_C = X_B_C[:2]
    theta_B_C = X_B_C[2]

    # Negative, since rotation direction in 3d in x-z plane (CW) is opposite to that in 2d (CCW).
    R_A_B = rot2d(-theta_A_B)
    p_A_C = p_A_B + R_A_B @ p_B_C

    X_A_C = np.array([p_A_C[0], p_A_C[1], theta_A_B + theta_B_C])
    return X_A_C


def transform2d_jax(X_A_B: np.ndarray | jnp.ndarray, X_B_C: np.ndarray | jnp.ndarray):
    """
    X_A_B: [ px, pz, theta ]
    pos_B_C: [ px, py, pz ]
    """
    assert X_A_B.shape == (3,)
    assert X_B_C.shape == (3,)

    p_A_B = X_A_B[:2]
    theta_A_B = X_A_B[2]

    p_B_C = X_B_C[:2]
    theta_B_C = X_B_C[2]

    # Negative, since rotation direction in 3d in x-z plane (CW) is opposite to that in 2d (CCW).
    R_A_B = rot2d(-theta_A_B)
    p_A_C = p_A_B + R_A_B @ p_B_C

    X_A_C = jnp.array([p_A_C[0], p_A_C[1], theta_A_B + theta_B_C])
    return X_A_C


def invtrans2d(X_A_B: np.ndarray):
    p_A_B = X_A_B[:2]
    theta_A_B = X_A_B[2]
    R_A_B = rot2d(-theta_A_B)

    p_B_A = -R_A_B.T @ p_A_B
    theta_B_A = -theta_A_B

    return np.array([p_B_A[0], p_B_A[1], theta_B_A])


def invtrans2d_jax(X_A_B: np.ndarray):
    p_A_B = X_A_B[:2]
    theta_A_B = X_A_B[2]
    R_A_B = rot2d(-theta_A_B)

    p_B_A = -R_A_B.T @ p_A_B
    theta_B_A = -theta_A_B

    return jnp.array([p_B_A[0], p_B_A[1], theta_B_A])
