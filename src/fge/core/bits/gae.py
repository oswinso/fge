import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Bool, Float


def compute_gae(
    gamma: float,
    lambd: float,
    T_rew: Float[ArrayLike, "T"],
    T_V_curr: Float[ArrayLike, "Tp1"],
    T_V_next: Float[ArrayLike, "Tp1"],
    T_nextvalid: Bool[ArrayLike, "T"],
) -> tuple[Float[ArrayLike, "T"], Float[ArrayLike, "T"]]:
    """
    Computes the Generalized Advantage Estimation (GAE) for a sequence of rewards and value predictions.

    GAE is used in reinforcement learning to compute advantage estimates, which balance bias and variance
    by incorporating both immediate rewards and future value predictions.

    Args:
        gamma (float): Discount factor for future rewards (0 <= gamma <= 1).
        lambd (float): Smoothing parameter for GAE (0 <= lambd <= 1).
        T_rew (Float[ArrayLike, "T"]): Rewards for each timestep, shape (T,).
        T_V_curr (Float[ArrayLike, "Tp1"]): Value predictions for the current timestep, shape (T,).
        T_V_next (Float[ArrayLike, "Tp1"]): Value predictions for the next timestep, shape (T,).
        T_nextvalid (Bool[ArrayLike, "T"]): Boolean mask indicating valid transitions, shape (T,).

    Returns:
        tuple:
            - T_Q_gae (Float[ArrayLike, "T"]): Estimated Q-values for each timestep, shape (T,).
            - T_A_gae (Float[ArrayLike, "T"]): Advantage estimates for each timestep, shape (T,).
    """

    def body(gae, deltaterm):
        """
        Computes the GAE for a single timestep.

        Args:
            gae (float): The GAE value from the previous timestep.
            deltaterm (tuple): A tuple containing:
                - delta (float): Temporal difference error for the current timestep.
                - isvalid (bool): Boolean indicating if the transition is valid.

        Returns:
            tuple: Updated GAE value for the current timestep.
        """
        delta, isvalid = deltaterm
        gae_prev = delta + gamma * lambd * gae * isvalid
        return gae_prev, gae_prev

    (T,) = T_rew.shape
    assert T_V_curr.shape == T_V_next.shape == T_nextvalid.shape == (T,)

    # Compute temporal difference errors for each timestep.
    T_delta = T_rew + gamma * T_V_next * T_nextvalid - T_V_curr

    # Prepare inputs for the scan operation, excluding the last timestep.
    deltaterm_input = T_delta[:-1], T_nextvalid[:-1]

    # Perform a reverse scan to compute GAE values for all timesteps except the last.
    # Called Tm1_gae because it computes the GAE for T-1 timesteps.
    _, Tm1_gae = lax.scan(
        body, T_delta[-1], deltaterm_input, length=T - 1, reverse=True
    )

    # Concatenate the last timestep's GAE value to the results.
    T_A_gae = jnp.concatenate([Tm1_gae, T_delta[-1, None]], axis=0)
    assert T_A_gae.shape == (T,)

    # Compute Q-values by adding the current value predictions to the advantages.
    T_Q_gae = T_A_gae + T_V_curr

    return T_Q_gae, T_A_gae


def compute_qvals(
    gamma: float, T_rew: Float[ArrayLike, "T"], T_nextvalid: Bool[ArrayLike, "T"]
) -> Float[ArrayLike, "T"]:
    """
    Computes the Q-values for a sequence of rewards and value predictions.

    Args:
        gamma (float): Discount factor for future rewards (0 <= gamma <= 1).
        T_rew (Float[ArrayLike, "T"]): Rewards for each timestep, shape (T,).
        T_V_curr (Float[ArrayLike, "Tp1"]): Value predictions for the current timestep, shape (T,).
        T_nextvalid (Bool[ArrayLike, "T"]): Boolean mask indicating valid transitions, shape (T,).

    Returns:
        Float[ArrayLike, "T"]: Estimated Q-values for each timestep, shape (T,).
    """

    def body(q, deltaterm):
        """
        Computes the Q-value for a single timestep.

        Args:
            q (float): The Q-value from the previous timestep.
            deltaterm (tuple): A tuple containing:
                - delta (float): Temporal difference error for the current timestep.
                - isvalid (bool): Boolean indicating if the transition is valid.

        Returns:
            float: Updated Q-value for the current timestep.
        """
        delta, isvalid = deltaterm
        q_prev = delta + gamma * q * isvalid
        return q_prev, q_prev

    (T,) = T_rew.shape
    assert T_nextvalid.shape == (T,)
    # Compute temporal difference errors for each timestep.
    T_delta = T_rew
    # Prepare inputs for the scan operation, excluding the last timestep.
    deltaterm_input = T_delta[:-1], T_nextvalid[:-1]
    # Perform a reverse scan to compute Q-values for all timesteps except the last.
    # Called Tm1_q because it computes the Q-values for T-1 timesteps.
    _, Tm1_q = lax.scan(body, T_delta[-1], deltaterm_input, length=T - 1, reverse=True)
    # Concatenate the last timestep's Q-value to the results.
    T_q = jnp.concatenate([Tm1_q, T_delta[-1, None]], axis=0)
    assert T_q.shape == (T,)
    return T_q
