import numpy as np


def fictitious_play(
    MN_payoff: np.ndarray, iters: int, rng: np.random.Generator | None
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a two-player zero-sum game using fictitious play.

    :param MN_payoff: (m, n) matrix of payoffs for player M. Player N's payoffs are the negative of this matrix.
    """
    m, n = MN_payoff.shape
    m_counts = np.zeros(m, dtype=np.int32)
    n_counts = np.zeros(n, dtype=np.int32)

    m_payoff = np.zeros(m, dtype=np.float64)
    n_payoff = np.zeros(n, dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng(seed=12345)

    assert iters > 1
    for _ in range(iters):
        # Randomly choose action based on current payoffs.
        action_1 = rng.choice(np.flatnonzero(m_payoff == m_payoff.max()))
        action_2 = rng.choice(np.flatnonzero(n_payoff == n_payoff.max()))

        m_payoff += MN_payoff[:, action_2]
        n_payoff += -MN_payoff[action_1, :]

        m_counts[action_1] += 1
        n_counts[action_2] += 1

    m_strategy = m_counts / m_counts.sum()
    n_strategy = n_counts / n_counts.sum()

    return m_strategy, n_strategy
