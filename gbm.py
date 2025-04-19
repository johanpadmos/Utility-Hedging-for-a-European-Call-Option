import numpy as np
import pandas as pd


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    n_periods: int,
    n_paths: int = 1,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Simulates Geometric Brownian Motion (GBM) for a given stock price.

    Args:
        S0 (float): Initial stock price ($$$ per share).
        mu (float): Expected return (drift).
        sigma (float): Volatility (stardard deviation of returns).
        dt (float): Time step (fraction of a year).
        n_periods (int): Number of periods to estimate.
        n_paths (int, optional): Number of Monte Carlo paths. Default to 1.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with shape (n_periods + 1, n_paths), each column representing a price path.

    Example Usage:
        >>> gbm_paths = simulate_gbm(S0=100, mu=0.05, sigma=0.2, dt=1/252, n_periods=252, n_paths=100)
    """

    rng = np.random.default_rng(random_state)

    # Generate standard normal random shocks
    Z = rng.standard_normal((n_periods, n_paths))

    # Compute log-returns using the exact GBM formulation:
    # dS/S = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Calculate cumulative log returns and compute the price paths
    cumulative_log_returns = np.vstack(
        [np.zeros((1, n_paths)), np.cumsum(log_returns, axis=0)]
    )
    S = S0 * np.exp(cumulative_log_returns)

    return pd.DataFrame(S)
