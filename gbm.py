import numpy as np
import pandas as pd
from scipy.stats import norm


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

    if random_state:
        np.random.seed(random_state)

    # Generate standard normal random shocks
    Z = norm.rvs(size=(n_periods, n_paths))

    # Compute price changes using the GBM formula
    dS = mu * dt + sigma * np.sqrt(dt) * Z  # GBM equation component
    S = np.zeros((n_periods + 1, n_paths))  # Initialise price matrix
    S[0] = S0  # Set initial price for all paths

    # Apply GBM formula recursively (vectorised)
    for t in range(1, n_periods + 1):
        S[t] = S[t - 1] * (1 + dS[t - 1])  # This follows S_{t + 1} = S_t * exp(dS)

    return pd.DataFrame(S)
