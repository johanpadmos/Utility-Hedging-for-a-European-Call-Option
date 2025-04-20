# gbm.py
"""
Log‑space simulation of geometric Brownian motion (GBM).

Example
-------
>>> from gbm import simulate_gbm
>>> paths = simulate_gbm(
...     S0=100.0, mu=0.05, sigma=0.20,
...     dt=1/252, n_periods=252, n_paths=3, random_state=42
... )
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

__all__: Final = ["simulate_gbm"]


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    n_periods: int,
    n_paths: int = 1,
    random_state: int | None = None,
    *,
    return_df: bool = True,
) -> pd.DataFrame | np.ndarray:
    """
    Generate Monte-Carlo paths for GBM with constant mu and sigma.

    Parameters
    ----------
    S0 : float
        Initial asset price (must be > 0).
    mu : float
        Drift (annualised, in decimals).
    sigma : float
        Volatility (annualised, in decimals; must be ≥ 0).
    dt : float
        Length of one simulation step in **years**.
    n_periods : int
        Number of time steps (must be ≥ 0).
    n_paths : int, default 1
        Number of independent paths.
    random_state : int | None, default None
        Seed for NumPy's random-number generator.
    return_df : bool, keyword-only, default True
        If *True* return a ``pandas.DataFrame`` whose index is time, columns are
        path IDs; if *False* return a ``numpy.ndarray`` shaped
        ``(n_paths, n_periods + 1)``.

    Returns
    -------
    pandas.DataFrame | numpy.ndarray
        Simulated price paths starting at ``t = 0``.

    Notes
    -----
    * The scheme is *exact* for GBM because the coefficients are constant.
    * When ``sigma == 0`` the path is deterministic:
      :math:`S_t = S_0 e^{\mu t}`.
    """
    # ----------------------- basic validation ------------------------------
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if n_periods < 0:
        raise ValueError("n_periods must be non‑negative.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    if n_periods == 0:
        single = np.full((n_paths, 1), S0)
        return pd.DataFrame(single.T, index=[0.0]) if return_df else single

    # ----------------------- deterministic path for sigma == 0 -------------
    if sigma == 0.0:
        t = np.linspace(0.0, n_periods * dt, n_periods + 1)
        price = S0 * np.exp(mu * t)  # shape (n_periods + 1,)
        out = np.tile(price, (n_paths, 1))  # (n_paths, n_periods + 1)
    else:
        # ----------------------- stochastic path ---------------------------
        rng = np.random.default_rng(random_state)
        Z = rng.standard_normal(size=(n_paths, n_periods))
        dW = np.sqrt(dt) * Z

        drift = (mu - 0.5 * sigma**2) * dt
        logS = np.log(S0) + np.cumsum(drift + sigma * dW, axis=1)
        out = np.concatenate((np.full((n_paths, 1), np.log(S0)), logS), axis=1)
        out = np.exp(out)  # back to price level

    # ----------------------- output formatting -----------------------------
    if return_df:
        index = np.linspace(0.0, n_periods * dt, n_periods + 1)
        return pd.DataFrame(out.T, index=index)  # rows=time
    return out  # ndarray rows=paths
