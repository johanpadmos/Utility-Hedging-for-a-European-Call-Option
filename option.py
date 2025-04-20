# black_scholes.py
"""
Black-Scholes pricing and Greeks for European options (continuous yield q).
"""

import numpy as np
from scipy.stats import norm

__all__ = ["black_scholes_price", "black_scholes_greeks"]


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """
    Return (d1, d2) used by Black–Scholes formulas.
    """
    if T <= 0 or sigma <= 0:
        # These values are never used when T<=0 or sigma<=0,
        # but return something sensible to avoid NaNs in edge handling.
        return 0.0, 0.0

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    option_type: str,
    q: float = 0.0,
) -> float:
    """
    Black–Scholes price for a European call or put.

    Parameters
    ----------
    S, K : float
        Spot price and strike.
    T : float
        Time to expiry in years.
    sigma : float
        Volatility (annualised, in decimals).
    r : float
        Continuous risk‑free rate.
    option_type : {"call", "put"}
        Option flavour.
    q : float, default 0.0
        Continuous dividend yield.

    Returns
    -------
    float
        Option present value.

    Notes
    -----
    * If T <= 0 the intrinsic value is returned.
    * If sigma <= 0 the value reduces to the discounted deterministic payoff.
    """

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    # Edge case: at or past expiry
    if T <= 0.0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic

    # Edge case: zero volatility ⇒ deterministic forward payoff
    if sigma <= 0.0:
        forward = S * np.exp((r - q) * T)
        payoff = (
            max(forward - K, 0.0) if option_type == "call" else max(K - forward, 0.0)
        )
        return np.exp(-r * T) * payoff

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:  # put
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    option_type: str,
    q: float = 0.0,
) -> dict:
    """
    Return Delta, Gamma, Vega, Theta, Rho for a European option.

    For T==0 or sigma==0, Greeks whose theoretical value
    is undefined are returned as np.nan.
    """

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    # Edge cases where Greeks blow up or are undefined
    if T <= 0.0 or sigma <= 0.0:
        return {k: np.nan for k in ("Delta", "Gamma", "Vega", "Theta", "Rho")}

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    sqrtT = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    # Delta
    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
    else:  # put
        delta = disc_q * (norm.cdf(d1) - 1)

    # Gamma & Vega (same for call and put)
    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT  # per 1.0 volatility (not %)

    # Theta
    common_term = -S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
    if option_type == "call":
        theta = (
            common_term - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:  # put
        theta = (
            common_term
            + r * K * disc_r * norm.cdf(-d2)  # sign flip
            - q * S * disc_q * norm.cdf(-d1)  # sign flip
        )
        rho = -K * T * disc_r * norm.cdf(-d2)  # sign flip

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho,
    }
