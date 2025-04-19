import numpy as np
from scipy.stats import norm

__all__ = ["black_scholes_price", "black_scholes_greeks"]


def _d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> tuple[float, float]:
    """
    Computes d1 and d2 used in the Black-Scholes model.

    Args:
        S (float): Current stock price ($ per share).
        K (float): Strike price ($ per share).
        T (float): Time to expiration (years).
        r (float): Continuously compounded risk-free interest rate (decimal, e.g., 0.05 for 5% p.a.).
        q (float): Continuously compounded dividend yield (decimal).
        sigma (float): Implied annualised volatility (decimal, e.g., 0.2 for 20% p.a.).

    Returns:
        tuple: A tuple containing d1 and d2.

    Example:
        >>> _d1_d2(S=100, K=110, T=1, r=0.05, q=0.03, sigma=0.2)
    """
    if T <= 0 or sigma <= 0:
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
    Computes the Black-Scholes price of a European option.

    Args:
        S (float): Current stock price ($ per share).
        K (float): Strike price ($ per share).
        T (float): Time to expiration (years).
        sigma (float): Implied annualised volatility (decimal, e.g., 0.2 for 20% p.a.).
        r (float): Continuously compounded risk-free interest rate (decimal, e.g., 0.05 for 5% p.a.).
        option_type (str): "call" or "put".
        q (float, optional): Continuously compounded dividend yield (decimal). Defaults to 0.

    Returns:
        float: Option price ($ per share).

    Raises:
        ValueError: If `option_type` is not 'call' or 'put'.

    Example:
        >>> black_scholes_price(S=100, K=110, T=1, sigma=0.2, r=0.05, option_type="put")
    """

    if option_type not in {"call", "put"}:
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")

    # Edge case: at or past expiry
    if T <= 0.0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic

    # Edge case: zero volatility -> deterministic forward payoff
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
    Computes the key Greeks for a European option using the Black-Scholes model.

    Args:
        S (float): Current stock price ($ per share).
        K (float): Strike price ($ per share).
        T (float): Time to expiration (years).
        sigma (float): Implied annualised volatility (decimal, e.g. 0.2 for 20% p.a.).
        r (float): Continuously comounded risk-free interest rate (decimal, e.g., 0.05 for 5% p.a.).
        option_type (str): "call" or "put".
        q (float, optional): Continuous dividend yield (decimal, e.g. 0.03 for 3% p.a.). Defaults to 0.

    Returns:
        dict: A dictionary containing Delta, Gamma, Vega, Theta, and Rho.

    Raises:
        ValueError: If `option_type` is not 'call' or 'put'.

    Examples:
        >>> greeks(S=100, K=110, T=1, sigma=0.2, r=0.05, option_type="put", q=0.03)
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
