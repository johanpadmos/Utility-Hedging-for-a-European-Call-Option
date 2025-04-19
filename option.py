import numpy as np
from scipy.stats import norm


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

    # Compute d1 and d2
    d1 = (np.log(S / K) + T * (r - q + 0.5 * sigma**2)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # option_type == "put"
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return round(price, 6)


def greeks(
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
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")

    # Compute d1 and d2
    d1 = (np.log(S / K) + T * (r - q + 0.5 * sigma**2)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Compute Greeks
    delta = (
        np.exp(-q * T) * norm.cdf(d1)
        if option_type == "call"
        else np.exp(-q * T) * (norm.cdf(d1) - 1)
    )
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    theta = (
        (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
        + q * S * np.exp(-q * T) * norm.cdf(d1 if option_type == "call" else -d1)
    )
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)

    return {
        "Delta": round(delta, 6),
        "Gamma": round(gamma, 6),
        "Vega": round(vega, 6),
        "Theta": round(theta, 6),
        "Rho": round(rho, 6),
    }
