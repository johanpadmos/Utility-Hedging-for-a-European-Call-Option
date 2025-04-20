# hedging.py
"""
Delta- and utility-based hedging simulators.

Example
-------
>>> from hedging import delta_hedging
>>> results = delta_hedging(
...     S0=100.0, K=105.0, T=1.0,
...     sigma=0.20, r=0.03, dt=1/252,
...     n_periods=252, hedging_frequency=21,
...     option_type="call",
... )
"""

from __future__ import annotations

from typing import Callable, Final, Optional

import numpy as np
import pandas as pd

from gbm import simulate_gbm
from option import black_scholes_greeks, black_scholes_price
from transaction import dynamic_trans_cost as _static_trans_cost  # for typing only

__all__: Final = [
    "delta_hedging",
    "utility_based_hedging",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

VolFunc = Callable[[int, float], float]
TransCostFunc = Callable[[int, float], float]  # same (time_idx, price) signature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_prices(
    prices: pd.Series | pd.DataFrame | np.ndarray,
) -> pd.Series:
    """
    Convert any 1‑D price container to a pandas Series with a RangeIndex.
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices.flatten())
    if not isinstance(prices, pd.Series):
        raise TypeError("stock_prices must be 1‑D ndarray, Series or DataFrame.")
    return prices.reset_index(drop=True)


def _resolve_sigma(
    vol_model: VolFunc | bool,
    fallback_sigma: float,
    t_idx: int,
    price: float,
) -> float:
    """
    Either call the `vol_model` or return the static `fallback_sigma`.
    """
    return vol_model(t_idx, price) if callable(vol_model) else fallback_sigma


def _resolve_trans_cost(
    tc_model: TransCostFunc | bool,
    fallback_tc: float,
    t_idx: int,
    price: float,
) -> float:
    """
    Either call the `tc_model` or return the static `fallback_tc`.
    """
    return tc_model(t_idx, price) if callable(tc_model) else fallback_tc


# ---------------------------------------------------------------------------
# Delta‑hedging
# ---------------------------------------------------------------------------


def delta_hedging(
    *,
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    dt: float,
    n_periods: int,
    hedging_frequency: int,
    option_type: str,
    q: float = 0.0,
    transaction_cost: float = 0.0,
    random_state: int | None = None,
    dynamic_vol: VolFunc | bool = False,
    dynamic_trans_cost: TransCostFunc | bool = False,
    stock_prices: Optional[pd.Series | pd.DataFrame | np.ndarray] = None,
) -> dict[str, object]:
    """
    Simulate a *classic* delta‑hedging strategy for one European option.

    Returns
    -------
    dict
        Keys: ``stock_prices``, ``option_prices``, ``delta_values``,
        ``cash_account``, ``total_pnl``, ``transaction_costs``.
    """

    # ---------------------------- price path ------------------------------
    if stock_prices is None:
        prices = simulate_gbm(
            S0,
            mu=r,  # risk‑neutral drift
            sigma=sigma,
            dt=dt,
            n_periods=n_periods,
            n_paths=1,
            random_state=random_state,
        )
    else:
        prices = stock_prices

    stock_prices_ser = _coerce_prices(prices)

    # ---------------------------- containers ------------------------------
    option_prices: list[float] = []
    delta_vals: list[float] = []
    cash_account: list[float] = []

    trans_cost_cum = 0.0

    # ---------------------------- t = 0 -----------------------------------
    sigma_t = _resolve_sigma(dynamic_vol, sigma, 0, S0)
    opt_price_0 = black_scholes_price(S0, K, T, sigma_t, r, option_type, q)
    delta_t = black_scholes_greeks(S0, K, T, sigma_t, r, option_type, q)["Delta"]

    stock_pos = -delta_t  # short option ⇒ hold −Δ shares
    tc0 = _resolve_trans_cost(dynamic_trans_cost, transaction_cost, 0, S0)
    cash = opt_price_0 - stock_pos * S0 * (1 + tc0)

    # record
    option_prices.append(opt_price_0)
    delta_vals.append(delta_t)
    cash_account.append(cash)

    # -------------------- main simulation loop ----------------------------
    for i in range(1, n_periods + 1):
        t = i * dt
        S_t = float(stock_prices_ser.iloc[i])
        T_rem = max(T - t, 0.0)

        sigma_t = _resolve_sigma(dynamic_vol, sigma, i, S_t)

        if T_rem > 0:
            opt_price = black_scholes_price(S_t, K, T_rem, sigma_t, r, option_type, q)
            delta_new = black_scholes_greeks(S_t, K, T_rem, sigma_t, r, option_type, q)[
                "Delta"
            ]
        else:  # final payoff
            opt_price = max(S_t - K, 0) if option_type == "call" else max(K - S_t, 0)
            delta_new = 0.0

        option_prices.append(opt_price)
        delta_vals.append(delta_new)

        # ------------- rebalance if on a hedge date or final step ----------
        if i % hedging_frequency == 0 or i == n_periods:
            desired_pos = -delta_new
            trade = desired_pos - stock_pos

            tc = _resolve_trans_cost(dynamic_trans_cost, transaction_cost, i, S_t)
            cash -= trade * S_t + abs(trade * S_t) * tc  # pay for shares + costs
            trans_cost_cum += abs(trade * S_t) * tc
            stock_pos = desired_pos

        cash_account.append(cash)

    # ---------------------------- closure ---------------------------------
    S_final = float(stock_prices_ser.iloc[-1])
    payoff = max(S_final - K, 0) if option_type == "call" else max(K - S_final, 0)
    total_pnl = cash + stock_pos * S_final - payoff

    return {
        "stock_prices": stock_prices_ser,
        "option_prices": option_prices,
        "delta_values": delta_vals,
        "cash_account": cash_account,
        "total_pnl": total_pnl,
        "transaction_costs": trans_cost_cum,
    }


# ---------------------------------------------------------------------------
# Utility‑based hedging with no‑trade band
# ---------------------------------------------------------------------------


def utility_based_hedging(
    *,
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    dt: float,
    n_periods: int,
    hedging_frequency: int,
    option_type: str,
    q: float = 0.0,
    transaction_cost: float = 0.0,
    risk_aversion: float = 1.0,  # reserved for future extensions
    no_trade_multiplier: float = 1.0,
    random_state: int | None = None,
    dynamic_vol: VolFunc | bool = False,
    dynamic_trans_cost: TransCostFunc | bool = False,
    stock_prices: Optional[pd.Series | pd.DataFrame | np.ndarray] = None,
) -> dict[str, object]:
    """
    Utility‑based hedging with *no‑trade* region (Zakamouline‑style).

    Returns
    -------
    dict
        Keys: ``stock_prices``, ``option_prices``, ``frictionless_deltas``,
        ``hedge_positions``, ``cash_account``, ``total_pnl``,
        ``transaction_costs``.
    """

    # price path
    if stock_prices is None:
        prices = simulate_gbm(
            S0,
            mu=r,
            sigma=sigma,
            dt=dt,
            n_periods=n_periods,
            n_paths=1,
            random_state=random_state,
        )
    else:
        prices = stock_prices

    stock_prices_ser = _coerce_prices(prices)

    no_trade_half = no_trade_multiplier * transaction_cost ** (1 / 3)

    # containers
    option_prices: list[float] = []
    frictionless_deltas: list[float] = []
    hedge_pos_list: list[float] = []
    cash_account: list[float] = []

    trans_cost_cum = 0.0

    # t = 0
    sigma_t = _resolve_sigma(dynamic_vol, sigma, 0, S0)
    opt_price_0 = black_scholes_price(S0, K, T, sigma_t, r, option_type, q)
    delta_star = black_scholes_greeks(S0, K, T, sigma_t, r, option_type, q)["Delta"]

    hedge_pos = delta_star
    tc0 = _resolve_trans_cost(dynamic_trans_cost, transaction_cost, 0, S0)
    cash = opt_price_0 - hedge_pos * S0 * (1 + tc0)

    option_prices.append(opt_price_0)
    frictionless_deltas.append(delta_star)
    hedge_pos_list.append(hedge_pos)
    cash_account.append(cash)

    # main loop
    for i in range(1, n_periods + 1):
        S_t = float(stock_prices_ser.iloc[i])
        t = i * dt
        T_rem = max(T - t, 0.0)

        sigma_t = _resolve_sigma(dynamic_vol, sigma, i, S_t)

        if T_rem > 0:
            opt_price = black_scholes_price(S_t, K, T_rem, sigma_t, r, option_type, q)
            delta_star = black_scholes_greeks(
                S_t, K, T_rem, sigma_t, r, option_type, q
            )["Delta"]
        else:
            opt_price = max(S_t - K, 0) if option_type == "call" else max(K - S_t, 0)
            delta_star = 0.0

        option_prices.append(opt_price)
        frictionless_deltas.append(delta_star)

        # no‑trade band
        lower, upper = delta_star - no_trade_half, delta_star + no_trade_half

        if i % hedging_frequency == 0 or i == n_periods:
            if hedge_pos < lower:
                desired = lower
            elif hedge_pos > upper:
                desired = upper
            else:
                desired = hedge_pos

            trade = desired - hedge_pos
            tc = _resolve_trans_cost(dynamic_trans_cost, transaction_cost, i, S_t)
            cost = abs(trade * S_t) * tc
            cash -= trade * S_t + cost
            trans_cost_cum += cost

            hedge_pos = desired

        hedge_pos_list.append(hedge_pos)
        cash_account.append(cash)

    # wrap‑up
    S_final = float(stock_prices_ser.iloc[-1])
    payoff = max(S_final - K, 0) if option_type == "call" else max(K - S_final, 0)
    total_pnl = cash + hedge_pos * S_final - payoff

    return {
        "stock_prices": stock_prices_ser,
        "option_prices": option_prices,
        "frictionless_deltas": frictionless_deltas,
        "hedge_positions": hedge_pos_list,
        "cash_account": cash_account,
        "total_pnl": total_pnl,
        "transaction_costs": trans_cost_cum,
    }
