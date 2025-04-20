# transaction.py
"""
Dynamic transaction‑cost helper.

Only the symbol listed in ``__all__`` is public.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["dynamic_trans_cost"]

# Type alias for clarity
VolFunc = Callable[[int, float], float]


def dynamic_trans_cost(
    time_index: int,
    current_price: float,
    dynamic_vol: VolFunc,
    *,
    base_cost: float = 0.001,  # 0.10 % of notional
    beta: float = 1.0,  # sensitivity to volatility
) -> float:
    """
    Transaction cost that scales linearly with *implied* volatility.

    Parameters
    ----------
    time_index : int
        Current simulation or trading step.
    current_price : float
        Current asset price.  *Not* used internally but forwarded to
        ``dynamic_vol`` so that custom vol surfaces can depend on it.
    dynamic_vol : Callable[[int, float], float]
        Function returning *non-negative* volatility at (time_index, price).
        Units must match those assumed in ``beta`` (e.g. per-step vol).
    base_cost : float, keyword-only, default 0.001
        Baseline proportional cost (e.g. 0.001 = 0.10% of notional).
    beta : float, keyword-only, default 1.0
        Linear sensitivity.  **Must be non-negative** to avoid negative costs.

    Returns
    -------
    float
        Effective transaction cost for one trade.

    Raises
    ------
    ValueError
        If `base_cost < 0`, `beta < 0`, or `dynamic_vol` returns a negative value.
    """
    if base_cost < 0:
        raise ValueError("base_cost must be non-negative.")
    if beta < 0:
        raise ValueError("beta must be non-negative.")

    vol = dynamic_vol(time_index, current_price)
    if vol < 0:
        raise ValueError("dynamic_vol returned a negative volatility.")

    cost = base_cost * (1.0 + beta * vol)
    return cost
