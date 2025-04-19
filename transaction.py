from typing import Callable


def dynamic_trans_cost_function(
    time_index: int,
    current_price: float,
    dynamic_vol: Callable[[int, float], float],
    base_cost: float = 0.001,
    beta: float = 1.0,
) -> float:
    """
    Computes a dynamic transaction cost based on the current implied volatility.

    Args:
        time_index (int): The current time step index.
        current_price (float): The current stock price (not used directly here, but could be).
        dynamic_vol (callable): A function that returns the current volatility given time_index and current_price.
        base_cost (float): The base transaction cost (e.g., 0.001 for 0.1%).
        beta (float): Sensitivity parameter that scales the effect of volatility on the cost.

    Returns:
        float: The dynamic transaction cost for the current trade.
    """
    # Get the current implied volatility (σ_imp) using the dynamic_vol callable.
    current_vol = dynamic_vol(time_index, current_price)
    # Calculate the effective transaction cost.
    return base_cost * (1 + beta * current_vol)
