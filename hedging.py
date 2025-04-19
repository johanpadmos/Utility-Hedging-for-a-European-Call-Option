from typing import Callable, Optional, Union

import pandas as pd

from gbm import simulate_gbm
from option import black_scholes_price, greeks


def delta_hedging(
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
    random_state: int = None,
    dynamic_vol: Union[bool, Callable[[int, float], float]] = False,
    dynamic_trans_cost: Union[bool, Callable[[int, float], float]] = False,
    stock_prices: Optional[pd.Series] = None,
) -> dict:
    """
    Simulates a delta hedging strategy for a European option (call or put) over the entire strategy duration.

    This function generates a stock price path using a Geometric Brownian Motion (GBM) model
    unless a pre-computed price series is provided via `stock_prices`. It then computes the option
    prices and corresponding delta values at each time step, rebalancing the hedging portfolio
    according to the specified hedging frequency. Transaction costs are incorporated into the cash
    account when trades occur.

    Args:
        S0 (float): Initial stock price (used if stock_prices is not provided).
        K (float): Strike price of the option.
        T (float): Time to expiration (in years).
        sigma (float): Static volatility for GBM simulation (used if dynamic_vol is False).
        r (float): Risk-free interest rate (annualized).
        dt (float): Time step as a fraction of a year (e.g., 1/252 for daily steps).
        n_periods (int): Total number of periods for the simulation.
        hedging_frequency (int): Number of periods between each hedging rebalance.
        option_type (str): "call" or "put" indicating the type of option.
        q (float, optional): Continuous dividend yield. Defaults to 0.0.
        transaction_cost (float, optional): Static proportional transaction cost per trade.
            Defaults to 0.0.
        random_state (int, optional): Seed for random number generation for reproducibility.
        dynamic_vol (bool or callable, optional): If True or a callable is provided, a dynamic
            volatility model (e.g., GARCH) will be used to update sigma over time. If False,
            the static sigma value is used. Defaults to False.
        dynamic_trans_cost (bool or callable, optional): If True or a callable is provided, a dynamic
            transaction cost model will be used. If False, the static transaction_cost value is used.
            Defaults to False.
        stock_prices (pd.Series, optional): Pre-computed stock price series to use instead of
            generating one via GBM. If provided, its length should match n_periods + 1.

    Returns:
        dict: A dictionary containing the following simulation outputs:
            - "stock_prices": The stock price series used (generated or provided).
            - "option_prices": The computed option prices over time.
            - "delta_values": The delta (hedge ratio) values at each time step.
            - "cash_account": The evolution of the cash account over time, including trade costs.
            - "total_pnl": The total profit and loss of the hedging strategy.
            - "transaction_costs": The cumulative transaction costs incurred.
    """

    # Helper function to obtain transaction cost factor at a given time step and price.
    def get_trans_cost(time_index: int, current_price: float) -> float:
        if callable(dynamic_trans_cost):
            return dynamic_trans_cost(time_index, current_price)
        else:
            return transaction_cost

    # If a pre-computed stock price series is not provided, simulate one using the GBM function.
    if stock_prices is None:
        simulated = simulate_gbm(
            S0,
            mu=r,  #  note mu=r as we assume risk neutral pricing
            sigma=sigma,
            dt=dt,
            n_periods=n_periods,
            n_paths=1,
            random_state=random_state,
        )
        # Ensure we have a pd.Series
        if isinstance(simulated, pd.DataFrame):
            stock_prices = simulated.iloc[:, 0]
        else:
            stock_prices = simulated
    else:
        # If provided as a DataFrame, take the first column.
        if isinstance(stock_prices, pd.DataFrame):
            stock_prices = stock_prices.iloc[:, 0]

    # Initialise lists to track simulation outputs.
    option_prices = []
    delta_values = []
    cash_account = []

    # Initialise total transaction cost accumulator.
    transaction_costs_total = 0.0

    # --- INITIALISATION AT TIME 0 ---
    t0 = 0.0
    T_remaining = T - t0

    # Determine current volatility: use dynamic_vol callable if provided; otherwise static.
    current_sigma = sigma
    if callable(dynamic_vol):
        current_sigma = dynamic_vol(0, S0)

    # Compute initial option price and Greeks using the provided option_type.
    option_price_0 = black_scholes_price(
        S0, K, T_remaining, current_sigma, r, option_type, q
    )
    greeks_dict = greeks(S0, K, T_remaining, current_sigma, r, option_type, q)
    delta_current = greeks_dict["Delta"]

    # For a short option position, the hedge is to hold -delta shares.
    stock_position = -delta_current

    # Compute initial transaction cost for setting up the hedge.
    trans_cost = get_trans_cost(0, S0)
    # Assume we receive the option premium when selling the option.
    cash = option_price_0 - (stock_position * S0 * (1 + trans_cost))

    # Record initial values.
    option_prices.append(option_price_0)
    delta_values.append(delta_current)
    cash_account.append(cash)

    # --- MAIN LOOP: Iterate Through Each Time Step ---
    for i in range(1, n_periods + 1):
        t = i * dt
        T_remaining = T - t
        S_t = stock_prices.iloc[i]

        # Update volatility if dynamic.
        current_sigma = sigma
        if callable(dynamic_vol):
            current_sigma = dynamic_vol(i, S_t)

        # Compute option price and Greeks at current time.
        if T_remaining > 0:
            option_price = black_scholes_price(
                S_t, K, T_remaining, current_sigma, r, option_type, q
            )
            greeks_dict = greeks(S_t, K, T_remaining, current_sigma, r, option_type, q)
            delta_new = greeks_dict["Delta"]
        else:
            # At expiration: for a call, payoff = max(S-K, 0); for a put, payoff = max(K-S, 0)
            option_price = max(S_t - K, 0) if option_type == "call" else max(K - S_t, 0)
            delta_new = 0.0

        option_prices.append(option_price)
        delta_values.append(delta_new)

        # Rebalance the hedge at specified hedging frequency (or at final time).
        if i % hedging_frequency == 0 or i == n_periods:
            desired_stock_position = -delta_new
            trade_shares = desired_stock_position - stock_position

            # Compute transaction cost for this trade.
            trans_cost = get_trans_cost(i, S_t)
            trade_cost = abs(trade_shares * S_t) * trans_cost
            transaction_costs_total += trade_cost

            # Update cash: subtract cost of buying (or add if selling) shares and transaction cost.
            cash -= trade_shares * S_t + trade_cost
            stock_position = desired_stock_position

        cash_account.append(cash)

    # --- FINAL PORTFOLIO CALCULATION ---
    S_final = stock_prices.iloc[-1]
    # At expiration, compute the option payoff based on option type.
    if option_type == "call":
        option_payoff = max(S_final - K, 0)
    else:
        option_payoff = max(K - S_final, 0)
    portfolio_value = cash + stock_position * S_final - option_payoff
    total_pnl = portfolio_value

    return {
        "stock_prices": stock_prices,
        "option_prices": option_prices,
        "delta_values": delta_values,
        "cash_account": cash_account,
        "total_pnl": total_pnl,
        "transaction_costs": transaction_costs_total,
    }


def utility_based_hedging(
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
    risk_aversion: float = 1.0,
    no_trade_multiplier: float = 1.0,
    random_state: int = None,
    dynamic_vol: Union[bool, Callable[[int, float], float]] = False,
    dynamic_trans_cost: Union[bool, Callable[[int, float], float]] = False,
    stock_prices: Optional[pd.Series] = None,
) -> dict:
    """
    Simulates a utility-based hedging strategy for a European option over the entire strategy duration.

    This function uses a no-trade (inaction) region approach inspired by Zakamouline (2006). Instead
    of continuously rebalancing to the frictionless (ideal) hedge ratio, the hedge is adjusted only
    when the deviation between the current hedge and the frictionless hedge exceeds a threshold.

    The threshold (half-width of the no-trade region) is given by:

        b = no_trade_multiplier * (transaction_cost)^(1/3)

    The ideal hedge is computed as:

        ideal_hedge = -target_delta

    and the no-trade region is defined as:

        [ideal_hedge - b,  ideal_hedge + b]

    Args:
        S0 (float): Initial stock price (used if stock_prices is not provided).
        K (float): Strike price of the option.
        T (float): Time to expiration (in years).
        sigma (float): Static volatility for GBM simulation (used if dynamic_vol is False).
        r (float): Risk-free interest rate (annualized).
        dt (float): Time step as a fraction of a year (e.g., 1/252 for daily steps).
        n_periods (int): Total number of periods for the simulation.
        hedging_frequency (int): Number of periods between each hedging rebalance.
        option_type (str): "call" or "put".
        q (float, optional): Continuous dividend yield. Defaults to 0.0.
        transaction_cost (float, optional): Static proportional transaction cost per trade.
            Defaults to 0.0.
        risk_aversion (float, optional): Risk aversion parameter (may be used for further extensions). Defaults to 1.0.
        no_trade_multiplier (float, optional): Multiplier to scale the no-trade region half-width. Defaults to 1.0.
        random_state (int, optional): Seed for random number generation for reproducibility.
        dynamic_vol (bool or callable, optional): If True or a callable is provided, a dynamic
            volatility model (e.g., GARCH) will be used to update sigma over time. If False,
            the static sigma value is used. Defaults to False.
        dynamic_trans_cost (bool or callable, optional): If True or a callable is provided, a dynamic
            transaction cost model will be used. If False, the static transaction_cost value is used.
            Defaults to False.
        stock_prices (pd.Series, optional): Pre-computed stock price series to use instead of
            generating one via GBM. If provided, its length should match n_periods + 1.

    Returns:
        dict: A dictionary containing:
            - "stock_prices": The stock price series used (generated or provided).
            - "option_prices": The computed option prices over time.
            - "frictionless_deltas": The frictionless (ideal) delta values at each time step.
            - "hedge_positions": The actual hedge positions (stock holdings) over time.
            - "cash_account": The evolution of the cash account over time (including trade costs).
            - "total_pnl": The total profit and loss of the hedging strategy.
            - "transaction_costs": The cumulative transaction costs incurred.
    """

    def get_trans_cost(time_index: int, current_price: float) -> float:
        if callable(dynamic_trans_cost):
            return dynamic_trans_cost(time_index, current_price)
        else:
            return transaction_cost

    # Define no-trade half-width based on the asymptotic result.
    no_trade_half_width = no_trade_multiplier * (transaction_cost) ** (1 / 3)

    # Obtain stock price series via GBM if not provided.
    if stock_prices is None:
        simulated = simulate_gbm(
            S0,
            mu=r,  # risk-neutral assumption: mu = r
            sigma=sigma,
            dt=dt,
            n_periods=n_periods,
            n_paths=1,
            random_state=random_state,
        )
        if isinstance(simulated, pd.DataFrame):
            stock_prices = simulated.iloc[:, 0]
        else:
            stock_prices = simulated
    else:
        if isinstance(stock_prices, pd.DataFrame):
            stock_prices = stock_prices.iloc[:, 0]

    # Initialize output lists.
    option_prices = []
    frictionless_deltas = []
    hedge_positions = []
    cash_account = []
    transaction_costs_total = 0.0

    # --- INITIALIZATION AT TIME 0 ---
    t0 = 0.0
    T_remaining = T - t0
    current_sigma = sigma
    if callable(dynamic_vol):
        current_sigma = dynamic_vol(0, S0)

    option_price_0 = black_scholes_price(
        S0, K, T_remaining, current_sigma, r, option_type, q
    )
    greeks_dict = greeks(S0, K, T_remaining, current_sigma, r, option_type, q)
    target_delta = greeks_dict["Delta"]
    # For a short option, ideal hedge is:
    ideal_hedge = target_delta

    # Set initial hedge position to the ideal hedge.
    hedge_position = ideal_hedge
    frictionless_deltas.append(target_delta)
    option_prices.append(option_price_0)
    hedge_positions.append(hedge_position)

    # Set initial cash: assume you receive the option premium when selling the option.
    trans_cost = get_trans_cost(0, S0)
    cash = option_price_0 - (hedge_position * S0 * (1 + trans_cost))
    cash_account.append(cash)

    # --- MAIN LOOP: Iterate Through Each Time Step ---
    for i in range(1, n_periods + 1):
        t = i * dt
        T_remaining = T - t
        S_t = stock_prices.iloc[i]

        # Update volatility if using a dynamic model.
        current_sigma = sigma
        if callable(dynamic_vol):
            current_sigma = dynamic_vol(i, S_t)

        # Compute frictionless option price and delta.
        if T_remaining > 0:
            option_price = black_scholes_price(
                S_t, K, T_remaining, current_sigma, r, option_type, q
            )
            greeks_dict = greeks(S_t, K, T_remaining, current_sigma, r, option_type, q)
            target_delta = greeks_dict["Delta"]
        else:
            # At expiration: payoff and delta are determined by the option's payoff.
            if option_type == "call":
                option_price = max(S_t - K, 0)
            else:
                option_price = max(K - S_t, 0)
            target_delta = 0.0

        frictionless_deltas.append(target_delta)
        option_prices.append(option_price)
        # Recalculate ideal hedge as the negative of frictionless delta.
        ideal_hedge = target_delta

        # Define the no-trade region directly around the ideal hedge.
        lower_bound = ideal_hedge - no_trade_half_width
        upper_bound = ideal_hedge + no_trade_half_width

        # Rebalance only at the specified frequency or at final time.
        if i % hedging_frequency == 0 or i == n_periods:
            if hedge_position < lower_bound:
                desired_position = lower_bound
            elif hedge_position > upper_bound:
                desired_position = upper_bound
            else:
                desired_position = hedge_position

            trade_shares = desired_position - hedge_position
            trans_cost = get_trans_cost(i, S_t)
            trade_cost = abs(trade_shares * S_t) * trans_cost
            transaction_costs_total += trade_cost

            cash -= trade_shares * S_t + trade_cost
            hedge_position = desired_position

        hedge_positions.append(hedge_position)
        cash_account.append(cash)

    # --- FINAL PORTFOLIO CALCULATION ---
    S_final = stock_prices.iloc[-1]
    if option_type == "call":
        option_payoff = max(S_final - K, 0)
    else:
        option_payoff = max(K - S_final, 0)
    portfolio_value = cash + hedge_position * S_final - option_payoff
    total_pnl = portfolio_value

    return {
        "stock_prices": stock_prices,
        "option_prices": option_prices,
        "frictionless_deltas": frictionless_deltas,
        "hedge_positions": hedge_positions,
        "cash_account": cash_account,
        "total_pnl": total_pnl,
        "transaction_costs": transaction_costs_total,
    }
