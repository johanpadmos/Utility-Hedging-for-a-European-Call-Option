import numpy as np


class GARCHVolatilityEstimator:
    def __init__(self, omega: float, alpha: float, beta: float, initial_vol: float):
        """
        Initializes a simple GARCH(1,1) volatility estimator.

        The GARCH(1,1) model is given by:
            sigma_t^2 = omega + alpha * (r_{t-1})^2 + beta * sigma_{t-1}^2

        Args:
            omega (float): The constant term in the GARCH model.
            alpha (float): The coefficient for the squared return (r_{t-1}^2).
            beta (float): The coefficient for the lagged variance (sigma_{t-1}^2).
            initial_vol (float): The initial volatility estimate.
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.current_vol = initial_vol
        self.last_price = None

    def update(self, price: float) -> float:
        """
        Updates the volatility estimate given a new stock price.

        Computes the log return from the previous price, updates the volatility
        using the GARCH(1,1) formula, and returns the new volatility estimate.

        Args:
            price (float): The current stock price.

        Returns:
            float: The updated volatility estimate.
        """
        if self.last_price is None:
            # At the first call, there's no return to compute.
            self.last_price = price
            return self.current_vol

        # Compute the log return: r = log(price_t / price_{t-1})
        r = np.log(price / self.last_price)
        # Update the volatility: sigma_t^2 = omega + alpha * r^2 + beta * sigma_{t-1}^2
        self.current_vol = np.sqrt(
            self.omega + self.alpha * r**2 + self.beta * (self.current_vol**2)
        )
        self.last_price = price
        return self.current_vol
