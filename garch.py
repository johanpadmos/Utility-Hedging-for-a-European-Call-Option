# garch.py
"""
GARCH(1, 1) volatility estimator usable as a drop-in **dynamic_vol** callable.

Example
-------
>>> from garch import GARCHVolatilityEstimator
>>> est = GARCHVolatilityEstimator(omega=1e-6, alpha=0.05, beta=0.9,
...                                initial_vol=0.20)
>>> sigma = est(0, 100.0)          # first call → returns initial_vol
>>> sigma = est(1, 101.0)          # updates and returns σ₁
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__: Final = ["GARCHVolatilityEstimator"]


class GARCHVolatilityEstimator:
    """
    Online GARCH(1, 1) σ‑estimator.

    The conditional variance is

    .. math::
        \sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2,

    where :math:`r_t = \ln(P_t/P_{t-1})`.

    Parameters
    ----------
    omega, alpha, beta : float
        Model coefficients.  Requires ``alpha + beta < 1`` for covariance
        stationarity; **not** enforced programmatically, but will raise a
        warning if violated.
    initial_vol : float
        σ₀ used until the first update.
    """

    def __init__(
        self,
        omega: float,
        alpha: float,
        beta: float,
        initial_vol: float,
    ) -> None:
        if omega < 0 or alpha < 0 or beta < 0:
            raise ValueError("ω, α, β must be non‑negative.")
        if alpha + beta >= 1:
            import warnings

            warnings.warn(
                "alpha + beta ≥ 1 — GARCH variance is non‑stationary.",
                RuntimeWarning,
                stacklevel=2,
            )
        if initial_vol <= 0:
            raise ValueError("initial_vol must be positive.")

        self._omega: float = omega
        self._alpha: float = alpha
        self._beta: float = beta
        self._sigma: float = initial_vol  # σ_{t-1} on entry
        self._last_price: float | None = None

    # ------------------------------------------------------------------ public

    def __call__(self, time_index: int, price: float) -> float:
        """
        Make the estimator itself a ``Callable[[int, float], float]`` so it fits
        directly into :func:`dynamic_trans_cost`.

        * ``time_index`` is accepted for signature compatibility but otherwise
          ignored — the GARCH recursion depends only on price history.
        * Returns the **updated** σₜ after incorporating ``price``.
        """
        return self.update(price)

    def update(self, price: float) -> float:
        """
        Update internal state with a new price and return the fresh σₜ.

        Notes
        -----
        * If this is the first call (no previous price), σ₀ is returned and the
          price is stored for the next step.
        * Prices must be strictly positive (log‑returns undefined otherwise).
        """
        if price <= 0.0:
            raise ValueError("price must be positive.")

        if self._last_price is None:
            # First observation: no return yet, keep σ₀.
            self._last_price = price
            return self._sigma

        # Log‑return r_{t-1}
        r = np.log(price / self._last_price)

        # Conditional variance update (keep σ ≥ 0 by construction)
        var_t = self._omega + self._alpha * r**2 + self._beta * self._sigma**2
        self._sigma = float(np.sqrt(var_t))

        # Roll price forward
        self._last_price = price
        return self._sigma

    # ------------------------------------------------------------- properties

    @property
    def sigma(self) -> float:
        """Current volatility estimate (σₜ)."""
        return self._sigma

    @property
    def params(self) -> tuple[float, float, float]:
        """Model parameters as ``(omega, alpha, beta)``."""
        return self._omega, self._alpha, self._beta
