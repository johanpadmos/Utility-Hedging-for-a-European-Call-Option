# Utility-Hedging-for-a-European-Call-Option

## Hedging Toolkit – Quantitative Option Simulation

A **pure‑Python sandbox** for experimenting with delta‑ and utility‑based
hedging, transaction‑cost modelling, and stochastic volatility.  
Everything is self‑contained: no external data feeds, no heavy dependencies –
just NumPy, Pandas, SciPy, Matplotlib and a few hundred lines of well‑typed
code.

---

## ✨ Key features

| Module | What it delivers |
|--------|------------------|
| **`option.py`** | Black‑Scholes price + Greeks with dividends, edge‑case guards. |
| **`gbm.py`** | Vectorised log‑Euler simulator for geometric Brownian motion. |
| **`garch.py`** | Online GARCH (1, 1) estimator that plugs straight into the cost/hedge engine (`VolFunc`). |
| **`transaction.py`** | Flexible *dynamic* proportional cost: linear in volatility. |
| **`hedging.py`** | • Classic delta‑hedge<br>• Utility/no‑trade‑band hedge à la Zakamouline<br>• Clean public API + detailed cash‑flow tracking. |

---

## 📦 Installation

```bash
git clone https://github.com/your‑handle/hedging‑toolkit.git
cd hedging‑toolkit
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt