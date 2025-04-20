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
```

## Requirements

* Python 3.9+
* `numpy`, `pandas`, `matplotlib`, `scipy`
* Optional: `tqdm` (for progress bars)
* Optional: A working LaTeX installation for `text.usetex = True` plots


## 🚀 Quick start (script)

```python
from hedging import delta_hedging
from garch import GARCHVolatilityEstimator          # optional dynamic vol
from transaction import dynamic_trans_cost          # optional dynamic cost

results = delta_hedging(
    S0=100, 
    K=100, 
    T=1.0,
    sigma=0.20, 
    r=0.05, 
    q=0.0,
    dt=1/252, 
    n_periods=252,
    hedging_frequency=21,            # monthly re‑hedge
    option_type="put",
    transaction_cost=0.001,          # 10 bps per trade
    dynamic_vol=False,               # could pass GARCHVolatilityEstimator(...)
    dynamic_trans_cost=False,
    random_state=42,
)

print("Final P&L:", results["total_pnl"])
```

You can plug in dynamic volatility (e.g. GARCH) and dynamic cost models by passing your own callables:

```python
from garch import GARCHVolatilityEstimator
from transaction import dynamic_trans_cost

dynamic_vol = GARCHVolatilityEstimator(omega=1e-6, alpha=0.05, beta=0.9, initial_vol=0.2)

results = delta_hedging(..., dynamic_vol=dynamic_vol, dynamic_trans_cost=dynamic_trans_cost)
```

## 📓 Interactive Notebook

To explore and visualise strategies interactively:

```bash
cd notebooks/
jupyter lab
```
Open hedging_demo.ipynb to:

* Simulate delta-hedging vs utility-based hedging
* Visualise cash account evolution
* Analyse how the no-trade region reduces transaction costs
* Generate publication-quality plots using LaTeX fonts

## 🧩 Extending

The architecture is modular and easy to build on:

| Want to… | Do this… |
|----------|----------|
| Use a custom volatility surface | Define my_vol(t_idx, price) and pass as dynamic_vol= |
| Change transaction cost model | Define my_cost(t_idx, price) and pass as dynamic_trans_cost= |
| Switch stochastic models | 	Replace simulate_gbm with your own generator. Only requirement: output a 1-D price Series. |
| Try alternative hedging rules | Fork hedging.py and implement a new strategy loop. |

## 📚 References

* Black & Scholes (1973) — _The pricing of options and corporate liabilities_
* Zakamouline (2006) — _Hedging with a threshold strategy_
* Engle (1982) — _Autoregressive conditional heteroskedasticity with estimates of the variance of United Kingdom inflation_

## ⚖️ License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this code as you wish.
See the LICENSE file for full legal details.