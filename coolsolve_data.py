"""Load the CoolSolve parametric sweep (CSV) or fall back to a labelled DEMO dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

CSV = Path("coolsolve_sweep_clean.csv")


def load():
    """Return (DataFrame, X, Y, Z, is_demo)."""
    if CSV.exists():
        d = pd.read_csv(CSV)
        num = d.select_dtypes("number")
        x, y = [c for c in ("T_ev_1", "T_cd_1") if c in num.columns][:2] or list(num.columns[:2])
        targets = [c for c in num.columns if c.lower().startswith(("cop", "q_dot_ev", "w_dot")) and c not in (x, y)]
        return num, x, y, (targets or [num.columns[-1]])[0], False
    rng = np.random.default_rng(0)
    g = pd.MultiIndex.from_product([range(-20, 11, 5), range(30, 55, 5)],
                                   names=["T_ev_1", "T_cd_1"]).to_frame(index=False)
    te, tc = g.T_ev_1 + 273.15, g.T_cd_1 + 273.15
    g["COP_1"] = 0.6 * te / (tc - te) + rng.normal(0, .05, len(g))
    g["W_dot_1"] = 2.0 + 0.045 * (tc - te) + rng.normal(0, .02, len(g))
    g["Q_dot_ev_1"] = g.COP_1 * g.W_dot_1
    return g, "T_ev_1", "T_cd_1", "COP_1", True


def grid(d, x, y, z):
    p = d.pivot_table(index=y, columns=x, values=z)
    xx, yy = np.meshgrid(p.columns.values, p.index.values)
    return p, xx, yy, p.values
