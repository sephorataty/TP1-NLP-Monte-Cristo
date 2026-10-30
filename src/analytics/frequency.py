"""Analyse fréquentielle : top N, loi de Zipf, rangs, hapax."""
from collections import Counter

import numpy as np
import pandas as pd


def frequency_table(freq: Counter, k: int = 20) -> pd.DataFrame:
    total = sum(freq.values())
    return pd.DataFrame([{"rang": r, "mot": m, "fréquence": n, "% total": round(n / total * 100, 3)}
                         for r, (m, n) in enumerate(freq.most_common(k), 1)])


def zipf_fit(freq: Counter, max_rank: int | None = None) -> dict:
    """Régression linéaire log10(fréquence) ~ log10(rang). Zipf idéal : pente = -1."""
    f = np.array(sorted(freq.values(), reverse=True))
    r = np.arange(1, len(f) + 1)
    if max_rank:
        f, r = f[:max_rank], r[:max_rank]
    pente, intercept = np.polyfit(np.log10(r), np.log10(f), 1)
    return {"pente": pente, "intercept": intercept, "rangs": r, "frequences": f}


def rangs(freq: Counter) -> dict:
    """{mot: rang} (1 = plus fréquent)."""
    return {m: i for i, (m, _) in enumerate(freq.most_common(), 1)}


def hapax(freq: Counter) -> list:
    return [m for m, n in freq.items() if n == 1]
