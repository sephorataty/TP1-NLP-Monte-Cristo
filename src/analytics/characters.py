"""Analyse des personnages : contexte lexical, mots distinctifs, présence au fil du texte."""
import re
from collections import Counter

import pandas as pd

from ..preprocessing.tokenizer import tokenize_words


def phrases_personnage(phrases_nettoyees: list, perso: str) -> list:
    return [p for p in phrases_nettoyees if re.search(rf"\b{perso}\b", p)]


def tokens_personnage(phrases_nettoyees: list, perso: str, stop_words: set, min_len: int = 4):
    """Tokens des phrases mentionnant le personnage (hors stop words et hors son nom)."""
    sel = phrases_personnage(phrases_nettoyees, perso)
    toks = [t for p in sel for t in tokenize_words(p)
            if t not in stop_words and len(t) >= min_len and t != perso]
    return sel, toks


def mots_distinctifs(tokens: list, freq_reference: Counter, k: int = 10, min_n: int = 4) -> list:
    """Mots sur-représentés par rapport au corpus de référence (ratio de fréquences relatives)."""
    c, tot = Counter(tokens), len(tokens)
    tot_ref = sum(freq_reference.values())
    score = {m: (n / tot) / ((freq_reference[m] + 1) / tot_ref) for m, n in c.items() if n >= min_n}
    return [m for m, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]


def distribution_personnages(blocs: list, persos: list, index=None) -> pd.DataFrame:
    """Nombre d'occurrences de chaque personnage dans chaque bloc."""
    return pd.DataFrame({p: [b.split().count(p) for b in blocs] for p in persos}, index=index)
