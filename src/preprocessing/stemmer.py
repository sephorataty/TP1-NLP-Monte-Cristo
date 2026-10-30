"""Racinisation (Snowball Stemmer, français)."""
from collections import Counter, defaultdict

import pandas as pd
from nltk.stem import SnowballStemmer


def get_stemmer(language: str = "french") -> SnowballStemmer:
    return SnowballStemmer(language)


def stem_tokens(tokens: list, stemmer=None) -> list:
    stemmer = stemmer or get_stemmer()
    return [stemmer.stem(t) for t in tokens]


def familles_par_racine(tokens, stemmer=None) -> dict:
    """{racine: {mots ayant cette racine}}."""
    stemmer = stemmer or get_stemmer()
    groupes = defaultdict(set)
    for mot in set(tokens):
        groupes[stemmer.stem(mot)].add(mot)
    return groupes


def stem_table(mots: list, vocabulaire: set, freq: Counter | None = None, stemmer=None) -> pd.DataFrame:
    """Tableau mot / radical / intelligible (= radical attesté dans le vocabulaire)."""
    stemmer = stemmer or get_stemmer()
    df = pd.DataFrame({"Mot original": mots, "Radical (Snowball)": [stemmer.stem(m) for m in mots]})
    df["Intelligible ?"] = ["oui" if r in vocabulaire else "non" for r in df["Radical (Snowball)"]]
    if freq is not None:
        df["Fréq. du mot"] = [freq[m] for m in mots]
    return df


def racines_thematiques(freq_stemmed: Counter, groupes: dict, prefixes: dict) -> pd.DataFrame:
    """Somme des occurrences des racines commençant par chaque préfixe."""
    rows = []
    for label, pref in prefixes.items():
        racines = {r: n for r, n in freq_stemmed.items() if r.startswith(pref)}
        mots = sorted({m for r in racines for m in groupes.get(r, ())})
        rows.append({"racine": label, "occurrences": sum(racines.values()),
                     "nb racines": len(racines), "exemples de mots": mots[:7]})
    return pd.DataFrame(rows).sort_values("occurrences", ascending=False)
