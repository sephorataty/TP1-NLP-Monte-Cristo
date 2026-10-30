"""Lemmatisation, POS tagging et noms propres avec SpaCy."""
import re
from collections import Counter

import pandas as pd

from ..utils.config import LETTRES

POS_LABELS = {"NOUN": "Nom commun", "VERB": "Verbe", "ADJ": "Adjectif", "ADV": "Adverbe",
              "PROPN": "Nom propre", "DET": "Déterminant", "PRON": "Pronom", "ADP": "Préposition",
              "PUNCT": "Ponctuation", "CCONJ": "Conj. coordination", "SCONJ": "Conj. subordination",
              "AUX": "Auxiliaire", "NUM": "Numéral", "X": "Autre", "INTJ": "Interjection", "SYM": "Symbole"}


def analyse_spacy(nlp, text: str):
    """Renvoie (doc, DataFrame token/lemme/POS/est_stop/est_ponc)."""
    doc = nlp(text)
    df = pd.DataFrame([{"token": t.text, "lemme": t.lemma_, "POS": t.pos_,
                        "est_stop": t.is_stop, "est_ponc": t.is_punct}
                       for t in doc if not t.is_space])
    return doc, df


def lemmes_propres(doc, min_len: int = 3) -> list:
    return [t.lemma_.lower() for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space and len(t.text) >= min_len]


def lemmes_par_pos(doc, pos: str, exclude: set = frozenset(), lemma: bool = True,
                   skip_stop: bool = True) -> Counter:
    """Compte les lemmes (ou les formes) d'une catégorie grammaticale donnée."""
    return Counter((t.lemma_.lower() if lemma else t.text) for t in doc
                   if t.pos_ == pos and t.is_alpha and not (skip_stop and t.is_stop)
                   and t.text.lower() not in exclude)


def pos_distribution(doc) -> pd.DataFrame:
    c = Counter(t.pos_ for t in doc if not t.is_space)
    tot = sum(c.values())
    return pd.DataFrame([{"POS": p, "catégorie": POS_LABELS.get(p, p), "n": n, "%": round(n / tot * 100, 1)}
                         for p, n in c.most_common()])


def extraire_propn(nlp, textes: list, exclude: set = frozenset(), batch_size: int = 64) -> Counter:
    """Noms propres sur un ensemble de textes BRUTS (les majuscules sont nécessaires)."""
    out = []
    for d in nlp.pipe(textes, batch_size=batch_size, disable=["parser", "ner", "lemmatizer"]):
        out.extend(t.text for t in d if t.pos_ == "PROPN" and len(t.text) > 2
                   and t.text.isalpha() and t.text.lower() not in exclude)
    return Counter(out)


def stem_vs_lemma(nlp, stemmer, mots: list, phrases: list, attendu: dict) -> pd.DataFrame:
    """Compare radical et lemme, le lemme étant calculé DANS une phrase réelle du corpus."""
    contexte = {}
    for p in phrases:
        toks = set(re.findall(rf"[{LETTRES}]+", p.lower()))
        for m in mots:
            if m not in contexte and m in toks:
                contexte[m] = p
    lem = {}
    for m, p in contexte.items():
        for t in nlp(p):
            if t.text.lower() == m:
                lem[m] = (t.lemma_.lower(), t.pos_)
                break

    def ok(x, y):
        return "oui" if x.replace("oe", "œ") in y.split(" ") or x == y else "non"

    df = pd.DataFrame({"Mot original": mots,
                       "Stemming (Snowball)": [stemmer.stem(m) for m in mots],
                       "Lemme (SpaCy)": [lem.get(m, ("—", ""))[0] for m in mots],
                       "POS": [lem.get(m, ("", "—"))[1] for m in mots]})
    df["Attendu"] = df["Mot original"].map(attendu)
    df["Stem correct ?"] = [ok(a, b) for a, b in zip(df["Stemming (Snowball)"], df["Attendu"])]
    df["Lemme correct ?"] = [ok(a, b) for a, b in zip(df["Lemme (SpaCy)"], df["Attendu"])]
    return df
