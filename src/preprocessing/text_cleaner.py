"""Nettoyage et normalisation du texte (français, corpus Gutenberg)."""
import re

import pandas as pd

from ..utils.config import LETTRES


def clean_text(text: str) -> str:
    """Version du sujet (conservée pour comparaison).
    Défaut connu : les apostrophes typographiques sont supprimées (étape 4)
    avant d'être normalisées (étape 5)."""
    text = text.lower()                                     # 1. minuscules
    text = re.sub(r"_([^_]+)_", r"\1", text)                # 2. italiques Gutenberg _mot_
    text = re.sub(r"[\n\r\t]+", " ", text)                  # 3. sauts de ligne
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ' ]", " ", text)  # 4. ponctuation
    text = re.sub(r"[‘’`]", "'", text)                      # 5. apostrophes
    text = re.sub(r"\s+", " ", text)                        # 6. espaces multiples
    return text.strip()                                     # 7. strip


_INVERSION = re.compile(
    r"-(?:t-)?(je|tu|il|elle|on|nous|vous|ils|elles|le|la|les|lui|moi|toi|leur|y|en|ce)\b")
_ELISION = re.compile(r"\b(qu|lorsqu|puisqu|jusqu|quoiqu|presqu|[ldjmnstc])'")


def clean_text_v2(text: str, remove_numbers: bool = True, keep_compounds: bool = True,
                  split_elisions: bool = True) -> str:
    """Nettoyage retenu pour le pipeline.
    - apostrophes normalisées AVANT le filtrage ;
    - nombres isolés supprimés (24, 1815) ;
    - mots composés conservés (monte-cristo, peut-être) ;
    - inversions séparées (dit-il -> dit il) ;
    - élisions séparées (l'homme -> l homme ; aujourd'hui préservé)."""
    text = text.lower()
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"[‘’`´]", "'", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    if remove_numbers:
        text = re.sub(r"\b\d+\b", " ", text)
    if keep_compounds:
        text = _INVERSION.sub(r" \1", text)
    if split_elisions:
        text = _ELISION.sub(r"\1 ", text)
    allowed = f"{LETTRES}'\\- " if keep_compounds else f"{LETTRES}' "
    text = re.sub(rf"[^{allowed}]", " ", text)
    text = re.sub(rf"(?<![{LETTRES}])-+|-+(?![{LETTRES}])", " ", text)   # tirets isolés (--)
    text = re.sub(rf"(?<![{LETTRES}])'|'(?![{LETTRES}])", " ", text)     # apostrophes isolées
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_leger(text: str) -> str:
    """Minuscules + espaces multiples uniquement."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def prep_brut(text: str) -> str:
    """Texte brut lisible par SpaCy / sent_tokenize : garde ponctuation et majuscules,
    retire seulement les _italiques_ et remplace les tirets de dialogue '--'."""
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\s*--\s*", " — ", text)
    return re.sub(r"\s+", " ", text).strip()


def comparer_versions(versions: dict) -> pd.DataFrame:
    """{'nom': texte} -> tableau caractères / mots / types distincts."""
    return pd.DataFrame([{"version": k, "caractères": len(v), "mots (.split)": len(v.split()),
                          "types distincts": len(set(v.split()))} for k, v in versions.items()])
