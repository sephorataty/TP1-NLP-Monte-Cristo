"""Téléchargement, extraction et découpage du corpus Gutenberg."""
import re
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR, GUTENBERG_URL, MIRROR_URL

_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK[^*]*\*\*\*")
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK[^*]*\*\*\*")


def telecharger_corpus(raw_path: Path = DATA_DIR / "17989-0.txt",
                       sources=((GUTENBERG_URL, "utf-8"), (MIRROR_URL, "latin-1")),
                       timeout: int = 30):
    """Renvoie (texte_brut, source). Utilise le cache local s'il existe,
    sinon essaie chaque URL dans l'ordre et met le résultat en cache."""
    raw_path = Path(raw_path)
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8"), f"cache local ({raw_path.name})"
    for url, enc in sources:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            r.encoding = enc
            txt = r.text.replace("\r\n", "\n")
            raw_path.write_text(txt, encoding="utf-8")
            return txt, url
        except Exception as exc:  # réseau bloqué, 404, etc.
            print(f"Échec {url} : {exc}")
    raise RuntimeError("Corpus introuvable (aucune source accessible)")


def extraire_roman(raw_text: str):
    """Retire l'en-tête et la licence Gutenberg.
    Robuste aux deux formats de marqueurs (THE/THIS, avec ou sans titre).
    Renvoie (roman, marqueur_debut, marqueur_fin)."""
    m_debut, m_fin = _START.search(raw_text), _END.search(raw_text)
    if not (m_debut and m_fin):
        raise ValueError("Marqueurs Gutenberg introuvables")
    roman = raw_text[m_debut.end():m_fin.start()]
    roman = re.sub(r"^\s*Produced by[^\n]*\n", "", roman)
    roman = re.split(r"\n\s*End of (?:the )?Project Gutenberg", roman)[0].strip()
    return roman, m_debut.group(), m_fin.group()


def decouper_chapitres(roman: str) -> dict:
    """Découpe sur les numéros romains isolés : {'I': {'titre', 'texte'}, ...}."""
    parts = re.split(r"\n\n([IVXLC]+)\n\n", "\n\n" + roman)
    chapitres = {}
    for i in range(1, len(parts) - 1, 2):
        contenu = parts[i + 1].strip()
        chapitres[parts[i]] = {"titre": contenu.split("\n")[0], "texte": contenu}
    return chapitres


def chapitres_dataframe(chapitres: dict) -> pd.DataFrame:
    return pd.DataFrame([{"chapitre": k, "titre": v["titre"], "nb_mots": len(v["texte"].split())}
                         for k, v in chapitres.items()])


def lignes_contenant(texte: str, mot: str, n: int | None = None) -> list:
    lignes = [l.strip() for l in texte.split("\n") if mot.lower() in l.lower()]
    return lignes if n is None else lignes[:n]


def stats_corpus(roman: str, mots_par_page: int = 300) -> dict:
    nb_mots = len(roman.split())
    return {"caracteres": len(roman), "lignes": roman.count("\n"), "mots_split": nb_mots,
            "pages_estimees": round(nb_mots / mots_par_page)}
