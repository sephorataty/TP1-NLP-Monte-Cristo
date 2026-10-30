"""Chemins et constantes partagés par tout le projet."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
OUT_DIR = PROJECT_ROOT / "outputs"

GUTENBERG_URL = "https://www.gutenberg.org/files/17989/17989-0.txt"
# Miroir GitHub officiel du Project Gutenberg (même texte, encodage latin-1)
MIRROR_URL = ("https://raw.githubusercontent.com/GITenberg/"
              "Le-comte-de-Monte-Cristo-Tome-I_17989/master/17989-8.txt")

# Lettres françaises autorisées par les regex de nettoyage
LETTRES = "a-zàâçéèêëîïôûùüÿñæœ"

for _d in (DATA_DIR, FIG_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)
