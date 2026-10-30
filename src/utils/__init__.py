"""Utilitaires : configuration, chargement du corpus, figures."""
from .config import PROJECT_ROOT, DATA_DIR, FIG_DIR, OUT_DIR, LETTRES
from .corpus_loader import (telecharger_corpus, extraire_roman, decouper_chapitres,
                            lignes_contenant, stats_corpus)
from .plotting import save_fig

__all__ = ["PROJECT_ROOT", "DATA_DIR", "FIG_DIR", "OUT_DIR", "LETTRES",
           "telecharger_corpus", "extraire_roman", "decouper_chapitres",
           "lignes_contenant", "stats_corpus", "save_fig"]
