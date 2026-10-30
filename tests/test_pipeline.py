"""Tests unitaires du package src (pytest)."""
from collections import Counter

import pytest

from src.utils.corpus_loader import extraire_roman, decouper_chapitres
from src.preprocessing.text_cleaner import clean_text, clean_text_v2, clean_text_leger
from src.preprocessing.stop_words_filter import filter_tokens
from src.preprocessing.stemmer import get_stemmer
from src.analytics.frequency import zipf_fit, hapax, rangs
from src.vectorization.documents import decouper_blocs
from src.vectorization.tfidf_vectorizer import build_tfidf, TfidfSearchEngine


@pytest.mark.parametrize("debut,fin", [
    ("*** START OF THE PROJECT GUTENBERG EBOOK 17989 ***", "*** END OF THE PROJECT GUTENBERG EBOOK 17989 ***"),
    ("*** START OF THIS PROJECT GUTENBERG EBOOK LE COMTE DE MONTE-CRISTO, TOME I ***",
     "*** END OF THIS PROJECT GUTENBERG EBOOK LE COMTE DE MONTE-CRISTO, TOME I ***"),
])
def test_extraire_roman_deux_formats(debut, fin):
    roman, _, _ = extraire_roman(f"licence\n{debut}\nLe texte du roman.\n{fin}\nlicence")
    assert roman == "Le texte du roman."


def test_decouper_chapitres():
    ch = decouper_chapitres("Titre\n\nI\n\nMarseille.\n\nTexte un.\n\nII\n\nLe père.\n\nTexte deux.")
    assert list(ch) == ["I", "II"] and ch["II"]["titre"] == "Le père."


def test_clean_text_sujet_coupe_monte_cristo():
    assert clean_text("Monte-Cristo") == "monte cristo"


def test_clean_text_v2():
    out = clean_text_v2("«Ah! c'est vous, Dantès!» (numéro 34) -- Monte-Cristo, dit-il, aujourd'hui.")
    assert "monte-cristo" in out.split()
    assert "34" not in out and "--" not in out
    assert "dit" in out.split() and "il" in out.split()
    assert "aujourd'hui" in out.split()
    assert "c" in out.split()            # élision séparée : c'est -> c est


def test_clean_text_v2_apostrophe_typographique():
    assert clean_text_v2("l’abbé") == "l abbé"


def test_clean_text_leger():
    assert clean_text_leger("  Edmond   DANTÈS\n") == "edmond dantès"


def test_filter_tokens():
    assert filter_tokens(["le", "dantès", "de", "vu", "abbé"], {"le", "de"}) == ["dantès", "abbé"]


def test_stemmer_faux_amis():
    st = get_stemmer()
    assert st.stem("mère") == st.stem("mer")
    assert st.stem("porte") == st.stem("port")


def test_frequences():
    freq = Counter({"de": 100, "le": 50, "et": 33, "rare": 1, "unique": 1})
    assert hapax(freq) == ["rare", "unique"]
    assert rangs(freq)["de"] == 1
    assert zipf_fit(freq)["pente"] < 0


def test_decouper_blocs():
    blocs, noms = decouper_blocs(" ".join(str(i) for i in range(10)), taille=4)
    assert len(blocs) == 3 and noms[0] == "Bloc_01" and blocs[-1] == "8 9"
    blocs, _ = decouper_blocs(" ".join(["mot"] * 100), taille=10, max_blocs=3)
    assert len(blocs) == 3


def test_moteur_recherche_tfidf():
    docs = ["prison château geôlier cachot prison", "navire capitaine mer marin navire",
            "fiançailles mariage repas mercédès", "prison navire repas"]
    vec, X, _ = build_tfidf(docs, ["A", "B", "C", "D"], min_df=1, max_df=1.0, max_features=None)
    moteur = TfidfSearchEngine(vec, X, ["A", "B", "C", "D"])
    assert moteur.search("geôlier prison")[0][0] == "A"
    assert moteur.search("mer capitaine")[0][0] == "B"
