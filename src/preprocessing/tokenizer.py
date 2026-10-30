"""Tokenisation en mots / phrases, n-grammes, comparaison NLTK vs SpaCy."""
from collections import Counter

import nltk
import pandas as pd
from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize

NLTK_RESOURCES = ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4")


def ensure_nltk_resources(resources=NLTK_RESOURCES):
    for res in resources:
        nltk.download(res, quiet=True)


def load_spacy(model: str = "fr_core_news_sm"):
    import spacy
    return spacy.load(model)


def tokenize_words(text: str, language: str = "french") -> list:
    return word_tokenize(text, language=language)


def tokenize_sentences(text: str, language: str = "french") -> list:
    """À appliquer sur le texte BRUT (la ponctuation délimite les phrases)."""
    return sent_tokenize(text, language=language)


def spacy_tokens(nlp, text: str) -> list:
    return [t.text for t in nlp(text) if not t.is_space]


def compare_tokenizers(nlp, text: str, k: int = 10) -> dict:
    """Compare NLTK et SpaCy sur un même texte."""
    tk_n, tk_s = tokenize_words(text), spacy_tokens(nlp, text)
    return {"nltk": tk_n, "spacy": tk_s,
            "seulement_nltk": (Counter(tk_n) - Counter(tk_s)).most_common(k),
            "seulement_spacy": (Counter(tk_s) - Counter(tk_n)).most_common(k)}


def top_ngrams(tokens: list, n: int = 2, k: int = 20, exclude: set | None = None,
               min_len: int = 0) -> list:
    """N-grammes les plus fréquents, en excluant éventuellement des mots (stop words)."""
    if exclude is not None:
        tokens = [t for t in tokens if t not in exclude and len(t) >= min_len]
    return Counter(ngrams(tokens, n)).most_common(k)


def tokens_dataframe(tokens: list) -> pd.DataFrame:
    df = pd.DataFrame({"position": range(len(tokens)), "token": tokens})
    df["longueur"] = df["token"].str.len()
    return df
