"""Stop words génériques (NLTK + SpaCy) et contextuels (spécifiques au roman)."""
from nltk.corpus import stopwords

# Mots ajoutés pour ce corpus : incises de dialogue, formules de politesse, adverbes vides
STOP_CONTEXTUELS = {
    "dit", "disant", "répondit", "reprit", "ajouta", "cria",
    "monsieur", "madame", "mademoiselle", "car", "alors", "puis",
    "bien", "encore", "tout", "très", "plus", "non", "oui", "si",
}


def get_stopwords(nlp=None, language: str = "french"):
    """Renvoie (stop_words_nltk, stop_words_spacy)."""
    sw_nltk = set(stopwords.words(language))
    sw_spacy = set(nlp.Defaults.stop_words) if nlp is not None else set()
    return sw_nltk, sw_spacy


def build_custom_stopwords(nlp=None, extra=STOP_CONTEXTUELS) -> set:
    sw_nltk, sw_spacy = get_stopwords(nlp)
    return sw_nltk | sw_spacy | set(extra)


def filter_tokens(tokens: list, stop_words: set, min_len: int = 3) -> list:
    """Retire les stop words et les tokens de moins de `min_len` caractères."""
    return [t for t in tokens if t.lower() not in stop_words and len(t) >= min_len]


def compare_stopword_lists(a: set, b: set) -> dict:
    return {"A": len(a), "B": len(b), "communs": len(a & b), "union": len(a | b),
            "seulement_A": sorted(a - b), "seulement_B": sorted(b - a)}
