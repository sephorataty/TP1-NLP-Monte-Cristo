"""Matrice Document-Terme (Bag of Words)."""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Mots d'au moins 3 caractères, tirets autorisés (monte-cristo)
TOKEN_PATTERN = r"(?u)\b[\w-]{3,}\b"


def build_bow(documents: list, noms: list, stop_words=None, min_df=2, max_df=0.90,
              max_features=300, token_pattern=TOKEN_PATTERN, **kwargs):
    """Renvoie (vectorizer, X sparse, DataFrame documents × termes)."""
    vec = CountVectorizer(lowercase=True, stop_words=list(stop_words) if stop_words else None,
                          min_df=min_df, max_df=max_df, max_features=max_features,
                          token_pattern=token_pattern, **kwargs)
    X = vec.fit_transform(documents)
    df = pd.DataFrame(X.toarray(), index=noms, columns=vec.get_feature_names_out())
    return vec, X, df


def densite(X) -> float:
    """% de cellules non nulles d'une matrice creuse."""
    return X.nnz / (X.shape[0] * X.shape[1]) * 100
