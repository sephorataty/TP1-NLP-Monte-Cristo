"""Pondération TF-IDF, similarité cosinus et mini moteur de recherche."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .bow_vectorizer import TOKEN_PATTERN


def build_tfidf(documents: list, noms: list, stop_words=None, min_df=2, max_df=0.90,
                max_features=300, sublinear_tf=True, token_pattern=TOKEN_PATTERN, **kwargs):
    """Renvoie (vectorizer, X sparse, DataFrame arrondi à 4 décimales)."""
    vec = TfidfVectorizer(lowercase=True, stop_words=list(stop_words) if stop_words else None,
                          min_df=min_df, max_df=max_df, max_features=max_features,
                          sublinear_tf=sublinear_tf, use_idf=True, smooth_idf=True,
                          token_pattern=token_pattern, **kwargs)
    X = vec.fit_transform(documents)
    df = pd.DataFrame(X.toarray().round(4), index=noms, columns=vec.get_feature_names_out())
    return vec, X, df


def top_terms_per_doc(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    out = pd.DataFrame({d: list(df.loc[d].sort_values(ascending=False).head(k).index) for d in df.index}).T
    out.columns = [f"#{i}" for i in range(1, k + 1)]
    return out


def similarity_matrix(X, noms: list) -> pd.DataFrame:
    return pd.DataFrame(cosine_similarity(X).round(3), index=noms, columns=noms)


def top_pairs(sim_df: pd.DataFrame) -> pd.Series:
    """Paires de documents distinctes triées par similarité décroissante."""
    mask = np.triu(np.ones(sim_df.shape, dtype=bool), k=1)
    return sim_df.where(mask).stack().dropna().sort_values(ascending=False)


def chevauchement_top(df: pd.DataFrame, k: int = 5) -> float:
    """Part moyenne de mots communs entre les top-k de deux documents (plus bas = plus discriminant)."""
    tops = [set(df.loc[d].sort_values(ascending=False).head(k).index) for d in df.index]
    n = len(tops)
    return float(np.mean([len(tops[i] & tops[j]) / k for i in range(n) for j in range(i + 1, n)]))


class TfidfSearchEngine:
    """Moteur de recherche par similarité cosinus dans un espace vectoriel (TF-IDF ou BoW)."""

    def __init__(self, vectorizer, X, noms: list, preprocess=None):
        self.vectorizer, self.X, self.noms = vectorizer, X, noms
        self.preprocess = preprocess or (lambda s: s)

    def termes_reconnus(self, requete: str) -> list:
        analyzer = self.vectorizer.build_analyzer()
        return [t for t in analyzer(self.preprocess(requete)) if t in self.vectorizer.vocabulary_]

    def scores(self, requete: str) -> np.ndarray:
        return cosine_similarity(self.vectorizer.transform([self.preprocess(requete)]), self.X)[0]

    def search(self, requete: str, k: int = 3) -> list:
        sc = self.scores(requete)
        return sorted(zip(self.noms, sc.round(4)), key=lambda x: -x[1])[:k]
