"""Représentations vectorielles : documents, Bag of Words, TF-IDF, recherche."""
from .documents import decouper_blocs, index_chapitres, chapitres_par_bloc
from .bow_vectorizer import build_bow, densite, TOKEN_PATTERN
from .tfidf_vectorizer import (build_tfidf, top_terms_per_doc, similarity_matrix, top_pairs,
                               chevauchement_top, TfidfSearchEngine)
