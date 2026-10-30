# TP 1 — Introduction au NLP : *Le Comte de Monte-Cristo*

**Master 2 Intelligence Artificielle — Traitement Automatique du Langage Naturel**

Pipeline NLP complet (prétraitement, représentation et analyse) appliqué au **Tome I du *Comte de Monte-Cristo*** d'Alexandre Dumas (Project Gutenberg #17989).

| Livrable | Fichier |
|---|---|
| Notebook de démonstration exécuté (vitrine) | [`TP1_NLP_Monte_Cristo.ipynb`](TP1_NLP_Monte_Cristo.ipynb) |
| Package réutilisable | [`src/`](src/) |
| Rapport avec toutes les réponses (Word / PDF) | [`rapport/TP1_NLP_Master2_IA_Reponses.docx`](rapport/TP1_NLP_Master2_IA_Reponses.docx) · [`.pdf`](rapport/TP1_NLP_Master2_IA_Reponses.pdf) |
| Figures · exports CSV | [`figures/`](figures/) · [`outputs/`](outputs/) |

## Architecture du projet

Le code est organisé en **packages réutilisables** : le notebook ne contient que les appels, les affichages
et les commentaires ; toute la logique est dans `src/` et peut être réutilisée pour analyser un autre corpus.

```
TP1-NLP-Monte-Cristo/
├── TP1_NLP_Monte_Cristo.ipynb      # Notebook de démonstration (vitrine), exécuté
├── README.md
├── requirements.txt
├── .gitignore                      # exclut nlp_env/, caches…
│
├── src/                            # Package source principal
│   ├── __init__.py
│   ├── utils/                      # Données et configuration
│   │   ├── config.py               #   chemins, URLs Gutenberg, alphabet français
│   │   ├── corpus_loader.py        #   téléchargement + cache, extraction (regex robuste), chapitres
│   │   └── plotting.py             #   sauvegarde homogène des figures
│   ├── preprocessing/              # Nettoyage et normalisation lexicale
│   │   ├── text_cleaner.py         #   clean_text (sujet), clean_text_v2, clean_text_leger, prep_brut
│   │   ├── tokenizer.py            #   mots / phrases, n-grammes, comparaison NLTK vs SpaCy
│   │   ├── stop_words_filter.py    #   stop words NLTK ∪ SpaCy ∪ contextuels, filtrage
│   │   ├── stemmer.py              #   Snowball, familles de mots, racines thématiques
│   │   └── lemmatizer.py           #   lemmes, POS, noms propres, stemming vs lemmatisation
│   ├── analytics/                  # Analyses et visualisations
│   │   ├── frequency.py            #   top N, loi de Zipf (régression), rangs, hapax
│   │   ├── characters.py           #   contexte des personnages, mots distinctifs, présence par bloc
│   │   └── visualization.py        #   barres, Zipf, WordClouds, heatmaps
│   ├── vectorization/              # Représentations vectorielles
│   │   ├── documents.py            #   découpage en blocs, repérage des chapitres
│   │   ├── bow_vectorizer.py       #   matrice Document-Terme (CountVectorizer), densité
│   │   └── tfidf_vectorizer.py     #   TF-IDF, similarité cosinus, TfidfSearchEngine
│   └── generation_rapport/
│       └── rapport_docx.py         #   remplit le sujet .docx avec réponses et figures
│
├── tests/test_pipeline.py          # 12 tests unitaires (pytest)
├── notebook_src/tp1_nlp.py         # source texte du notebook (jupytext)
├── data/                           # corpus brut + roman extrait (cache)
├── figures/                        # 12 graphiques générés
├── outputs/                        # exports CSV
└── rapport/                        # rapport final .docx / .pdf
```

Exemple de réutilisation sur un autre texte :

```python
from src.utils import telecharger_corpus, extraire_roman
from src.preprocessing import clean_text_v2, tokenize_words, build_custom_stopwords, filter_tokens, load_spacy
from src.vectorization import decouper_blocs, build_tfidf, TfidfSearchEngine

roman, *_ = extraire_roman(texte_gutenberg)
blocs, noms = decouper_blocs(clean_text_v2(roman), taille=2000)
vec, X, df = build_tfidf(blocs, noms, stop_words=build_custom_stopwords(load_spacy()))
TfidfSearchEngine(vec, X, noms, preprocess=clean_text_v2).search("château prison")
```

## Pipeline

```
Gutenberg → extraction → nettoyage → tokenisation → fréquences / Zipf → stop words
         → WordCloud → stemming (Snowball) → lemmatisation + POS (SpaCy)
         → matrice Document-Terme (BoW) → TF-IDF → similarité cosinus / moteur de recherche
```

## Résultats clés

| Indicateur | Valeur |
|---|---|
| Roman extrait | 727 472 caractères · 121 578 mots (`.split`) · 31 chapitres |
| Tokens NLTK après nettoyage | 130 542 · vocabulaire 11 142 · TTR 0,085 |
| Loi de Zipf | pente log-log **–1,15** (théorie : –1) |
| Hapax | 5 434 = 48,8 % du vocabulaire |
| Suppression des stop words | –61,4 % des tokens |
| Stemming Snowball | –47,7 % du vocabulaire |
| Stemming vs lemmatisation (15 mots) | 1/15 corrects vs **13/15** |
| BoW / TF-IDF | 20 blocs × 300 termes · 54,9 % de zéros (93,7 % sans filtrage) |
| TF-IDF vs BoW | chevauchement des top-5 entre blocs : 0,22 → **0,09** |
| Requête « château if prison dantès » | Bloc_15 = chapitre VIII *Le château d'If* |

<p align="center">
  <img src="figures/07_wordcloud_roman.png" width="80%"><br>
  <img src="figures/10_personnages_fil_du_roman.png" width="80%">
</p>

## Corrections apportées au code du sujet

1. **Marqueurs Gutenberg** : le fichier réel contient `*** START OF THIS PROJECT GUTENBERG EBOOK LE COMTE DE MONTE-CRISTO, TOME I ***` et non `... EBOOK 17989 ***` → `split(DEBUT)[1]` levait une `IndexError`. Remplacé par une regex robuste.
2. **`clean_text`** : les apostrophes typographiques étaient supprimées *avant* d'être normalisées (étapes 4/5 inversées). Ajout de la gestion des élisions (`l'abbé` → `l abbé`), des mots composés (`monte-cristo` conservé), des inversions (`dit-il`) et des nombres.
3. **`sent_tokenize(roman_clean)`** : sur un texte sans ponctuation, on obtient **une seule phrase** → les WordClouds par personnage auraient été identiques. La segmentation en phrases se fait désormais sur le texte brut.
4. **PROPN** : la détection des noms propres a besoin des majuscules → SpaCy appliqué au texte brut.

## Exécution

```bash
python -m venv nlp_env
source nlp_env/bin/activate            # Windows : nlp_env\Scripts\activate
pip install -r requirements.txt
python -m spacy download fr_core_news_sm

jupyter notebook TP1_NLP_Monte_Cristo.ipynb      # le notebook (≈ 30 s d'exécution)
python -m pytest tests                           # tests unitaires
python -m src.generation_rapport.rapport_docx sujet.docx rapport/sortie.docx   # rapport Word
```

Le corpus est mis en cache dans `data/` (le chargeur tente gutenberg.org puis le miroir GitHub du Project Gutenberg).
Pour régénérer le notebook depuis sa source : `jupytext --to ipynb notebook_src/tp1_nlp.py -o TP1_NLP_Monte_Cristo.ipynb`.

## Environnement testé

Python 3.13 · NLTK 3.10.4 · SpaCy 3.8 (`fr_core_news_sm` 3.8.0) · scikit-learn 1.9.1 · pandas 3.0.5 · Matplotlib 3.11 · WordCloud 1.9.6

---
*Corpus : Project Gutenberg, domaine public.*
