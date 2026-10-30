# %% [markdown]
# # TP 1 — Introduction au NLP
# ## Prétraitement, Représentation et Analyse de Textes
# **Master 2 Intelligence Artificielle — Traitement Automatique du Langage Naturel**
#
# *Corpus : Le Comte de Monte-Cristo, Tome I — Alexandre Dumas (Project Gutenberg #17989)*
#
# Ce notebook est la **vitrine** du projet : toute la logique réutilisable est dans le package `src/`,
# le notebook se contente d'appeler les fonctions, d'afficher les résultats et de les commenter.
#
# ```
# src/
# ├── utils/            config.py · corpus_loader.py · plotting.py
# ├── preprocessing/    text_cleaner.py · tokenizer.py · stop_words_filter.py · stemmer.py · lemmatizer.py
# ├── analytics/        frequency.py · characters.py · visualization.py
# ├── vectorization/    documents.py · bow_vectorizer.py · tfidf_vectorizer.py
# └── generation_rapport/ rapport_docx.py
# ```
#
# > Les réponses rédigées aux questions sont dans `rapport/TP1_NLP_Master2_IA_Reponses.docx`.

# %% [markdown]
# ---
# # Partie 1 — Installation et Configuration
# ```bash
# python -m venv nlp_env && source nlp_env/bin/activate   # Windows : nlp_env\Scripts\activate
# pip install -r requirements.txt
# python -m spacy download fr_core_news_sm
# ```

# %%
import re
import sys
import time
import random
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))   # rend le package src importable depuis le notebook

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk, spacy, sklearn, wordcloud
from nltk import bigrams, trigrams

from src.utils import (telecharger_corpus, extraire_roman, decouper_chapitres, lignes_contenant,
                       stats_corpus, save_fig, OUT_DIR, DATA_DIR)
from src.utils.corpus_loader import chapitres_dataframe
from src.preprocessing import (clean_text, clean_text_v2, clean_text_leger, prep_brut, comparer_versions,
                               ensure_nltk_resources, load_spacy, tokenize_words, tokenize_sentences,
                               compare_tokenizers, top_ngrams, tokens_dataframe,
                               get_stopwords, build_custom_stopwords, filter_tokens, compare_stopword_lists,
                               get_stemmer, stem_tokens, familles_par_racine, stem_table, racines_thematiques,
                               analyse_spacy, lemmes_propres, lemmes_par_pos, pos_distribution,
                               extraire_propn, stem_vs_lemma)
from src.analytics import (frequency_table, zipf_fit, rangs, hapax, tokens_personnage, mots_distinctifs,
                           distribution_personnages, plot_top_frequences, plot_zipf, make_wordcloud,
                           show_wordcloud, wordcloud_grid, plot_heatmap)
from src.vectorization import (decouper_blocs, index_chapitres, chapitres_par_bloc, build_bow, densite,
                               build_tfidf, top_terms_per_doc, similarity_matrix, top_pairs,
                               chevauchement_top, TfidfSearchEngine)
from src.vectorization.documents import chapitre_du_mot

warnings.filterwarnings('ignore')
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)
plt.rcParams['figure.dpi'] = 100

# %% [markdown]
# ### 1.3 / 1.4 Ressources NLTK et modèle SpaCy — Travail demandé a), b), c)

# %%
ensure_nltk_resources()          # punkt, punkt_tab, stopwords, wordnet, omw-1.4
nlp = load_spacy('fr_core_news_sm')
print('Modèle SpaCy chargé avec succès ! Pipeline :', nlp.pipe_names)

# b) Versions
print(f'\nPython      : {sys.version.split()[0]}')
print(f'NLTK        : {nltk.__version__}')
print(f'SpaCy       : {spacy.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'Pandas      : {pd.__version__}')
print(f'Matplotlib  : {matplotlib.__version__}')
print(f'WordCloud   : {wordcloud.__version__}')

# c) Vérification du modèle
doc = nlp("Edmond Dantès fut injustement emprisonné au château d'If.")
print('\n', [(t.text, t.pos_) for t in doc])

# %% [markdown]
# ---
# # Partie 2 — Chargement du corpus
#
# **Remarque technique.** Le code du sujet utilise `*** START OF THE PROJECT GUTENBERG EBOOK 17989 ***`,
# alors que le fichier réel contient `*** START OF THIS PROJECT GUTENBERG EBOOK LE COMTE DE MONTE-CRISTO, TOME I ***`
# → `split(DEBUT)[1]` lève une `IndexError`. `extraire_roman()` utilise une **regex** robuste aux deux formats.
# `telecharger_corpus()` met le fichier en cache dans `data/` et bascule sur le miroir GitHub du Project Gutenberg
# si gutenberg.org est inaccessible.

# %%
raw_text, source = telecharger_corpus()
print(f'Téléchargement terminé (source : {source}).')
print(f'Taille brute : {len(raw_text):,} caractères')
print(f'Aperçu :\n{raw_text[:300]}')

roman, marq_debut, marq_fin = extraire_roman(raw_text)
print('\nMarqueur début :', marq_debut)
print('Marqueur fin   :', marq_fin)
(DATA_DIR / 'monte_cristo.txt').write_text(roman, encoding='utf-8')   # 2.3 sauvegarde locale

# %% [markdown]
# ### Travail demandé — Partie 2

# %%
st = stats_corpus(roman)
print(f"a) Nombre total de caractères : {st['caracteres']:,}")
print(f"   Nombre total de lignes       : {st['lignes']:,}")
print(f'\n--- 500 premiers caractères ---\n{roman[:500]}')
print(f'\n--- 200 derniers caractères (fichier Gutenberg brut) ---\n{raw_text[-200:]}')
print(f'\n--- 200 derniers caractères du roman extrait ---\n{roman[-200:]}')

# %%
nb_mots_split = st['mots_split']
print(f'b) Nombre de mots (.split()) : {nb_mots_split:,}')
print(f"c) Occurrences de \"dantès\" : {roman.lower().count('dantès')}")
lignes_v = lignes_contenant(roman, 'vengeance')
print(f'd) {len(lignes_v)} lignes contiennent "vengeance". Les 5 premières :')
for l in lignes_v[:5]:
    print('   -', l)
print(f"e) Estimation : {st['pages_estimees']} pages (300 mots/page)")

# %%
chapitres = decouper_chapitres(roman)
print(f'{len(chapitres)} chapitres détectés')
print(chapitres_dataframe(chapitres).to_string(index=False))

# %% [markdown]
# ---
# # Partie 3 — Nettoyage du texte (`src/preprocessing/text_cleaner.py`)

# %%
print(repr(roman[1000:1200]))     # 3.1 caractères cachés

# %% [markdown]
# ### a) Fonction du sujet (`clean_text`)
# *Remarque :* dans la version du sujet, les apostrophes typographiques `’` sont supprimées (étape 4)
# **avant** d'être normalisées (étape 5). La version finale `clean_text_v2` corrige cet ordre.

# %%
roman_clean_tp = clean_text(roman)
print(f'a) Taille avant : {len(roman):,} caractères')
print(f'   Taille après : {len(roman_clean_tp):,} caractères')
print(f'   Réduction    : {(1 - len(roman_clean_tp)/len(roman))*100:.1f}%')
print(f'\nAperçu nettoyé :\n{roman_clean_tp[:400]}')

extrait = roman[2800:3100]
print('\nAVANT NETTOYAGE :'); print(extrait)
print('\nAPRÈS NETTOYAGE :'); print(clean_text(extrait))

# %%
# b) Extrait bruité / c) Monte-Cristo
bruite = '«Ah! c\'est vous, Dantès!» cria l\'homme... (numéro 34)\n\n-- Il est mort!!'
print('b) Avant :', repr(bruite))
print('   Après :', repr(clean_text(bruite)))
print('\nc)', repr(clean_text("Le comte de Monte-Cristo arriva.")))
print('   Occurrences "monte-cristo" dans le roman brut :', roman.lower().count('monte-cristo'))

# %% [markdown]
# **c)** `Monte-Cristo` devient **`monte cristo` (2 tokens)**. Solution dans `clean_text_v2` : tiret autorisé
# **entre deux lettres**, tirets isolés (`--`) supprimés, inversions verbales séparées (`dit-il` → `dit il`).
# **d)** nombres isolés supprimés. On sépare aussi les **élisions** (`l'homme` → `l homme`).

# %%
for s in [bruite, "Le comte de Monte-Cristo arriva le 24 février 1815.",
          "--Eh bien! dit-il, qu'y a-t-il? s'écria-t-il aujourd'hui."]:
    print(repr(s)); print('  ->', repr(clean_text_v2(s)))

# %%
# e) Version légère + comparaison
roman_clean = clean_text_v2(roman)        # <- texte utilisé dans toute la suite
roman_leger = clean_text_leger(roman)
print(comparer_versions({'brut': roman, 'clean_text (sujet)': roman_clean_tp,
                         'clean_text_v2 (finale)': roman_clean, 'clean_text_leger': roman_leger}).to_string(index=False))
print('\nExtrait version légère :', roman_leger[2000:2250])
print('\nExtrait version finale :', roman_clean[1900:2150])
print('\n"monte-cristo" conservé ?', roman_clean.count('monte-cristo'), 'occurrences')

# %% [markdown]
# ---
# # Partie 4 — Tokenisation (`src/preprocessing/tokenizer.py`)

# %%
t0 = time.time()
tokens = tokenize_words(roman_clean)
t_nltk = time.time() - t0
print(f'Nombre total de tokens       : {len(tokens):,}')
print(f'Vocabulaire (tokens uniques) : {len(set(tokens)):,}')
print(f'Richesse lexicale (TTR)      : {len(set(tokens))/len(tokens):.4f}')
print(f'Temps NLTK                   : {t_nltk:.2f} s')
print('\n50 premiers tokens :'); print(tokens[:50])

# %% [markdown]
# ### c) NLTK vs SpaCy (5 000 premiers caractères)

# %%
cmp_clean = compare_tokenizers(nlp, roman_clean[:5000], k=8)
cmp_brut = compare_tokenizers(nlp, roman[:5000], k=10)
print(f"Texte NETTOYÉ : NLTK = {len(cmp_clean['nltk'])} tokens | SpaCy = {len(cmp_clean['spacy'])} tokens")
print('  Tokens propres à NLTK  :', cmp_clean['seulement_nltk'])
print('  Tokens propres à SpaCy :', cmp_clean['seulement_spacy'])
print(f"\nTexte BRUT    : NLTK = {len(cmp_brut['nltk'])} tokens | SpaCy = {len(cmp_brut['spacy'])} tokens")
print('  Tokens propres à NLTK  :', cmp_brut['seulement_nltk'])
print('  Tokens propres à SpaCy :', cmp_brut['seulement_spacy'])

for p in ["L'abbé Faria s'écria : « Dantès ! »", "Aujourd'hui, M. Morrel est-il arrivé à Marseille ?",
          "Le 24 février 1815, la vigie de Notre-Dame-de-la-Garde signala le trois-mâts le Pharaon."]:
    n_, s_ = tokenize_words(p), [t.text for t in nlp(p)]
    print(f'\n{p}\n  NLTK  ({len(n_)}) : {n_}\n  SpaCy ({len(s_)}) : {s_}')

# %% [markdown]
# ### 4.3 Phrases
# `sent_tokenize` sur le texte **nettoyé** (code du sujet) ne trouve qu'une phrase : la ponctuation a disparu.
# On segmente donc le texte **brut**.

# %%
print(f'Sur texte nettoyé : {len(tokenize_sentences(roman_clean[:20000]))} phrase(s) détectée(s) !')
phrases = tokenize_sentences(re.sub(r'\s*\n\s*', ' ', roman[:20000]))
print(f'Sur texte brut    : {len(phrases)} phrases (20 000 premiers caractères)')
print(f"Longueur moyenne d'une phrase : {sum(len(p.split()) for p in phrases)/len(phrases):.1f} mots")
for p in [p for p in phrases if len(p.split()) > 12][:3]:
    print(f' - {p[:100]}...')
phrases_roman = tokenize_sentences(prep_brut(roman))     # réutilisées en parties 7 et 9
print(f'\nNombre total de phrases du roman : {len(phrases_roman):,}')

# %%
# 4.4 N-grammes + d)
print('Bigrammes (20 premiers) :', list(bigrams(tokens[:200]))[:20])
print('\nTrigrammes (10 premiers) :', list(trigrams(tokens[:200]))[:10])
print('\nd) 20 bigrammes les plus fréquents (tous tokens) :')
for (a, b), n in top_ngrams(tokens, 2, 20):
    print(f'   {a} {b:<15} {n}')

sw_nltk, sw_spacy = get_stopwords(nlp)
print('\nBigrammes les plus fréquents (sans stop words) :')
for (a, b), n in top_ngrams(tokens, 2, 20, exclude=sw_nltk | sw_spacy, min_len=3):
    print(f'   {a} {b:<18} {n}')

# %%
# e) DataFrame des tokens
df_tok = tokens_dataframe(tokens)
print(df_tok.head(10).to_string(index=False))
print(f'\nLongueur moyenne des tokens             : {df_tok.longueur.mean():.2f} caractères')
print(f'Longueur moyenne des types (vocabulaire): {pd.Series(list(set(tokens))).str.len().mean():.2f} caractères')
print('Token le plus long :', df_tok.loc[df_tok.longueur.idxmax(), 'token'])
print(df_tok.longueur.describe().round(2).to_string())

# %% [markdown]
# ---
# # Partie 5 — Analyse fréquentielle (`src/analytics/frequency.py`)

# %%
freq = Counter(tokens)
print('a) TOP 20 mots les plus fréquents :')
print(frequency_table(freq, 20).to_string(index=False))
top20 = freq.most_common(20)
print(f'\nLes 20 premiers mots couvrent {sum(n for _, n in top20)/len(tokens)*100:.1f}% des tokens.')

# %%
plot_top_frequences(top20, 'Top 20 mots les plus fréquents — Le Comte de Monte-Cristo (Tome I)',
                    name='05_top20_frequences')

# %%
# c) Loi de Zipf
fit = zipf_fit(freq)
fit_1000 = zipf_fit(freq, max_rank=1000)
plot_zipf(fit, 'Loi de Zipf — Le Comte de Monte-Cristo', name='05_loi_de_zipf')
print(f"Pente log-log globale        : {fit['pente']:.3f}  (Zipf idéal : -1)")
print(f"Pente sur les 1000 premiers  : {fit_1000['pente']:.3f}")
top5 = freq.most_common(5)
for i, (m, n) in enumerate(top5):
    print(f'  Rang {i+1}: {m:<5} ({n} fois) — ratio avec rang 1 : {top5[0][1]/n:.2f}')

# %%
# d) Mots thématiques / e) Hapax
classement = rangs(freq)
for m in ['dantès', 'vengeance', 'liberté', 'prison']:
    print(f"d) {m:<10} fréquence = {freq[m]:>4}   rang = {classement.get(m, '—')}")
hap = hapax(freq)
print(f'\ne) Hapax legomena : {len(hap):,} mots = {len(hap)/len(freq)*100:.1f}% du vocabulaire '
      f'mais {len(hap)/len(tokens)*100:.2f}% des tokens')
random.seed(0)
print('   Exemples :', random.sample(hap, 25))

# %% [markdown]
# ---
# # Partie 6 — Stop words (`src/preprocessing/stop_words_filter.py`)

# %%
c_sw = compare_stopword_lists(sw_nltk, sw_spacy)
print(f"a) Stop words NLTK  : {c_sw['A']}\n   Stop words SpaCy : {c_sw['B']}")
print(f"   Communs          : {c_sw['communs']}\n   Union            : {c_sw['union']}")
print(f"\nDans NLTK, absents de SpaCy ({len(c_sw['seulement_A'])}) : {c_sw['seulement_A'][:25]}")
print(f"Dans SpaCy, absents de NLTK ({len(c_sw['seulement_B'])}) : {c_sw['seulement_B'][:25]}")

# %%
sw_custom = build_custom_stopwords(nlp)          # NLTK ∪ SpaCy ∪ stop words contextuels
tokens_no_stop = filter_tokens(tokens, sw_custom, min_len=3)
freq_clean = Counter(tokens_no_stop)
print(f'Tokens avant : {len(tokens):,}')
print(f'Tokens après : {len(tokens_no_stop):,}')
print(f'c) Réduction (tokens)     : {(len(tokens)-len(tokens_no_stop))/len(tokens)*100:.1f}%')
print(f'   Réduction (vocabulaire): {(1-len(set(tokens_no_stop))/len(set(tokens)))*100:.1f}% '
      f'({len(set(tokens)):,} -> {len(set(tokens_no_stop)):,} types)')
print('\nb) Top 20 après suppression des stop words :')
for mot, n in freq_clean.most_common(20):
    print(f'   {mot:<20} {n}')

print(f'\n{"Rang":<5} {"Avant":<22} {"Après"}'); print('-' * 45)
for i, (a, b) in enumerate(zip([m for m, _ in freq.most_common(10)], [m for m, _ in freq_clean.most_common(10)]), 1):
    print(f'{i:<5} {a:<22} {b}')

# %%
# d) Justification des stop words contextuels
for m in ['dit', 'répondit', 'reprit', 'monsieur', 'madame']:
    print(f'{m:<10} {freq[m]:>5} occurrences (rang {classement.get(m)})')
print('\nMots qui suivent "monsieur" :', Counter(b for a, b in bigrams(tokens) if a == 'monsieur').most_common(8))
print('Mots qui suivent "dit/répondit" :',
      Counter(b for a, b in bigrams(tokens) if a in ('dit', 'répondit')).most_common(8))

# %%
# e) Personnages
persos = ['dantès', 'edmond', 'danglars', 'fernand', 'villefort', 'faria', 'abbé', 'mercédès',
          'morrel', 'caderousse', 'franz', 'albert', 'noirtier']
rang_clean = rangs(freq_clean)
df_persos = pd.DataFrame({'personnage': persos, 'fréquence': [freq_clean[p] for p in persos],
                          'rang (après stop words)': [rang_clean.get(p) for p in persos]}
                         ).sort_values('fréquence', ascending=False)
print(df_persos.to_string(index=False))
plt.figure(figsize=(10, 4.5))
sns.barplot(data=df_persos, x='personnage', y='fréquence', color='steelblue')
plt.title('Fréquence des personnages (après suppression des stop words)')
plt.xticks(rotation=30); plt.tight_layout(); save_fig('06_personnages')

# %% [markdown]
# ---
# # Partie 7 — WordCloud (`src/analytics/visualization.py`, `src/analytics/characters.py`)

# %%
texte_clean = ' '.join(tokens_no_stop)
wc = make_wordcloud(texte_clean, max_words=150, min_font_size=8)
show_wordcloud(wc, 'WordCloud — Le Comte de Monte-Cristo (Tome I)', name='07_wordcloud_roman')
print('a) 5 mots les plus visibles :', list(wc.words_.keys())[:5])

# %% [markdown]
# ### b) Par personnage
# Le code du sujet appelle `sent_tokenize(roman_clean)` → **une seule phrase** (tout le roman) : les nuages seraient
# identiques. On segmente le texte **brut** en phrases, puis on nettoie chaque phrase ; le nom du personnage est
# retiré de son nuage et on calcule ses **mots distinctifs**.

# %%
phrases_clean_list = [clean_text_v2(p) for p in phrases_roman]
clouds, titres = [], []
for perso, cmap in zip(['dantès', 'danglars', 'villefort'], ['Blues', 'Reds', 'Greens']):
    sel, toks = tokens_personnage(phrases_clean_list, perso, sw_custom, min_len=4)
    clouds.append(make_wordcloud(' '.join(toks), width=600, height=450, colormap=cmap))
    titres.append(f'{perso.capitalize()} ({len(sel)} phrases)')
    print(f'{perso.capitalize():<10} {len(sel):>4} phrases | top fréquents : {[m for m, _ in Counter(toks).most_common(8)]}')
    print(f'{"":<10} mots distinctifs : {mots_distinctifs(toks, freq_clean)}')
wordcloud_grid(clouds, titres, 'Champ lexical des personnages principaux', name='07_wordcloud_personnages')

# %% [markdown]
# ### c) Noms propres (PROPN) — sur le texte brut, car la détection a besoin des majuscules

# %%
t0 = time.time()
paragraphes = [prep_brut(p) for p in re.split(r'\n\s*\n', roman) if p.strip()]
freq_propn = extraire_propn(nlp, paragraphes, exclude=sw_custom)
print(f'SpaCy sur tout le roman : {time.time()-t0:.1f} s ({len(paragraphes):,} paragraphes)')
print('Top 30 PROPN :', freq_propn.most_common(30))
show_wordcloud(make_wordcloud(frequencies=freq_propn, colormap='viridis', max_words=100),
               'Noms propres (POS = PROPN, SpaCy)', name='07_wordcloud_noms_propres')

# %%
# d) Paramètres
wordcloud_grid([make_wordcloud(texte_clean, width=800, height=450, colormap='RdYlGn', max_words=mw)
                for mw in (50, 200)],
               ["colormap='RdYlGn', max_words=50", "colormap='RdYlGn', max_words=200"],
               name='07_wordcloud_parametres')

# %%
# e) Début vs fin
debut, fin = tokens_no_stop[:3000], tokens_no_stop[-3000:]
wordcloud_grid([make_wordcloud(' '.join(debut), width=800, height=450, colormap='Blues'),
                make_wordcloud(' '.join(fin), width=800, height=450, colormap='Oranges')],
               ['3000 premiers mots', '3000 derniers mots'], name='07_wordcloud_debut_fin')
print('Top début :', [m for m, _ in Counter(debut).most_common(15)])
print('Top fin   :', [m for m, _ in Counter(fin).most_common(15)])
communs = set(debut) & set(fin)
print(f'Vocabulaire commun : {len(communs)} mots sur {len(set(debut))} (début) / {len(set(fin))} (fin) '
      f'— Jaccard = {len(communs)/len(set(debut)|set(fin)):.2f}')

# %% [markdown]
# ---
# # Partie 8 — Stemming (`src/preprocessing/stemmer.py`)

# %%
stemmer = get_stemmer('french')
familles = {'vengeance': ['vengeance', 'venger', 'vengeur', 'vindicatif'],
            'prison': ['prison', 'prisonnier', 'emprisonner', 'emprisonnement'],
            'liberté': ['liberté', 'libérer', 'libérateur', 'libération'],
            'mer': ['mer', 'marin', 'maritime', 'navire', 'naviguer']}
print(f'{"Famille":<11}{"Mot original":<20} {"Racine (stem)"}'); print('-' * 50)
print(f'{"":<11}{"dantès":<20} {stemmer.stem("dantès")}')
for fam, ms in familles.items():
    for m in ms:
        print(f'{fam:<11}{m:<20} {stemmer.stem(m)}')
    rac = sorted({stemmer.stem(m) for m in ms})
    print(f'{"":<11}-> {len(rac)} racine(s) distincte(s) : {rac}')

# %%
# b) Corpus
tokens_stemmed = stem_tokens(tokens_no_stop, stemmer)
freq_stemmed = Counter(tokens_stemmed)
v_av, v_ap = len(set(tokens_no_stop)), len(set(tokens_stemmed))
print(f'Vocabulaire avant stemming : {v_av:,} mots')
print(f'Vocabulaire après stemming : {v_ap:,} racines')
print(f'b) Réduction du vocabulaire : {(1 - v_ap/v_av)*100:.1f}%')
print('\nTop 20 racines :')
for r, n in freq_stemmed.most_common(20):
    print(f'   {r:<20} {n}')

groupes = familles_par_racine(tokens_no_stop, stemmer)
print('\nFamilles de mots (10 plus grandes) :')
for racine, ms in sorted(groupes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f'  [{racine:<12}] ({len(ms):>2}) -> {sorted(ms)[:8]}')

# %%
# c) Faux amis du stemmer
for groupe in [('porte', 'porter', 'port'), ('sort', 'sortir', 'sortie'), ('mari', 'marin', 'marine'),
               ('vers', 'verser', 'verre'), ('livre', 'livrer', 'libre'), ('somme', 'sommeil', 'sommes'),
               ('pas', 'passer', 'passion'), ('mère', 'mer', 'merci'), ('été', 'étaient', 'état')]:
    print('  ', ' | '.join(f'{m} -> {stemmer.stem(m)}' for m in groupe))

# %%
# d) 15 mots caractéristiques / e) racines thématiques
mots_15 = ['vengeance', 'prisonnier', 'château', 'liberté', 'navire', 'trésor', 'abbé', 'capitaine',
           'mariage', 'procureur', 'emprisonné', 'évasion', 'fiançailles', 'cachot', 'marseille']
print(stem_table(mots_15, set(tokens), freq, stemmer).to_string(index=False))
print("\n(« Intelligible » = le radical est lui-même un mot attesté dans le roman)\n")
print(racines_thematiques(freq_stemmed, groupes,
                          {'veng-': 'veng', 'pris-': 'pris', 'liber-': 'liber', 'dant-': 'dant'}).to_string(index=False))

# %% [markdown]
# ---
# # Partie 9 — Lemmatisation (`src/preprocessing/lemmatizer.py`)

# %%
chap1 = prep_brut(chapitres['I']['texte'])
t0 = time.time()
doc, df_tokens = analyse_spacy(nlp, chap1)
t_ch1 = time.time() - t0
df_tokens.to_csv(OUT_DIR / 'chapitre1_spacy_tokens.csv', index=False)
print(f'Chapitre I : « {chapitres["I"]["titre"]} » — {len(df_tokens):,} tokens analysés en {t_ch1:.2f} s')
print('(DataFrame complet exporté dans outputs/chapitre1_spacy_tokens.csv)\n')
print(df_tokens.head(40).to_string(index=False))

# %%
print('Top 20 lemmes (chapitre I) :')
for l, n in Counter(lemmes_propres(doc)).most_common(20):
    print(f'   {l:<20} {n}')
print('\nb) Top 15 NOMS (lemmatisés) :', lemmes_par_pos(doc, 'NOUN').most_common(15))
print('\nc) Noms propres :', lemmes_par_pos(doc, 'PROPN', exclude=sw_custom, lemma=False,
                                           skip_stop=False).most_common(25))
print('\n   (complément NER) :', Counter((e.text, e.label_) for e in doc.ents).most_common(15))

# %%
# 9.3 POS
df_pos = pos_distribution(doc)
print(df_pos.to_string(index=False))
plt.figure(figsize=(10, 4.5))
plt.bar(df_pos['catégorie'], df_pos['n'], color='steelblue'); plt.ylabel('Nombre de tokens')
plt.title('Distribution des catégories grammaticales — Chapitre I'); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); save_fig('09_pos_chapitre1')

# %%
# d) Stemming vs lemmatisation (lemme calculé dans une phrase réelle du roman)
cibles = ['navire', 'prisonnier', 'vengeance', 'chevaux', 'yeux', 'allait', 'fut', 'meilleurs',
          'aperçut', 'gendarmes', 'prisonniers', 'disait', 'émotion', 'jeunes', 'dantès']
attendu = {'navire': 'navire', 'prisonnier': 'prisonnier', 'vengeance': 'vengeance', 'chevaux': 'cheval',
           'yeux': 'œil', 'allait': 'aller', 'fut': 'être', 'meilleurs': 'bon (ou meilleur)',
           'aperçut': 'apercevoir', 'gendarmes': 'gendarme', 'prisonniers': 'prisonnier', 'disait': 'dire',
           'émotion': 'émotion', 'jeunes': 'jeune', 'dantès': 'dantès'}
df_cmp = stem_vs_lemma(nlp, stemmer, cibles, phrases_roman, attendu)
print(df_cmp.to_string(index=False))
print(f"\nStemming correct : {(df_cmp['Stem correct ?']=='oui').sum()}/15 | "
      f"Lemmatisation correcte : {(df_cmp['Lemme correct ?']=='oui').sum()}/15")
df_cmp.to_csv(OUT_DIR / 'stemming_vs_lemmatisation.csv', index=False)

# %%
# Coût computationnel (même texte : chapitre I)
t0 = time.time(); _ = stem_tokens(tokenize_words(chap1), stemmer); tn = time.time() - t0
t0 = time.time(); _ = nlp(chap1); ts = time.time() - t0
print(f'Chapitre I ({len(chap1.split()):,} mots) : NLTK tokenisation+stemming = {tn*1000:.0f} ms | '
      f'SpaCy pipeline complet = {ts*1000:.0f} ms  -> ×{ts/tn:.0f}')

# %% [markdown]
# ---
# # Partie 10 — Matrice Document-Terme & BoW (`src/vectorization/`)

# %%
TAILLE = 2000
blocs, noms_blocs = decouper_blocs(roman_clean, TAILLE, max_blocs=20)
mots_roman = roman_clean.split()
idx_chap = index_chapitres(chapitres, clean_text_v2)
chap_blocs = chapitres_par_bloc(len(blocs), TAILLE, idx_chap)
print(f'Nombre de blocs : {len(blocs)} (couvrent {len(blocs)*TAILLE:,} mots sur {len(mots_roman):,}, '
      f'soit {len(blocs)*TAILLE/len(mots_roman)*100:.0f}% du roman)')
print(f'Taille du premier bloc : {len(blocs[0].split())} mots')
print('Chapitres couverts :', dict(zip(noms_blocs, chap_blocs)))

# %%
vectorizer, X_bow, bow_df = build_bow(blocs, noms_blocs, stop_words=sw_custom)
print(f'a) Matrice BoW : {X_bow.shape[0]} documents × {X_bow.shape[1]} mots')
print(bow_df.iloc[:5, :8])
n_dantes = sum('dantès' in b.split() for b in blocs)
print(f"\n'dantès' dans le vocabulaire ? {'dantès' in bow_df.columns} (présent dans {n_dantes}/{len(blocs)} blocs "
      f"= {n_dantes/len(blocs):.0%} < max_df 90 % -> conservé)")
trop_frequents = sorted(w for w in set(' '.join(blocs).split())
                        if w not in sw_custom and len(w) > 2 and sum(w in b.split() for b in blocs)/len(blocs) > 0.9)
print('Mots (hors stop words) supprimés par max_df, présents dans > 90 % des blocs :', trop_frequents)

# %%
print('Mots les plus fréquents (somme sur tous les blocs) :')
print(bow_df.sum().sort_values(ascending=False).head(10).to_string())
print('\nTop mots dans Bloc_01 :')
print(bow_df.loc['Bloc_01'].sort_values(ascending=False).head(10).to_string())
rich = pd.DataFrame({'types (300 mots BoW)': (bow_df > 0).sum(axis=1),
                     'types (bloc complet)': [len(set(b.split())) for b in blocs],
                     'types hors stop words': [len({w for w in b.split() if w not in sw_custom and len(w) > 2})
                                               for b in blocs]})
print('\nc) Richesse lexicale par bloc :'); print(rich.sort_values('types (bloc complet)', ascending=False).head(5))

# %%
# d) Personnages par bloc (comptage direct dans le texte)
compte_persos = distribution_personnages(blocs, ['dantès', 'danglars', 'faria', 'villefort', 'fernand', 'mercédès'],
                                         index=noms_blocs)
compte_persos['chapitres'] = chap_blocs
print(compte_persos.to_string())
for p in ['dantès', 'danglars', 'faria']:
    s = compte_persos[p]
    print(f'{p:<9} -> max dans {s.idxmax()} ({s.max()}) | top 3 : {list(s.sort_values(ascending=False).head(3).index)}')

# %%
# d) complément : roman ENTIER, pour voir Faria
blocs_all, _ = decouper_blocs(roman_clean, TAILLE)
dist = distribution_personnages(blocs_all, ['dantès', 'danglars', 'villefort', 'faria', 'abbé', 'franz'],
                                index=[f'B{i+1:02d}' for i in range(len(blocs_all))])
ax = dist.plot(figsize=(14, 5), marker='o', ms=3)
ax.set_title(f'Présence des personnages au fil du Tome I ({len(blocs_all)} blocs de 2000 mots)')
ax.set_xlabel('Bloc'); ax.set_ylabel('Occurrences')
plt.tight_layout(); save_fig('10_personnages_fil_du_roman')
for p in dist.columns:
    i = dist[p].idxmax()
    print(f'{p:<10} max dans {i} ({dist[p].max()}) — chapitre {chapitre_du_mot((int(i[1:])-1)*TAILLE, idx_chap)}')

# %%
# e) Heatmap
top_mots = bow_df.sum().sort_values(ascending=False).head(15).index
plot_heatmap(bow_df[top_mots], 'Matrice BoW — 15 mots les plus fréquents', 'Mots', 'Blocs du roman',
             name='10_heatmap_bow', annot=True)

# %%
# f) Sparsité
dens = densite(X_bow)
print(f'f) Densité (300 mots, 20 blocs)  : {dens:.2f}%  -> {100-dens:.2f}% de zéros')
_, X_full, _ = build_bow(blocs_all, list(range(len(blocs_all))), stop_words=None, min_df=1, max_df=1.0,
                         max_features=None, token_pattern=r"(?u)\b[\w-]+\b")
dens_full = densite(X_full)
print(f'   Matrice SANS filtrage, roman entier : {X_full.shape[0]} × {X_full.shape[1]:,} '
      f'-> densité {dens_full:.2f}% / {100-dens_full:.2f}% de zéros')
print(f'   Mémoire dense : {X_full.shape[0]*X_full.shape[1]*8/1e6:.1f} Mo vs sparse CSR : '
      f'{(X_full.data.nbytes + X_full.indices.nbytes + X_full.indptr.nbytes)/1e6:.2f} Mo')

# %% [markdown]
# ---
# # Partie 11 — TF-IDF & Comparaison finale (`src/vectorization/tfidf_vectorizer.py`)

# %%
tfidf_vec, X_tfidf, tfidf_df = build_tfidf(blocs, noms_blocs, stop_words=sw_custom)
print(f'Matrice TF-IDF : {X_tfidf.shape}')
print(tfidf_df.iloc[:3, :6])
print('\na) Bloc_01 — 15 meilleurs scores TF-IDF :')
print(tfidf_df.loc['Bloc_01'].sort_values(ascending=False).head(15).to_string())

# %%
# b) Top 5 par bloc
top5_bloc = top_terms_per_doc(tfidf_df, 5)
top5_bloc.insert(0, 'chapitres', chap_blocs)
print(top5_bloc.to_string())
top5_bloc.to_csv(OUT_DIR / 'tfidf_top5_par_bloc.csv')

# %%
# c) Similarité cosinus
sim_df = similarity_matrix(X_tfidf, noms_blocs)
plot_heatmap(sim_df, 'Similarité cosinus entre les blocs (TF-IDF)', name='11_similarite_cosinus',
             cmap='Blues', figsize=(12, 10))
print('Blocs les plus similaires au Bloc_01 :')
print(sim_df['Bloc_01'].sort_values(ascending=False)[1:6].to_string())
paires = top_pairs(sim_df)
print('\nc) Les 5 paires de blocs les plus proches :'); print(paires.head(5).to_string())
print('\n   Les 3 paires les plus éloignées :'); print(paires.tail(3).to_string())

# %%
# d) Mini moteur de recherche
moteur = TfidfSearchEngine(tfidf_vec, X_tfidf, noms_blocs, preprocess=clean_text_v2)
for q in ['château if prison dantès', 'mariage fiançailles mercédès', 'complot lettre dénonciation',
          'navire capitaine mer']:
    print(f"Requête : '{q}'  (termes reconnus : {moteur.termes_reconnus(q)})")
    for b, sc in moteur.search(q):
        print(f'    {b} (ch. {chap_blocs[noms_blocs.index(b)]}) score = {sc}')

# %%
# d) complément : même requête dans l'espace BoW
moteur_bow = TfidfSearchEngine(vectorizer, X_bow, noms_blocs, preprocess=clean_text_v2)
q = 'château if prison dantès'
for nom, m in [('BoW', moteur_bow), ('TF-IDF', moteur)]:
    sc = np.sort(m.scores(q))
    print(f'{nom:<7}: {[(b, float(s)) for b, s in m.search(q, 5)]}')
    print(f'{"":<7}  écart 1er / 2e = ×{sc[-1]/sc[-2]:.2f}')

# %%
# e) BoW vs TF-IDF
for mot_test in ['dantès', 'armateur', 'leclère']:
    print(f"{mot_test:<9} Bloc_01 -> BoW = {bow_df.loc['Bloc_01', mot_test]:>3} | "
          f"TF-IDF = {tfidf_df.loc['Bloc_01', mot_test]:.4f} | rang BoW = "
          f"{list(bow_df.loc['Bloc_01'].sort_values(ascending=False).index).index(mot_test)+1} | rang TF-IDF = "
          f"{list(tfidf_df.loc['Bloc_01'].sort_values(ascending=False).index).index(mot_test)+1} | "
          f"IDF = {tfidf_vec.idf_[tfidf_vec.vocabulary_[mot_test]]:.3f}")
cmp = pd.DataFrame({'BoW Bloc_01': list(bow_df.loc['Bloc_01'].sort_values(ascending=False).head(10).index),
                    'TF-IDF Bloc_01': list(tfidf_df.loc['Bloc_01'].sort_values(ascending=False).head(10).index),
                    'BoW global': list(bow_df.sum().sort_values(ascending=False).head(10).index),
                    'TF-IDF global': list(tfidf_df.sum().sort_values(ascending=False).head(10).index)})
print(cmp.to_string())
print(f'\nChevauchement moyen des top-5 entre blocs : BoW = {chevauchement_top(bow_df):.2f} | '
      f'TF-IDF = {chevauchement_top(tfidf_df):.2f}  (plus bas = plus discriminant)')

# %% [markdown]
# ---
# # Récapitulatif des indicateurs clés

# %%
recap = pd.DataFrame([
    ('Caractères (roman extrait)', f'{len(roman):,}'),
    ('Mots (.split)', f'{nb_mots_split:,}'),
    ('Tokens NLTK (après nettoyage)', f'{len(tokens):,}'),
    ('Vocabulaire', f'{len(set(tokens)):,}'),
    ('Richesse lexicale (TTR)', f'{len(set(tokens))/len(tokens):.4f}'),
    ('Hapax', f'{len(hap):,} ({len(hap)/len(freq)*100:.1f}% du vocab.)'),
    ('Pente de Zipf', f"{fit['pente']:.2f}"),
    ('Tokens après stop words', f'{len(tokens_no_stop):,} (-{(1-len(tokens_no_stop)/len(tokens))*100:.1f}%)'),
    ('Réduction vocab. par stemming', f'{(1-v_ap/v_av)*100:.1f}%'),
    ('Matrice BoW / TF-IDF', f'{X_bow.shape[0]} × {X_bow.shape[1]}'),
    ('Densité BoW', f'{dens:.1f}%'),
], columns=['Indicateur', 'Valeur'])
print(recap.to_string(index=False))
recap.to_csv(OUT_DIR / 'recapitulatif.csv', index=False)
