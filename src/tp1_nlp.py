# %% [markdown]
# # TP 1 — Introduction au NLP
# ## Prétraitement, Représentation et Analyse de Textes
# **Master 2 Intelligence Artificielle — Traitement Automatique du Langage Naturel**
#
# *Corpus : Le Comte de Monte-Cristo, Tome I — Alexandre Dumas (Project Gutenberg #17989)*
#
# | Étape | Contenu |
# |---|---|
# | 1 | Installation & configuration |
# | 2 | Chargement du corpus |
# | 3 | Nettoyage du texte |
# | 4 | Tokenisation (NLTK / SpaCy, phrases, n-grammes) |
# | 5 | Analyse fréquentielle & loi de Zipf |
# | 6 | Suppression des stop words |
# | 7 | WordClouds |
# | 8 | Stemming (Snowball) |
# | 9 | Lemmatisation & POS (SpaCy) |
# | 10 | Matrice Document-Terme & Bag of Words |
# | 11 | TF-IDF, similarité cosinus, mini moteur de recherche |
#
# > Les réponses rédigées aux questions de réflexion sont dans `rapport/TP1_NLP_Master2_IA_Reponses.docx`
# > et résumées dans les cellules *Analyse* de ce notebook.

# %% [markdown]
# ---
# # Partie 1 — Installation et Configuration
# ```bash
# pip install -r requirements.txt
# python -m spacy download fr_core_news_sm
# ```

# %%
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from wordcloud import WordCloud

import nltk
import spacy
import sklearn
from nltk import bigrams, trigrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)
plt.rcParams['figure.dpi'] = 100

FIG_DIR = Path('figures'); FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path('outputs'); OUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path('data'); DATA_DIR.mkdir(exist_ok=True)

def save_fig(name):
    """Sauvegarde la figure courante dans figures/ puis l'affiche."""
    plt.savefig(FIG_DIR / f'{name}.png', dpi=110, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ### 1.3 Ressources NLTK

# %%
for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(res, quiet=True)
print('Ressources NLTK OK')

# %% [markdown]
# ### 1.4 Modèle SpaCy — Travail demandé a), b), c)

# %%
nlp = spacy.load('fr_core_news_sm')
print('Modèle SpaCy chargé avec succès !')
print('Pipeline :', nlp.pipe_names)

# b) Versions
import wordcloud as _wc
print(f'\nPython      : {os.sys.version.split()[0]}')
print(f'NLTK        : {nltk.__version__}')
print(f'SpaCy       : {spacy.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'Pandas      : {pd.__version__}')
print(f'Matplotlib  : {matplotlib.__version__}')
print(f'WordCloud   : {_wc.__version__}')

# c) Vérification du modèle SpaCy
doc = nlp("Edmond Dantès fut injustement emprisonné au château d'If.")
print('\n', [(t.text, t.pos_) for t in doc])

# %% [markdown]
# ---
# # Partie 2 — Chargement du corpus
#
# **Remarque technique importante.** Le code du sujet utilise les marqueurs
# `*** START OF THE PROJECT GUTENBERG EBOOK 17989 ***`. Or le fichier 17989 réel contient
# `*** START OF THIS PROJECT GUTENBERG EBOOK LE COMTE DE MONTE-CRISTO, TOME I ***`.
# Avec `split(DEBUT)[1]`, le code d'origine lève une `IndexError`.
# On utilise donc une **expression régulière** robuste aux deux formats (THE/THIS, avec ou sans titre).
#
# Le fichier est mis en cache dans `data/` ; si gutenberg.org est inaccessible (blocage réseau / proxy),
# on utilise le miroir GitHub officiel du Project Gutenberg (GITenberg) — même texte.

# %%
URL = 'https://www.gutenberg.org/files/17989/17989-0.txt'
MIRROR = ('https://raw.githubusercontent.com/GITenberg/'
          'Le-comte-de-Monte-Cristo-Tome-I_17989/master/17989-8.txt')
RAW_PATH = DATA_DIR / '17989-0.txt'

def telecharger_corpus():
    if RAW_PATH.exists():
        return RAW_PATH.read_text(encoding='utf-8'), f'cache local ({RAW_PATH})'
    for url, enc in [(URL, 'utf-8'), (MIRROR, 'latin-1')]:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = enc
            txt = r.text.replace('\r\n', '\n')
            RAW_PATH.write_text(txt, encoding='utf-8')
            return txt, url
        except Exception as e:
            print(f'Échec {url} : {e}')
    raise RuntimeError('Corpus introuvable')

raw_text, source = telecharger_corpus()
print(f'Téléchargement terminé (source : {source}).')
print(f'Taille brute : {len(raw_text):,} caractères')
print(f'Aperçu :\n{raw_text[:300]}')

# %% [markdown]
# ### 2.2 Extraction du contenu (sans les métadonnées Gutenberg)

# %%
m_debut = re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK[^*]*\*\*\*', raw_text)
m_fin   = re.search(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK[^*]*\*\*\*', raw_text)
print('Marqueur début :', m_debut.group())
print('Marqueur fin   :', m_fin.group())

roman = raw_text[m_debut.end():m_fin.start()]
# On retire aussi la ligne de crédit "Produced by ..." et la mention de fin "End of the Project Gutenberg EBook"
roman = re.sub(r'^\s*Produced by[^\n]*\n', '', roman)
roman = re.split(r'\n\s*End of (?:the )?Project Gutenberg', roman)[0].strip()

# Sauvegarde locale (2.3)
(DATA_DIR / 'monte_cristo.txt').write_text(roman, encoding='utf-8')

# %% [markdown]
# ### Travail demandé — Partie 2

# %%
# a)
print(f'a) Nombre total de caractères : {len(roman):,}')
print(f'   Nombre total de lignes       : {roman.count(chr(10)):,}')
print(f'\n--- 500 premiers caractères ---\n{roman[:500]}')
print(f'\n--- 200 derniers caractères (fichier Gutenberg brut) ---\n{raw_text[-200:]}')
print(f'\n--- 200 derniers caractères du roman extrait ---\n{roman[-200:]}')

# %%
# b) Estimation rapide du nombre de mots
nb_mots_split = len(roman.split())
print(f'b) Nombre de mots (.split()) : {nb_mots_split:,}')

# c) Occurrences de Dantès
nb_dantes = roman.lower().count('dantès')
print(f'c) Occurrences de "dantès" : {nb_dantes}')

# d) 5 premières lignes contenant "vengeance"
lignes_vengeance = [l.strip() for l in roman.split('\n') if 'vengeance' in l.lower()]
print(f'd) {len(lignes_vengeance)} lignes contiennent "vengeance". Les 5 premières :')
for l in lignes_vengeance[:5]:
    print('   -', l)

# e) Nombre de pages
print(f'e) Estimation : {nb_mots_split/300:.0f} pages (300 mots/page)')

# %%
# Complément : découpage en chapitres (utile pour les parties 9 et 10)
parts = re.split(r'\n\n([IVXLC]+)\n\n', '\n\n' + roman)
chapitres = {}
for i in range(1, len(parts) - 1, 2):
    num, contenu = parts[i], parts[i + 1].strip()
    titre = contenu.split('\n')[0]
    chapitres[num] = {'titre': titre, 'texte': contenu}
df_chap = pd.DataFrame([{'chapitre': k, 'titre': v['titre'], 'nb_mots': len(v['texte'].split())}
                        for k, v in chapitres.items()])
print(f'{len(chapitres)} chapitres détectés')
print(df_chap.to_string(index=False))

# %% [markdown]
# ---
# # Partie 3 — Nettoyage du texte

# %%
# 3.1 Observation du texte brut
print(repr(roman[1000:1200]))

# %% [markdown]
# ### 3.2 Fonction de nettoyage (version du sujet)
# *Remarque :* dans la version du sujet, la normalisation des apostrophes typographiques (étape 5)
# intervient **après** la suppression des caractères non autorisés (étape 4) : une apostrophe `’`
# est donc déjà remplacée par un espace et l'étape 5 ne sert à rien. On corrige cet ordre dans la version finale.

# %%
def clean_text(text):
    """Nettoie un texte pour une analyse NLP en français (version du sujet)."""
    text = text.lower()                                     # 1. minuscules
    text = re.sub(r'_([^_]+)_', r'\1', text)                # 2. italiques Gutenberg _mot_
    text = re.sub(r'[\n\r\t]+', ' ', text)                  # 3. sauts de ligne / tabulations
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ' ]", ' ', text)  # 4. ponctuation / caractères spéciaux
    text = re.sub(r"[‘’`]", "'", text)                      # 5. apostrophes typographiques
    text = re.sub(r'\s+', ' ', text)                        # 6. espaces multiples
    return text.strip()                                     # 7. strip

roman_clean_tp = clean_text(roman)
print(f'a) Taille avant : {len(roman):,} caractères')
print(f'   Taille après : {len(roman_clean_tp):,} caractères')
print(f'   Réduction    : {(1 - len(roman_clean_tp)/len(roman))*100:.1f}%')
print(f'\nAperçu nettoyé :\n{roman_clean_tp[:400]}')

# %%
# 3.3 Vérification sur un extrait
extrait = roman[2800:3100]
print('AVANT NETTOYAGE :'); print(extrait)
print('\nAPRÈS NETTOYAGE :'); print(clean_text(extrait))

# %%
# b) Extrait bruité
bruite = '«Ah! c\'est vous, Dantès!» cria l\'homme... (numéro 34)\n\n-- Il est mort!!'
print('b) Avant :', repr(bruite))
print('   Après :', repr(clean_text(bruite)))

# c) Monte-Cristo
print('\nc)', repr(clean_text("Le comte de Monte-Cristo arriva.")))
print('   Occurrences "monte-cristo" dans le roman brut :', roman.lower().count('monte-cristo'))

# %% [markdown]
# **c)** Le tiret est remplacé par un espace : `Monte-Cristo` devient **`monte cristo` (2 tokens)**, jamais `montécristo`.
# Solution : autoriser le tiret **entre deux lettres** dans la regex, puis supprimer seulement les tirets isolés
# (ex. `--` de dialogue). On sépare en revanche les inversions verbales (`dit-il`, `s'écria-t-il`) pour ne pas
# créer de faux tokens.

# %%
LETTRES = "a-zàâçéèêëîïôûùüÿñæœ"

def clean_text_v2(text):
    """Version finale utilisée dans la suite du TP.
    Améliorations : nombres supprimés (d), mots composés conservés (c),
    apostrophes normalisées AVANT filtrage, élisions séparées (l'homme -> l homme)."""
    text = text.lower()
    text = re.sub(r'_([^_]+)_', r'\1', text)                    # italiques Gutenberg
    text = re.sub(r"[‘’`´]", "'", text)                         # apostrophes typographiques (avant filtrage !)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)                        # d) nombres isolés (24, 1815…)
    # inversions verbales : dit-il, s'écria-t-il, voyez-vous -> on garde le verbe et le pronom séparés
    text = re.sub(r'-(?:t-)?(je|tu|il|elle|on|nous|vous|ils|elles|le|la|les|lui|moi|toi|leur|y|en|ce)\b',
                  r' \1', text)
    # élisions : l'homme -> l homme ; qu'il -> qu il (aujourd'hui est préservé : pas de frontière de mot avant d')
    text = re.sub(r"\b(qu|lorsqu|puisqu|jusqu|quoiqu|presqu|[ldjmnstc])'", r'\1 ', text)
    text = re.sub(rf"[^{LETTRES}'\- ]", ' ', text)             # ponctuation (on garde - et ')
    text = re.sub(rf"(?<![{LETTRES}])-+|-+(?![{LETTRES}])", ' ', text)  # tirets isolés (--, - )
    text = re.sub(rf"(?<![{LETTRES}])'|'(?![{LETTRES}])", ' ', text)    # apostrophes isolées
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

for s in [bruite, "Le comte de Monte-Cristo arriva le 24 février 1815.",
          "--Eh bien! dit-il, qu'y a-t-il? s'écria-t-il aujourd'hui."]:
    print(repr(s)); print('  ->', repr(clean_text_v2(s)))

# %%
# e) Version légère
def clean_text_leger(text):
    text = text.lower()
    return re.sub(r'\s+', ' ', text).strip()

roman_clean = clean_text_v2(roman)        # <- texte utilisé dans toute la suite
roman_leger = clean_text_leger(roman)

comparaison = pd.DataFrame({
    'version': ['brut', 'clean_text (sujet)', 'clean_text_v2 (finale)', 'clean_text_leger'],
    'caractères': [len(roman), len(roman_clean_tp), len(roman_clean), len(roman_leger)],
    'mots (.split)': [len(roman.split()), len(roman_clean_tp.split()), len(roman_clean.split()), len(roman_leger.split())],
    'types distincts': [len(set(roman.split())), len(set(roman_clean_tp.split())),
                        len(set(roman_clean.split())), len(set(roman_leger.split()))],
})
print(comparaison.to_string(index=False))
print('\nExtrait version légère :', roman_leger[2000:2250])
print('\nExtrait version finale :', roman_clean[1900:2150])
print('\n"monte-cristo" conservé ?', roman_clean.count('monte-cristo'), 'occurrences')

# %% [markdown]
# ---
# # Partie 4 — Tokenisation

# %%
# 4.1 / a) b)
t0 = time.time()
tokens = word_tokenize(roman_clean, language='french')
t_nltk = time.time() - t0
print(f'Nombre total de tokens       : {len(tokens):,}')
print(f'Vocabulaire (tokens uniques) : {len(set(tokens)):,}')
print(f'Richesse lexicale (TTR)      : {len(set(tokens))/len(tokens):.4f}')
print(f'Temps NLTK                   : {t_nltk:.2f} s')
print('\n50 premiers tokens :'); print(tokens[:50])

# %% [markdown]
# ### 4.2 / c) NLTK vs SpaCy sur le premier chapitre (5000 premiers caractères)

# %%
def compare_tokenizers(texte, label):
    tk_nltk = word_tokenize(texte, language='french')
    tk_spacy = [t.text for t in nlp(texte) if not t.is_space]
    only_n = Counter(tk_nltk) - Counter(tk_spacy)
    only_s = Counter(tk_spacy) - Counter(tk_nltk)
    return tk_nltk, tk_spacy, only_n, only_s

extrait_ch1 = roman_clean[:5000]
n1, s1, on1, os1 = compare_tokenizers(extrait_ch1, 'nettoyé')
extrait_brut = roman[:5000]
n2, s2, on2, os2 = compare_tokenizers(extrait_brut, 'brut')

print(f'Texte NETTOYÉ : NLTK = {len(n1)} tokens | SpaCy = {len(s1)} tokens')
print('  Tokens propres à NLTK  :', on1.most_common(8))
print('  Tokens propres à SpaCy :', os1.most_common(8))
print(f'\nTexte BRUT    : NLTK = {len(n2)} tokens | SpaCy = {len(s2)} tokens')
print('  Tokens propres à NLTK  :', on2.most_common(10))
print('  Tokens propres à SpaCy :', os2.most_common(10))
print('\n30 premiers tokens SpaCy (brut) :', s2[:30])

# %%
phrases_test = ["L'abbé Faria s'écria : « Dantès ! »", "Aujourd'hui, M. Morrel est-il arrivé à Marseille ?",
                "Le 24 février 1815, la vigie de Notre-Dame-de-la-Garde signala le trois-mâts le Pharaon."]
rows = []
for p in phrases_test:
    rows.append({'phrase': p, 'NLTK': word_tokenize(p, language='french'),
                 'SpaCy': [t.text for t in nlp(p)]})
for r in rows:
    print(f"\n{r['phrase']}\n  NLTK  ({len(r['NLTK'])}) : {r['NLTK']}\n  SpaCy ({len(r['SpaCy'])}) : {r['SpaCy']}")

# %% [markdown]
# ### 4.3 Tokenisation en phrases
# Appliquer `sent_tokenize` sur le texte **nettoyé** (code du sujet) ne fonctionne pas : la ponctuation
# a été supprimée, il n'y a plus de frontières de phrases. On le démontre, puis on tokenise le texte **brut**.

# %%
phrases_clean = sent_tokenize(roman_clean[:20000], language='french')
print(f'Sur texte nettoyé : {len(phrases_clean)} phrase(s) détectée(s) !')

texte_phrases = re.sub(r'\s*\n\s*', ' ', roman[:20000])
phrases = sent_tokenize(texte_phrases, language='french')
print(f'Sur texte brut    : {len(phrases)} phrases (20 000 premiers caractères)')
print(f"Longueur moyenne d'une phrase : {sum(len(p.split()) for p in phrases)/len(phrases):.1f} mots")
for p in [p for p in phrases if len(p.split()) > 12][:3]:
    print(f' - {p[:100]}...')

def prep_brut(t):
    """Texte brut lisible par SpaCy / sent_tokenize : retire les _italiques_ Gutenberg
    et remplace les tirets de dialogue '--' par un tiret cadratin."""
    t = re.sub(r'_([^_]+)_', r'\1', t)
    t = re.sub(r'\s*--\s*', ' — ', t)
    return re.sub(r'\s+', ' ', t).strip()

# Toutes les phrases du roman (réutilisées en partie 7)
phrases_roman = sent_tokenize(prep_brut(roman), language='french')
print(f'\nNombre total de phrases du roman : {len(phrases_roman):,}')

# %%
# 4.4 N-grammes + d)
bi = list(bigrams(tokens[:200]));  print('Bigrammes (20 premiers) :', bi[:20])
tri = list(trigrams(tokens[:200])); print('\nTrigrammes (10 premiers) :', tri[:10])

top_bi = Counter(bigrams(tokens)).most_common(20)
print('\nd) 20 bigrammes les plus fréquents (tous tokens) :')
for (a, b), n in top_bi:
    print(f'   {a} {b:<15} {n}')

# %%
# d) complément : bigrammes sans mots-outils -> expressions et noms de personnages
sw_base = set(stopwords.words('french')) | nlp.Defaults.stop_words
tok_lex = [t for t in tokens if t not in sw_base and len(t) > 2]
top_bi_lex = Counter(bigrams(tok_lex)).most_common(20)
print('Bigrammes les plus fréquents (sans stop words) :')
for (a, b), n in top_bi_lex:
    print(f'   {a} {b:<18} {n}')

# %%
# e) DataFrame des tokens
df_tok = pd.DataFrame({'position': range(len(tokens)), 'token': tokens})
df_tok['longueur'] = df_tok['token'].str.len()
print(df_tok.head(10).to_string(index=False))
print(f'\nLongueur moyenne des tokens            : {df_tok.longueur.mean():.2f} caractères')
print(f'Longueur moyenne des types (vocabulaire): {pd.Series(list(set(tokens))).str.len().mean():.2f} caractères')
print('Token le plus long :', df_tok.loc[df_tok.longueur.idxmax(), 'token'])
print(df_tok.longueur.describe().round(2).to_string())

# %% [markdown]
# ---
# # Partie 5 — Analyse fréquentielle

# %%
freq = Counter(tokens)
top20 = freq.most_common(20)
print('a) TOP 20 mots les plus fréquents :')
print(f'{"Rang":<6} {"Mot":<20} {"Fréquence":<10} {"% total"}')
print('-' * 50)
for rang, (mot, n) in enumerate(top20, 1):
    print(f'{rang:<6} {mot:<20} {n:<10} {n/len(tokens)*100:.3f}%')
print(f'\nLes 20 premiers mots couvrent {sum(n for _, n in top20)/len(tokens)*100:.1f}% des tokens.')

# %%
# b) Barres horizontales
mots = [m for m, n in top20]; freqs = [n for m, n in top20]
plt.figure(figsize=(12, 6))
bars = plt.barh(mots[::-1], freqs[::-1], color='steelblue', edgecolor='white')
for bar, val in zip(bars, freqs[::-1]):
    plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=10)
plt.xlabel('Fréquence')
plt.title('Top 20 mots les plus fréquents — Le Comte de Monte-Cristo (Tome I)')
plt.tight_layout(); save_fig('05_top20_frequences')

# %%
# c) Loi de Zipf + ajustement de la pente
all_freqs = np.array(sorted(freq.values(), reverse=True))
rangs = np.arange(1, len(all_freqs) + 1)
pente, intercept = np.polyfit(np.log10(rangs), np.log10(all_freqs), 1)
mask = rangs <= 1000
pente_1000, intercept_1000 = np.polyfit(np.log10(rangs[mask]), np.log10(all_freqs[mask]), 1)

plt.figure(figsize=(10, 5))
plt.loglog(rangs, all_freqs, 'b.', markersize=2, alpha=0.5, label='Observé')
plt.loglog(rangs, 10**intercept * rangs**pente, 'r--', label=f'Régression (pente = {pente:.2f})')
plt.loglog(rangs, all_freqs[0] / rangs, 'g:', label='Zipf théorique f = C / r')
plt.xlabel('Rang (log)'); plt.ylabel('Fréquence (log)')
plt.title('Loi de Zipf — Le Comte de Monte-Cristo')
plt.grid(True, which='both', alpha=0.3); plt.legend()
plt.tight_layout(); save_fig('05_loi_de_zipf')

print(f'Pente log-log globale        : {pente:.3f}  (Zipf idéal : -1)')
print(f'Pente sur les 1000 premiers  : {pente_1000:.3f}')
top5 = freq.most_common(5)
print('\nVérification (ratio f1/fr ≈ r attendu) :')
for i, (m, n) in enumerate(top5):
    print(f'  Rang {i+1}: {m:<5} ({n} fois) — ratio avec rang 1 : {top5[0][1]/n:.2f}')

# %%
# d) Mots thématiques
classement = {mot: r for r, (mot, _) in enumerate(freq.most_common(), 1)}
for m in ['dantès', 'vengeance', 'liberté', 'prison']:
    print(f"d) {m:<10} fréquence = {freq[m]:>4}   rang = {classement.get(m, '—')}")

# e) Hapax
hapax = [m for m, n in freq.items() if n == 1]
print(f'\ne) Hapax legomena : {len(hapax):,} mots = {len(hapax)/len(freq)*100:.1f}% du vocabulaire '
      f'mais {len(hapax)/len(tokens)*100:.2f}% des tokens')
import random; random.seed(0)
print('   Exemples :', random.sample(hapax, 25))

# %% [markdown]
# ---
# # Partie 6 — Suppression des Stop Words

# %%
sw_nltk = set(stopwords.words('french'))
sw_spacy = set(nlp.Defaults.stop_words)
print(f'a) Stop words NLTK  : {len(sw_nltk)}')
print(f'   Stop words SpaCy : {len(sw_spacy)}')
print(f'   Communs          : {len(sw_nltk & sw_spacy)}')
print(f'   Union            : {len(sw_nltk | sw_spacy)}')
print(f'\nExemples NLTK  : {sorted(sw_nltk)[:15]}')
print(f'Exemples SpaCy : {sorted(sw_spacy)[:15]}')
print(f'\nDans NLTK, absents de SpaCy ({len(sw_nltk - sw_spacy)}) : {sorted(sw_nltk - sw_spacy)[:25]}')
print(f'Dans SpaCy, absents de NLTK ({len(sw_spacy - sw_nltk)}) : {sorted(sw_spacy - sw_nltk)[:25]}')

# %%
# 6.2 Filtrage
sw_custom = sw_nltk | sw_spacy | {
    'dit', 'disant', 'répondit', 'reprit', 'ajouta', 'cria',
    'monsieur', 'madame', 'mademoiselle', 'car', 'alors', 'puis',
    'bien', 'encore', 'tout', 'très', 'plus', 'non', 'oui', 'si'
}
tokens_no_stop = [t for t in tokens if t.lower() not in sw_custom and len(t) > 2]
freq_clean = Counter(tokens_no_stop)

print(f'Tokens avant : {len(tokens):,}')
print(f'Tokens après : {len(tokens_no_stop):,}')
print(f'c) Réduction (tokens)     : {(len(tokens)-len(tokens_no_stop))/len(tokens)*100:.1f}%')
print(f'   Réduction (vocabulaire): {(1-len(set(tokens_no_stop))/len(set(tokens)))*100:.1f}% '
      f'({len(set(tokens)):,} -> {len(set(tokens_no_stop)):,} types)')
print('\nb) Top 20 après suppression des stop words :')
for mot, n in freq_clean.most_common(20):
    print(f'   {mot:<20} {n}')

# %%
# 6.3 Comparaison avant / après
print(f'{"Rang":<5} {"Avant":<22} {"Après"}'); print('-' * 45)
for i, (a, b) in enumerate(zip([m for m, _ in freq.most_common(10)],
                               [m for m, _ in freq_clean.most_common(10)]), 1):
    print(f'{i:<5} {a:<22} {b}')

# %%
# d) Justification des stop words contextuels
for m in ['dit', 'répondit', 'reprit', 'monsieur', 'madame']:
    print(f'{m:<10} {freq[m]:>5} occurrences (rang {classement.get(m)})')
apres_monsieur = Counter(b for a, b in bigrams(tokens) if a == 'monsieur').most_common(8)
print('\nMots qui suivent "monsieur" :', apres_monsieur)
apres_dit = Counter(b for a, b in bigrams(tokens) if a in ('dit', 'répondit')).most_common(8)
print('Mots qui suivent "dit/répondit" :', apres_dit)

# %%
# e) Personnages
persos = ['dantès', 'edmond', 'danglars', 'fernand', 'villefort', 'faria', 'abbé', 'mercédès',
          'morrel', 'caderousse', 'franz', 'albert', 'noirtier']
df_persos = pd.DataFrame({'personnage': persos,
                          'fréquence': [freq_clean[p] for p in persos],
                          'rang (après stop words)': [
                              [m for m, _ in freq_clean.most_common()].index(p) + 1 if p in freq_clean else None
                              for p in persos]}).sort_values('fréquence', ascending=False)
print(df_persos.to_string(index=False))

plt.figure(figsize=(10, 4.5))
sns.barplot(data=df_persos, x='personnage', y='fréquence', color='steelblue')
plt.title('Fréquence des personnages (après suppression des stop words)')
plt.xticks(rotation=30); plt.tight_layout(); save_fig('06_personnages')

# %% [markdown]
# ---
# # Partie 7 — Visualisation : WordCloud

# %%
# a) WordCloud du roman complet
texte_clean = ' '.join(tokens_no_stop)
wc = WordCloud(width=1200, height=600, background_color='white', colormap='Blues',
               collocations=False, max_words=150, min_font_size=8, random_state=42).generate(texte_clean)
plt.figure(figsize=(16, 8))
plt.imshow(wc, interpolation='bilinear'); plt.axis('off')
plt.title('WordCloud — Le Comte de Monte-Cristo (Tome I)', fontsize=16, pad=20)
plt.tight_layout(); save_fig('07_wordcloud_roman')
print('a) 5 mots les plus visibles :', list(wc.words_.keys())[:5])

# %% [markdown]
# ### b) WordClouds par personnage
# Le code du sujet appelle `sent_tokenize(roman_clean)` : sur un texte sans ponctuation on obtient **une seule
# « phrase »** = tout le roman, donc les nuages seraient identiques. On segmente donc le texte **brut** en phrases,
# puis on nettoie chaque phrase. On retire aussi le nom du personnage lui-même (il dominerait son propre nuage)
# et on calcule les **mots distinctifs** (sur-représentation par rapport au roman entier).

# %%
phrases_clean_list = [clean_text_v2(p) for p in phrases_roman]

def tokens_personnage(perso):
    sel = [p for p in phrases_clean_list if re.search(rf'\b{perso}\b', p)]
    toks = [t for p in sel for t in word_tokenize(p, language='french')
            if t not in sw_custom and len(t) > 3 and t != perso]
    return sel, toks

total_clean = sum(freq_clean.values())
def mots_distinctifs(toks, k=10, min_n=4):
    c = Counter(toks); tot = sum(c.values())
    score = {m: (n/tot) / ((freq_clean[m]+1)/total_clean) for m, n in c.items() if n >= min_n}
    return [m for m, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]

personnages = ['dantès', 'danglars', 'villefort']
colourmaps = ['Blues', 'Reds', 'Greens']
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, perso, cmap in zip(axes, personnages, colourmaps):
    sel, toks = tokens_personnage(perso)
    wcp = WordCloud(width=600, height=450, background_color='white', colormap=cmap,
                    collocations=False, random_state=42).generate(' '.join(toks))
    ax.imshow(wcp, interpolation='bilinear'); ax.axis('off')
    ax.set_title(f'{perso.capitalize()} ({len(sel)} phrases)', fontsize=13, fontweight='bold')
    print(f'{perso.capitalize():<10} {len(sel):>4} phrases | top fréquents : {[m for m,_ in Counter(toks).most_common(8)]}')
    print(f'{"":<10} mots distinctifs : {mots_distinctifs(toks)}')
plt.suptitle('Champ lexical des personnages principaux', fontsize=15, y=1.02)
plt.tight_layout(); save_fig('07_wordcloud_personnages')

# %% [markdown]
# ### c) WordCloud des noms propres (PROPN)
# La détection des noms propres repose sur la **majuscule** : on applique donc SpaCy sur le texte **brut**,
# par paragraphes (`nlp.pipe`), en désactivant les composants inutiles pour aller plus vite.

# %%
t0 = time.time()
paragraphes = [prep_brut(p) for p in re.split(r'\n\s*\n', roman) if p.strip()]
propn = []
for d in nlp.pipe(paragraphes, batch_size=64, disable=['parser', 'ner', 'lemmatizer']):
    propn.extend(t.text for t in d if t.pos_ == 'PROPN' and len(t.text) > 2 and t.text.isalpha()
                  and t.text.lower() not in sw_custom)
t_spacy_full = time.time() - t0
freq_propn = Counter(propn)
print(f'SpaCy sur tout le roman : {t_spacy_full:.1f} s ({len(paragraphes):,} paragraphes)')
print('Top 30 PROPN :', freq_propn.most_common(30))

wc_propn = WordCloud(width=1200, height=600, background_color='white', colormap='viridis',
                     collocations=False, max_words=100, random_state=42).generate_from_frequencies(freq_propn)
plt.figure(figsize=(16, 8)); plt.imshow(wc_propn, interpolation='bilinear'); plt.axis('off')
plt.title('Noms propres (POS = PROPN, SpaCy)', fontsize=16)
plt.tight_layout(); save_fig('07_wordcloud_noms_propres')

# %%
# d) Paramètres
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax, mw in zip(axes, [50, 200]):
    w = WordCloud(width=800, height=450, background_color='white', colormap='RdYlGn',
                  collocations=False, max_words=mw, random_state=42).generate(texte_clean)
    ax.imshow(w, interpolation='bilinear'); ax.axis('off')
    ax.set_title(f"colormap='RdYlGn', max_words={mw}", fontsize=13)
plt.tight_layout(); save_fig('07_wordcloud_parametres')

# %%
# e) Début vs fin du roman
debut, fin = tokens_no_stop[:3000], tokens_no_stop[-3000:]
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax, toks, titre, cmap in zip(axes, [debut, fin], ['3000 premiers mots', '3000 derniers mots'], ['Blues', 'Oranges']):
    w = WordCloud(width=800, height=450, background_color='white', colormap=cmap,
                  collocations=False, random_state=42).generate(' '.join(toks))
    ax.imshow(w, interpolation='bilinear'); ax.axis('off'); ax.set_title(titre, fontsize=14)
plt.tight_layout(); save_fig('07_wordcloud_debut_fin')
print('Top début :', [m for m, _ in Counter(debut).most_common(15)])
print('Top fin   :', [m for m, _ in Counter(fin).most_common(15)])
communs = set(debut) & set(fin)
print(f'Vocabulaire commun : {len(communs)} mots sur {len(set(debut))} (début) / {len(set(fin))} (fin) '
      f'— Jaccard = {len(communs)/len(set(debut)|set(fin)):.2f}')

# %% [markdown]
# ---
# # Partie 8 — Stemming (Snowball)

# %%
stemmer = SnowballStemmer('french')
familles = {
    'vengeance': ['vengeance', 'venger', 'vengeur', 'vindicatif'],
    'prison':    ['prison', 'prisonnier', 'emprisonner', 'emprisonnement'],
    'liberté':   ['liberté', 'libérer', 'libérateur', 'libération'],
    'mer':       ['mer', 'marin', 'maritime', 'navire', 'naviguer'],
}
print(f'{"Famille":<11}{"Mot original":<20} {"Racine (stem)"}'); print('-' * 50)
print(f'{"":<11}{"dantès":<20} {stemmer.stem("dantès")}')
for fam, ms in familles.items():
    for m in ms:
        print(f'{fam:<11}{m:<20} {stemmer.stem(m)}')
    print(f'{"":<11}-> {len({stemmer.stem(m) for m in ms})} racine(s) distincte(s) : {sorted({stemmer.stem(m) for m in ms})}')

# %%
# 8.2 / b) Application sur le corpus
tokens_stemmed = [stemmer.stem(t) for t in tokens_no_stop]
freq_stemmed = Counter(tokens_stemmed)
v_av, v_ap = len(set(tokens_no_stop)), len(set(tokens_stemmed))
print(f'Vocabulaire avant stemming : {v_av:,} mots')
print(f'Vocabulaire après stemming : {v_ap:,} racines')
print(f'b) Réduction du vocabulaire : {(1 - v_ap/v_av)*100:.1f}%')
print('\nTop 20 racines :')
for r, n in freq_stemmed.most_common(20):
    print(f'   {r:<20} {n}')

# %%
# 8.3 Familles de mots
groupes = defaultdict(set)
for mot in set(tokens_no_stop):
    groupes[stemmer.stem(mot)].add(mot)
top_familles = sorted(groupes.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print('Familles de mots (10 plus grandes) :')
for racine, ms in top_familles:
    print(f'  [{racine:<12}] ({len(ms):>2}) -> {sorted(ms)[:8]}')

# %%
# c) Faux amis du stemmer
faux_amis = [('porte', 'porter', 'port'), ('sort', 'sortir', 'sortie'), ('mari', 'marin', 'marine'),
             ('vers', 'verser', 'verre'), ('livre', 'livrer', 'libre'), ('somme', 'sommeil', 'sommes'),
             ('pas', 'passer', 'passion'), ('mère', 'mer', 'merci'), ('été', 'étaient', 'état')]
for groupe in faux_amis:
    print('  ', ' | '.join(f'{m} -> {stemmer.stem(m)}' for m in groupe))

# %%
# d) Tableau 15 mots caractéristiques
vocab_roman = set(tokens)
mots_15 = ['vengeance', 'prisonnier', 'château', 'liberté', 'navire', 'trésor', 'abbé', 'capitaine',
           'mariage', 'procureur', 'emprisonné', 'évasion', 'fiançailles', 'cachot', 'marseille']
df_stem = pd.DataFrame({'Mot original': mots_15, 'Radical (Snowball)': [stemmer.stem(m) for m in mots_15]})
df_stem['Intelligible ?'] = ['oui' if r in vocab_roman else 'non' for r in df_stem['Radical (Snowball)']]
df_stem['Fréq. du mot'] = [freq[m] for m in mots_15]
print(df_stem.to_string(index=False))
print("\n(« Intelligible » = le radical est lui-même un mot attesté dans le roman)")

# %%
# e) Thèmes par racine
themes = {'veng-': 'veng', 'pris-': 'pris', 'liber-': 'liber', 'dant-': 'dant'}
rows = []
for lab, pref in themes.items():
    racines = {r: n for r, n in freq_stemmed.items() if r.startswith(pref)}
    mots_f = sorted({m for r in racines for m in groupes[r]})
    rows.append({'racine': lab, 'occurrences': sum(racines.values()), 'nb racines': len(racines),
                 'exemples de mots': mots_f[:7]})
df_themes = pd.DataFrame(rows).sort_values('occurrences', ascending=False)
print(df_themes.to_string(index=False))

# %% [markdown]
# ---
# # Partie 9 — Lemmatisation avec SpaCy

# %%
# a) Premier chapitre complet
chap1 = prep_brut(chapitres['I']['texte'])
t0 = time.time(); doc = nlp(chap1); t_ch1 = time.time() - t0
rows = [{'token': t.text, 'lemme': t.lemma_, 'POS': t.pos_, 'est_stop': t.is_stop, 'est_ponc': t.is_punct}
        for t in doc if not t.is_space]
df_tokens = pd.DataFrame(rows)
df_tokens.to_csv(OUT_DIR / 'chapitre1_spacy_tokens.csv', index=False)
print(f'Chapitre I : « {chapitres["I"]["titre"]} » — {len(df_tokens):,} tokens analysés en {t_ch1:.2f} s')
print('(DataFrame complet exporté dans outputs/chapitre1_spacy_tokens.csv)\n')
print(df_tokens.head(40).to_string(index=False))

# %%
# 9.2 Lemmes propres
lemmes_propres = [t.lemma_.lower() for t in doc
                  if not t.is_stop and not t.is_punct and not t.is_space and len(t.text) > 2]
freq_lemmes = Counter(lemmes_propres)
print('Top 20 lemmes (chapitre I) :')
for l, n in freq_lemmes.most_common(20):
    print(f'   {l:<20} {n}')

# %%
# b) NOUN / c) PROPN
noms = Counter(t.lemma_.lower() for t in doc if t.pos_ == 'NOUN' and not t.is_stop and t.is_alpha)
print('b) Top 15 NOMS (lemmatisés) :', noms.most_common(15))
propres = Counter(t.text for t in doc if t.pos_ == 'PROPN' and t.is_alpha and t.text.lower() not in sw_custom)
print('\nc) Noms propres :', propres.most_common(25))
ents = Counter((e.text, e.label_) for e in doc.ents)
print('\n   (complément NER) :', ents.most_common(15))

# %%
# 9.3 POS
pos_counts = Counter(t.pos_ for t in doc if not t.is_space)
pos_labels = {'NOUN': 'Nom commun', 'VERB': 'Verbe', 'ADJ': 'Adjectif', 'ADV': 'Adverbe',
              'PROPN': 'Nom propre', 'DET': 'Déterminant', 'PRON': 'Pronom', 'ADP': 'Préposition',
              'PUNCT': 'Ponctuation', 'CCONJ': 'Conj. coordination', 'SCONJ': 'Conj. subordination',
              'AUX': 'Auxiliaire', 'NUM': 'Numéral', 'X': 'Autre', 'INTJ': 'Interjection', 'SYM': 'Symbole'}
tot = sum(pos_counts.values())
print('Distribution POS (chapitre I) :')
for pos, n in pos_counts.most_common():
    print(f'  {pos_labels.get(pos, pos):<22} {n:>5} ({n/tot*100:.1f}%)')
plt.figure(figsize=(10, 4.5))
pc = pd.Series({pos_labels.get(k, k): v for k, v in pos_counts.most_common()})
pc.plot(kind='bar', color='steelblue'); plt.ylabel('Nombre de tokens')
plt.title('Distribution des catégories grammaticales — Chapitre I'); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); save_fig('09_pos_chapitre1')

# %%
# d) Stemming vs Lemmatisation, lemme pris EN CONTEXTE dans le roman
cibles = ['navire', 'prisonnier', 'vengeance', 'chevaux', 'yeux', 'allait', 'fut', 'meilleurs',
          'aperçut', 'gendarmes', 'prisonniers', 'disait', 'émotion', 'jeunes', 'dantès']
contexte = {}
for p in phrases_roman:
    toks_p = set(re.findall(rf"[{LETTRES}]+", p.lower()))
    for c in cibles:
        if c not in contexte and c in toks_p:
            contexte[c] = p
lem_ctx = {}
for c, p in contexte.items():
    for t in nlp(p):
        if t.text.lower() == c:
            lem_ctx[c] = (t.lemma_.lower(), t.pos_); break
df_cmp = pd.DataFrame({'Mot original': cibles,
                       'Stemming (Snowball)': [stemmer.stem(c) for c in cibles],
                       'Lemme (SpaCy)': [lem_ctx.get(c, ('—', ''))[0] for c in cibles],
                       'POS': [lem_ctx.get(c, ('', '—'))[1] for c in cibles]})
# Jugement linguistique (forme canonique attendue dans un dictionnaire)
attendu = {'navire': 'navire', 'prisonnier': 'prisonnier', 'vengeance': 'vengeance', 'chevaux': 'cheval',
           'yeux': 'œil', 'allait': 'aller', 'fut': 'être', 'meilleurs': 'bon (ou meilleur)',
           'aperçut': 'apercevoir', 'gendarmes': 'gendarme', 'prisonniers': 'prisonnier', 'disait': 'dire',
           'émotion': 'émotion', 'jeunes': 'jeune', 'dantès': 'dantès'}
df_cmp['Attendu'] = df_cmp['Mot original'].map(attendu)
def ok(x, y):
    return 'oui' if x.replace('oe', 'œ') in y.split(' ') or x == y else 'non'
df_cmp['Stem correct ?'] = [ok(a, b) for a, b in zip(df_cmp['Stemming (Snowball)'], df_cmp['Attendu'])]
df_cmp['Lemme correct ?'] = [ok(a, b) for a, b in zip(df_cmp['Lemme (SpaCy)'], df_cmp['Attendu'])]
print(df_cmp.to_string(index=False))
print(f"\nStemming correct : {(df_cmp['Stem correct ?']=='oui').sum()}/15 | "
      f"Lemmatisation correcte : {(df_cmp['Lemme correct ?']=='oui').sum()}/15")
df_cmp.to_csv(OUT_DIR / 'stemming_vs_lemmatisation.csv', index=False)

# %%
# Coût computationnel SpaCy vs NLTK (même texte : chapitre I)
t0 = time.time(); _ = word_tokenize(chap1, language='french'); _ = [stemmer.stem(w) for w in _]; tn = time.time() - t0
t0 = time.time(); _ = nlp(chap1); ts = time.time() - t0
print(f'Chapitre I ({len(chap1.split()):,} mots) : NLTK tokenisation+stemming = {tn*1000:.0f} ms | '
      f'SpaCy pipeline complet = {ts*1000:.0f} ms  -> ×{ts/tn:.0f}')

# %% [markdown]
# ---
# # Partie 10 — Matrice Document-Terme & Bag of Words

# %%
# 10.1 Blocs de 2000 mots (20 premiers blocs, comme demandé)
mots_roman = roman_clean.split()
taille_bloc = 2000
blocs = [' '.join(mots_roman[i:i+taille_bloc])
         for i in range(0, min(len(mots_roman), taille_bloc*20), taille_bloc)]
noms_blocs = [f'Bloc_{i+1:02d}' for i in range(len(blocs))]
print(f'Nombre de blocs : {len(blocs)} (couvrent {len(blocs)*taille_bloc:,} mots sur {len(mots_roman):,}, '
      f'soit {len(blocs)*taille_bloc/len(mots_roman)*100:.0f}% du roman)')
print(f'Taille du premier bloc : {len(blocs[0].split())} mots')

# Repère : chapitre de début de chaque bloc
cum, debut_chap = 0, []
for k, v in chapitres.items():
    debut_chap.append((cum, k)); cum += len(clean_text_v2(v['texte']).split())
def chap_du_mot(i):
    return [k for d, k in debut_chap if d <= i][-1]
print('Chapitres couverts :', {nb: f'ch. {chap_du_mot(i*taille_bloc)}–{chap_du_mot(min((i+1)*taille_bloc, len(mots_roman))-1)}'
                               for i, nb in enumerate(noms_blocs)})

# %%
# 10.2 Matrice BoW / a)
vectorizer = CountVectorizer(lowercase=True, stop_words=list(sw_custom),
                             min_df=2, max_df=0.90, max_features=300,
                             token_pattern=r"(?u)\b[\w-]{3,}\b")
X_bow = vectorizer.fit_transform(blocs)
vocab = vectorizer.get_feature_names_out()
bow_df = pd.DataFrame(X_bow.toarray(), index=noms_blocs, columns=vocab)
print(f'a) Matrice BoW : {X_bow.shape[0]} documents × {X_bow.shape[1]} mots')
print(bow_df.iloc[:5, :8])
n_dantes = sum('dantès' in b.split() for b in blocs)
print(f"\n'dantès' dans le vocabulaire ? {'dantès' in vocab} (présent dans {n_dantes}/{len(blocs)} blocs "
      f"= {n_dantes/len(blocs):.0%} < max_df 90 % -> conservé)")
trop_frequents = sorted(w for w in set(' '.join(blocs).split())
                        if w not in sw_custom and len(w) > 2 and sum(w in b.split() for b in blocs) / len(blocs) > 0.9)
print('Mots (hors stop words) supprimés par max_df, présents dans > 90 % des blocs :', trop_frequents)

# %%
# 10.3 Analyse
print('Mots les plus fréquents (somme sur tous les blocs) :')
print(bow_df.sum().sort_values(ascending=False).head(10).to_string())
print('\nTop mots dans Bloc_01 :')
print(bow_df.loc['Bloc_01'].sort_values(ascending=False).head(10).to_string())

# c) Richesse lexicale par bloc
rich = pd.DataFrame({
    'types (300 mots BoW)': (bow_df > 0).sum(axis=1),
    'types (bloc complet)': [len(set(b.split())) for b in blocs],
    'types hors stop words': [len({w for w in b.split() if w not in sw_custom and len(w) > 2}) for b in blocs],
})
print('\nc) Richesse lexicale par bloc :'); print(rich.sort_values('types (bloc complet)', ascending=False).head(5))

# %%
# d) Où apparaissent les personnages ? (comptage direct dans les blocs, car 'dantès' est filtré par max_df)
compte_persos = pd.DataFrame({p: [b.split().count(p) for b in blocs]
                              for p in ['dantès', 'danglars', 'faria', 'villefort', 'fernand', 'mercédès']},
                             index=noms_blocs)
compte_persos['chapitres'] = [f'{chap_du_mot(i*taille_bloc)}–{chap_du_mot((i+1)*taille_bloc-1)}' for i in range(len(blocs))]
print(compte_persos.to_string())
for p in ['dantès', 'danglars', 'faria']:
    s = compte_persos[p]
    print(f'{p:<9} -> max dans {s.idxmax()} ({s.max()}) | top 3 : {list(s.sort_values(ascending=False).head(3).index)}')

# %%
# d) complément : sur le roman ENTIER (tous les blocs), pour voir Faria
blocs_all = [' '.join(mots_roman[i:i+taille_bloc]) for i in range(0, len(mots_roman), taille_bloc)]
dist = pd.DataFrame({p: [b.split().count(p) for b in blocs_all]
                     for p in ['dantès', 'danglars', 'villefort', 'faria', 'abbé', 'franz']},
                    index=[f'B{i+1:02d}' for i in range(len(blocs_all))])
ax = dist.plot(figsize=(14, 5), marker='o', ms=3)
ax.set_title(f'Présence des personnages au fil du Tome I ({len(blocs_all)} blocs de 2000 mots)')
ax.set_xlabel('Bloc'); ax.set_ylabel('Occurrences')
plt.tight_layout(); save_fig('10_personnages_fil_du_roman')
for p in dist.columns:
    s = dist[p]; i = s.idxmax()
    print(f'{p:<10} max dans {i} ({s.max()}) — chapitre {chap_du_mot((int(i[1:])-1)*taille_bloc)}')

# %%
# e) Heatmap
top_mots = bow_df.sum().sort_values(ascending=False).head(15).index
plt.figure(figsize=(14, 8))
sns.heatmap(bow_df[top_mots], cmap='YlOrRd', linewidths=0.3, annot=True, fmt='d',
            xticklabels=True, yticklabels=True)
plt.title('Matrice BoW — 15 mots les plus fréquents'); plt.xlabel('Mots'); plt.ylabel('Blocs du roman')
plt.tight_layout(); save_fig('10_heatmap_bow')

# %%
# f) Sparsité
densite = (bow_df > 0).sum().sum() / bow_df.size * 100
print(f'f) Densité (300 mots, 20 blocs)  : {densite:.2f}%  -> {100-densite:.2f}% de zéros')
cv_full = CountVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
X_full = cv_full.fit_transform(blocs_all)
dens_full = X_full.nnz / (X_full.shape[0] * X_full.shape[1]) * 100
print(f'   Matrice SANS filtrage, roman entier : {X_full.shape[0]} × {X_full.shape[1]:,} '
      f'-> densité {dens_full:.2f}% / {100-dens_full:.2f}% de zéros')
print(f'   Mémoire dense : {X_full.shape[0]*X_full.shape[1]*8/1e6:.1f} Mo vs sparse CSR : '
      f'{(X_full.data.nbytes + X_full.indices.nbytes + X_full.indptr.nbytes)/1e6:.2f} Mo')

# %% [markdown]
# ---
# # Partie 11 — TF-IDF & Comparaison finale

# %%
tfidf_vec = TfidfVectorizer(lowercase=True, stop_words=list(sw_custom), min_df=2, max_df=0.90,
                            max_features=300, sublinear_tf=True, use_idf=True, smooth_idf=True,
                            token_pattern=r"(?u)\b[\w-]{3,}\b")
X_tfidf = tfidf_vec.fit_transform(blocs)
vocab_tfidf = tfidf_vec.get_feature_names_out()
tfidf_df = pd.DataFrame(X_tfidf.toarray().round(4), index=noms_blocs, columns=vocab_tfidf)
print(f'Matrice TF-IDF : {X_tfidf.shape}')
print(tfidf_df.iloc[:3, :6])

# a) Scores du Bloc_01
print('\na) Bloc_01 — 15 meilleurs scores TF-IDF :')
print(tfidf_df.loc['Bloc_01'].sort_values(ascending=False).head(15).to_string())

# %%
# b) Top 5 par bloc
top5_bloc = pd.DataFrame({b: list(tfidf_df.loc[b].sort_values(ascending=False).head(5).index) for b in noms_blocs}).T
top5_bloc.columns = [f'#{i}' for i in range(1, 6)]
top5_bloc.insert(0, 'chapitres', compte_persos['chapitres'])
print(top5_bloc.to_string())
top5_bloc.to_csv(OUT_DIR / 'tfidf_top5_par_bloc.csv')

# %%
# c) Similarité cosinus
sim_matrix = cosine_similarity(X_tfidf)
sim_df = pd.DataFrame(sim_matrix.round(3), index=noms_blocs, columns=noms_blocs)
plt.figure(figsize=(12, 10))
sns.heatmap(sim_df, cmap='Blues', annot=False, xticklabels=True, yticklabels=True, linewidths=0.2)
plt.title('Similarité cosinus entre les blocs (TF-IDF)')
plt.tight_layout(); save_fig('11_similarite_cosinus')

print('Blocs les plus similaires au Bloc_01 :')
print(sim_df['Bloc_01'].sort_values(ascending=False)[1:6].to_string())
s = sim_df.where(np.triu(np.ones(sim_df.shape, dtype=bool), k=1)).stack().dropna().sort_values(ascending=False)
print('\nc) Les 5 paires de blocs les plus proches :'); print(s.head(5).to_string())
print('\n   Les 3 paires les plus éloignées :'); print(s.tail(3).to_string())

# %%
# d) Mini moteur de recherche
def rechercher(requete, k=3):
    v = tfidf_vec.transform([clean_text_v2(requete)])
    scores = cosine_similarity(v, X_tfidf)[0]
    termes = [t for t in tfidf_vec.build_analyzer()(clean_text_v2(requete)) if t in tfidf_vec.vocabulary_]
    return sorted(zip(noms_blocs, scores.round(4)), key=lambda x: -x[1])[:k], termes

for q in ['château if prison dantès', 'mariage fiançailles mercédès', 'complot lettre dénonciation',
          'navire capitaine mer']:
    res, termes = rechercher(q)
    print(f"Requête : '{q}'  (termes reconnus : {termes})")
    for b, sc in res:
        print(f'    {b} (ch. {compte_persos.loc[b, "chapitres"]}) score = {sc}')

# %%
# d) complément : même requête avec la matrice BoW
q_bow = vectorizer.transform(['château if prison dantès'])
sc_bow = cosine_similarity(q_bow, X_bow)[0]
print('BoW    :', [(b, round(float(x), 3)) for b, x in sorted(zip(noms_blocs, sc_bow), key=lambda x: -x[1])[:5]])
sc_tf = cosine_similarity(tfidf_vec.transform(['château if prison dantès']), X_tfidf)[0]
print('TF-IDF :', [(b, round(float(x), 3)) for b, x in sorted(zip(noms_blocs, sc_tf), key=lambda x: -x[1])[:5]])
for nom, sc in [('BoW', sc_bow), ('TF-IDF', sc_tf)]:
    s1, s2 = sorted(sc)[-1], sorted(sc)[-2]
    print(f'{nom:<7} écart 1er / 2e = ×{s1/s2:.2f}')

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

# Pouvoir discriminant : combien de blocs partagent le même top-5 ?
def chevauchement(df):
    tops = [set(df.loc[b].sort_values(ascending=False).head(5).index) for b in noms_blocs]
    return np.mean([len(tops[i] & tops[j]) / 5 for i in range(len(tops)) for j in range(i+1, len(tops))])
print(f'\nChevauchement moyen des top-5 entre blocs : BoW = {chevauchement(bow_df):.2f} | '
      f'TF-IDF = {chevauchement(tfidf_df):.2f}  (plus bas = plus discriminant)')
sim_bow = cosine_similarity(X_bow)
iu = np.triu_indices(len(blocs), 1)
print(f'Similarité cosinus moyenne entre blocs : BoW = {sim_bow[iu].mean():.3f} | TF-IDF = {sim_matrix[iu].mean():.3f}')

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
    ('Hapax', f'{len(hapax):,} ({len(hapax)/len(freq)*100:.1f}% du vocab.)'),
    ('Pente de Zipf', f'{pente:.2f}'),
    ('Tokens après stop words', f'{len(tokens_no_stop):,} (-{(1-len(tokens_no_stop)/len(tokens))*100:.1f}%)'),
    ('Réduction vocab. par stemming', f'{(1-v_ap/v_av)*100:.1f}%'),
    ('Matrice BoW / TF-IDF', f'{X_bow.shape[0]} × {X_bow.shape[1]}'),
    ('Densité BoW', f'{densite:.1f}%'),
], columns=['Indicateur', 'Valeur'])
print(recap.to_string(index=False))
recap.to_csv(OUT_DIR / 'recapitulatif.csv', index=False)
