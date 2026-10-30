"""Remplit le sujet TP1 (.docx) avec les réponses, résultats et figures du notebook.
Usage : python -m src.generation_rapport.rapport_docx <sujet.docx> <sortie.docx> [url_github]
"""
import re
import sys
import copy
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC, DST = sys.argv[1], sys.argv[2]
REPO = sys.argv[3] if len(sys.argv) > 3 else 'https://github.com/sephorataty/TP1-NLP-Monte-Cristo'
ROOT = Path(__file__).resolve().parents[2]  # racine du projet
FIG = ROOT / 'figures'

TXT = RGBColor(0x1F, 0x1F, 0x1F)
ACC = RGBColor(0x1F, 0x4E, 0x79)


# ---------------------------------------------------------------- helpers
def _font(run, size=10, bold=False, italic=False, color=TXT, mono=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    name = 'Consolas' if mono else 'Calibri'
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.insert(0, rf)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rf.set(qn(a), name)


def _spacing(p, before=0, after=60, line=None):
    pf = p.paragraph_format
    pf.space_before = Pt(before / 20)
    pf.space_after = Pt(after / 20)
    if line:
        pf.line_spacing = line


def _rich(p, text, size=10, base_bold=False, color=TXT):
    """**gras** et `code` en ligne."""
    for part in re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', text):
        if not part:
            continue
        if part.startswith('**'):
            _font(p.add_run(part[2:-2]), size, bold=True, color=color)
        elif part.startswith('`'):
            _font(p.add_run(part[1:-1]), size - 0.5, mono=True, color=RGBColor(0x8B, 0x1A, 0x1A))
        else:
            _font(p.add_run(part), size, bold=base_bold, color=color)


def _borders(tbl, color='BFBFBF'):
    tblPr = tbl._tbl.tblPr
    b = OxmlElement('w:tblBorders')
    for e in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{e}')
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '4'); el.set(qn('w:color'), color)
        b.append(el)
    after = None
    for tag in ('w:tblInd', 'w:tblCellSpacing', 'w:jc', 'w:tblW', 'w:tblStyleColBandSize',
                'w:tblStyleRowBandSize', 'w:bidiVisual', 'w:tblOverlap', 'w:tblpPr', 'w:tblStyle'):
        after = tblPr.find(qn(tag))
        if after is not None:
            break
    if after is not None:
        after.addnext(b)
    else:
        tblPr.insert(0, b)


def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement('w:shd')
    sh.set(qn('w:val'), 'clear'); sh.set(qn('w:color'), 'auto'); sh.set(qn('w:fill'), fill)
    tcPr.append(sh)


def _width(cell, inches):
    tcPr = cell._tc.get_or_add_tcPr()
    w = OxmlElement('w:tcW'); w.set(qn('w:w'), str(int(inches * 1440))); w.set(qn('w:type'), 'dxa')
    for old in tcPr.findall(qn('w:tcW')):
        tcPr.remove(old)
    tcPr.insert(0, w)


class Cell:
    def __init__(self, cell):
        self.c = cell
        # garde "Réponse :" et retire les paragraphes vides de remplissage
        for p in list(cell.paragraphs)[1:]:
            if not p.text.strip():
                p._p.getparent().remove(p._p)

    def h(self, text):
        p = self.c.add_paragraph(); _spacing(p, before=100, after=40)
        _font(p.add_run(text), 10.5, bold=True, color=ACC)

    def p(self, text):
        p = self.c.add_paragraph(); _spacing(p, after=60, line=1.08)
        _rich(p, text)

    def b(self, text):
        p = self.c.add_paragraph(); _spacing(p, after=30, line=1.05)
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.first_line_indent = Inches(-0.14)
        _rich(p, '– ' + text)

    def code(self, text):
        for line in text.strip('\n').split('\n'):
            p = self.c.add_paragraph(); _spacing(p, after=0)
            p.paragraph_format.left_indent = Inches(0.15)
            _font(p.add_run(line if line else ' '), 8.5, mono=True, color=RGBColor(0x33, 0x33, 0x33))
        p = self.c.add_paragraph(); _spacing(p, after=20)

    def img(self, name, width=5.9, caption=None):
        p = self.c.add_paragraph(); _spacing(p, before=40, after=20)
        p.alignment = 1
        p.add_run().add_picture(str(FIG / name), width=Inches(width))
        if caption:
            q = self.c.add_paragraph(); _spacing(q, after=80); q.alignment = 1
            _font(q.add_run(caption), 8.5, italic=True, color=RGBColor(0x59, 0x59, 0x59))

    def table(self, header, rows, widths=None, size=8.5):
        t = self.c.add_table(rows=1 + len(rows), cols=len(header))
        _borders(t)
        total = 6.1
        widths = widths or [total / len(header)] * len(header)
        for j, htxt in enumerate(header):
            cell = t.cell(0, j); _shade(cell, 'DCE6F1'); _width(cell, widths[j])
            cell.paragraphs[0].text = ''
            _font(cell.paragraphs[0].add_run(str(htxt)), size, bold=True)
        for i, r in enumerate(rows, 1):
            for j, v in enumerate(r):
                cell = t.cell(i, j); _width(cell, widths[j])
                para = cell.paragraphs[0]; _spacing(para, after=0)
                _rich(para, str(v), size=size)
        # largeurs fixes : tblW + tblGrid + tblLayout fixed
        tblPr = t._tbl.tblPr
        tw = tblPr.find(qn('w:tblW'))
        if tw is None:
            tw = OxmlElement('w:tblW'); tblPr.append(tw)
        tw.set(qn('w:w'), str(int(sum(widths) * 1440))); tw.set(qn('w:type'), 'dxa')
        lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed')
        look = tblPr.find(qn('w:tblLook'))
        if look is not None:
            look.addprevious(lay)
        else:
            tblPr.append(lay)
        for gc, w in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), widths):
            gc.set(qn('w:w'), str(int(w * 1440)))
        return t


# ---------------------------------------------------------------- document
doc = Document(SRC)
reps = [i for i, t in enumerate(doc.tables)
        if len(t.rows) == 1 and len(t.columns) == 1 and t.cell(0, 0).text.strip().startswith('Réponse')]
assert len(reps) == 27, reps
R = [Cell(doc.tables[i].cell(0, 0)) for i in reps]

# ---------- en-tête : note de rendu
info = doc.tables[0]
anchor = info._tbl
note_p = OxmlElement('w:p')
anchor.addnext(note_p)
from docx.text.paragraph import Paragraph
np_ = Paragraph(note_p, info._parent)
_spacing(np_, before=120, after=60)
_rich(np_, f'**Rendu — code source complet et exécuté :** {REPO}  '
           '(notebook de démonstration `TP1_NLP_Monte_Cristo.ipynb`, package réutilisable `src/` organisé en '
           '`utils`, `preprocessing`, `analytics`, `vectorization` et `generation_rapport`, 12 tests unitaires, '
           'figures, données et ce rapport). '
           'Tous les chiffres ci-dessous proviennent de l\'exécution réelle du notebook sur le Tome I du roman '
           '(Gutenberg #17989), avec Python 3.13, NLTK 3.10, SpaCy 3.8 (fr_core_news_sm) et scikit-learn 1.9.',
      size=9.5, color=ACC)

# =========================================================== PARTIE 1
c = R[0]
c.h('a) Installations')
c.p('Toutes les librairies s\'installent et s\'importent sans erreur (`pip install -r requirements.txt` puis '
    '`python -m spacy download fr_core_news_sm`). Les 5 ressources NLTK (punkt, punkt_tab, stopwords, wordnet, '
    'omw-1.4) sont téléchargées ; `punkt_tab` est bien indispensable avec NLTK ≥ 3.9, sinon `word_tokenize` lève '
    'l\'erreur `LookupError` décrite dans le sujet.')
c.h('b) Versions installées')
c.table(['Python', 'NLTK', 'SpaCy', 'scikit-learn', 'pandas', 'Matplotlib', 'WordCloud', 'Modèle SpaCy'],
        [['3.13', '3.10.4', '3.8.18', '1.9.1', '3.0.5', '3.11.2', '1.9.6', 'fr_core_news_sm 3.8.0']],
        widths=[0.55, 0.6, 0.6, 0.8, 0.6, 0.75, 0.75, 1.45])
c.h('c) Vérification du modèle SpaCy')
c.code("[('Edmond', 'PROPN'), ('Dantès', 'PROPN'), ('fut', 'AUX'), ('injustement', 'ADV'),\n"
       " ('emprisonné', 'VERB'), ('au', 'ADP'), ('château', 'NOUN'), (\"d'\", 'ADP'), ('If', 'PROPN'), ('.', 'PUNCT')]")
c.p('Le modèle est opérationnel (pipeline : tok2vec, morphologizer, parser, attribute_ruler, lemmatizer, ner). '
    'L\'étiquetage est entièrement correct : les trois noms propres sont reconnus, « fut » est bien un auxiliaire, '
    '« emprisonné » un participe verbal, et l\'élision « d\' » est séparée de « If » et étiquetée préposition.')

# =========================================================== PARTIE 2
c = R[1]
c.h('Remarque préalable : corpus et marqueurs Gutenberg')
c.p('Le fichier 17989 correspond au **Tome I** du roman (sur 4), soit les chapitres I à XXXI (de l\'arrivée du '
    'Pharaon à Marseille jusqu\'à « Simbad le marin »). Par ailleurs, les marqueurs indiqués dans le sujet '
    '(`*** START OF THE PROJECT GUTENBERG EBOOK 17989 ***`) n\'existent pas dans le fichier : le vrai marqueur est '
    '`*** START OF THIS PROJECT GUTENBERG EBOOK LE COMTE DE MONTE-CRISTO, TOME I ***`. Le code d\'origine lève donc '
    'une `IndexError` sur `split(DEBUT)[1]`. Nous utilisons une expression régulière robuste aux deux formats '
    '(THE/THIS, avec ou sans titre), et un cache local + un miroir GitHub du Project Gutenberg si le site est inaccessible.')
c.h('a) Statistiques de base')
c.table(['Mesure', 'Valeur'],
        [['Caractères (fichier brut)', '747 000'], ['Caractères (roman extrait)', '**727 472**'],
         ['Lignes (roman extrait)', '**16 686**'], ['Métadonnées supprimées', '19 528 caractères (licence en anglais)']],
        widths=[3.0, 3.1])
c.p('Les **500 premiers caractères** contiennent la page de titre (« Alexandre Dumas — LE COMTE DE MONTE-CRISTO — '
    'Tome I (1845-1846) ») puis la table des matières (« I Marseille.--L\'arrivée. II Le père et le fils. III Les '
    'Catalans… »). Les **200 derniers caractères du fichier brut** sont la fin de la licence Gutenberg '
    '(« …subscribe to our email newsletter… *** END: FULL LICENSE *** ») : sans extraction, ce texte anglais serait '
    'analysé. Après extraction, le roman se termine bien par « …sous les enchantements de ce rêve inouï. FIN DU TOME PREMIER ».')
c.h('b) à e)')
c.b('**b)** `len(roman.split())` = **121 578 mots** (estimation brute).')
c.b('**c)** `roman.lower().count(\'dantès\')` = **777 occurrences**.')
c.b('**d)** 12 lignes contiennent « vengeance ». Les 5 premières : « que Fernand surtout était terrible dans sa '
    'vengeance.» » ; « «À la bonne heure, continua Danglars; ainsi votre vengeance aurait le » ; « menaçant et fort '
    'pour toutes les vengeances; alors il manifesta à M. » ; « qui, pour lui aussi, était devenu messager d\'une rude '
    'vengeance. Alors, » ; « Il se disait que c\'était la haine des hommes et non la vengeance de Dieu ».')
c.b('**e)** 121 578 / 300 ≈ **405 pages** pour le Tome I (≈ 1 600 pages pour l\'œuvre complète en 4 tomes).')
c.p('Le découpage en chapitres (regex sur les numéros romains isolés) détecte bien les **31 chapitres**, de 1 866 mots '
    '(ch. XX) à 9 363 mots (ch. XXXI).')

c = R[2]
c.h('Q1. Pourquoi supprimer les métadonnées Gutenberg ?')
c.p('L\'en-tête et surtout la licence finale représentent ≈ 19 500 caractères **en anglais** qui n\'appartiennent pas '
    'à l\'œuvre. Les conserver : (1) ajoute au vocabulaire des mots parasites (project, gutenberg, ebook, license, '
    'foundation…) ; (2) fausse toutes les statistiques (nombre de mots, fréquences, richesse lexicale, hapax) ; '
    '(3) pollue les représentations BoW/TF-IDF du dernier bloc et la détection de langue ; (4) rend les résultats non '
    'comparables d\'une édition à l\'autre. Le principe général : **analyser uniquement le signal**, jamais l\'emballage.')
c.h('Q2. `.split()` vs `word_tokenize`')
c.p('`.split()` coupe seulement sur les espaces : la ponctuation reste collée aux mots, si bien que « Dantès! », '
    '« Dantès, », « «Dantès » deviennent des types différents. Sur le texte brut, `.split()` donne 121 578 mots mais '
    '**20 220 types**, contre **11 142 types** après nettoyage + `word_tokenize`. `word_tokenize` applique des règles '
    'linguistiques (séparation de la ponctuation, abréviations comme « M. », contractions, nombres). '
    '**On préfère `.split()`** pour une estimation rapide ou sur un texte déjà nettoyé (c\'est ce que fait la partie 10), '
    'et **`word_tokenize`** dès que l\'on fait de l\'analyse (fréquences, vocabulaire, vectorisation).')
c.h('Q3. La fréquence d\'un nom est-elle un indicateur d\'importance ?')
c.p('Seulement partiellement. Dantès est bien le plus fréquent (777 + 247 « Edmond »), mais : (1) les personnages '
    'sont souvent désignés autrement (« le jeune homme » : 140 bigrammes, « le prisonnier », « le numéro 34 », pronoms) ; '
    '(2) dans ce tome, Dantès change d\'identité (abbé Busoni, Simbad le marin) : au-delà du chapitre XXV son nom '
    'disparaît presque alors qu\'il est toujours au centre du récit (voir partie 10) ; (3) l\'abbé Faria, pivot du roman, '
    'n\'apparaît que **105 fois** sous son nom mais 214 fois comme « l\'abbé ». L\'importance narrative relève du rôle '
    '(qui agit, sur qui), ce que mesurent mieux la résolution de coréférence, la NER ou l\'analyse de réseaux de personnages.')

# =========================================================== PARTIE 3
c = R[3]
c.h('a) Nettoyage du roman')
c.p('Avec la fonction du sujet : **727 472 → 695 417 caractères (–4,4 %)**. La réduction est faible car on remplace '
    'la ponctuation par des espaces au lieu de la supprimer : l\'essentiel du texte est conservé.')
c.h('b) Extrait bruité')
c.code("Avant : «Ah! c'est vous, Dantès!» cria l'homme... (numéro 34)\\n\\n-- Il est mort!!\n"
       "Après : ah c'est vous dantès cria l'homme numéro il est mort")
c.p('Il ne reste que des mots en minuscules : guillemets français, points d\'exclamation, points de suspension, '
    'parenthèses, tirets de dialogue et retours à la ligne ont disparu. Le nombre « 34 » a lui aussi disparu, car la '
    'classe de caractères autorisés de l\'étape 4 ne contient pas de chiffres. On perd ainsi « le numéro 34 », '
    'qui est pourtant l\'identité de Dantès au château d\'If (titre du chapitre XV).')
c.h('c) « Monte-Cristo »')
c.p('Le tiret est remplacé par un espace : on obtient **« monte cristo » (2 tokens)**, jamais « montécristo ». '
    '**Solution retenue (`clean_text_v2`)** : autoriser le tiret uniquement **entre deux lettres** puis supprimer les '
    'tirets isolés (`--` de dialogue). « monte-cristo » est alors conservé en un seul token (41 occurrences), tout comme '
    '« peut-être », « cent-jours » ou « saint-pierre-ès-liens ». Les inversions verbales sont séparées (« dit-il » → '
    '« dit il », « s\'écria-t-il » → « s écria il ») pour ne pas créer de faux mots. Alternatives : remplacer les '
    'expressions connues par `monte_cristo` avant le nettoyage, ou utiliser `nltk.tokenize.MWETokenizer`.')
c.h('d) Suppression des nombres + corrections apportées')
c.p('Ajout de `re.sub(r\'\\b\\d+\\b\', \' \', text)` (24, 1815…). Comme dit plus haut, cette étape est redondante avec la '
    'regex du sujet, mais elle devient nécessaire dès que l\'on élargit la classe de caractères autorisés ; nous la '
    'gardons explicite. Nous avons aussi corrigé deux défauts de la fonction d\'origine :')
c.b('l\'ordre des étapes 4 et 5 : les apostrophes typographiques (’) sont supprimées **avant** d\'être normalisées, '
    'donc l\'étape 5 ne sert à rien. On normalise désormais avant de filtrer ;')
c.b('les **élisions** : « l\'abbé » et « abbé » étaient deux types distincts. On sépare l\'élision (« l\'abbé » → '
    '« l abbé »), en préservant « aujourd\'hui » (pas de frontière de mot avant le d\').')
c.h('e) Version légère et comparaison')
c.table(['Version', 'Caractères', 'Mots (.split)', 'Types distincts'],
        [['Texte brut', '727 472', '121 578', '20 220'],
         ['clean_text (sujet)', '695 417', '123 564', '11 959'],
         ['**clean_text_v2 (retenue)**', '695 115', '130 542', '**11 142**'],
         ['clean_text_leger', '722 818', '121 578', '19 807']],
        widths=[2.2, 1.2, 1.3, 1.4])
c.p('La version légère (minuscules + espaces) ne réduit presque pas le vocabulaire (19 807 types) : « dantès, », '
    '« dantès! », « _pharaon_ » restent distincts. Le nettoyage complet **divise le vocabulaire par 1,8**. Le nombre de '
    'mots augmente légèrement avec v2 (130 542) car les élisions et les inversions sont séparées en mots autonomes.')

c = R[4]
c.h('Q1. Pourquoi nettoyer ? Trois raisons')
c.b('**Réduire le bruit et la dimension** : de 20 220 à 11 142 types (–45 %), donc des matrices plus petites et moins creuses.')
c.b('**Uniformiser** des formes identiques : « Dantès » / « dantès » / « Dantès! » doivent être comptées ensemble.')
c.b('**Retirer les artefacts non linguistiques** : italiques Gutenberg `_Pharaon_`, tirets de dialogue `--`, '
    'table des matières, numérotation, mentions de licence.')
c.h('Q2. Problèmes concrets sans nettoyage (sur ce roman)')
c.b('« Dantès », « Dantès! », « Dantès, », « «Dantès » comptés comme 4 mots différents ;')
c.b('« Pharaon » (nom du navire en italique) distinct de « Pharaon » ;')
c.b('les tirets de dialogue « -- » deviennent des tokens : 11 occurrences dans les seuls 5 000 premiers caractères ;')
c.b('des collages comme « Marseille.--L\'arrivée » (titre du chapitre I) produisent un token absurde '
    '(SpaCy l\'étiquette même comme adjectif) ;')
c.b('« l\'abbé » et « abbé » séparés : sans traitement des élisions, l\'abbé Faria est sous-compté.')
c.h('Q3. Le nettoyage supprime-t-il de l\'information utile ?')
c.p('Oui. La **ponctuation** porte l\'émotion et la modalité (« !! », « ? » : utile en analyse de sentiment), '
    'marque le dialogue (guillemets, tirets) et surtout délimite les phrases : nous l\'avons constaté, `sent_tokenize` '
    'sur le texte nettoyé ne trouve **qu\'une seule phrase** au lieu de 237. Les **majuscules** servent à la '
    'détection des noms propres (on a dû travailler sur le texte brut pour le WordCloud des PROPN). Les **nombres** '
    'portent les dates (24 février 1815) et les identités (numéros 34 et 27).')
c.h('Q4. Faut-il toujours tout appliquer ?')
c.p('Non, le nettoyage dépend de la tâche. Exemples contre-productifs : supprimer la ponctuation **avant** la '
    'segmentation en phrases (démontré en 4.3) ; passer en minuscules avant une **extraction d\'entités nommées** '
    '(« Pharaon » le navire devient « pharaon ») ; supprimer les nombres pour une extraction de dates ; supprimer « ne… '
    'pas » en analyse de sentiment. Les modèles de type BERT/CamemBERT ont d\'ailleurs leur propre tokeniseur et '
    'travaillent sur le texte quasi brut.')

# =========================================================== PARTIE 4
c = R[5]
c.h('a) et b) Tokenisation NLTK du roman complet')
c.table(['Tokens', 'Vocabulaire (types)', 'Richesse lexicale (TTR)', 'Temps'],
        [['**130 542**', '**11 142**', '**0,0854**', '0,5 s']], widths=[1.4, 1.6, 1.8, 1.3])
c.code("['alexandre', 'dumas', 'le', 'comte', 'de', 'monte-cristo', 'tome', 'i', 'table', 'des', 'matières',\n"
       " 'alexandre', 'dumas', 'i', 'marseille', 'l', 'arrivée', 'ii', 'le', 'père', 'et', 'le', 'fils', 'iii',\n"
       " 'les', 'catalans', 'iv', 'complot', 'v', 'le', 'repas', 'des', 'fiançailles', 'vi', 'le', 'substitut',\n"
       " 'du', 'procureur', 'du', 'roi', 'vii', 'l', 'interrogatoire', 'viii', 'le', 'château', 'd', 'if', 'ix', 'le']")
c.p('Un TTR de 0,085 signifie qu\'en moyenne chaque mot distinct est répété ≈ 12 fois : c\'est normal pour un long '
    'texte (le TTR diminue mécaniquement avec la taille du corpus, il ne faut comparer que des textes de même longueur).')
c.h('c) NLTK vs SpaCy sur le début du roman (5 000 caractères)')
c.table(['Critère', 'NLTK (word_tokenize)', 'SpaCy (fr_core_news_sm)'],
        [['Tokens (texte nettoyé)', '920', '932'],
         ['Tokens (texte brut)', '975', '1 039 (+6,6 %)'],
         ['Élisions « l\'abbé », « s\'écria »', 'un seul token : `l\'abbé`', 'deux tokens : `l\'` + `abbé`'],
         ['Inversion « est-il »', '`est-il`', '`est` + `-il`'],
         ['Mots composés (monte-cristo, cent-jours)', 'conservés', 'découpés : `monte` `-` `cristo`'],
         ['Tirets de dialogue « -- »', '`--` (1 token)', '`-` séparés ou collés au mot voisin'],
         ['Italiques `_Pharaon_`', '`_Pharaon_`', '`_` + `Pharaon` + `_`'],
         ['Notre-Dame-de-la-Garde, trois-mâts, M.', 'identiques', 'identiques'],
         ['Informations supplémentaires', 'aucune', 'POS, lemme, stop word, entités…']],
        widths=[2.2, 1.9, 2.0])
c.h('d) 20 bigrammes les plus fréquents')
c.p('Sur tous les tokens, ce sont des groupes grammaticaux : **de la** (589), **qu il** (575), **c est** (385), à la '
    '(384), de l (347), d un (253), à l (223), que je (222), d une (209), **l abbé** (201), il avait (195), que vous (186)… '
    'Seul « l abbé » renvoie au contenu. En filtrant les mots vides, des **noms de personnages et des expressions '
    'caractéristiques** apparaissent :')
c.table(['Bigramme', 'n', 'Bigramme', 'n', 'Bigramme', 'n'],
        [['jeune homme', '140', 'procureur roi', '31', 'capitaine leclère', '17'],
         ['louis xviii', '45', 'ministre police', '22', 'thomson french', '17'],
         ['monsieur morrel', '41', 'abbé faria', '22', 'substitut procureur', '16'],
         ['jeune fille', '40', 'edmond dantès', '21', 'île monte-cristo', '18'],
         ['île elbe', '36', 'monsieur villefort', '21', 'écria dantès', '19']],
        widths=[1.35, 0.6, 1.35, 0.6, 1.5, 0.7])
c.p('« jeune homme » est la désignation récurrente de Dantès ; « louis xviii », « île d\'Elbe », « ministre de la '
    'police » situent l\'intrigue politique (Restauration / Cent-Jours) ; « procureur du roi » et « substitut » renvoient '
    'à Villefort ; « Thomson et French » est la maison de banque de Rome.')
c.h('e) DataFrame des tokens')
c.p('DataFrame de 130 542 lignes [position, token, longueur]. **Longueur moyenne : 4,32 caractères** (médiane 4, max 21 '
    'pour « saint-pierre-ès-liens »). Mais la longueur moyenne des **types** est de 7,60 caractères : les mots courts '
    '(de, le, la, il, à) sont peu nombreux mais extrêmement répétés, ce qui fait baisser la moyenne par occurrence.')

c = R[6]
c.h('Q1. Token vs mot')
c.p('Un **token** est l\'unité produite par un tokeniseur selon des règles techniques : mot, signe de ponctuation, '
    'nombre, élision, sous-mot. Un **mot** est une unité linguistique. Les deux ne coïncident pas toujours : '
    '« aujourd\'hui » = 1 mot = 1 token ; « l\'homme » = 2 mots, mais 1 token pour NLTK et 2 pour SpaCy ; « dit-il » = '
    '2 mots ; pour BERT, « Dantès » peut devenir plusieurs sous-mots. On distingue aussi le token (occurrence, 130 542) '
    'du type (forme distincte, 11 142).')
c.h('Q2. Pourquoi la tokenisation est-elle fondamentale ?')
c.p('C\'est elle qui définit les unités sur lesquelles tout le reste opère : comptages, stop words, stemming, '
    'lemmatisation, vocabulaire et donc **les dimensions** des matrices BoW/TF-IDF. Une erreur à cette étape se '
    'propage dans tout le pipeline (ex. « l\'abbé » non séparé → « abbé » sous-compté, Faria sous-estimé).')
c.h('Q3. NLTK et SpaCy produisent-ils les mêmes tokens ?')
c.p('Non (975 vs 1 039 tokens sur le même extrait brut). **Différence principale** : SpaCy possède des règles '
    'd\'exception propres au français et **sépare les élisions** (l\', d\', s\', qu\') ainsi que les pronoms inversés '
    '(-il), alors que NLTK (Punkt + règles de type Treebank pensées pour l\'anglais) les laisse collés. À l\'inverse, '
    'NLTK garde les mots composés à tiret, que SpaCy découpe. SpaCy est aussi plus lent mais fournit POS, lemmes et entités.')
c.h('Q4. Quand les bigrammes sont-ils plus utiles que les mots isolés ?')
c.p('Pour les **expressions figées et les entités multi-mots** dont le sens n\'est pas la somme des mots : « jeune '
    'homme » (140) ne veut pas dire « jeune » + « homme » séparément, « procureur du roi », « château d\'If », « île '
    'd\'Elbe », « Louis XVIII ». Aussi pour la **négation** (« pas coupable ») en analyse de sentiment, et dans un moteur '
    'de recherche (requête « abbé Faria » : mieux vaut chercher le bigramme que les deux mots indépendamment).')

# =========================================================== PARTIE 5
c = R[7]
c.h('a) Top 20 (rang, mot, fréquence, %)')
top20 = [('de', 4839, 3.707), ('le', 3207, 2.457), ('et', 3196, 2.448), ('la', 2967, 2.273), ('il', 2859, 2.190),
         ('à', 2815, 2.156), ('l', 2387, 1.829), ('que', 1963, 1.504), ('un', 1896, 1.452), ('vous', 1665, 1.275),
         ('en', 1471, 1.127), ('d', 1380, 1.057), ('les', 1359, 1.041), ('qui', 1279, 0.980), ('je', 1278, 0.979),
         ('une', 1210, 0.927), ('qu', 1136, 0.870), ('ce', 1078, 0.826), ('est', 1025, 0.785), ('dit', 938, 0.719)]
rows = []
for i in range(10):
    a, b = top20[i], top20[i + 10]
    rows.append([i + 1, a[0], a[1], f'{a[2]:.3f} %'.replace('.', ','), i + 11, b[0], b[1], f'{b[2]:.3f} %'.replace('.', ',')])
c.table(['Rang', 'Mot', 'Fréq.', '%', 'Rang', 'Mot', 'Fréq.', '%'], rows,
        widths=[0.5, 0.75, 0.75, 1.05, 0.5, 0.75, 0.75, 1.05])
c.p('Ces 20 mots couvrent à eux seuls **30,6 % des tokens**.')
c.h('b) Graphique du top 20')
c.img('05_top20_frequences.png', 5.6)
c.h('c) Loi de Zipf')
c.img('05_loi_de_zipf.png', 5.4)
c.p('Le nuage de points forme une **droite quasi parfaite** en log-log. La pente ajustée est **–1,15** (–1,09 sur les '
    '1 000 premiers rangs), proche du –1 théorique : **la loi de Zipf est vérifiée**. Les écarts sont classiques : en tête, '
    'le ratio f₁/f₂ vaut 1,51 au lieu de 2 (les mots-outils les plus fréquents sont « tassés », ce que modélise mieux '
    'la loi de Zipf-Mandelbrot) ; en queue, des paliers horizontaux apparaissent (des milliers de mots à fréquence 1, 2, 3).')
c.h('d) Mots thématiques')
c.table(['Mot', 'Fréquence', 'Rang'],
        [['dantès', '777', '30'], ['prison', '67', '220'], ['liberté', '49', '301'], ['vengeance', '11', '1 126']],
        widths=[2.0, 2.0, 2.1])
c.p('« vengeance » est étonnamment rare : le Tome I raconte l\'ascension, la trahison, la prison et l\'évasion de '
    'Dantès ; la vengeance proprement dite se déroule dans les tomes II à IV. La prison domine bien le vocabulaire thématique.')
c.h('e) Hapax legomena')
c.p('**5 434 mots n\'apparaissent qu\'une fois, soit 48,8 % du vocabulaire** mais seulement 4,16 % des tokens. '
    'Exemples : bêche, raviserait, langoureux, scindaient, indubitablement, exterminateur, licite, planton, cavités. '
    'Ils mélangent du **vocabulaire riche et informatif** (exterminateur, cavités), des **formes conjuguées rares** '
    '(raviserait, arriverai : la lemmatisation les ramènerait à des lemmes fréquents) et du **bruit**. Statistiquement, '
    'ils sont peu exploitables (un seul exemple) et doublent la dimension des matrices : on les filtre souvent '
    '(`min_df=2`), mais ils sont précieux en stylométrie et en recherche d\'information.')

c = R[8]
c.h('Q1. Le top 20 est-il informatif ?')
c.p('Non : ce sont tous des **mots-outils** (articles, prépositions, pronoms, auxiliaires). On obtiendrait presque la '
    'même liste avec n\'importe quel roman français. Deux indices de style seulement : « vous » (10e) et « je » (15e) '
    'révèlent l\'importance du dialogue, confirmée par « dit » (20e). Rien ne permet de deviner le sujet du roman.')
c.h('Q2. La loi de Zipf et sa signification pour le NLP')
c.p('Zipf énonce que la fréquence d\'un mot est inversement proportionnelle à son rang : f(r) ≈ C / r^s avec s ≈ 1 '
    '(droite de pente –1 en log-log). Conséquences pour le NLP : quelques dizaines de mots couvrent un tiers du texte '
    '(→ intérêt des stop words), tandis qu\'une très longue traîne de mots rares (→ hapax) rend les matrices '
    '**creuses**, impose de tronquer le vocabulaire (`max_features`, `min_df`), de lisser les probabilités des modèles '
    'de langue et de gérer les mots hors vocabulaire. C\'est aussi la justification de la pondération IDF.')
c.h('Q3. Hapax : définition, faut-il les garder ?')
c.p('Un hapax est un mot qui n\'apparaît **qu\'une seule fois** dans le corpus (48,8 % du vocabulaire ici). Pour une '
    'analyse statistique (fréquences, BoW, classification, topic modeling) on les **supprime** en général : ils '
    'augmentent la dimension sans apporter d\'information généralisable. On les **garde** pour la stylométrie '
    '(richesse lexicale d\'un auteur), l\'attribution d\'auteur ou un moteur de recherche (un mot rare est très '
    'discriminant). Une lemmatisation préalable en élimine une partie.')

# =========================================================== PARTIE 6
c = R[9]
c.h('a) NLTK vs SpaCy')
c.table(['NLTK', 'SpaCy', 'Communs', 'Union', 'Seulement NLTK', 'Seulement SpaCy'],
        [['157', '**507**', '85', '579', '72', '422']], widths=[0.8, 0.8, 0.9, 0.8, 1.3, 1.5])
c.p('**SpaCy est bien plus complète** : elle ajoute des adverbes et connecteurs (ainsi, alors, afin, après, assez, '
    'autrement…), parfois en double sans accent (« anterieur »/« antérieur »). NLTK contient surtout les formes '
    'conjuguées d\'être et avoir (eusse, aurions, ayants) et les lettres élidées (c, d, j, l…), absentes de SpaCy. '
    'Les deux listes sont complémentaires : nous utilisons leur union.')
c.h('b) Top 20 après suppression')
c.table(['Mot', 'n', 'Mot', 'n', 'Mot', 'n', 'Mot', 'n'],
        [['**dantès**', '777', '**caderousse**', '219', 'fois', '167', 'demanda', '155'],
         ['homme', '424', 'jeune', '217', '**danglars**', '166', 'yeux', '154'],
         ['**morrel**', '273', '**abbé**', '214', 'rien', '165', 'fit', '141'],
         ['**villefort**', '258', 'père', '193', '**fernand**', '161', 'point', '139'],
         ['**edmond**', '247', 'faire', '225', 'temps', '157', 'jour', '138']],
        widths=[0.95, 0.55, 0.95, 0.55, 0.95, 0.55, 0.95, 0.65])
c.p('Les personnages apparaissent enfin : **Dantès/Edmond, Morrel, Villefort, Caderousse, l\'abbé (Faria), '
    'Danglars, Fernand** (Mercédès est 21e). Restent quelques mots peu informatifs (faire, fois, point, fit, demanda).')
c.h('c) Taux de réduction')
c.p('**130 542 → 50 389 tokens : réduction de 61,4 %**. En revanche, le vocabulaire ne baisse que de 4,3 % '
    '(11 142 → 10 668 types) : les stop words sont **peu nombreux mais extrêmement fréquents** (loi de Zipf).')
c.h('d) Pourquoi « dit », « répondit », « monsieur » ?')
c.p('« dit » apparaît 938 fois (20e mot du roman), « répondit » 137, « reprit » 119, « monsieur » 333. Ce sont les '
    '**incises de dialogue** (« dit Caderousse », « répondit Dantès » : après dit/répondit viennent surtout il, le, '
    'Caderousse, Dantès, Danglars) et une **formule de politesse** (« monsieur Morrel », « monsieur de Villefort »). '
    'Ils sont répartis uniformément et masquent les mots thématiques. Ils ne sont pourtant **pas totalement '
    'non-informatifs** : ils mesurent la part de dialogue, et « monsieur » traduit la hiérarchie sociale du XIXe siècle. '
    'La liste est aussi incohérente : « demanda » (155), « fit », « point » (« ne… point ») ou « faire » auraient dû y figurer.')
c.h('e) Personnages')
c.img('06_personnages.png', 5.3)
c.p('**Dantès domine largement** (777, et 1 024 avec « Edmond »), devant Morrel (273), Villefort (258), Caderousse '
    '(219), Danglars (166), Fernand (161), Mercédès (138) et Faria (105, mais 214 « abbé »).')

c = R[10]
c.h('Q1. Mots disparus')
c.p('Articles (le, la, les, un), prépositions (de, à, en), conjonctions (et, que), pronoms (il, je, vous), '
    'auxiliaires (est, avait) et élisions (l, d, qu). Ils sont **inutiles pour identifier les thèmes**, mais '
    'indispensables à la syntaxe et au sens (négation, rôle des personnages, temps).')
c.h('Q2. Pourquoi des stop words spécifiques au domaine ?')
c.p('Certains mots sont fréquents **partout dans un corpus donné** sans discriminer les documents : « dit » et '
    '« monsieur » dans un roman dialogué du XIXe siècle, « patient » dans un corpus médical, « produit » dans des avis '
    'clients. Ils se comportent comme des mots vides dans ce contexte précis. C\'est l\'équivalent manuel de `max_df` '
    '(qui a d\'ailleurs retiré automatiquement homme, jeune, yeux, faire, fois en partie 10).')
c.h('Q3. Tâche où il ne faut PAS supprimer les stop words')
c.p('**L\'analyse de sentiment** : « je ne suis pas coupable » devient « coupable », sens inversé. Également la '
    'traduction automatique, les questions-réponses, les modèles de langue, la recherche d\'expressions exactes '
    '(« le comte de Monte-Cristo ») et la **stylométrie / attribution d\'auteur**, où la fréquence des mots-outils est '
    'justement le meilleur marqueur du style d\'un auteur.')
c.h('Q4. Est-ce réversible ? Impact sur la compréhension')
c.p('**Non**, c\'est une perte d\'information irréversible (sauf à garder le texte original). Exemple : « Je ne l\'ai '
    'pas trahi » → « trahi » : le sens est inversé. « Il est mort » → « mort ». On perd la négation, les relations '
    'sujet/objet, les temps : la phrase devient un sac de mots-clés, utile pour la recherche thématique mais plus '
    'pour la compréhension.')

# =========================================================== PARTIE 7
c = R[11]
c.h('a) WordCloud du roman complet')
c.img('07_wordcloud_roman.png', 5.9)
c.p('Les 5 mots les plus visibles : **dantès, homme, morrel, villefort, jeune**. C\'est cohérent avec le Tome I : le héros, '
    'son armateur Morrel, le procureur Villefort qui l\'emprisonne, et « jeune homme », désignation constante de '
    'Dantès. « homme » et « jeune » restent cependant génériques.')
c.h('b) WordClouds par personnage')
c.p('**Correction du code fourni** : `sent_tokenize(roman_clean)` ne trouve qu\'**une seule phrase** car la ponctuation a '
    'été supprimée ; les trois nuages auraient été identiques (tout le roman). Nous segmentons le texte **brut** en '
    'phrases (6 598), puis nettoyons chaque phrase. Le nom du personnage est retiré de son propre nuage, et nous '
    'calculons les **mots distinctifs** (sur-représentation par rapport au roman entier).')
c.img('07_wordcloud_personnages.png', 6.0)
c.table(['Personnage', 'Phrases', 'Mots distinctifs (sur-représentés)', 'Interprétation'],
        [['Dantès', '743', 'abbé, faria, prisonnier, fragments, dalles, pioche, crique', 'prison, tunnel, évasion, trésor'],
         ['Danglars', '162', 'écriture, dénonciation, fernand, caderousse, capitaine', 'le complot et la lettre anonyme'],
         ['Villefort', '252', 'certificat, prévenus, pétition, juge, saint-méran, renée, napoléon',
          'justice, carrière, mariage, politique']],
        widths=[0.85, 0.65, 2.7, 1.9])
c.h('c) Noms propres (POS = PROPN)')
c.p('La détection des noms propres s\'appuie sur la majuscule : on applique donc SpaCy sur le texte **brut** '
    '(4 201 paragraphes, `nlp.pipe`, parser et NER désactivés : 6,7 s).')
c.img('07_wordcloud_noms_propres.png', 5.6)
c.p('**Personnages** : Dantès (287), Morrel, Villefort, Edmond, Caderousse, Danglars, Fernand, Mercédès, Franz, '
    'Jacopo, Faria, Gaetano, Blacas, Julie, Emmanuel. **Lieux** : Marseille, Paris, France, île d\'Elbe, Monte-Cristo. '
    'Également Dieu, Pharaon (le navire), Louis (XVIII), Napoléon. **Limites** : « Monte » et « Cristo » sont séparés, '
    'et des noms communs en début de phrase sont mal étiquetés (« mer », « rocher », « calme »).')
c.h('d) Paramètres')
c.img('07_wordcloud_parametres.png', 6.0)
c.p('**max_words=50 est le plus lisible** : peu de mots, grands, hiérarchie claire. Avec 200 mots, le nuage est plus '
    'exhaustif mais la majorité des mots devient illisible. La palette **RdYlGn** est peu adaptée : le jaune pâle sur fond '
    'blanc manque de contraste. Meilleur réglage : 50 à 100 mots avec une palette séquentielle sombre (Blues, viridis).')
c.h('e) Début vs fin du roman')
c.img('07_wordcloud_debut_fin.png', 6.0)
c.p('Les vocabulaires sont **très différents** (indice de Jaccard = 0,16). Début : dantès, père, capitaine, morrel, '
    'armateur, danglars, mercédès, fernand → Marseille, le navire, la famille, le triangle amoureux. Fin : franz, '
    'gaetano, barque, feu, hôte, excellence, simbad → l\'Italie, Franz d\'Épinay et le mystérieux « Simbad le marin ». '
    'Cela révèle la **structure narrative** : changement de décor, de personnages et de point de vue. Le héros a disparu '
    'du vocabulaire sous son propre nom, il est devenu « l\'hôte », « Son Excellence », « Simbad ».')

c = R[12]
c.h('Q1. Les WordClouds suffisent-ils ?')
c.p('Non, ce sont des outils d\'**exploration**. Limites : pas de valeur quantitative précise (tailles difficiles à '
    'comparer), les mots longs paraissent plus importants, aucune information de contexte, d\'ordre ou de négation, '
    'forte dépendance au prétraitement, mise en page aléatoire, mots génériques fréquents surreprésentés (homme, '
    'faire), expressions multi-mots cassées (« monte » / « cristo »). Surtout, la fréquence brute ne dit pas ce qui est '
    '**distinctif** : il faut des mesures relatives (TF-IDF, log-ratio, comme fait en b).')
c.h('Q2. Thèmes dominants')
c.b('**Les personnages et leurs relations** : Dantès, Morrel, Villefort, Danglars, Fernand, Mercédès, Caderousse ;')
c.b('**la mer et la navigation** : Pharaon, capitaine, navire, armateur, marin, barque, île ;')
c.b('**la justice et la prison** : procureur, substitut, prisonnier, cachot, geôlier, château (d\'If), abbé ;')
c.b('**la politique** : roi, sire, Napoléon, Louis XVIII, île d\'Elbe, ministre ;')
c.b('**la famille et l\'amour** : père, mère, fils, fiançailles, Mercédès ; puis **le trésor et l\'Italie** en fin de tome.')
c.h('Q3. Pourquoi supprimer les stop words avant ?')
c.p('La taille d\'un mot étant proportionnelle à sa fréquence, « de, le, et, la, il » (30 % des tokens) occuperaient '
    'tout le nuage et écraseraient les mots de contenu, dans un nombre de places limité (`max_words`). Le nuage serait '
    'identique pour n\'importe quel texte français et ne dirait rien du roman.')

# =========================================================== PARTIE 8
c = R[13]
c.h('a) Familles de mots')
c.table(['Famille', 'Mots → racines', 'Racines distinctes'],
        [['vengeance', 'vengeance→vengeanc, venger→veng, vengeur→vengeur, vindicatif→vindiqu', '4'],
         ['prison', 'prison→prison, prisonnier→prisonni, emprisonner→emprison, emprisonnement→emprison', '3'],
         ['liberté', 'liberté→libert, libérer→liber, libérateur→liber, libération→liber', '2'],
         ['mer', 'mer→mer, marin→marin, maritime→maritim, navire→navir, naviguer→navigu', '5']],
        widths=[0.9, 4.2, 1.0])
c.p('**Non**, aucune famille ne partage une racine unique. Le stemmer gère bien la flexion (prisonnier / '
    'prisonniers → prisonni, libérer / libération → liber) mais mal la **dérivation** (vengeance ≠ venger) ; et '
    'navire / mer / marin n\'ont de toute façon pas la même étymologie : un stemmer ne capte pas la parenté sémantique.')
c.h('b) Réduction du vocabulaire')
c.p('**10 668 mots → 5 581 racines : –47,7 %**. Le gain est **très significatif** : la dimension est presque divisée '
    'par deux, surtout grâce à la conjugaison (familles de 20 formes : trouva, trouvait, trouvaient, trouver… → '
    '« trouv »). Top racines : dantes (777), homm (509), morrel, demand (266), villefort, edmond, **port (234)**, jeun, fair.')
c.h('c) Faux amis du stemmer')
c.table(['Mots de sens différents', 'Racine commune'],
        [['**mère** (parent) / **mer** (océan)', 'mer'],
         ['**porte** (de la cellule) / **porter** (verbe) / **port** (de Marseille)', 'port'],
         ['**sort** (destin) / **sortir** / **sortie**', 'sort'],
         ['**livre** (ouvrage) / **livrer** (remettre)', 'livr'],
         ['**sommes** (verbe être) / **somme** (montant, ou sommeil)', 'somm']],
        widths=[4.4, 1.7])
c.p('Le cas « port » est grave pour ce roman : la 7e racine la plus fréquente (234) mélange le port de Marseille, la '
    'porte du cachot et le verbe porter.')
c.h('d) 15 mots caractéristiques')
stem_rows = [('vengeance', 'vengeanc', 'non'), ('prisonnier', 'prisonni', 'non'), ('château', 'château', 'oui'),
             ('liberté', 'libert', 'non'), ('navire', 'navir', 'non'), ('trésor', 'trésor', 'oui'),
             ('abbé', 'abbé', 'oui'), ('capitaine', 'capitain', 'non'), ('mariage', 'mariag', 'non'),
             ('procureur', 'procureur', 'oui'), ('emprisonné', 'emprison', 'non'), ('évasion', 'évas', 'non'),
             ('fiançailles', 'fiançaill', 'non'), ('cachot', 'cachot', 'oui'), ('marseille', 'marseil', 'non')]
c.table(['Mot original', 'Radical (Snowball)', 'Intelligible ?', 'Mot original', 'Radical (Snowball)', 'Intelligible ?'],
        [list(stem_rows[i]) + (list(stem_rows[i + 8]) if i + 8 < 15 else ['', '', '']) for i in range(8)],
        widths=[1.05, 1.0, 0.95, 1.05, 1.0, 1.05])
c.p('Seuls **5/15** radicaux sont des mots réels (ceux que le stemmer ne modifie pas) ; les 10 autres sont tronqués et '
    'illisibles (vengeanc, prisonni, fiançaill, marseil).')
c.h('e) Fréquences des racines thématiques')
c.table(['Racine', 'Occurrences', 'Mots regroupés'],
        [['dant-', '780', 'dantès, dante'], ['pris-', '251', 'prison, prisonnier(s), prisons… mais aussi **pris, prise** (prendre !)'],
         ['liber-', '49', 'liberté'], ['veng-', '19', 'vengeance(s), venger, vengeur, vengent…']],
        widths=[0.9, 1.0, 4.2])
c.p('Après le personnage principal, **le thème de la prison domine** nettement le Tome I, loin devant la liberté et la '
    'vengeance. Mais la racine « pris- » est elle-même un faux ami (elle inclut le participe « pris » du verbe prendre) : '
    'le chiffre de 251 surestime le thème carcéral.')

c = R[14]
c.h('Q1. Avantages du stemming')
c.p('Il réduit fortement le vocabulaire (–47,7 %), regroupe les flexions (prisonnier/prisonniers, toutes les formes '
    'conjuguées), réduit la sparsité des matrices, améliore le **rappel** d\'un moteur de recherche (« emprisonné » '
    'retrouve « emprisonnement »). Il est très rapide, ne nécessite ni dictionnaire ni modèle, et reste disponible pour '
    'de nombreuses langues.')
c.h('Q2. Limites (exemples concrets)')
c.b('**Sur-racinisation** (sens différents fusionnés) : mère/mer → « mer », porte/port/porter → « port », pris/prison → « pris- » ;')
c.b('**Sous-racinisation** (même famille séparée) : venger → « veng » vs vengeance → « vengeanc » ; formes irrégulières '
    'non reconnues : fut (≠ être), allait (≠ aller), yeux (≠ œil), aperçut inchangé ;')
c.b('**Radicaux illisibles et noms propres abîmés** : prisonni, fiançaill, caderouss, marseil, et « dantès » → « dantes » (accent perdu).')
c.h('Q3. Pourquoi Snowball plutôt que Porter pour le français ?')
c.p('L\'algorithme de Porter (1980) a été conçu **pour l\'anglais** : ses règles visent -ing, -ed, -ation, -ness. '
    'Appliqué au français, il laisse les terminaisons françaises intactes ou coupe à tort. Snowball (« Porter 2 », '
    'conçu par Porter lui-même) est un cadre multilingue avec un **jeu de règles spécifique au français** : suffixes '
    '-ement, -ité, -euse, -ation, terminaisons verbales (-aient, -erons, -issait), gestion des voyelles accentuées et '
    'des régions RV/R1/R2 propres à la morphologie française.')
c.h('Q4. Quand préférer le stemming à la lemmatisation ?')
c.p('Pour les **gros volumes ou le temps réel** (moteurs de recherche, indexation de millions de documents) où la '
    'vitesse compte (NLTK est ≈ 8 fois plus rapide ici) ; quand la lisibilité des termes importe peu (features BoW/TF-IDF '
    'pour un classifieur) ; quand on privilégie le rappel ; ou pour une langue sans bon lemmatiseur.')

# =========================================================== PARTIE 9
c = R[15]
c.h('a) SpaCy sur le chapitre I complet')
c.p('Chapitre I « Marseille. — L\'arrivée » : **4 189 tokens analysés en 0,46 s**. Avant l\'analyse, on retire les '
    'italiques `_Pharaon_` et on remplace les tirets de dialogue `--` (sinon SpaCy produit « Marseille.--L\' » étiqueté '
    'ADJ). Le DataFrame complet (token, lemme, POS, est_stop, est_ponc) est exporté dans `outputs/chapitre1_spacy_tokens.csv` ; extrait :')
c.table(['token', 'lemme', 'POS', 'est_stop', 'est_ponc'],
        [['Le', 'le', 'DET', 'True', 'False'], ['vigie', 'vigie', 'NOUN', 'False', 'False'],
         ['Notre-Dame', 'Notre-Dame', 'PROPN', 'False', 'False'], ['signala', 'signala', 'ADJ ✗', 'False', 'False'],
         ['trois-mâts', 'trois-mâts', 'NOUN', 'False', 'False'], ['venant', 'venir', 'VERB', 'False', 'False'],
         ['Smyrne', 'Smyrne', 'PROPN', 'False', 'False'], ['Trieste', 'trieste', 'NOUN ✗', 'False', 'False']],
        widths=[1.3, 1.3, 1.2, 1.15, 1.15])
c.p('Erreurs visibles : « signala » (passé simple) pris pour un adjectif et non lemmatisé, « Trieste » pris pour un nom commun.')
c.h('b) Top 15 noms communs lemmatisés')
c.p('monsieur (32), **armateur (28)**, homme (19), **capitaine (16)**, dantè (14, erreur), **bâtiment (11)**, île (9), '
    'père (9), **navire (8)**, oeil (7), ordre (7), **pilote (6)**, **port (6)**, fils (6), paquet (6). Le lexique est '
    'maritime et commercial : c\'est l\'arrivée du Pharaon et le rapport de Dantès à son armateur. '
    '« dantè » révèle une erreur : Dantès est parfois étiqueté NOUN et le lemmatiseur retire le « s » final comme un pluriel.')
c.h('c) Noms propres')
c.p('**Personnages** : Morrel (21), Danglars (17), Dantès (15), Leclère (10, le capitaine mort en mer), Edmond (10), '
    'Mercédès (3). **Lieux** : Marseille, île d\'Elbe et Porto-Ferrajo (escale du navire et lettre de Napoléon), Paris, '
    'Naples, Smyrne, If, Jaros, Pomègue (îles devant Marseille), la Réserve. **Navire** : Pharaon (8). Quelques erreurs : '
    '« jeta », « Pardon », « Excusez » étiquetés PROPN à cause de la majuscule après un tiret de dialogue.')
c.h('d) Stemming vs lemmatisation (lemme pris dans le contexte réel du roman)')
lem = [('navire', 'navir', 'navire', 'non', 'oui'), ('prisonnier', 'prisonni', 'prisonnier', 'non', 'oui'),
       ('vengeance', 'vengeanc', 'vengeance', 'non', 'oui'), ('chevaux', 'cheval', 'cheval', 'oui', 'oui'),
       ('yeux', 'yeux', 'yeux', 'non', 'non (œil)'), ('allait', 'allait', 'aller', 'non', 'oui'),
       ('fut', 'fut', 'être', 'non', 'oui'), ('meilleurs', 'meilleur', 'meilleur', 'non', 'non (bon)'),
       ('aperçut', 'aperçut', 'apercevoir', 'non', 'oui'), ('gendarmes', 'gendarm', 'gendarme', 'non', 'oui'),
       ('prisonniers', 'prisonni', 'prisonnier', 'non', 'oui'), ('disait', 'dis', 'dire', 'non', 'oui'),
       ('émotion', 'émot', 'émotion', 'non', 'oui'), ('jeunes', 'jeun', 'jeune', 'non', 'oui'),
       ('dantès', 'dantes', 'dantès', 'non', 'oui')]
c.table(['Mot original', 'Stemming (Snowball)', 'Lemme (SpaCy)', 'Stem correct ?', 'Lemme correct ?'],
        [list(r) for r in lem], widths=[1.25, 1.35, 1.25, 1.05, 1.2])
c.p('**Score : stemming 1/15, lemmatisation 13/15.** Les deux échecs de SpaCy : « yeux » (étiqueté ADJ dans sa phrase, '
    'alors qu\'ailleurs il donne bien « oeil ») et « meilleurs », laissé au comparatif.')
c.h('e) Méthode la plus adaptée à l\'analyse littéraire : la lemmatisation')
c.b('**fut → être** (stem « fut ») et **aperçut → apercevoir** (stem inchangé) : le récit est au passé simple, très '
    'irrégulier ; seul le lemmatiseur regroupe ces formes avec leur verbe.')
c.b('**disait → dire** (stem « dis », confondu avec « dis ») et **allait → aller** : verbes de parole et de mouvement, '
    'omniprésents dans le dialogue, correctement regroupés.')
c.b('**mère / mer** : le stemmer les fusionne (« mer »), le lemmatiseur garde deux lemmes distincts, indispensable dans '
    'un roman où la mer et la famille sont deux thèmes majeurs. Et les résultats restent lisibles (« prisonnier » et '
    'non « prisonni »), ce qui compte pour une interprétation littéraire.')

c = R[16]
c.h('Q1. Différence fondamentale')
c.p('Le **stemming** tronque les suffixes par règles heuristiques, **sans contexte ni dictionnaire** : le résultat n\'est '
    'pas forcément un mot (vengeanc). La **lemmatisation** s\'appuie sur une analyse morphologique, un dictionnaire et la '
    '**catégorie grammaticale** pour renvoyer la forme canonique (infinitif, masculin singulier) : le résultat est '
    'toujours un mot de la langue (fut → être, chevaux → cheval).')
c.h('Q2. Pourquoi le POS compte : l\'exemple de « vole »')
c.p('« vole » peut être un verbe (« l\'oiseau vole », lemme **voler**) alors que le nom correspondant est « **vol** » '
    '(« un vol ») : sans POS, on ne sait pas s\'il faut ramener la forme vers le verbe ou le nom. Les homographes sont '
    'nombreux dans le roman : « porte » (NOUN, la porte du cachot) vs « porte » (VERB, porter), « été » (la saison vs '
    'participe d\'être), « sort » (le destin vs sortir), « livre ». Le stemmer, lui, les réduit tous à la même racine. '
    'Le POS ne résout cependant pas tout : « voler » (dérober) et « voler » (dans les airs) restent un seul lemme '
    '(c\'est le rôle de la désambiguïsation lexicale).')
c.h('Q3. Meilleure méthode pour un corpus littéraire du XIXe siècle')
c.p('La **lemmatisation** (13/15 contre 1/15 ici) : le récit au passé simple et à l\'imparfait (fut, aperçut, allait, '
    'disait) multiplie les formes irrégulières que le stemmer ne sait pas traiter. Une réserve : `fr_core_news_sm` est '
    'entraîné sur des textes contemporains (presse, web) ; il commet des erreurs sur les noms propres anciens (« dantè »), '
    'le passé simple (« signala » → ADJ) et les tournures littéraires. Un modèle plus grand (`fr_core_news_lg`, '
    '`fr_dep_news_trf`) ou Stanza améliorerait les résultats.')
c.h('Q4. Coût computationnel SpaCy vs NLTK')
c.p('Sur le chapitre I : NLTK (tokenisation + stemming) **56 ms** contre SpaCy (pipeline complet) **440 ms**, soit '
    '**≈ 8 fois plus lent**, plus le chargement du modèle et sa mémoire. Sur le roman entier, SpaCy prend 6,7 s même en '
    'désactivant parser et NER. Ce compromis est acceptable lorsque la **qualité linguistique** prime et que le corpus '
    'est de taille raisonnable ou traité hors ligne (analyse littéraire, extraction d\'informations), en optimisant avec '
    '`nlp.pipe`, la désactivation des composants inutiles et le multiprocessing. Il ne l\'est pas pour de l\'indexation '
    'massive en temps réel.')

# =========================================================== PARTIE 10
c = R[17]
c.h('a) Dimensions')
c.p('Matrice BoW : **20 documents × 300 mots**. Les 20 blocs de 2 000 mots couvrent 40 000 mots, soit **31 % du '
    'Tome I (chapitres I à XII)**. Remarque : `max_df=0.90` a retiré automatiquement homme, jeune, yeux, faire et '
    'fois (présents dans plus de 90 % des blocs) ; « dantès » (17/20 blocs = 85 %) est conservé.')
c.h('b) Ligne, colonne, cellule')
c.p('Une **ligne** = un document (un bloc de 2 000 mots) représenté par son vecteur de 300 compteurs. Une **colonne** = '
    'un terme du vocabulaire et sa distribution à travers les blocs. Une **cellule** = le nombre d\'occurrences de ce '
    'terme dans ce bloc (ex. « armateur » apparaît 15 fois dans Bloc_01).')
c.h('c) Bloc au vocabulaire le plus riche')
c.p('**Bloc_17** (chapitres IX–X : fin du repas de fiançailles et cabinet des Tuileries) : **751 types distincts** '
    '(592 hors stop words), et 151 des 300 mots du vocabulaire BoW non nuls. Il mélange deux scènes et deux registres '
    '(Marseille / la cour de Louis XVIII).')
c.h('d) Où apparaissent Dantès, Danglars et Faria ?')
c.table(['Personnage', 'Bloc max', 'Top 3 blocs', 'Correspondance dans le roman'],
        [['dantès', 'Bloc_14 (28)', '14, 06, 02', 'ch. VII–VIII : interrogatoire et transfert au château d\'If ; absent des blocs 10, 18, 20 (salons parisiens)'],
         ['danglars', 'Bloc_09 (27)', '09, 06, 07', 'ch. III–VI : le complot et l\'arrestation au repas de fiançailles ; quasi absent ensuite (4 occurrences au bloc 17 seulement)'],
         ['faria', '— (0 partout)', '—', 'Faria n\'entre en scène qu\'au ch. XV, hors des 20 premiers blocs']],
        widths=[0.85, 1.05, 0.9, 3.3])
c.p('Pour Faria, nous avons étendu l\'analyse au **roman entier (66 blocs)** : son pic est au bloc 35 (ch. XVIII, « Le '
    'trésor »). Les pics correspondent exactement aux parties où les personnages agissent. On voit même « abbé » culminer '
    'au ch. XXVI (l\'abbé Busoni, c\'est-à-dire Dantès déguisé) et « franz » au ch. XXXI, alors que « dantès » s\'effondre.')
c.img('10_personnages_fil_du_roman.png', 6.0)
c.h('e) Heatmap')
c.img('10_heatmap_bow.png', 5.6)
c.p('La heatmap fait apparaître **deux grands actes** : blocs 1–9 (Marseille) dominés par danglars, fernand, '
    'caderousse, mercédès, père ; blocs 10–20 dominés par villefort, roi, sire (Paris, la politique, la justice). '
    '« dantès » est présent presque partout mais s\'éteint dans les blocs 10, 18 et 20. Le BoW suffit déjà à retrouver la '
    'segmentation narrative.')
c.h('f) Sparsité')
c.p('Sur la matrice filtrée (20 × 300) : **54,9 % de zéros** (densité 45,1 %), une valeur faible car `max_features` ne '
    'garde que les mots fréquents. Sans filtrage, sur le roman entier : **66 × 11 142 avec 93,7 % de zéros** ; en '
    'mémoire, 5,9 Mo en dense contre 0,56 Mo en format creux (CSR). La sparsité pose problème car elle augmente la mémoire et '
    'le calcul, rend les distances peu fiables (malédiction de la dimension : la plupart des documents ne partagent '
    'presque aucun terme), favorise le sur-apprentissage, et certains algorithmes exigent des données denses (KNN, PCA '
    'classique, réseaux de neurones). Solutions : formats creux, régularisation, réduction de dimension (SVD/LSA), embeddings denses.')

c = R[18]
c.h('Q1. Ligne et colonne')
c.p('Une **ligne** est la représentation vectorielle d\'un document (son « profil lexical »). Une **colonne** est la '
    'distribution d\'un terme sur l\'ensemble des documents (où et combien il apparaît).')
c.h('Q2. Pourquoi la matrice est-elle creuse ?')
c.p('À cause de la **loi de Zipf** : chaque document n\'utilise qu\'une petite fraction du vocabulaire (≈ 750 types par '
    'bloc de 2 000 mots, contre 11 142 au total), et près de la moitié du vocabulaire est faite d\'hapax qui n\'apparaissent '
    'que dans un seul document.')
c.h('Q3. Avantages du Bag of Words')
c.p('Simple, rapide, sans apprentissage ; **interprétable** (chaque dimension est un mot) ; compatible avec tous les '
    'algorithmes de ML ; stockage efficace en format creux ; excellente base de référence en classification (Naive Bayes, '
    'régression logistique). Ici, il a suffi à retrouver la structure du roman et la présence des personnages.')
c.h('Q4. Informations perdues')
c.p('**L\'ordre des mots et la syntaxe** : « Danglars dénonce Dantès » = « Dantès dénonce Danglars ». Le **contexte et la '
    'négation**, la **sémantique** (prison, cachot, geôle sont des dimensions orthogonales, sans aucune proximité), la '
    'polysémie (port/porte), la coréférence (« le jeune homme » = Dantès), et les mots hors vocabulaire. Les comptages '
    'dépendent aussi de la longueur des documents.')

# =========================================================== PARTIE 11
c = R[19]
c.h('a) Matrice TF-IDF et scores du Bloc_01')
c.p('Matrice **20 × 300** (sublinear_tf, smooth_idf, normalisation L2). Meilleurs scores du Bloc_01 : armateur (0,198), '
    'navire (0,194), bâtiment (0,188), morrel (0,187), leclère (0,170), marin (0,168), poste (0,158), danglars, capitaine, '
    'île (0,152), port (0,148), château (0,148). Cela résume parfaitement le chapitre I : l\'arrivée du Pharaon, la mort '
    'du capitaine Leclère, Dantès qui rend compte à l\'armateur Morrel.')
c.h('b) Top 5 TF-IDF par bloc')
tf = [('01', 'I', 'armateur, navire, bâtiment, morrel, leclère'), ('02', 'I–II', 'armateur, leclère, capitaine, morrel, danglars'),
      ('03', 'II', 'argent, vieillard, caderousse, capitaine, vin'), ('04', 'II–III', 'fernand, mère, larmes, attendez, mercédès'),
      ('05', 'III', 'fernand, caderousse, catalan, danglars, malheur'), ('06', 'III–IV', 'fernand, tuer, caderousse, danglars, papier'),
      ('07', 'IV–V', 'fernand, caderousse, danglars, repas, viens'), ('08', 'V', 'commissaire, fernand, danglars, caderousse, bruit'),
      ('09', 'V–VI', 'danglars, caderousse, fernand, armateur, morrel'), ('10', 'VI', 'marquise, napoléon, comte, villefort, saint-méran'),
      ('11', 'VI–VII', 'renée, marquise, tuer, villefort, marquis'), ('12', 'VII', 'justice, villefort, politiques, morrel, substitut'),
      ('13', 'VII', 'villefort, ordres, lettre, juge, noirtier'), ('14', 'VII–VIII', 'gendarmes, barque, palais, commissaire, lumière'),
      ('15', 'VIII', 'geôlier, gendarme, château, gouverneur, gendarmes'), ('16', 'VIII–IX', 'geôlier, gouverneur, marquis, impossible, marquise'),
      ('17', 'IX–X', 'sire, cabinet, blacas, xviii, louis'), ('18', 'X', 'sire, duc, blacas, majesté, baron'),
      ('19', 'X–XI', 'sire, ministre, blacas, louis, xviii'), ('20', 'XI–XII', 'général, sire, ministre, villefort, noirtier')]
c.table(['Bloc', 'Ch.', 'Top 5 TF-IDF', 'Bloc', 'Ch.', 'Top 5 TF-IDF'],
        [[f'B{tf[i][0]}', tf[i][1], tf[i][2], f'B{tf[i+10][0]}', tf[i+10][1], tf[i+10][2]] for i in range(10)],
        widths=[0.4, 0.55, 2.1, 0.4, 0.55, 2.1], size=8)
c.p('**Oui, ces mots sont très pertinents** : ils résument chaque passage. Misère du père Dantès (argent, vieillard, vin), '
    'les Catalans (fernand, mère, larmes, mercédès), le complot (tuer, papier), l\'arrestation (commissaire), les '
    'fiançailles de Villefort (marquise, saint-méran, renée), l\'interrogatoire et la lettre pour Noirtier (lettre, juge), '
    'le château d\'If (geôlier, gouverneur), le cabinet de Louis XVIII (sire, blacas). Un peu de bruit subsiste (attendez, '
    'viens, impossible, bruit).')
c.h('c) Similarité cosinus')
c.img('11_similarite_cosinus.png', 4.9)
c.p('Paires les plus proches : **Bloc_18–19 (0,709)** et 18–20 (0,653) : le cabinet des Tuileries ; 05–06 (0,637) et 04–05 '
    '(0,618) : les Catalans et le complot ; 01–02 (0,596) : l\'arrivée du Pharaon. Les plus éloignées : 06–18 (0,203), '
    '05–20, 15–19. Les blocs **consécutifs** sont les plus similaires (continuité des scènes) et la heatmap montre trois '
    'groupes : Marseille (1–9), Villefort et la prison (10–16), les Tuileries (17–20).')
c.h('d) Mini moteur de recherche')
c.table(['Requête', 'Termes reconnus', 'Top 3 (score)', 'Pertinent ?'],
        [['château if prison dantès', 'château, prison, dantès', 'B15 (0,296), B01, B14', 'oui : B15 = ch. VIII « Le château d\'If »'],
         ['navire capitaine mer', 'les 3', 'B01 (0,253), B13, B02', 'oui : arrivée du Pharaon'],
         ['mariage fiançailles mercédès', 'fiançailles, mercédès', 'B06, B08, B07', 'oui : repas des fiançailles'],
         ['complot lettre dénonciation', 'lettre seulement', 'B13, B11, B16', 'partiel : mots hors vocabulaire']],
        widths=[1.6, 1.3, 1.4, 1.8])
c.p('Le moteur fonctionne. « if » est ignoré (moins de 3 lettres, hors vocabulaire), et « complot » et « dénonciation » '
    'ne sont pas dans les 300 mots retenus : c\'est une limite de `max_features`.')
c.h('e) BoW vs TF-IDF sur le Bloc_01')
c.table(['Rang', 'BoW', 'TF-IDF'],
        [['1', 'armateur', 'armateur'], ['2', 'morrel', 'navire'], ['3', '**dantès**', 'bâtiment'], ['4', 'bâtiment', 'morrel'],
         ['5', 'île', '**leclère**'], ['6', 'danglars', 'marin'], ['7', 'navire', 'poste']], widths=[0.8, 2.6, 2.7])
c.p('**TF-IDF est plus discriminant**. « dantès » passe du 3e rang (BoW) au **23e** (IDF = 1,15, car présent dans 17 '
    'blocs sur 20) ; « leclère », propre à ce passage, monte du 13e au **5e** (IDF = 2,25). Mesuré globalement, le '
    'chevauchement moyen des top-5 entre deux blocs passe de **0,22 (BoW) à 0,09 (TF-IDF)** : les mots-clés TF-IDF '
    'caractérisent beaucoup mieux chaque bloc.')

c = R[20]
c.h('Q1. Pourquoi TF-IDF est meilleur pour un moteur de recherche')
c.p('Parce qu\'il pondère chaque terme par sa **rareté** : les mots présents partout (dantès, homme) pèsent peu, les mots '
    'spécifiques (geôlier, gouverneur) pèsent beaucoup. La requête est donc orientée vers le document où ses termes sont '
    '**caractéristiques**, et pas seulement fréquents. La normalisation L2 et le `sublinear_tf` limitent aussi l\'effet '
    'de la longueur et des répétitions. Notre requête « château if prison dantès » trouve ainsi directement le chapitre du château d\'If.')
c.h('Q2. Que signifie un score TF-IDF élevé ?')
c.p('Le mot est **fréquent dans ce document et rare dans le reste du corpus** : c\'est un mot-signature du document '
    '(« geôlier » pour le Bloc_15, « leclère » pour le Bloc_01).')
c.h('Q3. Similarité cosinus')
c.p('C\'est le cosinus de l\'angle entre deux vecteurs : cos(A, B) = A·B / (‖A‖ × ‖B‖). Elle mesure l\'orientation et non '
    'la longueur, donc elle est insensible à la taille des documents. Pour des vecteurs TF-IDF (positifs), elle varie '
    'de 0 (aucun terme commun) à **1 = similarité totale** (mêmes proportions de termes ; c\'est la diagonale de la heatmap).')
c.h('Q4. Quand TF-IDF donne-t-il de mauvais résultats ?')
c.b('**Synonymes et paraphrases** : « cachot », « prison », « geôle » n\'ont aucun lien ; une requête « évasion » ne trouve pas un texte parlant de « fuite » ;')
c.b('**mots hors vocabulaire** (« dénonciation », « if » dans notre test) et textes très courts (requêtes, tweets) ;')
c.b('**termes importants mais omniprésents** : leur IDF devient faible (« dantès ») ou ils sont supprimés (max_df a retiré « homme ») ;')
c.b('**petit corpus** (20 documents) : IDF instable ; **polysémie** (port/porte) ; **ordre et négation** ignorés ; corpus multilingue.')

# =========================================================== SYNTHÈSE (tableau)
synth = doc.tables[108]
evals = [
    'BoW plus simple, mais TF-IDF ne coûte qu\'un paramètre de plus (même API scikit-learn, même vocabulaire).',
    'Confirmé : sur le Bloc_01, « dantès » passe du rang 3 (BoW) au rang 23 (TF-IDF), « leclère » du rang 13 au rang 5.',
    'Aucun pour les deux : « Danglars dénonce Dantès » = « Dantès dénonce Danglars ».',
    'TF-IDF plus discriminant : chevauchement moyen des top-5 entre blocs 0,22 (BoW) contre 0,09 (TF-IDF).',
    'Identique (20 × 300). Sans filtrage : 66 × 11 142, avec 93,7 % de zéros.',
    'Les deux sont lisibles ; le top TF-IDF résume chaque bloc (geôlier, gouverneur → château d\'If).',
    'Ignoré par les deux ; on peut le compenser en partie avec des n-grammes (ngram_range=(1,2)).',
    'Nuance : l\'IDF donne aux mots rares un poids maximal (risque de sur-valoriser un hapax) ; ici min_df=2 les élimine des deux.',
    'BoW : base de référence et Naive Bayes. TF-IDF : notre moteur de recherche a trouvé le bon chapitre.',
    'Pour le français : CamemBERT / FlauBERT, et sentence-CamemBERT pour la recherche sémantique.',
]
for i, e in enumerate(evals, 1):
    cell = synth.cell(i, 3)
    para = cell.paragraphs[0]
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    _rich(para, e, size=8.5)

# =========================================================== QUESTIONS DE SYNTHÈSE
c = R[21]
c.p('Le pipeline transforme progressivement un fichier texte brut en vecteurs numériques exploitables par le machine learning :')
c.table(['#', 'Étape', 'Objectif', 'Entrée → Sortie'],
        [['1', 'Chargement', 'Obtenir le texte de l\'œuvre seule (sans licence)', 'URL / fichier → chaîne (727 472 car.)'],
         ['2', 'Nettoyage', 'Supprimer le bruit, normaliser (minuscules, ponctuation, élisions, nombres)', 'chaîne → chaîne normalisée'],
         ['3', 'Tokenisation', 'Découper en unités d\'analyse (mots, phrases, n-grammes)', 'chaîne → liste de 130 542 tokens'],
         ['4', 'Fréquences', 'Décrire statistiquement le corpus (top mots, Zipf, hapax)', 'tokens → Counter (mot → effectif)'],
         ['5', 'Stop words', 'Retirer les mots-outils peu informatifs', '130 542 → 50 389 tokens'],
         ['6', 'WordCloud', 'Explorer visuellement les thèmes', 'tokens filtrés → image'],
         ['7', 'Stemming', 'Regrouper les variantes par racine (rapide)', '10 668 mots → 5 581 racines'],
         ['8', 'Lemmatisation', 'Forme canonique selon le POS (précis)', 'tokens → lemmes + POS'],
         ['9', 'BoW', 'Représenter chaque document par ses comptages', 'textes → matrice 20 × 300 d\'entiers'],
         ['10', 'TF-IDF', 'Pondérer par la rareté pour discriminer et comparer', 'textes → matrice 20 × 300 pondérée → similarités']],
        widths=[0.3, 1.0, 2.6, 2.2])

c = R[22]
c.p('Le **nettoyage** transforme le texte (chaîne → chaîne) : il supprime ou normalise des caractères (minuscules, '
    'ponctuation, nombres). La **tokenisation** segmente (chaîne → liste d\'unités) sans modifier le contenu.')
c.b('**Tokeniser sans nettoyer : oui**, et c\'est même parfois nécessaire. SpaCy et `sent_tokenize` sont conçus pour '
    'du texte brut (la segmentation en phrases et la NER ont besoin de la ponctuation et des majuscules), de même que les '
    'tokeniseurs de BERT. Sur le texte brut, on obtient cependant des tokens bruités (« -- », « Pharaon »).')
c.b('**Nettoyer sans tokeniser : oui en partie**, car les regex opèrent sur la chaîne. Mais une partie du nettoyage '
    '(stop words, stemming, filtre de longueur) se fait **au niveau du token** : on doit d\'abord tokeniser.')
c.p('**L\'ordre compte** : nous avons montré que `sent_tokenize` après suppression de la ponctuation trouve 1 phrase au '
    'lieu de 237. En pratique, on segmente en phrases sur le brut, puis on nettoie, puis on tokenise en mots.')

c = R[23]
c.p('Dans ce roman très dialogué du XIXe siècle, « dit » (938 occurrences, 20e mot du texte), « répondit » (137) et '
    '« monsieur » (333) sont des **incises de dialogue et des formules de politesse** réparties uniformément : ils ne '
    'distinguent aucun passage et occupaient le haut des classements et des WordClouds. Les retirer fait apparaître '
    'les personnages et les thèmes.')
c.p('**Pour des tweets : non.** « répondit » ou « monsieur » y sont rares ; les mots vides propres aux tweets sont '
    'plutôt « rt », « lol », « mdr », les URL, les @mentions. On y garderait les émojis et la négation, porteurs de '
    'sentiment. **Pour des avis clients : non plus**, et il faudrait être prudent avec les candidats évidents (« produit », '
    '« commande ») qui peuvent porter l\'aspect évalué (« livraison » est essentiel en analyse par aspect). Une liste de '
    'stop words se construit **pour chaque corpus et chaque tâche**, de préférence à partir des données (fréquence '
    'documentaire, `max_df`) plutôt qu\'à la main.')

c = R[24]
c.p('Le **stemming** est rapide, sans dictionnaire, mais produit des racines illisibles et des confusions ; la '
    '**lemmatisation** est 8 fois plus lente mais linguistiquement correcte grâce au POS (13/15 corrects contre 1/15 '
    'sur notre tableau). Trois exemples du roman où la lemmatisation l\'emporte :')
c.b('**« fut » → être** (stem : « fut ») et « aperçut » → apercevoir (stem inchangé) : le passé simple irrégulier, '
    'omniprésent dans le récit, n\'est pas reconnu par le stemmer.')
c.b('**« disait » → dire** (stem : « dis », confondu avec « dis ») et « allait » → aller (stem : « allait ») : les verbes '
    'de parole et de mouvement sont correctement regroupés.')
c.b('**« mère » / « mer »** : fusionnés en « mer » par le stemmer, distincts pour le lemmatiseur ; de même « prisonniers » '
    '→ « prisonnier » (lisible) au lieu de « prisonni ».')
c.p('La lemmatisation n\'est pas infaillible pour autant (« Dantès » → « dantè », « yeux » non ramené à « œil »).')

c = R[25]
c.p('**Intuition** : un moteur de recherche doit récompenser les mots qui **caractérisent** un document, pas ceux qui '
    'sont partout. Avec la requête « château if prison dantès » sur nos 20 blocs :')
c.b('en **BoW**, le Bloc_15 sort aussi en tête (0,551) mais l\'écart est faible (Bloc_14 : 0,426), et les résultats '
    'suivants (Bloc_06, Bloc_02, Bloc_08 : les Catalans, l\'arrivée, l\'arrestation) ne remontent que parce que Dantès y '
    'est souvent nommé (27 et 26 fois) : « dantès », présent dans 17 blocs sur 20, pèse autant que « prison » ;')
c.b('en **TF-IDF**, « dantès » a un IDF faible (1,15) et ce sont « château » et « prison » qui comptent. Le **Bloc_15 '
    '(chapitre VIII « Le château d\'If »)** l\'emporte nettement (0,296, soit 1,9 fois le 2e résultat contre 1,3 en BoW), '
    'suivi de blocs qui mentionnent réellement le château d\'If (Bloc_01 : « rasa le château d\'If », Bloc_14 : le transfert).')
c.p('C\'est comme chercher un livre en bibliothèque : demander « un livre avec le mot “le” » est inutile, alors que '
    '« geôlier » mène directement au bon rayon. Même mécanisme pour « leclère » (rang 13 → 5 dans le Bloc_01).')

c = R[26]
c.p('BoW et TF-IDF ne voient que des **comptages de mots sans ordre ni sens**. Trois questions hors de leur portée :')
c.b('**« Qui est Simbad le marin ? Qui est l\'abbé Busoni ? »** Il s\'agit de Dantès sous d\'autres identités. Il faut '
    'résoudre la **coréférence** et suivre une entité à travers ses noms : BoW compte « simbad », « abbé » et « dantès » '
    'comme des dimensions indépendantes.')
c.b('**« Danglars est-il sincère quand il félicite Dantès d\'être nommé capitaine ? »** C\'est une question d\'**ironie** '
    'et d\'intention cachée : les mots sont positifs, l\'intention est hostile.')
c.b('**« La lettre de dénonciation est-elle la cause de l\'arrestation, et qui l\'a écrite, qui l\'a postée ? »** Il '
    's\'agit de **temporalité, de causalité et de rôles** (Danglars écrit, Fernand poste) : « Danglars dénonce Dantès » '
    'et « Dantès dénonce Danglars » ont le même vecteur.')
c.p('**Approches modernes** : **Word2Vec / FastText** capturent la similarité sémantique (prison ≈ cachot ≈ geôle) mais '
    'restent statiques (un vecteur par mot, sans contexte). **BERT (CamemBERT, FlauBERT pour le français)** produit des '
    'représentations **contextuelles** qui permettent la NER, la coréférence, la détection de sentiment ou d\'ironie et '
    'les questions-réponses. **GPT et les LLM** vont plus loin : compréhension de longs contextes, raisonnement '
    'narratif, résumé et réponse en langage naturel à ces trois questions.')

doc.save(DST)
print('OK ->', DST)
