"""Découpage du texte en documents (blocs de mots) et repérage des chapitres."""


def decouper_blocs(texte: str, taille: int = 2000, max_blocs: int | None = None):
    """Renvoie (blocs, noms). max_blocs=None : tout le texte."""
    mots = texte.split()
    fin = len(mots) if max_blocs is None else min(len(mots), taille * max_blocs)
    blocs = [" ".join(mots[i:i + taille]) for i in range(0, fin, taille)]
    return blocs, [f"Bloc_{i + 1:02d}" for i in range(len(blocs))]


def index_chapitres(chapitres: dict, nettoyeur) -> list:
    """[(position du premier mot, numéro de chapitre)] dans le texte nettoyé."""
    cum, idx = 0, []
    for k, v in chapitres.items():
        idx.append((cum, k))
        cum += len(nettoyeur(v["texte"]).split())
    return idx


def chapitre_du_mot(i: int, idx: list) -> str:
    return [k for d, k in idx if d <= i][-1]


def chapitres_par_bloc(n_blocs: int, taille: int, idx: list) -> list:
    return [f"{chapitre_du_mot(i * taille, idx)}–{chapitre_du_mot((i + 1) * taille - 1, idx)}"
            for i in range(n_blocs)]
