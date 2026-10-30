"""Sauvegarde homogène des figures."""
import matplotlib.pyplot as plt

from .config import FIG_DIR


def save_fig(name: str, fig_dir=FIG_DIR, dpi: int = 110, show: bool = True):
    """Enregistre la figure courante dans figures/<name>.png puis l'affiche."""
    plt.savefig(fig_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
