"""Graphiques réutilisables : barres, Zipf, WordClouds, heatmaps."""
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from ..utils.plotting import save_fig


def plot_top_frequences(top: list, title: str, name: str | None = None, color: str = "steelblue"):
    mots, freqs = [m for m, _ in top], [n for _, n in top]
    plt.figure(figsize=(12, 6))
    bars = plt.barh(mots[::-1], freqs[::-1], color=color, edgecolor="white")
    for bar, val in zip(bars, freqs[::-1]):
        plt.text(bar.get_width() + max(freqs) * 0.01, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=10)
    plt.xlabel("Fréquence"); plt.title(title); plt.tight_layout()
    if name:
        save_fig(name)


def plot_zipf(fit: dict, title: str, name: str | None = None):
    r, f = fit["rangs"], fit["frequences"]
    plt.figure(figsize=(10, 5))
    plt.loglog(r, f, "b.", markersize=2, alpha=0.5, label="Observé")
    plt.loglog(r, 10 ** fit["intercept"] * r ** fit["pente"], "r--",
               label=f"Régression (pente = {fit['pente']:.2f})")
    plt.loglog(r, f[0] / r, "g:", label="Zipf théorique f = C / r")
    plt.xlabel("Rang (log)"); plt.ylabel("Fréquence (log)"); plt.title(title)
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    if name:
        save_fig(name)


def make_wordcloud(text: str | None = None, frequencies: dict | None = None, **kwargs) -> WordCloud:
    params = dict(width=1200, height=600, background_color="white", colormap="Blues",
                  collocations=False, random_state=42)
    params.update(kwargs)
    wc = WordCloud(**params)
    return wc.generate_from_frequencies(frequencies) if frequencies is not None else wc.generate(text)


def show_wordcloud(wc: WordCloud, title: str, name: str | None = None, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
    plt.title(title, fontsize=16, pad=20); plt.tight_layout()
    if name:
        save_fig(name)


def wordcloud_grid(clouds: list, titles: list, suptitle: str | None = None, name: str | None = None,
                   figsize=(20, 6)):
    fig, axes = plt.subplots(1, len(clouds), figsize=figsize)
    for ax, wc, t in zip(axes, clouds, titles):
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(t, fontsize=13, fontweight="bold")
    if suptitle:
        plt.suptitle(suptitle, fontsize=15, y=1.02)
    plt.tight_layout()
    if name:
        save_fig(name)


def plot_heatmap(df, title: str, xlabel: str = "", ylabel: str = "", name: str | None = None,
                 cmap: str = "YlOrRd", annot: bool = False, fmt: str = "d", figsize=(14, 8)):
    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap=cmap, linewidths=0.3, annot=annot, fmt=fmt, xticklabels=True, yticklabels=True)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout()
    if name:
        save_fig(name)
