"""Analyses statistiques et visualisations."""
from .frequency import frequency_table, zipf_fit, rangs, hapax
from .characters import phrases_personnage, tokens_personnage, mots_distinctifs, distribution_personnages
from .visualization import (plot_top_frequences, plot_zipf, make_wordcloud, show_wordcloud,
                            wordcloud_grid, plot_heatmap)
