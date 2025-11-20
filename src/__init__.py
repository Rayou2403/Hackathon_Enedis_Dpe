"""
Package utilitaire pour le projet Hackathon DPE x Enedis.

Ce package expose les fonctions principales utilisées par l'application Streamlit :
- data_prep : chargement du fichier dbz.csv
- features  : calcul des indicateurs conso DPE vs Enedis & impact des classes DPE
"""

from .data_prep import load_dbz, base_clean_df
from .features import (
    add_conso_features,
    compute_dpe_vs_real_stats,
    summarize_subset,
    gain_entre_classes,
    prepare_scatter_sample,
    prepare_boxplot_ratio,
    prepare_usages_long_format,
)

__all__ = [
    # data_prep
    "load_dbz",
    "base_clean_df",
    # features
    "add_conso_features",
    "compute_dpe_vs_real_stats",
    "summarize_subset",
    "gain_entre_classes",
    "prepare_scatter_sample",
    "prepare_boxplot_ratio",
    "prepare_usages_long_format",
]
