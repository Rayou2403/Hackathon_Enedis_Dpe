# src/data_prep.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DBZ_PATH = BASE_DIR / "processed" / "dbz.csv"


def load_dbz(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Charge le fichier dbz.csv contenant l'intersection Enedis x DPE.

    Paramètres
    ----------
    path : chemin vers le CSV (facultatif).
           Si None, on utilise processed/dbz.csv à la racine du projet.

    Retour
    ------
    DataFrame pandas.
    """
    csv_path = Path(path) if path is not None else DBZ_PATH
    df = pd.read_csv(csv_path)

    # Normalisation minimale des types sur quelques colonnes numériques
    for col in ["annee_enedis", "annee_dpe_matched", "annee_construction"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "surface_habitable" in df.columns:
        df["surface_habitable"] = pd.to_numeric(
            df["surface_habitable"], errors="coerce"
        )

    return df


def base_clean_df() -> pd.DataFrame:
    """
    Pipeline minimal pour l'application :
    - charge dbz.csv
    - laisse toutes les lignes, le matching temporel ayant été fait en amont.

    Toute la logique de filtrage (conso > 0, outliers, etc.) est gérée
    dans src.features.add_conso_features().
    """
    df = load_dbz()
    return df
