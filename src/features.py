# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# 1. Ajout des features de consommation (réel vs DPE)
# ------------------------------------------------------------------
def add_conso_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les variables de comparaison conso réelle / DPE et filtre les valeurs aberrantes.

    - conso_reelle_kwh : conso Enedis annualisée par logement
    - conso_dpe_kwh    : conso estimée DPE (5 usages)
    - ecart_kwh_logement, ecart_relatif, ratio_reel_sur_dpe
    - conso_reelle_kwh_m2, conso_dpe_kwh_m2 (si surface dispo)
    """
    df = df.copy()

    if "conso_logement_kwh" not in df.columns:
        raise KeyError("Colonne 'conso_logement_kwh' absente de dbz.csv")
    if "conso_5_usages_ef" not in df.columns:
        raise KeyError("Colonne 'conso_5_usages_ef' absente de dbz.csv")

    # 1) Colonnes de base
    df["conso_reelle_kwh"] = df["conso_logement_kwh"]
    df["conso_dpe_kwh"] = df["conso_5_usages_ef"]

    mask = (
        df["conso_reelle_kwh"].notna()
        & df["conso_dpe_kwh"].notna()
        & (df["conso_reelle_kwh"] > 0)
        & (df["conso_dpe_kwh"] > 0)
    )
    df = df[mask].copy()

    # 2) Filtre léger des DPE aberrants (au-dessus du 99.5e percentile)
    upper = df["conso_dpe_kwh"].quantile(0.995)
    df = df[df["conso_dpe_kwh"] <= upper].copy()

    # 3) Écarts et ratios
    df["ecart_kwh_logement"] = df["conso_reelle_kwh"] - df["conso_dpe_kwh"]
    df["ecart_relatif"] = df["ecart_kwh_logement"] / df["conso_dpe_kwh"]
    df["ratio_reel_sur_dpe"] = df["conso_reelle_kwh"] / df["conso_dpe_kwh"]

    # 4) kWh/m² si surface disponible
    if "surface_habitable" in df.columns:
        df["conso_reelle_kwh_m2"] = df["conso_reelle_kwh"] / df["surface_habitable"]
        df["conso_dpe_kwh_m2"] = df["conso_dpe_kwh"] / df["surface_habitable"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df.dropna(
        subset=["ecart_kwh_logement", "ecart_relatif", "ratio_reel_sur_dpe"]
    )


# ------------------------------------------------------------------
# 2. Statistiques globales + par groupe
# ------------------------------------------------------------------
@dataclass
class DpeStatsResult:
    global_stats: pd.DataFrame
    biais_moyen_kwh: float
    std_biais_kwh: float
    ratio_moyen: float
    by_dpe: pd.DataFrame
    by_type: pd.DataFrame
    by_periode: pd.DataFrame


def compute_dpe_vs_real_stats(df: pd.DataFrame) -> DpeStatsResult:
    """
    Regroupe les statistiques de comparaison DPE vs conso réelle.
    """
    num_cols = [
        "conso_reelle_kwh",
        "conso_dpe_kwh",
        "ecart_kwh_logement",
        "ecart_relatif",
        "ratio_reel_sur_dpe",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    global_stats = df[num_cols].describe()

    biais_moyen = df["ecart_kwh_logement"].mean()
    std_biais = df["ecart_kwh_logement"].std()
    ratio_moyen = df["ratio_reel_sur_dpe"].mean()

    # Par classe DPE
    if "etiquette_dpe" in df.columns:
        by_dpe = (
            df.groupby("etiquette_dpe")
            .agg(
                n=("conso_reelle_kwh", "size"),
                conso_reelle_moy=("conso_reelle_kwh", "mean"),
                conso_dpe_moy=("conso_dpe_kwh", "mean"),
                ecart_moy_kwh=("ecart_kwh_logement", "mean"),
                ecart_med_kwh=("ecart_kwh_logement", "median"),
                ecart_std_kwh=("ecart_kwh_logement", "std"),
                ratio_moy=("ratio_reel_sur_dpe", "mean"),
                ratio_med=("ratio_reel_sur_dpe", "median"),
            )
            .reset_index()
            .sort_values("etiquette_dpe")
        )
    else:
        by_dpe = pd.DataFrame()

    # Par type de bâtiment
    if "type_batiment" in df.columns:
        by_type = (
            df.groupby("type_batiment")
            .agg(
                n=("conso_reelle_kwh", "size"),
                conso_reelle_moy=("conso_reelle_kwh", "mean"),
                conso_dpe_moy=("conso_dpe_kwh", "mean"),
                ecart_moy_kwh=("ecart_kwh_logement", "mean"),
                ecart_std_kwh=("ecart_kwh_logement", "std"),
                ratio_moy=("ratio_reel_sur_dpe", "mean"),
            )
            .reset_index()
        )
    else:
        by_type = pd.DataFrame()

    # Par période de construction
    if "periode_construction" in df.columns:
        by_periode = (
            df.groupby("periode_construction")
            .agg(
                n=("conso_reelle_kwh", "size"),
                conso_reelle_moy=("conso_reelle_kwh", "mean"),
                conso_dpe_moy=("conso_dpe_kwh", "mean"),
                ecart_moy_kwh=("ecart_kwh_logement", "mean"),
                ecart_std_kwh=("ecart_kwh_logement", "std"),
                ratio_moy=("ratio_reel_sur_dpe", "mean"),
            )
            .reset_index()
        )
    else:
        by_periode = pd.DataFrame()

    return DpeStatsResult(
        global_stats=global_stats,
        biais_moyen_kwh=biais_moyen,
        std_biais_kwh=std_biais,
        ratio_moyen=ratio_moyen,
        by_dpe=by_dpe,
        by_type=by_type,
        by_periode=by_periode,
    )


def summarize_subset(df: pd.DataFrame) -> str:
    """
    Résumé textuel des écarts DPE / réel sur un sous-échantillon (pour l'utilisateur).
    """
    n = len(df)
    if n == 0:
        return "Aucun logement dans le sous-échantillon sélectionné."

    conso_reelle_moy = df["conso_reelle_kwh"].mean()
    conso_dpe_moy = df["conso_dpe_kwh"].mean()
    ecart_moy = df["ecart_kwh_logement"].mean()
    ratio_moy = df["ratio_reel_sur_dpe"].mean()

    q25, q50, q75 = df["ecart_kwh_logement"].quantile([0.25, 0.5, 0.75])

    ratio_pct = (ratio_moy - 1) * 100

    texte = (
        f"Sur les **{n} logements** correspondant à vos filtres :\n\n"
        f"- la consommation réelle moyenne est de **{conso_reelle_moy:,.0f} kWh/an/logement**, "
        f"contre **{conso_dpe_moy:,.0f} kWh/an/logement** estimés dans les DPE ;\n"
        f"- en moyenne, la consommation réelle est **{abs(ecart_moy):,.0f} kWh/an** "
        f"{'en dessous' if ecart_moy < 0 else 'au-dessus'} des estimations DPE ;\n"
        f"- cela correspond à un ratio moyen réel / DPE de **{ratio_moy:.2f}** "
        f"(soit {ratio_pct:+.0f} % par rapport à la valeur DPE).\n\n"
        f"La variabilité est importante : 50 % des logements ont un écart "
        f"entre **{q25:,.0f} kWh/an** et **{q75:,.0f} kWh/an**, avec un écart médian "
        f"de **{q50:,.0f} kWh/an**."
    )

    return texte.replace(",", " ")


# ------------------------------------------------------------------
# 3. Impact d'un changement de classe DPE
# ------------------------------------------------------------------
def gain_entre_classes(
    df: pd.DataFrame, classe_depart: str, classe_arrivee: str
) -> Dict[str, float]:
    """
    Écart moyen de consommation réelle entre deux classes DPE
    (classe_depart -> classe_arrivee) sur un sous-échantillon.

    On travaille sur des moyennes observées (approche transversale :
    logements de classe X vs logements de classe Y).
    """
    df = df.copy()
    subset_depart = df[df["etiquette_dpe"] == classe_depart]
    subset_arrivee = df[df["etiquette_dpe"] == classe_arrivee]

    if subset_depart.empty or subset_arrivee.empty:
        return {}

    conso_depart = subset_depart["conso_reelle_kwh"].mean()
    conso_arrivee = subset_arrivee["conso_reelle_kwh"].mean()
    gain_kwh = conso_depart - conso_arrivee  # > 0 si on consomme moins après

    return {
        "n_depart": len(subset_depart),
        "n_arrivee": len(subset_arrivee),
        "conso_depart": conso_depart,
        "conso_arrivee": conso_arrivee,
        "gain_kwh": gain_kwh,
    }


# ------------------------------------------------------------------
# 4. Jeux pour dataviz
# ------------------------------------------------------------------
def prepare_scatter_sample(df: pd.DataFrame, n_max: int = 5000) -> pd.DataFrame:
    """
    Échantillon pour un nuage de points conso DPE vs conso réelle.
    """
    cols_needed = ["conso_dpe_kwh", "conso_reelle_kwh"]
    df = df.dropna(subset=[c for c in cols_needed if c in df.columns])
    if len(df) > n_max:
        return df.sample(n_max, random_state=0)
    return df


def prepare_boxplot_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jeu de données pour un boxplot du ratio réel / DPE par classe DPE.
    """
    if "etiquette_dpe" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["etiquette_dpe", "ratio_reel_sur_dpe"])


def prepare_usages_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formate les consommations par usage en 'long' pour éventuelles viz
    (chauffage, ECS, éclairage, etc.) si les colonnes existent.
    """
    usage_cols = [
        "conso_chauffage_ef",
        "conso_ecs_ef",
        "conso_refroidissement_ef",
        "conso_eclairage_ef",
        "conso_auxiliaires_ef",
    ]
    usage_cols = [c for c in usage_cols if c in df.columns]
    if not usage_cols:
        return pd.DataFrame()

    id_vars = []
    if "etiquette_dpe" in df.columns:
        id_vars.append("etiquette_dpe")

    long_df = df.melt(
        id_vars=id_vars if id_vars else None,
        value_vars=usage_cols,
        var_name="usage",
        value_name="conso_kwh",
    )
    return long_df.dropna(subset=["conso_kwh"])
