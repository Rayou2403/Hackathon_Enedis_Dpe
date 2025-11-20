# src/models.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Dossier racine du projet et chemin du modèle
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.joblib"

# Variables explicatives candidates
# ⚠️ On enlève 'periode_construction' pour éviter l'incohérence avec l'année
CANDIDATE_FEATURES: List[str] = [
    "conso_dpe_kwh",       # consommation estimée par le DPE (kWh/an)
    "surface_habitable",   # m²
    "annee_construction",  # année (numérique)
    "etiquette_dpe",       # classe A–G
    "type_batiment",       # maison / appartement / ...
    "code_region",         # code INSEE de région
]


def _prepare_training_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prépare X, y et les listes de variables numériques / catégorielles.

    - Cible : conso_reelle_kwh (dérivée de conso_logement_kwh)
    - Features : sous-ensemble de CANDIDATE_FEATURES présentes dans le DataFrame
    """
    df = df.copy()

    if "conso_logement_kwh" not in df.columns:
        raise KeyError("Colonne 'conso_logement_kwh' absente du DataFrame.")
    if "conso_5_usages_ef" not in df.columns:
        raise KeyError("Colonne 'conso_5_usages_ef' absente du DataFrame.")

    # Cible et feature principale DPE
    df["conso_reelle_kwh"] = df["conso_logement_kwh"]
    df["conso_dpe_kwh"] = df["conso_5_usages_ef"]

    # On garde les lignes cohérentes
    mask = (
        df["conso_reelle_kwh"].notna()
        & df["conso_dpe_kwh"].notna()
        & (df["conso_reelle_kwh"] > 0)
        & (df["conso_dpe_kwh"] > 0)
    )
    df = df[mask].copy()

    # Sélection des features effectivement disponibles
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if not features:
        raise ValueError("Aucune feature candidate présente dans les données.")

    X = df[features].copy()
    y = df["conso_reelle_kwh"].copy()

    # On enlève les lignes avec NaN dans les features
    mask_non_nan = X.notna().all(axis=1)
    X = X[mask_non_nan]
    y = y[mask_non_nan]

    # Variables catégorielles / numériques
    cat_cols = [c for c in features if c in ["etiquette_dpe", "type_batiment", "code_region"]]
    num_cols = [c for c in features if c not in cat_cols]

    return X, y, num_cols, cat_cols


def train_and_save_model(df: pd.DataFrame) -> Dict[str, float | str]:
    """
    Entraîne plusieurs modèles, choisit le meilleur (MAE) et le sauvegarde.

    Sauvegarde dans le fichier joblib :
    - le pipeline complet (prétraitement + modèle),
    - la liste des features,
    - le nom du meilleur modèle,
    - quelques métriques, y compris un baseline DPE.
    """
    X, y, num_cols, cat_cols = _prepare_training_data(df)

    # Préprocesseur : numériques en passthrough, catégorielles en OneHotEncoder
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Baseline : prédire la conso DPE (tel quel)
    if "conso_dpe_kwh" in X_test.columns:
        baseline_pred = X_test["conso_dpe_kwh"].values
        mae_baseline = mean_absolute_error(y_test, baseline_pred)
        rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))
    else:
        mae_baseline = np.nan
        rmse_baseline = np.nan

    # Modèles candidats
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=0,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=0,
        ),
    }

    results = []
    best_pipe = None
    best_name = None
    best_mae = np.inf
    best_rmse = np.inf
    best_r2 = -np.inf

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append((name, mae, rmse, r2))

        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            best_r2 = r2
            best_pipe = pipe
            best_name = name

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # On sauvegarde aussi les métriques pour les afficher dans Streamlit
    metrics = {
        "baseline_MAE_kWh": float(mae_baseline) if not np.isnan(mae_baseline) else None,
        "baseline_RMSE_kWh": float(rmse_baseline) if not np.isnan(rmse_baseline) else None,
        "model_MAE_kWh": float(best_mae),
        "model_RMSE_kWh": float(best_rmse),
        "model_R2": float(best_r2),
        "best_model_name": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(
        {
            "pipeline": best_pipe,
            "features": X.columns.tolist(),
            "best_name": best_name,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    return metrics


def load_model() -> Dict:
    """Charge le modèle. S'il n'existe pas encore, l'entraîne à la volée."""
    if not MODEL_PATH.exists():
        # Import local pour éviter les import circulaires
        from .data_prep import base_clean_df
        from .features import add_conso_features

        df = base_clean_df()
        df = add_conso_features(df)
        train_and_save_model(df)

    return joblib.load(MODEL_PATH)


def predict_conso(model_obj: Dict, user_data: Dict[str, object]) -> float:
    """
    Prédit la consommation réelle à partir d'un dict d'inputs utilisateur.

    - model_obj : dictionnaire chargé depuis le .joblib
    - user_data : ex {
          "conso_dpe_kwh": 12000,
          "surface_habitable": 80,
          "annee_construction": 1980,
          "etiquette_dpe": "E",
          "type_batiment": "Appartement",
          "code_region": "11",
      }
    """
    pipeline = model_obj["pipeline"]
    features = model_obj["features"]

    # On construit un DataFrame d'une ligne avec les features attendues
    row = {col: user_data.get(col, np.nan) for col in features}
    X = pd.DataFrame([row], columns=features)

    y_pred = float(pipeline.predict(X)[0])
    # On s'assure d'avoir une conso positive
    return max(y_pred, 0.0)
