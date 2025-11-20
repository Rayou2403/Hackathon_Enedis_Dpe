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

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.joblib"

CANDIDATE_FEATURES: List[str] = [
    "conso_dpe_kwh",
    "surface_habitable",
    "annee_construction",
    "etiquette_dpe",
    "type_batiment",
    "periode_construction",
    "code_region",
]


def _prepare_training_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prépare X, y, listes num/cat pour l'entraînement.
    """
    df = df.copy()
    df["conso_reelle_kwh"] = df["conso_logement_kwh"]
    df["conso_dpe_kwh"] = df["conso_5_usages_ef"]

    mask = (
        df["conso_reelle_kwh"].notna()
        & df["conso_dpe_kwh"].notna()
        & (df["conso_reelle_kwh"] > 0)
        & (df["conso_dpe_kwh"] > 0)
    )
    df = df[mask].copy()

    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    X = df[features].copy()
    y = df["conso_reelle_kwh"].copy()

    mask_non_nan = X.notna().all(axis=1)
    X = X[mask_non_nan]
    y = y[mask_non_nan]

    cat_cols = [
        c
        for c in features
        if c in ["etiquette_dpe", "type_batiment", "periode_construction", "code_region"]
    ]
    num_cols = [c for c in features if c not in cat_cols]

    return X, y, num_cols, cat_cols


def train_and_save_model(df: pd.DataFrame) -> Dict[str, float | str]:
    """
    Entraîne plusieurs modèles, choisit le meilleur (MAE) et le sauvegarde.
    """
    X, y, num_cols, cat_cols = _prepare_training_data(df)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            random_state=0,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150,
            max_depth=3,
            random_state=0,
        ),
    }

    results = []
    best_pipe = None
    best_name = None
    best_mae = np.inf

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
            best_pipe = pipe
            best_name = name

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipe,
            "features": X.columns.tolist(),
            "best_name": best_name,
        },
        MODEL_PATH,
    )

    # Résultats du meilleur modèle
    best_row = sorted(results, key=lambda r: r[1])[0]
    return {
        "best_model": best_row[0],
        "MAE_kWh": best_row[1],
        "RMSE_kWh": best_row[2],
        "R2": best_row[3],
    }


def load_model() -> Dict:
    """
    Charge le modèle. S'il n'existe pas encore, l'entraîne à la volée.
    """
    if not MODEL_PATH.exists():
        # Import local pour éviter les import circulaires
        from .data_prep import base_clean_df
        from .features import add_conso_features

        df = base_clean_df(max_delta_years=2)
        df = add_conso_features(df)
        train_and_save_model(df)

    return joblib.load(MODEL_PATH)


def predict_conso(model_obj: Dict, user_data: Dict[str, object]) -> float:
    """
    Prédit la consommation réelle à partir d'un dict d'inputs utilisateur.
    """
    pipeline = model_obj["pipeline"]
    features = model_obj["features"]

    row = {col: user_data.get(col, np.nan) for col in features}
    X = pd.DataFrame([row], columns=features)

    return float(pipeline.predict(X)[0])
