"""Chronological model fitting and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data_pipeline import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import joblib


@dataclass
class FittedModel:
    estimator: object
    kind: str
    features: List[str]
    categorical_features: List[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = _prepare_features(frame)
        return np.asarray(self.estimator.predict(x), dtype=float)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == "catboost":
            self.estimator.save_model(str(path.with_suffix(".cbm")))
            metadata = {
                "kind": self.kind,
                "features": self.features,
                "categorical_features": self.categorical_features,
                "model_file": path.with_suffix(".cbm").name,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            joblib.dump(self, path.with_suffix(".joblib"))


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].fillna("unknown").astype(str)
    return x


def recency_weights(years: pd.Series, half_life_years: float = 5.0) -> np.ndarray:
    max_year = float(pd.to_numeric(years, errors="coerce").max())
    age = max_year - pd.to_numeric(years, errors="coerce").to_numpy(dtype=float)
    return np.exp(-math.log(2.0) * age / float(half_life_years))


def fit_model(
    train: pd.DataFrame,
    iterations: int = 320,
    random_seed: int = 42,
    prefer_catboost: bool = True,
) -> FittedModel:
    x = _prepare_features(train)
    y = pd.to_numeric(train["finish_pct"], errors="coerce").to_numpy(dtype=float)
    weights = recency_weights(train["year"], half_life_years=5.0)

    if prefer_catboost and CATBOOST_AVAILABLE:
        cat_indices = [x.columns.get_loc(col) for col in CATEGORICAL_FEATURES]
        estimator = CatBoostRegressor(
            loss_function="RMSE",
            iterations=int(iterations),
            depth=7,
            learning_rate=0.045,
            l2_leaf_reg=7.0,
            random_strength=0.45,
            bagging_temperature=0.4,
            border_count=96,
            random_seed=int(random_seed),
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        estimator.fit(x, y, cat_features=cat_indices, sample_weight=weights)
        return FittedModel(estimator, "catboost", list(FEATURES), list(CATEGORICAL_FEATURES))

    transformer = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    estimator = Pipeline(
        steps=[
            ("transform", transformer),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=max(180, int(iterations)),
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=2.0,
                    random_state=int(random_seed),
                ),
            ),
        ]
    )
    estimator.fit(x, y, model__sample_weight=weights)
    return FittedModel(estimator, "hist_gradient_boosting", list(FEATURES), list(CATEGORICAL_FEATURES))


def baseline_scores(frame: pd.DataFrame) -> np.ndarray:
    values = (
        0.35 * frame["team_roll_finish_10"].to_numpy(dtype=float)
        + 0.24 * frame["driver_roll_finish_10"].to_numpy(dtype=float)
        + 0.12 * frame["prev_team_rank_pct"].to_numpy(dtype=float)
        + 0.08 * frame["prev_driver_rank_pct"].to_numpy(dtype=float)
        + 0.11 * (1.0 - frame["team_roll_points_10"].to_numpy(dtype=float))
        + 0.06 * (1.0 - frame["driver_roll_points_10"].to_numpy(dtype=float))
        + 0.04 * frame["driver_roll_dnf_10"].to_numpy(dtype=float)
    )
    return values


def add_predicted_ranks(frame: pd.DataFrame, score_col: str = "pred_score") -> pd.DataFrame:
    out = frame.copy()
    out["predicted_position"] = out.groupby("raceId")[score_col].rank(method="first", ascending=True).astype(int)
    return out


def evaluate_races(frame: pd.DataFrame, score_col: str = "pred_score") -> Dict[str, float]:
    ranked = add_predicted_ranks(frame, score_col)
    ranked["rank_abs_error"] = (ranked["predicted_position"] - ranked["positionOrder"]).abs()
    winner_hits, winner_top3, podium_overlap, correlations = [], [], [], []
    for _, race in ranked.groupby("raceId"):
        race = race.sort_values("predicted_position")
        actual_winner = race.loc[race["positionOrder"].idxmin(), "driverRef"]
        predicted_order = race["driverRef"].tolist()
        winner_hits.append(float(predicted_order[0] == actual_winner))
        winner_top3.append(float(actual_winner in predicted_order[:3]))
        predicted_podium = set(predicted_order[:3])
        actual_podium = set(race.nsmallest(3, "positionOrder")["driverRef"])
        podium_overlap.append(len(predicted_podium & actual_podium) / 3.0)
        corr = spearmanr(race["predicted_position"], race["positionOrder"], nan_policy="omit").statistic
        if np.isfinite(corr):
            correlations.append(float(corr))
    return {
        "rank_mae": float(ranked["rank_abs_error"].mean()),
        "rank_median_ae": float(ranked["rank_abs_error"].median()),
        "winner_accuracy": float(np.mean(winner_hits)),
        "winner_top3_coverage": float(np.mean(winner_top3)),
        "podium_overlap": float(np.mean(podium_overlap)),
        "mean_spearman": float(np.mean(correlations)),
        "races": int(ranked["raceId"].nunique()),
        "rows": int(len(ranked)),
    }


def chronological_backtest(
    model_table: pd.DataFrame,
    test_years: Tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024),
    iterations: int = 260,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]], float]:
    all_predictions = []
    fold_rows = []
    for test_year in test_years:
        train = model_table[model_table["year"] < test_year].copy()
        test = model_table[model_table["year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        model = fit_model(train, iterations=iterations, random_seed=random_seed + test_year)
        test["pred_score"] = model.predict(test)
        test["baseline_score"] = baseline_scores(test)
        model_metrics = evaluate_races(test, "pred_score")
        baseline_metrics = evaluate_races(test, "baseline_score")
        fold_rows.append(
            {
                "test_year": test_year,
                "train_start_year": int(train["year"].min()),
                "train_end_year": int(train["year"].max()),
                "train_rows": len(train),
                "test_rows": len(test),
                "model_kind": model.kind,
                **{f"model_{k}": v for k, v in model_metrics.items()},
                **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
            }
        )
        all_predictions.append(test)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    overall = {
        "model": evaluate_races(predictions, "pred_score"),
        "baseline": evaluate_races(predictions, "baseline_score"),
    }
    residuals = predictions["finish_pct"].to_numpy(dtype=float) - predictions["pred_score"].to_numpy(dtype=float)
    residual_sigma = float(np.clip(np.nanstd(residuals), 0.07, 0.25))
    return predictions, fold_metrics, overall, residual_sigma


def feature_importance(model: FittedModel) -> pd.DataFrame:
    if model.kind == "catboost":
        values = model.estimator.get_feature_importance()
        return pd.DataFrame({"feature": FEATURES, "importance": values}).sort_values("importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame({"feature": FEATURES, "importance": np.nan})


def tune_ensemble_weight(
    predictions: pd.DataFrame,
    tuning_years: Tuple[int, ...] = (2019, 2020, 2021, 2022),
) -> Tuple[float, pd.DataFrame]:
    tuning = predictions[predictions["year"].isin(tuning_years)].copy()
    if tuning.empty:
        tuning = predictions.copy()
    rows = []
    for weight in np.linspace(0.0, 1.0, 21):
        tuning["ensemble_score"] = weight * tuning["pred_score"] + (1.0 - weight) * tuning["baseline_score"]
        metrics = evaluate_races(tuning, "ensemble_score")
        objective = metrics["rank_mae"] - 0.20 * metrics["winner_accuracy"] - 0.05 * metrics["mean_spearman"]
        rows.append({"model_weight": float(weight), "objective": float(objective), **metrics})
    table = pd.DataFrame(rows).sort_values(["objective", "rank_mae", "winner_accuracy"], ascending=[True, True, False])
    best = float(table.iloc[0]["model_weight"])
    return best, table.sort_values("model_weight").reset_index(drop=True)


def evaluate_ensemble(predictions: pd.DataFrame, model_weight: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = predictions.copy()
    out["ensemble_score"] = float(model_weight) * out["pred_score"] + (1.0 - float(model_weight)) * out["baseline_score"]
    return out, evaluate_races(out, "ensemble_score")
