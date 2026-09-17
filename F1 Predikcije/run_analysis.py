#!/usr/bin/env python3
"""Run the complete F1 prediction, validation, and reporting pipeline.

The default workflow is reproducible and uses only information available before
an evaluated race or season. Bundled official-result snapshots are used for
post-hoc comparison, for the sequential 2026 forecast, and for the explicitly labelled 2026 nowcast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib

from src.data_pipeline import (
    FEATURES,
    build_future_feature_table,
    load_historical_bundle,
)
from src.modeling import (
    baseline_scores,
    chronological_backtest,
    evaluate_ensemble,
    evaluate_races,
    feature_importance,
    fit_model,
    tune_ensemble_weight,
)
from src.forecasting import (
    compare_race_winners,
    compare_standings,
    simulate_season,
)
from src.reporting import (
    plot_actual_winner_rank,
    plot_backtest,
    plot_dataset_coverage,
    plot_feature_importance,
    plot_nowcast_intervals,
    plot_points_comparison,
    plot_rank_comparison,
    plot_remaining_winner_probabilities,
    save_json,
)

AS_OF_DATE = "2026-09-10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 2025-2026 prediction seminar pipeline")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing data/, src/, and outputs/.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=3000,
        help="Monte Carlo season simulations per scenario (default: 3000).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller models and fewer simulations for a fast smoke test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_inventory(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        years = None
        if "year" in frame.columns:
            values = pd.to_numeric(frame["year"], errors="coerce").dropna()
            if len(values):
                years = (int(values.min()), int(values.max()))
        rows.append(
            {
                "file": path.name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "min_year": years[0] if years else np.nan,
                "max_year": years[1] if years else np.nan,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def read_external(external_dir: Path, stem: str) -> pd.DataFrame:
    path = external_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required external snapshot is missing: {path}")
    return pd.read_csv(path)


def save_simulation(prefix: str, simulation, tables_dir: Path) -> None:
    simulation.driver_standings.to_csv(tables_dir / f"{prefix}_driver_standings.csv", index=False)
    simulation.constructor_standings.to_csv(tables_dir / f"{prefix}_constructor_standings.csv", index=False)
    simulation.race_driver_probabilities.to_csv(tables_dir / f"{prefix}_race_probabilities.csv", index=False)
    save_json(simulation.metadata, tables_dir / f"{prefix}_metadata.json")


def make_ensemble_year_table(predictions: pd.DataFrame, weight: float) -> pd.DataFrame:
    rows = []
    for year, group in predictions.groupby("year"):
        group = group.copy()
        group["ensemble_score"] = weight * group["pred_score"] + (1 - weight) * group["baseline_score"]
        metrics = evaluate_races(group, "ensemble_score")
        rows.append({"test_year": int(year), **{f"ensemble_{k}": v for k, v in metrics.items()}})
    return pd.DataFrame(rows).sort_values("test_year")


def add_forecast_scores(frame: pd.DataFrame, model, model_weight: float) -> pd.DataFrame:
    out = frame.copy()
    out["ml_score"] = model.predict(out)
    out["baseline_score"] = baseline_scores(out)
    out["pred_score"] = model_weight * out["ml_score"] + (1.0 - model_weight) * out["baseline_score"]
    return out


def points_mapping(frame: pd.DataFrame, key: str) -> Dict[str, float]:
    return {str(k): float(v) for k, v in zip(frame[key], frame["points"])}


def run() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    historical_dir = root / "data" / "historical"
    external_dir = root / "data" / "external"
    outputs_dir = root / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    models_dir = outputs_dir / "models"
    for directory in [tables_dir, figures_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("f1-seminar")
    simulations = 600 if args.quick else max(int(args.simulations), 500)
    backtest_iterations = 90 if args.quick else 240
    final_iterations = 180 if args.quick else 440

    log.info("1/8 Inspecting dataset and building leakage-safe features")
    inventory = dataset_inventory(historical_dir)
    inventory.to_csv(tables_dir / "dataset_inventory.csv", index=False)
    plot_dataset_coverage(inventory, figures_dir / "01_dataset_coverage.png")
    bundle = load_historical_bundle(historical_dir, min_year=2005)
    save_json(bundle.data_quality, tables_dir / "data_quality_summary.json")

    log.info("2/8 Running chronological backtest for 2019-2024")
    backtest_predictions, fold_metrics, backtest_overall, residual_sigma = chronological_backtest(
        bundle.model_table,
        iterations=backtest_iterations,
        random_seed=args.seed,
    )
    ensemble_weight, ensemble_grid = tune_ensemble_weight(backtest_predictions)
    backtest_predictions, ensemble_overall = evaluate_ensemble(backtest_predictions, ensemble_weight)
    ensemble_year = make_ensemble_year_table(backtest_predictions, ensemble_weight)
    backtest_overall["ensemble"] = ensemble_overall
    backtest_overall["ensemble_model_weight"] = ensemble_weight
    backtest_overall["simulation_residual_sigma"] = residual_sigma
    fold_metrics.to_csv(tables_dir / "backtest_by_year.csv", index=False)
    ensemble_grid.to_csv(tables_dir / "ensemble_weight_search.csv", index=False)
    ensemble_year.to_csv(tables_dir / "ensemble_backtest_by_year.csv", index=False)
    backtest_predictions[
        [
            "raceId", "year", "round", "name", "driverRef", "driver_name", "constructorRef",
            "positionOrder", "finish_pct", "pred_score", "baseline_score", "ensemble_score",
        ]
    ].to_csv(tables_dir / "backtest_predictions_2019_2024.csv", index=False)
    save_json(backtest_overall, tables_dir / "backtest_overall_metrics.json")
    plot_backtest(fold_metrics, ensemble_year, figures_dir / "02_backtest_rank_mae.png")

    log.info("3/8 Fitting final model on all 2005-2024 rows")
    model = fit_model(bundle.model_table, iterations=final_iterations, random_seed=args.seed)
    model.save(models_dir / "f1_finish_model")
    importance = feature_importance(model)
    importance.to_csv(tables_dir / "feature_importance.csv", index=False)
    plot_feature_importance(importance, figures_dir / "03_feature_importance.png")

    log.info("4/8 Producing strict 2025 forecast and comparing it with final results")
    cal25 = read_external(external_dir, "calendar_2025")
    roster25 = read_external(external_dir, "roster_2025")
    actual25d = read_external(external_dir, "actual_driver_standings_2025")
    actual25c = read_external(external_dir, "actual_constructor_standings_2025")
    actual25w = read_external(external_dir, "actual_race_winners_2025")
    f25 = build_future_feature_table(bundle, cal25, roster25)
    f25 = add_forecast_scores(f25, model, ensemble_weight)
    f25.to_csv(tables_dir / "features_and_scores_2025.csv", index=False)
    sim25 = simulate_season(
        f25,
        n_simulations=simulations,
        residual_sigma=min(residual_sigma, 0.21),
        random_seed=args.seed + 100,
        scenario_name="strict_2025_forecast_from_data_through_2024",
    )
    save_simulation("forecast_2025", sim25, tables_dir)
    comp25d, metrics25d = compare_standings(sim25.driver_standings, actual25d, entity="driver")
    comp25c, metrics25c = compare_standings(sim25.constructor_standings, actual25c, entity="constructor")
    comp25w, metrics25w = compare_race_winners(sim25.race_driver_probabilities, actual25w)
    comp25d.to_csv(tables_dir / "comparison_2025_driver_standings.csv", index=False)
    comp25c.to_csv(tables_dir / "comparison_2025_constructor_standings.csv", index=False)
    comp25w.to_csv(tables_dir / "comparison_2025_race_winners.csv", index=False)
    plot_rank_comparison(comp25d, figures_dir / "04_2025_rank_comparison.png", "Prognoza za 2025. naspram konačnog poretka vozača")
    plot_points_comparison(comp25d, figures_dir / "05_2025_points_comparison.png", "Očekivani naspram konačnih bodova u sezoni 2025.")
    plot_actual_winner_rank(comp25w, figures_dir / "06_2025_actual_winner_rank.png", "Sezona 2025: pozicija koju je model dao stvarnom pobjedniku")

    log.info("5/8 Producing strict 2026 forecast from data through 2024")
    cal26 = read_external(external_dir, "calendar_2026")
    roster26 = read_external(external_dir, "roster_2026")
    actual26d = read_external(external_dir, "actual_driver_standings_2026")
    actual26c = read_external(external_dir, "actual_constructor_standings_2026")
    actual26w = read_external(external_dir, "actual_race_winners_2026")
    f26_strict = build_future_feature_table(bundle, cal26, roster26)
    f26_strict = add_forecast_scores(f26_strict, model, ensemble_weight)
    f26_strict.to_csv(tables_dir / "features_and_scores_2026_strict_from_2024.csv", index=False)
    sim26_strict_full = simulate_season(
        f26_strict,
        n_simulations=simulations,
        residual_sigma=min(residual_sigma, 0.21),
        random_seed=args.seed + 180,
        scenario_name="strict_2026_two_year_forecast_from_data_through_2024",
    )
    save_simulation("forecast_2026_strict_full", sim26_strict_full, tables_dir)
    snapshot_round = int(actual26d["snapshot_round"].max())
    f26_strict_completed = f26_strict[f26_strict["round"] <= snapshot_round].copy()
    sim26_strict_to_date = simulate_season(
        f26_strict_completed,
        n_simulations=simulations,
        residual_sigma=min(residual_sigma, 0.21),
        random_seed=args.seed + 181,
        scenario_name="strict_2026_forecast_from_2024_evaluated_through_round_13",
    )
    save_simulation("forecast_2026_strict_to_round13", sim26_strict_to_date, tables_dir)
    comp26sd, metrics26sd = compare_standings(sim26_strict_to_date.driver_standings, actual26d, entity="driver")
    comp26sc, metrics26sc = compare_standings(sim26_strict_to_date.constructor_standings, actual26c, entity="constructor")
    comp26sw, metrics26sw = compare_race_winners(sim26_strict_to_date.race_driver_probabilities, actual26w)
    comp26sd.to_csv(tables_dir / "comparison_2026_strict_driver_standings_to_round13.csv", index=False)
    comp26sc.to_csv(tables_dir / "comparison_2026_strict_constructor_standings_to_round13.csv", index=False)
    comp26sw.to_csv(tables_dir / "comparison_2026_strict_race_winners_to_round13.csv", index=False)
    plot_rank_comparison(comp26sd, figures_dir / "07_2026_strict_rank_comparison.png", "Stroga prognoza 2026. iz podataka do 2024. naspram poretka poslije 13. utrke")
    plot_actual_winner_rank(comp26sw, figures_dir / "08_2026_strict_actual_winner_rank.png", "Stroga prognoza 2026: pozicija dodijeljena stvarnom pobjedniku")

    log.info("6/8 Producing sequential 2026 preseason forecast and comparison through round 13")
    f26_pre = build_future_feature_table(
        bundle,
        cal26,
        roster26,
        external_driver_standings=actual25d,
        external_constructor_standings=actual25c,
        external_winners=actual25w,
        assimilation_weight=0.70,
    )
    f26_pre = add_forecast_scores(f26_pre, model, ensemble_weight)
    f26_pre.to_csv(tables_dir / "features_and_scores_2026_preseason.csv", index=False)
    sim26_pre_full = simulate_season(
        f26_pre,
        n_simulations=simulations,
        residual_sigma=min(residual_sigma, 0.21),
        random_seed=args.seed + 200,
        scenario_name="2026_preseason_strength_forecast_using_2025_final_results",
    )
    save_simulation("forecast_2026_preseason_full", sim26_pre_full, tables_dir)
    f26_completed = f26_pre[f26_pre["round"] <= snapshot_round].copy()
    sim26_pre_to_date = simulate_season(
        f26_completed,
        n_simulations=simulations,
        residual_sigma=min(residual_sigma, 0.21),
        random_seed=args.seed + 201,
        scenario_name="2026_preseason_forecast_evaluated_through_round_13",
    )
    save_simulation("forecast_2026_preseason_to_round13", sim26_pre_to_date, tables_dir)
    comp26d, metrics26d = compare_standings(sim26_pre_to_date.driver_standings, actual26d, entity="driver")
    comp26c, metrics26c = compare_standings(sim26_pre_to_date.constructor_standings, actual26c, entity="constructor")
    comp26w, metrics26w = compare_race_winners(sim26_pre_to_date.race_driver_probabilities, actual26w)
    comp26d.to_csv(tables_dir / "comparison_2026_preseason_driver_standings_to_round13.csv", index=False)
    comp26c.to_csv(tables_dir / "comparison_2026_preseason_constructor_standings_to_round13.csv", index=False)
    comp26w.to_csv(tables_dir / "comparison_2026_preseason_race_winners_to_round13.csv", index=False)
    plot_rank_comparison(comp26d, figures_dir / "09_2026_preseason_rank_comparison.png", "Predsezonska prognoza 2026. naspram poretka poslije 13. utrke")
    plot_actual_winner_rank(comp26w, figures_dir / "10_2026_preseason_actual_winner_rank.png", "Predsezonska prognoza 2026: pozicija dodijeljena stvarnom pobjedniku")

    log.info("7/8 Producing 2026 nowcast for remaining rounds using results through Italy")
    f26_now = build_future_feature_table(
        bundle,
        cal26,
        roster26,
        external_driver_standings=actual26d,
        external_constructor_standings=actual26c,
        external_winners=actual26w,
        assimilation_weight=0.90,
    )
    f26_now = add_forecast_scores(f26_now, model, ensemble_weight)
    future_rounds = f26_now[f26_now["round"] > int(actual26d["snapshot_round"].max())].copy()
    future_rounds.to_csv(tables_dir / "features_and_scores_2026_remaining.csv", index=False)
    nowcast26 = simulate_season(
        future_rounds,
        n_simulations=simulations,
        residual_sigma=min(residual_sigma, 0.21),
        random_seed=args.seed + 300,
        starting_driver_points=points_mapping(actual26d, "driverRef"),
        starting_constructor_points=points_mapping(actual26c, "constructorRef"),
        scenario_name="2026_nowcast_after_round_13_with_actual_points_as_start",
    )
    name_map = dict(zip(actual26d["driverRef"], actual26d["driver_name"]))
    nowcast26.driver_standings["driver_name"] = nowcast26.driver_standings.apply(
        lambda row: name_map.get(row["driverRef"], row["driver_name"]), axis=1
    )
    team_name_map = dict(zip(actual26c["constructorRef"], actual26c["constructor_name"]))
    nowcast26.constructor_standings["constructor_name"] = nowcast26.constructor_standings.apply(
        lambda row: team_name_map.get(row["constructorRef"], row["constructor_name"]), axis=1
    )
    save_simulation("nowcast_2026_after_round13", nowcast26, tables_dir)
    top3_remaining = (
        nowcast26.race_driver_probabilities.sort_values(["round", "win_probability"], ascending=[True, False])
        .groupby("round")
        .head(3)
    )
    top3_remaining.to_csv(tables_dir / "nowcast_2026_remaining_top3_win_candidates.csv", index=False)
    plot_nowcast_intervals(nowcast26.driver_standings, figures_dir / "11_2026_nowcast_intervals.png", "Projektovani konačni poredak vozača za 2026. poslije 13. utrke")
    plot_remaining_winner_probabilities(nowcast26.race_driver_probabilities, figures_dir / "12_2026_remaining_win_probabilities.png")

    log.info("8/8 Saving reproducibility metadata and summary")
    software = pd.DataFrame(
        [
            ("Python", platform.python_version()),
            ("pandas", pd.__version__),
            ("numpy", np.__version__),
            ("scikit-learn", sklearn.__version__),
            ("scipy", scipy.__version__),
            ("matplotlib", matplotlib.__version__),
            ("model_backend", model.kind),
        ],
        columns=["component", "version"],
    )
    software.to_csv(tables_dir / "software_versions.csv", index=False)

    metrics = {
        "as_of_date": AS_OF_DATE,
        "dataset": bundle.data_quality,
        "backtest": backtest_overall,
        "ensemble_weight": ensemble_weight,
        "2025": {
            "driver_standings": metrics25d,
            "constructor_standings": metrics25c,
            "race_winners": metrics25w,
        },
        "2026_strict_from_2024_evaluated_through_round_13": {
            "driver_standings": metrics26sd,
            "constructor_standings": metrics26sc,
            "race_winners": metrics26sw,
        },
        "2026_preseason_evaluated_through_round_13": {
            "driver_standings": metrics26d,
            "constructor_standings": metrics26c,
            "race_winners": metrics26w,
        },
        "2026_nowcast": nowcast26.metadata,
        "model": {
            "backend": model.kind,
            "feature_count": len(FEATURES),
            "training_rows": int(len(bundle.model_table)),
            "training_start_year": int(bundle.model_table["year"].min()),
            "training_end_year": int(bundle.model_table["year"].max()),
        },
    }
    save_json(metrics, outputs_dir / "metrics.json")

    champion25 = sim25.driver_standings.iloc[0]
    champion26 = nowcast26.driver_standings.iloc[0]
    summary_lines = [
        "F1 prediction seminar - generated analysis summary",
        f"As-of date: {AS_OF_DATE}",
        "",
        f"Historical model rows: {len(bundle.model_table):,}",
        f"Chronological backtest races: {backtest_overall['ensemble']['races']}",
        f"Backtest ensemble rank MAE: {backtest_overall['ensemble']['rank_mae']:.3f}",
        f"Backtest ensemble winner accuracy: {100*backtest_overall['ensemble']['winner_accuracy']:.1f}%",
        f"Backtest actual winner in top 3: {100*backtest_overall['ensemble']['winner_top3_coverage']:.1f}%",
        "",
        f"2025 predicted champion: {champion25['driver_name']}",
        f"2025 actual champion: {actual25d.sort_values('position').iloc[0]['driver_name']}",
        f"2025 driver rank MAE: {metrics25d['rank_mae']:.3f}",
        f"2025 actual race winner in predicted top 3: {100*metrics25w['actual_winner_top3']:.1f}%",
        "",
        f"2026 strict-from-2024 rank MAE through round 13: {metrics26sd['rank_mae']:.3f}",
        f"2026 sequential preseason rank MAE through round 13: {metrics26d['rank_mae']:.3f}",
        "The 2026 regulation change created a structural break that neither historical scenario fully foresaw.",
        f"2026 nowcast champion leader: {champion26['driver_name']}",
        f"2026 nowcast title probability: {100*champion26['championship_probability']:.1f}%",
        f"2026 projected final points: {champion26['expected_points']:.1f}",
        "",
        "See outputs/tables, outputs/figures, and outputs/metrics.json for all details.",
    ]
    (outputs_dir / "analysis_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    log.info("Analysis complete. Main metrics written to %s", outputs_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
