"""Monte Carlo season simulation and forecast comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RACE_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
SPRINT_POINTS = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)


@dataclass
class SeasonSimulation:
    driver_standings: pd.DataFrame
    constructor_standings: pd.DataFrame
    race_driver_probabilities: pd.DataFrame
    metadata: Dict[str, object]


def _rank_matrix(points: np.ndarray) -> np.ndarray:
    tiny = np.arange(points.shape[1], dtype=float)[None, :] * -1e-10
    order = np.argsort(-(points + tiny), axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(points.shape[0])[:, None], order] = np.arange(1, points.shape[1] + 1)[None, :]
    return ranks


def simulate_season(
    entries: pd.DataFrame,
    score_col: str = "pred_score",
    n_simulations: int = 3000,
    residual_sigma: float = 0.14,
    random_seed: int = 42,
    starting_driver_points: Optional[Mapping[str, float]] = None,
    starting_constructor_points: Optional[Mapping[str, float]] = None,
    scenario_name: str = "forecast",
) -> SeasonSimulation:
    if entries.empty:
        raise ValueError("No race entries were supplied to simulate_season().")
    required = {
        "round", "race_name", "driverRef", "driver_name", "constructorRef", "constructor_name",
        "is_sprint", score_col, "driver_roll_dnf_10", "team_roll_dnf_10",
    }
    missing = sorted(required - set(entries.columns))
    if missing:
        raise ValueError(f"Missing required simulation columns: {missing}")

    entries = entries.copy().sort_values(["round", score_col, "driverRef"]).reset_index(drop=True)
    starting_driver_points = dict(starting_driver_points or {})
    starting_constructor_points = dict(starting_constructor_points or {})
    rng = np.random.default_rng(int(random_seed))

    driver_meta = entries[["driverRef", "driver_name"]].drop_duplicates("driverRef")
    missing_start_drivers = [d for d in starting_driver_points if d not in set(driver_meta["driverRef"])]
    if missing_start_drivers:
        driver_meta = pd.concat(
            [
                driver_meta,
                pd.DataFrame(
                    {"driverRef": missing_start_drivers, "driver_name": missing_start_drivers}
                ),
            ],
            ignore_index=True,
        )
    drivers = driver_meta["driverRef"].tolist()
    driver_names = dict(zip(driver_meta["driverRef"], driver_meta["driver_name"]))
    driver_index = {ref: idx for idx, ref in enumerate(drivers)}

    constructor_meta = entries[["constructorRef", "constructor_name"]].drop_duplicates("constructorRef")
    missing_start_teams = [c for c in starting_constructor_points if c not in set(constructor_meta["constructorRef"])]
    if missing_start_teams:
        constructor_meta = pd.concat(
            [
                constructor_meta,
                pd.DataFrame(
                    {"constructorRef": missing_start_teams, "constructor_name": missing_start_teams}
                ),
            ],
            ignore_index=True,
        )
    constructors = constructor_meta["constructorRef"].tolist()
    constructor_names = dict(zip(constructor_meta["constructorRef"], constructor_meta["constructor_name"]))
    constructor_index = {ref: idx for idx, ref in enumerate(constructors)}

    driver_points = np.zeros((n_simulations, len(drivers)), dtype=float)
    constructor_points = np.zeros((n_simulations, len(constructors)), dtype=float)
    for ref, points in starting_driver_points.items():
        driver_points[:, driver_index[ref]] = float(points)
    for ref, points in starting_constructor_points.items():
        constructor_points[:, constructor_index[ref]] = float(points)

    driver_season_effect = rng.normal(0.0, 0.025, size=(n_simulations, len(drivers)))
    constructor_season_effect = rng.normal(0.0, 0.032, size=(n_simulations, len(constructors)))
    race_rows = []

    for round_number, race in entries.groupby("round", sort=True):
        race = race.reset_index(drop=True)
        n = len(race)
        didx = np.array([driver_index[ref] for ref in race["driverRef"]], dtype=int)
        cidx = np.array([constructor_index[ref] for ref in race["constructorRef"]], dtype=int)
        base = race[score_col].to_numpy(dtype=float)[None, :]
        noise = rng.normal(0.0, residual_sigma, size=(n_simulations, n))
        scores = base + noise + driver_season_effect[:, didx] + constructor_season_effect[:, cidx]
        dnf_probability = (
            0.58 * race["driver_roll_dnf_10"].to_numpy(dtype=float)
            + 0.42 * race["team_roll_dnf_10"].to_numpy(dtype=float)
        )
        dnf_probability = np.clip(dnf_probability, 0.015, 0.30)
        dnf = rng.random((n_simulations, n)) < dnf_probability[None, :]
        scores = scores + dnf * (0.52 + 0.32 * rng.random((n_simulations, n)))

        order = np.argsort(scores, axis=1)
        positions = np.empty_like(order)
        positions[np.arange(n_simulations)[:, None], order] = np.arange(1, n + 1)[None, :]
        race_points = np.zeros((n_simulations, n), dtype=float)
        top_count = min(len(RACE_POINTS), n)
        race_points[np.arange(n_simulations)[:, None], order[:, :top_count]] = RACE_POINTS[:top_count]

        sprint_points = np.zeros((n_simulations, n), dtype=float)
        if int(race["is_sprint"].iloc[0]) == 1:
            sprint_noise = rng.normal(0.0, residual_sigma * 1.18, size=(n_simulations, n))
            sprint_scores = base + sprint_noise + driver_season_effect[:, didx] + constructor_season_effect[:, cidx]
            sprint_dnf = rng.random((n_simulations, n)) < np.clip(dnf_probability[None, :] * 0.65, 0.01, 0.20)
            sprint_scores = sprint_scores + sprint_dnf * (0.52 + 0.32 * rng.random((n_simulations, n)))
            sprint_order = np.argsort(sprint_scores, axis=1)
            sprint_top = min(len(SPRINT_POINTS), n)
            sprint_points[np.arange(n_simulations)[:, None], sprint_order[:, :sprint_top]] = SPRINT_POINTS[:sprint_top]

        total_weekend_points = race_points + sprint_points
        for local_idx, global_idx in enumerate(didx):
            driver_points[:, global_idx] += total_weekend_points[:, local_idx]
        for local_idx, global_idx in enumerate(cidx):
            constructor_points[:, global_idx] += total_weekend_points[:, local_idx]

        deterministic_order = np.argsort(race[score_col].to_numpy(dtype=float))
        deterministic_positions = np.empty(n, dtype=int)
        deterministic_positions[deterministic_order] = np.arange(1, n + 1)
        for local_idx, row in race.iterrows():
            race_rows.append(
                {
                    "round": int(round_number),
                    "race_name": row["race_name"],
                    "circuitRef": row.get("circuitRef", ""),
                    "driverRef": row["driverRef"],
                    "driver_name": row["driver_name"],
                    "constructorRef": row["constructorRef"],
                    "constructor_name": row["constructor_name"],
                    "pred_score": float(row[score_col]),
                    "deterministic_position": int(deterministic_positions[local_idx]),
                    "expected_position": float(positions[:, local_idx].mean()),
                    "win_probability": float(np.mean(positions[:, local_idx] == 1)),
                    "podium_probability": float(np.mean(positions[:, local_idx] <= 3)),
                    "points_probability": float(np.mean(positions[:, local_idx] <= 10)),
                    "expected_race_points": float(race_points[:, local_idx].mean()),
                    "expected_sprint_points": float(sprint_points[:, local_idx].mean()),
                    "expected_weekend_points": float(total_weekend_points[:, local_idx].mean()),
                    "dnf_probability_input": float(dnf_probability[local_idx]),
                }
            )

    driver_ranks = _rank_matrix(driver_points)
    constructor_ranks = _rank_matrix(constructor_points)
    driver_rows = []
    for ref, idx in driver_index.items():
        driver_rows.append(
            {
                "driverRef": ref,
                "driver_name": driver_names.get(ref, ref),
                "predicted_position": int(np.argsort(-driver_points.mean(axis=0)).tolist().index(idx) + 1),
                "expected_points": float(driver_points[:, idx].mean()),
                "points_p10": float(np.quantile(driver_points[:, idx], 0.10)),
                "points_p90": float(np.quantile(driver_points[:, idx], 0.90)),
                "expected_championship_position": float(driver_ranks[:, idx].mean()),
                "championship_probability": float(np.mean(driver_ranks[:, idx] == 1)),
                "top3_championship_probability": float(np.mean(driver_ranks[:, idx] <= 3)),
            }
        )
    driver_standings = pd.DataFrame(driver_rows).sort_values(
        ["expected_points", "championship_probability"], ascending=[False, False]
    ).reset_index(drop=True)
    driver_standings["predicted_position"] = np.arange(1, len(driver_standings) + 1)

    constructor_rows = []
    for ref, idx in constructor_index.items():
        constructor_rows.append(
            {
                "constructorRef": ref,
                "constructor_name": constructor_names.get(ref, ref),
                "predicted_position": 0,
                "expected_points": float(constructor_points[:, idx].mean()),
                "points_p10": float(np.quantile(constructor_points[:, idx], 0.10)),
                "points_p90": float(np.quantile(constructor_points[:, idx], 0.90)),
                "expected_championship_position": float(constructor_ranks[:, idx].mean()),
                "championship_probability": float(np.mean(constructor_ranks[:, idx] == 1)),
                "top3_championship_probability": float(np.mean(constructor_ranks[:, idx] <= 3)),
            }
        )
    constructor_standings = pd.DataFrame(constructor_rows).sort_values(
        ["expected_points", "championship_probability"], ascending=[False, False]
    ).reset_index(drop=True)
    constructor_standings["predicted_position"] = np.arange(1, len(constructor_standings) + 1)

    race_probabilities = pd.DataFrame(race_rows).sort_values(
        ["round", "win_probability", "expected_position"], ascending=[True, False, True]
    )
    return SeasonSimulation(
        driver_standings=driver_standings,
        constructor_standings=constructor_standings,
        race_driver_probabilities=race_probabilities,
        metadata={
            "scenario": scenario_name,
            "n_simulations": int(n_simulations),
            "residual_sigma": float(residual_sigma),
            "random_seed": int(random_seed),
            "first_round": int(entries["round"].min()),
            "last_round": int(entries["round"].max()),
            "races": int(entries["round"].nunique()),
            "drivers": int(len(drivers)),
            "constructors": int(len(constructors)),
        },
    )


def compare_standings(predicted: pd.DataFrame, actual: pd.DataFrame, entity: str = "driver") -> Tuple[pd.DataFrame, Dict[str, float]]:
    key = "driverRef" if entity == "driver" else "constructorRef"
    name_col = "driver_name" if entity == "driver" else "constructor_name"
    pred = predicted[[key, name_col, "predicted_position", "expected_points"]].copy()
    pred = pred.rename(columns={"predicted_position": "predicted_position", "expected_points": "predicted_points"})
    act = actual[[key, "position", "points"]].copy().rename(columns={"position": "actual_position", "points": "actual_points"})
    comparison = pred.merge(act, on=key, how="outer", indicator=True)
    comparison["rank_error"] = comparison["predicted_position"] - comparison["actual_position"]
    comparison["absolute_rank_error"] = comparison["rank_error"].abs()
    common = comparison[comparison["_merge"] == "both"].copy()
    if len(common) >= 2:
        correlation = spearmanr(common["predicted_position"], common["actual_position"], nan_policy="omit").statistic
    else:
        correlation = np.nan
    predicted_champion = pred.sort_values("predicted_position").iloc[0][key] if len(pred) else None
    actual_champion = act.sort_values("actual_position").iloc[0][key] if len(act) else None
    pred_top3 = set(pred.nsmallest(3, "predicted_position")[key])
    act_top3 = set(act.nsmallest(3, "actual_position")[key])
    metrics = {
        "rank_mae": float(common["absolute_rank_error"].mean()) if len(common) else np.nan,
        "rank_median_ae": float(common["absolute_rank_error"].median()) if len(common) else np.nan,
        "spearman": float(correlation) if np.isfinite(correlation) else np.nan,
        "champion_correct": float(predicted_champion == actual_champion),
        "top3_overlap": float(len(pred_top3 & act_top3) / 3.0),
        "common_entities": int(len(common)),
        "actual_entities": int(len(act)),
        "predicted_entities": int(len(pred)),
    }
    comparison = comparison.drop(columns=["_merge"]).sort_values("actual_position", na_position="last")
    return comparison, metrics


def compare_race_winners(race_probabilities: pd.DataFrame, actual_winners: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for round_number, race in race_probabilities.groupby("round", sort=True):
        ordered = race.sort_values(["win_probability", "expected_position"], ascending=[False, True]).reset_index(drop=True)
        for rank, row in ordered.iterrows():
            rows.append(
                {
                    "round": int(round_number),
                    "candidate_rank": int(rank + 1),
                    "candidate_driverRef": row["driverRef"],
                    "candidate_name": row["driver_name"],
                    "win_probability": row["win_probability"],
                }
            )
    candidates = pd.DataFrame(rows)
    top = candidates[candidates["candidate_rank"] == 1].rename(
        columns={"candidate_driverRef": "predicted_winner_ref", "candidate_name": "predicted_winner"}
    )
    comparison = actual_winners.merge(
        top[["round", "predicted_winner_ref", "predicted_winner", "win_probability"]],
        on="round",
        how="left",
    )
    actual_rank_map = candidates.merge(
        actual_winners[["round", "driverRef"]].rename(columns={"driverRef": "actual_driverRef"}),
        on="round",
        how="inner",
    )
    actual_rank_map = actual_rank_map[actual_rank_map["candidate_driverRef"] == actual_rank_map["actual_driverRef"]][
        ["round", "candidate_rank", "win_probability"]
    ].rename(columns={"candidate_rank": "actual_winner_predicted_rank", "win_probability": "actual_winner_probability"})
    comparison = comparison.merge(actual_rank_map, on="round", how="left")
    comparison["winner_correct"] = (comparison["driverRef"] == comparison["predicted_winner_ref"]).astype(int)
    metrics = {
        "winner_accuracy": float(comparison["winner_correct"].mean()),
        "actual_winner_top3": float((comparison["actual_winner_predicted_rank"] <= 3).mean()),
        "actual_winner_top5": float((comparison["actual_winner_predicted_rank"] <= 5).mean()),
        "mean_reciprocal_rank": float((1.0 / comparison["actual_winner_predicted_rank"]).mean()),
        "mean_actual_winner_probability": float(comparison["actual_winner_probability"].mean()),
        "races": int(len(comparison)),
    }
    return comparison, metrics
