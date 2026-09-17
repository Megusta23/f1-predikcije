"""Data loading and leakage-safe feature engineering for the F1 seminar project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


LINEAGE_MAP: Dict[str, str] = {
    "toro_rosso": "faenza",
    "alphatauri": "faenza",
    "rb": "faenza",
    "racing_bulls": "faenza",
    "aston_martin": "silverstone",
    "racing_point": "silverstone",
    "force_india": "silverstone",
    "spyker": "silverstone",
    "spyker_mf1": "silverstone",
    "midland": "silverstone",
    "jordan": "silverstone",
    "alpine": "enstone",
    "renault": "enstone",
    "lotus_f1": "enstone",
    "benetton": "enstone",
    "toleman": "enstone",
    "sauber": "sauber_audi",
    "bmw_sauber": "sauber_audi",
    "alfa": "sauber_audi",
    "audi": "sauber_audi",
    "mercedes": "mercedes",
    "brawn": "mercedes",
    "honda": "mercedes",
    "bar": "mercedes",
    "red_bull": "red_bull",
    "jaguar": "red_bull",
    "stewart": "red_bull",
    "cadillac": "cadillac",
}

NEW_DRIVER_DOBS: Dict[str, str] = {
    "antonelli": "2006-08-25",
    "hadjar": "2004-09-28",
    "bortoleto": "2004-10-14",
    "arvid_lindblad": "2007-08-08",
}

NUMERIC_FEATURES = [
    "year",
    "round_fraction",
    "field_size",
    "driver_age",
    "driver_starts_log",
    "driver_roll_finish_3",
    "driver_roll_finish_5",
    "driver_roll_finish_10",
    "driver_roll_finish_20",
    "driver_roll_grid_5",
    "driver_roll_grid_10",
    "driver_roll_points_5",
    "driver_roll_points_10",
    "driver_roll_win_20",
    "driver_roll_podium_10",
    "driver_roll_dnf_10",
    "driver_circuit_finish",
    "driver_circuit_starts_log",
    "team_roll_finish_3",
    "team_roll_finish_5",
    "team_roll_finish_10",
    "team_roll_points_5",
    "team_roll_points_10",
    "team_roll_win_10",
    "team_roll_podium_10",
    "team_roll_dnf_10",
    "team_circuit_finish",
    "team_circuit_starts_log",
    "prev_driver_rank_pct",
    "prev_team_rank_pct",
]

CATEGORICAL_FEATURES = ["driverRef", "constructorLineage", "circuitRef"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class HistoricalBundle:
    model_table: pd.DataFrame
    raw_results: pd.DataFrame
    driver_state: pd.DataFrame
    team_state: pd.DataFrame
    driver_circuit_state: pd.DataFrame
    team_circuit_state: pd.DataFrame
    rookie_defaults: Mapping[str, float]
    global_defaults: Mapping[str, float]
    driver_dobs: Mapping[str, pd.Timestamp]
    data_quality: Mapping[str, object]


def constructor_lineage(constructor_ref: str) -> str:
    ref = str(constructor_ref)
    return LINEAGE_MAP.get(ref, ref)


def _rolling_prior(
    frame: pd.DataFrame,
    group_cols: Iterable[str],
    source_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    return frame.groupby(list(group_cols), sort=False)[source_col].transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).mean()
    )


def _expanding_prior(
    frame: pd.DataFrame,
    group_cols: Iterable[str],
    source_col: str,
) -> pd.Series:
    return frame.groupby(list(group_cols), sort=False)[source_col].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )


def _rank_pct(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="min", ascending=False)
    denom = max(len(values) - 1, 1)
    return (ranks - 1) / denom


def load_historical_bundle(data_dir: Path, min_year: int = 2005) -> HistoricalBundle:
    data_dir = Path(data_dir)
    races = pd.read_csv(data_dir / "races.csv")
    results = pd.read_csv(data_dir / "results.csv")
    drivers = pd.read_csv(data_dir / "drivers.csv")
    constructors = pd.read_csv(data_dir / "constructors.csv")
    circuits = pd.read_csv(data_dir / "circuits.csv")
    status = pd.read_csv(data_dir / "status.csv")

    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    drivers["dob"] = pd.to_datetime(drivers["dob"], errors="coerce")

    race_cols = ["raceId", "year", "round", "circuitId", "name", "date"]
    driver_cols = ["driverId", "driverRef", "forename", "surname", "dob"]
    constructor_cols = ["constructorId", "constructorRef", "name"]

    merged = (
        results.merge(races[race_cols], on="raceId", how="left", validate="many_to_one")
        .merge(drivers[driver_cols], on="driverId", how="left", validate="many_to_one")
        .merge(
            constructors[constructor_cols].rename(columns={"name": "constructor_name"}),
            on="constructorId",
            how="left",
            validate="many_to_one",
        )
        .merge(circuits[["circuitId", "circuitRef", "name"]].rename(columns={"name": "circuit_name"}), on="circuitId", how="left", validate="many_to_one")
        .merge(status, on="statusId", how="left", validate="many_to_one")
    )

    quality = {
        "source_start_year": int(merged["year"].min()),
        "source_end_year": int(merged["year"].max()),
        "source_races": int(merged["raceId"].nunique()),
        "source_result_rows": int(len(merged)),
        "source_drivers": int(merged["driverId"].nunique()),
        "source_constructors": int(merged["constructorId"].nunique()),
        "duplicate_driver_race_rows": int(merged.duplicated(["raceId", "driverId"]).sum()),
        "missing_finish_order": int(merged["positionOrder"].isna().sum()),
        "missing_driver_ref": int(merged["driverRef"].isna().sum()),
        "missing_constructor_ref": int(merged["constructorRef"].isna().sum()),
        "missing_circuit_ref": int(merged["circuitRef"].isna().sum()),
    }

    merged = merged[
        (merged["year"] >= int(min_year))
        & merged["positionOrder"].notna()
        & merged["driverRef"].notna()
        & merged["constructorRef"].notna()
        & merged["circuitRef"].notna()
    ].copy()
    merged.sort_values(["date", "raceId", "positionOrder"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    merged["constructorLineage"] = merged["constructorRef"].map(constructor_lineage)
    merged["driver_name"] = merged["forename"].fillna("") + " " + merged["surname"].fillna("")
    merged["field_size"] = merged.groupby("raceId")["driverId"].transform("count").astype(float)
    denom = (merged["field_size"] - 1).clip(lower=1)
    merged["finish_pct"] = (merged["positionOrder"].astype(float) - 1) / denom
    grid_clean = pd.to_numeric(merged["grid"], errors="coerce").fillna(0).astype(float)
    grid_clean = grid_clean.where(grid_clean > 0, merged["field_size"])
    merged["grid_pct"] = (grid_clean - 1) / denom
    max_points = merged.groupby("raceId")["points"].transform("max").replace(0, np.nan)
    merged["points_norm"] = (pd.to_numeric(merged["points"], errors="coerce").fillna(0) / max_points).fillna(0).clip(0, 2)
    merged["win_flag"] = (merged["positionOrder"] == 1).astype(float)
    merged["podium_flag"] = (merged["positionOrder"] <= 3).astype(float)
    merged["top10_flag"] = (merged["positionOrder"] <= 10).astype(float)
    status_text = merged["status"].fillna("")
    merged["dnf_flag"] = (~status_text.str.match(r"^(Finished|\+\d+ Lap)", na=False)).astype(float)
    merged["driver_age"] = (merged["date"] - merged["dob"]).dt.days / 365.25
    merged["driver_age"] = merged["driver_age"].where(merged["driver_age"].between(16, 65))
    season_races = merged[["year", "raceId", "round"]].drop_duplicates().groupby("year")["round"].max()
    merged["round_fraction"] = merged["round"] / merged["year"].map(season_races).clip(lower=1)

    merged["driver_starts_prior"] = merged.groupby("driverRef", sort=False).cumcount()
    merged["driver_starts_log"] = np.log1p(merged["driver_starts_prior"])
    for window in [3, 5, 10, 20]:
        merged[f"driver_roll_finish_{window}"] = _rolling_prior(merged, ["driverRef"], "finish_pct", window)
    for window in [5, 10]:
        merged[f"driver_roll_grid_{window}"] = _rolling_prior(merged, ["driverRef"], "grid_pct", window)
        merged[f"driver_roll_points_{window}"] = _rolling_prior(merged, ["driverRef"], "points_norm", window)
    merged["driver_roll_win_20"] = _rolling_prior(merged, ["driverRef"], "win_flag", 20)
    merged["driver_roll_podium_10"] = _rolling_prior(merged, ["driverRef"], "podium_flag", 10)
    merged["driver_roll_dnf_10"] = _rolling_prior(merged, ["driverRef"], "dnf_flag", 10)
    merged["driver_circuit_starts"] = merged.groupby(["driverRef", "circuitRef"], sort=False).cumcount()
    merged["driver_circuit_starts_log"] = np.log1p(merged["driver_circuit_starts"])
    merged["driver_circuit_finish"] = _expanding_prior(merged, ["driverRef", "circuitRef"], "finish_pct")

    team_race = (
        merged.groupby(
            ["raceId", "date", "year", "round", "circuitRef", "constructorLineage"],
            as_index=False,
        )
        .agg(
            team_finish=("finish_pct", "mean"),
            team_points=("points_norm", "mean"),
            team_win=("win_flag", "max"),
            team_podium=("podium_flag", "mean"),
            team_dnf=("dnf_flag", "mean"),
        )
        .sort_values(["date", "raceId", "constructorLineage"])
        .reset_index(drop=True)
    )
    for window in [3, 5, 10]:
        team_race[f"team_roll_finish_{window}"] = _rolling_prior(team_race, ["constructorLineage"], "team_finish", window)
    for window in [5, 10]:
        team_race[f"team_roll_points_{window}"] = _rolling_prior(team_race, ["constructorLineage"], "team_points", window)
    team_race["team_roll_win_10"] = _rolling_prior(team_race, ["constructorLineage"], "team_win", 10)
    team_race["team_roll_podium_10"] = _rolling_prior(team_race, ["constructorLineage"], "team_podium", 10)
    team_race["team_roll_dnf_10"] = _rolling_prior(team_race, ["constructorLineage"], "team_dnf", 10)
    team_race["team_circuit_starts"] = team_race.groupby(["constructorLineage", "circuitRef"], sort=False).cumcount()
    team_race["team_circuit_starts_log"] = np.log1p(team_race["team_circuit_starts"])
    team_race["team_circuit_finish"] = _expanding_prior(team_race, ["constructorLineage", "circuitRef"], "team_finish")

    team_feature_cols = [
        "raceId",
        "constructorLineage",
        "team_roll_finish_3",
        "team_roll_finish_5",
        "team_roll_finish_10",
        "team_roll_points_5",
        "team_roll_points_10",
        "team_roll_win_10",
        "team_roll_podium_10",
        "team_roll_dnf_10",
        "team_circuit_finish",
        "team_circuit_starts_log",
    ]
    merged = merged.merge(team_race[team_feature_cols], on=["raceId", "constructorLineage"], how="left", validate="many_to_one")

    season_driver = merged.groupby(["year", "driverRef"], as_index=False)["points"].sum()
    season_driver["rank_pct"] = season_driver.groupby("year")["points"].transform(_rank_pct)
    season_driver["year"] += 1
    merged = merged.merge(
        season_driver[["year", "driverRef", "rank_pct"]].rename(columns={"rank_pct": "prev_driver_rank_pct"}),
        on=["year", "driverRef"],
        how="left",
        validate="many_to_one",
    )
    season_team = merged.groupby(["year", "constructorLineage"], as_index=False)["points"].sum()
    season_team["rank_pct"] = season_team.groupby("year")["points"].transform(_rank_pct)
    season_team["year"] += 1
    merged = merged.merge(
        season_team[["year", "constructorLineage", "rank_pct"]].rename(columns={"rank_pct": "prev_team_rank_pct"}),
        on=["year", "constructorLineage"],
        how="left",
        validate="many_to_one",
    )

    global_defaults = {}
    for col in NUMERIC_FEATURES:
        value = pd.to_numeric(merged[col], errors="coerce").median()
        global_defaults[col] = float(value) if pd.notna(value) else 0.5

    rookie_rows = merged[merged["driver_starts_prior"] < 5]
    rookie_defaults = {}
    for col in NUMERIC_FEATURES:
        value = pd.to_numeric(rookie_rows[col], errors="coerce").median()
        if pd.isna(value):
            value = global_defaults[col]
        rookie_defaults[col] = float(value)

    numeric_fill_cols = [c for c in NUMERIC_FEATURES if c not in {"year", "round_fraction", "field_size", "driver_age", "driver_starts_log", "driver_circuit_starts_log", "team_circuit_starts_log"}]
    for col in numeric_fill_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(global_defaults[col])
    merged["driver_age"] = merged["driver_age"].fillna(global_defaults["driver_age"])
    merged["prev_driver_rank_pct"] = merged["prev_driver_rank_pct"].fillna(rookie_defaults["prev_driver_rank_pct"])
    merged["prev_team_rank_pct"] = merged["prev_team_rank_pct"].fillna(global_defaults["prev_team_rank_pct"])
    merged[CATEGORICAL_FEATURES] = merged[CATEGORICAL_FEATURES].fillna("unknown").astype(str)

    driver_state = _build_driver_state(merged)
    team_state = _build_team_state(team_race, merged)
    driver_circuit_state = _build_driver_circuit_state(merged)
    team_circuit_state = _build_team_circuit_state(team_race)

    driver_dobs = {
        str(row.driverRef): row.dob
        for row in drivers[["driverRef", "dob"]].itertuples(index=False)
        if pd.notna(row.dob)
    }
    for ref, value in NEW_DRIVER_DOBS.items():
        driver_dobs[ref] = pd.Timestamp(value)

    quality.update(
        {
            "model_min_year": int(merged["year"].min()),
            "model_max_year": int(merged["year"].max()),
            "model_rows": int(len(merged)),
            "model_races": int(merged["raceId"].nunique()),
            "model_drivers": int(merged["driverRef"].nunique()),
            "model_lineages": int(merged["constructorLineage"].nunique()),
            "grid_zero_rows": int((pd.to_numeric(results["grid"], errors="coerce").fillna(0) == 0).sum()),
        }
    )

    return HistoricalBundle(
        model_table=merged,
        raw_results=merged,
        driver_state=driver_state,
        team_state=team_state,
        driver_circuit_state=driver_circuit_state,
        team_circuit_state=team_circuit_state,
        rookie_defaults=rookie_defaults,
        global_defaults=global_defaults,
        driver_dobs=driver_dobs,
        data_quality=quality,
    )


def _last_mean(group: pd.DataFrame, col: str, n: int) -> float:
    return float(pd.to_numeric(group[col], errors="coerce").tail(n).mean())


def _build_driver_state(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for driver_ref, group in merged.sort_values(["date", "raceId"]).groupby("driverRef", sort=False):
        row = {
            "driverRef": driver_ref,
            "last_year": int(group["year"].max()),
            "last_date": group["date"].max(),
            "starts": int(len(group)),
            "driver_roll_finish_3": _last_mean(group, "finish_pct", 3),
            "driver_roll_finish_5": _last_mean(group, "finish_pct", 5),
            "driver_roll_finish_10": _last_mean(group, "finish_pct", 10),
            "driver_roll_finish_20": _last_mean(group, "finish_pct", 20),
            "driver_roll_grid_5": _last_mean(group, "grid_pct", 5),
            "driver_roll_grid_10": _last_mean(group, "grid_pct", 10),
            "driver_roll_points_5": _last_mean(group, "points_norm", 5),
            "driver_roll_points_10": _last_mean(group, "points_norm", 10),
            "driver_roll_win_20": _last_mean(group, "win_flag", 20),
            "driver_roll_podium_10": _last_mean(group, "podium_flag", 10),
            "driver_roll_dnf_10": _last_mean(group, "dnf_flag", 10),
        }
        rows.append(row)
    state = pd.DataFrame(rows)
    latest_year = int(merged["year"].max())
    standings = merged[merged["year"] == latest_year].groupby("driverRef", as_index=False)["points"].sum()
    standings["prev_driver_rank_pct"] = _rank_pct(standings["points"])
    state = state.merge(standings[["driverRef", "prev_driver_rank_pct"]], on="driverRef", how="left")
    return state


def _build_team_state(team_race: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lineage, group in team_race.sort_values(["date", "raceId"]).groupby("constructorLineage", sort=False):
        rows.append(
            {
                "constructorLineage": lineage,
                "last_year": int(group["year"].max()),
                "last_date": group["date"].max(),
                "starts": int(len(group)),
                "team_roll_finish_3": _last_mean(group, "team_finish", 3),
                "team_roll_finish_5": _last_mean(group, "team_finish", 5),
                "team_roll_finish_10": _last_mean(group, "team_finish", 10),
                "team_roll_points_5": _last_mean(group, "team_points", 5),
                "team_roll_points_10": _last_mean(group, "team_points", 10),
                "team_roll_win_10": _last_mean(group, "team_win", 10),
                "team_roll_podium_10": _last_mean(group, "team_podium", 10),
                "team_roll_dnf_10": _last_mean(group, "team_dnf", 10),
            }
        )
    state = pd.DataFrame(rows)
    latest_year = int(merged["year"].max())
    standings = merged[merged["year"] == latest_year].groupby("constructorLineage", as_index=False)["points"].sum()
    standings["prev_team_rank_pct"] = _rank_pct(standings["points"])
    return state.merge(standings[["constructorLineage", "prev_team_rank_pct"]], on="constructorLineage", how="left")


def _build_driver_circuit_state(merged: pd.DataFrame) -> pd.DataFrame:
    return (
        merged.groupby(["driverRef", "circuitRef"], as_index=False)
        .agg(driver_circuit_finish=("finish_pct", "mean"), driver_circuit_starts=("raceId", "count"))
        .assign(driver_circuit_starts_log=lambda x: np.log1p(x["driver_circuit_starts"]))
    )


def _build_team_circuit_state(team_race: pd.DataFrame) -> pd.DataFrame:
    return (
        team_race.groupby(["constructorLineage", "circuitRef"], as_index=False)
        .agg(team_circuit_finish=("team_finish", "mean"), team_circuit_starts=("raceId", "count"))
        .assign(team_circuit_starts_log=lambda x: np.log1p(x["team_circuit_starts"]))
    )


def build_future_feature_table(
    bundle: HistoricalBundle,
    calendar: pd.DataFrame,
    roster: pd.DataFrame,
    external_driver_standings: Optional[pd.DataFrame] = None,
    external_constructor_standings: Optional[pd.DataFrame] = None,
    external_winners: Optional[pd.DataFrame] = None,
    assimilation_weight: float = 0.0,
) -> pd.DataFrame:
    calendar = calendar.copy()
    roster = roster.copy()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    season = int(calendar["season"].iloc[0])
    max_round = int(calendar["round"].max())

    rows = []
    for race in calendar.itertuples(index=False):
        active = roster[(roster["start_round"] <= race.round) & (roster["end_round"] >= race.round)]
        for entry in active.itertuples(index=False):
            rows.append(
                {
                    "year": season,
                    "season": season,
                    "round": int(race.round),
                    "round_fraction": float(race.round) / max_round,
                    "date": race.date,
                    "race_name": race.race_name,
                    "circuitRef": race.circuitRef,
                    "is_sprint": int(getattr(race, "is_sprint", 0)),
                    "completed": int(getattr(race, "completed", 0)),
                    "constructorRef": entry.constructorRef,
                    "constructor_name": entry.constructor_name,
                    "constructorLineage": constructor_lineage(entry.constructorRef),
                    "driverRef": entry.driverRef,
                    "driver_name": entry.driver_name,
                    "field_size": float(len(active)),
                }
            )
    future = pd.DataFrame(rows)

    driver_state = bundle.driver_state.set_index("driverRef")
    team_state = bundle.team_state.set_index("constructorLineage")
    dc = bundle.driver_circuit_state.set_index(["driverRef", "circuitRef"])
    tc = bundle.team_circuit_state.set_index(["constructorLineage", "circuitRef"])

    driver_cols = [
        "driver_roll_finish_3", "driver_roll_finish_5", "driver_roll_finish_10", "driver_roll_finish_20",
        "driver_roll_grid_5", "driver_roll_grid_10", "driver_roll_points_5", "driver_roll_points_10",
        "driver_roll_win_20", "driver_roll_podium_10", "driver_roll_dnf_10", "prev_driver_rank_pct",
    ]
    team_cols = [
        "team_roll_finish_3", "team_roll_finish_5", "team_roll_finish_10", "team_roll_points_5",
        "team_roll_points_10", "team_roll_win_10", "team_roll_podium_10", "team_roll_dnf_10",
        "prev_team_rank_pct",
    ]

    def driver_value(ref: str, col: str) -> float:
        if ref not in driver_state.index:
            return float(bundle.rookie_defaults.get(col, bundle.global_defaults.get(col, 0.5)))
        value = driver_state.at[ref, col]
        if pd.isna(value):
            value = bundle.rookie_defaults.get(col, bundle.global_defaults.get(col, 0.5))
        starts = float(driver_state.at[ref, "starts"])
        if starts < 12 and col.startswith("driver_"):
            weight = starts / (starts + 8.0)
            prior = bundle.rookie_defaults.get(col, bundle.global_defaults.get(col, 0.5))
            value = weight * float(value) + (1 - weight) * float(prior)
        return float(value)

    def team_value(lineage: str, col: str) -> float:
        if lineage not in team_state.index:
            if lineage == "cadillac":
                if col.startswith("team_roll_finish") or col == "prev_team_rank_pct":
                    return 0.88
                if col.startswith("team_roll_points") or col.startswith("team_roll_win") or col.startswith("team_roll_podium"):
                    return 0.01
                if col.startswith("team_roll_dnf"):
                    return 0.20
            return float(bundle.global_defaults.get(col, 0.5))
        value = team_state.at[lineage, col]
        return float(value) if pd.notna(value) else float(bundle.global_defaults.get(col, 0.5))

    for col in driver_cols:
        future[col] = [driver_value(ref, col) for ref in future["driverRef"]]
    for col in team_cols:
        future[col] = [team_value(ref, col) for ref in future["constructorLineage"]]

    future["driver_starts"] = [float(driver_state.at[ref, "starts"]) if ref in driver_state.index else 0.0 for ref in future["driverRef"]]
    future["driver_starts_log"] = np.log1p(future["driver_starts"])

    ages = []
    for ref, race_date in zip(future["driverRef"], future["date"]):
        dob = bundle.driver_dobs.get(str(ref))
        if dob is None or pd.isna(race_date):
            ages.append(bundle.global_defaults["driver_age"])
        else:
            ages.append((pd.Timestamp(race_date) - pd.Timestamp(dob)).days / 365.25)
    future["driver_age"] = ages

    dc_finish, dc_starts = [], []
    tc_finish, tc_starts = [], []
    for row in future.itertuples(index=False):
        dkey = (row.driverRef, row.circuitRef)
        if dkey in dc.index:
            value = dc.loc[dkey]
            dc_finish.append(float(value["driver_circuit_finish"]))
            dc_starts.append(float(value["driver_circuit_starts_log"]))
        else:
            dc_finish.append(float(bundle.global_defaults["driver_circuit_finish"]))
            dc_starts.append(0.0)
        tkey = (row.constructorLineage, row.circuitRef)
        if tkey in tc.index:
            value = tc.loc[tkey]
            tc_finish.append(float(value["team_circuit_finish"]))
            tc_starts.append(float(value["team_circuit_starts_log"]))
        else:
            tc_finish.append(float(bundle.global_defaults["team_circuit_finish"]))
            tc_starts.append(0.0)
    future["driver_circuit_finish"] = dc_finish
    future["driver_circuit_starts_log"] = dc_starts
    future["team_circuit_finish"] = tc_finish
    future["team_circuit_starts_log"] = tc_starts

    if assimilation_weight > 0 and external_driver_standings is not None and external_constructor_standings is not None:
        future = assimilate_standings(
            future,
            external_driver_standings,
            external_constructor_standings,
            external_winners,
            assimilation_weight,
        )

    for col in NUMERIC_FEATURES:
        future[col] = pd.to_numeric(future[col], errors="coerce").fillna(bundle.global_defaults.get(col, 0.5))
    future[CATEGORICAL_FEATURES] = future[CATEGORICAL_FEATURES].fillna("unknown").astype(str)
    return future


def assimilate_standings(
    future: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    winners: Optional[pd.DataFrame],
    weight: float,
) -> pd.DataFrame:
    out = future.copy()
    weight = float(np.clip(weight, 0, 1))
    ds = driver_standings.copy()
    cs = constructor_standings.copy()
    ds["rank_pct_external"] = (ds["position"] - 1) / max(len(ds) - 1, 1)
    cs["rank_pct_external"] = (cs["position"] - 1) / max(len(cs) - 1, 1)
    max_driver_points = max(float(ds["points"].max()), 1.0)
    max_team_points = max(float(cs["points"].max()), 1.0)
    ds["points_external"] = ds["points"] / max_driver_points
    cs["points_external"] = cs["points"] / max_team_points
    rounds = max(int(ds.get("snapshot_round", pd.Series([1])).max()), 1)
    win_counts = winners.groupby("driverRef").size().to_dict() if winners is not None and len(winners) else {}
    team_win_counts = winners.groupby("constructorRef").size().to_dict() if winners is not None and len(winners) else {}
    d_rank = ds.set_index("driverRef")["rank_pct_external"].to_dict()
    d_points = ds.set_index("driverRef")["points_external"].to_dict()
    t_rank = cs.set_index("constructorRef")["rank_pct_external"].to_dict()
    t_points = cs.set_index("constructorRef")["points_external"].to_dict()

    driver_finish_cols = ["driver_roll_finish_3", "driver_roll_finish_5", "driver_roll_finish_10", "driver_roll_finish_20", "driver_roll_grid_5", "driver_roll_grid_10"]
    for col in driver_finish_cols:
        proxy = out["driverRef"].map(d_rank).fillna(out[col]).clip(0, 1)
        out[col] = (1 - weight) * out[col] + weight * proxy
    for col in ["driver_roll_points_5", "driver_roll_points_10"]:
        proxy = out["driverRef"].map(d_points).fillna(out[col]).clip(0, 1)
        out[col] = (1 - weight) * out[col] + weight * proxy
    out["prev_driver_rank_pct"] = (1 - weight) * out["prev_driver_rank_pct"] + weight * out["driverRef"].map(d_rank).fillna(out["prev_driver_rank_pct"])
    win_proxy = out["driverRef"].map(lambda x: win_counts.get(x, 0) / rounds)
    out["driver_roll_win_20"] = (1 - weight) * out["driver_roll_win_20"] + weight * win_proxy
    podium_proxy = (out["driverRef"].map(d_points).fillna(0) ** 1.4).clip(0, 1)
    out["driver_roll_podium_10"] = (1 - weight) * out["driver_roll_podium_10"] + weight * podium_proxy

    team_finish_cols = ["team_roll_finish_3", "team_roll_finish_5", "team_roll_finish_10"]
    for col in team_finish_cols:
        proxy = out["constructorRef"].map(t_rank).fillna(out[col]).clip(0, 1)
        out[col] = (1 - weight) * out[col] + weight * proxy
    for col in ["team_roll_points_5", "team_roll_points_10"]:
        proxy = out["constructorRef"].map(t_points).fillna(out[col]).clip(0, 1)
        out[col] = (1 - weight) * out[col] + weight * proxy
    out["prev_team_rank_pct"] = (1 - weight) * out["prev_team_rank_pct"] + weight * out["constructorRef"].map(t_rank).fillna(out["prev_team_rank_pct"])
    team_win_proxy = out["constructorRef"].map(lambda x: team_win_counts.get(x, 0) / rounds)
    out["team_roll_win_10"] = (1 - weight) * out["team_roll_win_10"] + weight * team_win_proxy
    out["team_roll_podium_10"] = (1 - weight) * out["team_roll_podium_10"] + weight * (out["constructorRef"].map(t_points).fillna(0) ** 1.2).clip(0, 1)
    return out
