"""Output tables, charts, and compact machine-readable summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_json(data: Mapping[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        return value

    path.write_text(json.dumps(convert(dict(data)), indent=2, ensure_ascii=False), encoding="utf-8")


def plot_dataset_coverage(inventory: pd.DataFrame, path: Path) -> None:
    display = inventory.sort_values("rows", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh(display["file"], display["rows"])
    ax.set_xlabel("Broj redova")
    ax.set_title("Najveće tabele u dostavljenom F1 skupu podataka")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(fold_metrics: pd.DataFrame, ensemble_year_metrics: pd.DataFrame, path: Path) -> None:
    data = fold_metrics[["test_year", "model_rank_mae", "baseline_rank_mae"]].merge(
        ensemble_year_metrics[["test_year", "ensemble_rank_mae"]], on="test_year", how="left"
    )
    x = np.arange(len(data))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.0, 5.1))
    ax.bar(x - width, data["baseline_rank_mae"], width, label="Bazni model nedavne forme")
    ax.bar(x, data["model_rank_mae"], width, label="CatBoost")
    ax.bar(x + width, data["ensemble_rank_mae"], width, label="Ansambl")
    ax.set_xticks(x, data["test_year"].astype(str))
    ax.set_ylabel("Srednja apsolutna greška poretka")
    ax.set_xlabel("Hronološka testna sezona")
    ax.set_title("Hronološki backtest bez curenja podataka")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    data = importance.dropna(subset=["importance"]).head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.barh(data["feature"], data["importance"])
    ax.set_xlabel("Važnost obilježja prema CatBoost modelu")
    ax.set_title(f"Top {len(data)} ulaznih obilježja modela")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rank_comparison(comparison: pd.DataFrame, path: Path, title: str, top_n: int = 15) -> None:
    data = comparison.dropna(subset=["actual_position", "predicted_position"]).nsmallest(top_n, "actual_position").copy()
    data = data.sort_values("actual_position", ascending=False)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.8, max(5.6, 0.36 * len(data) + 1.8)))
    for idx, row in enumerate(data.itertuples(index=False)):
        ax.plot([row.actual_position, row.predicted_position], [idx, idx], linewidth=1.2, alpha=0.7)
    ax.scatter(data["actual_position"], y, label="Stvarni poredak", s=42)
    ax.scatter(data["predicted_position"], y, label="Predviđeni poredak", marker="x", s=52)
    ax.set_yticks(y, data["driver_name"])
    ax.set_xlabel("Pozicija u prvenstvu (manja vrijednost je bolja)")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_points_comparison(comparison: pd.DataFrame, path: Path, title: str, top_n: int = 10) -> None:
    data = comparison.dropna(subset=["actual_position", "predicted_points", "actual_points"]).nsmallest(top_n, "actual_position").copy()
    data = data.sort_values("actual_position")
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    ax.bar(x - width / 2, data["actual_points"], width, label="Stvarni bodovi")
    ax.bar(x + width / 2, data["predicted_points"], width, label="Očekivani bodovi")
    ax.set_xticks(x, data["driver_name"], rotation=35, ha="right")
    ax.set_ylabel("Bodovi")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_nowcast_intervals(standings: pd.DataFrame, path: Path, title: str, top_n: int = 10) -> None:
    data = standings.nsmallest(top_n, "predicted_position").sort_values("predicted_position", ascending=False)
    y = np.arange(len(data))
    left = data["expected_points"] - data["points_p10"]
    right = data["points_p90"] - data["expected_points"]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.errorbar(data["expected_points"], y, xerr=np.vstack([left, right]), fmt="o", capsize=3)
    ax.set_yticks(y, data["driver_name"])
    ax.set_xlabel("Projektovani konačni bodovi (10.-90. percentil)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_actual_winner_rank(comparison: pd.DataFrame, path: Path, title: str) -> None:
    data = comparison.sort_values("round")
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.plot(data["round"], data["actual_winner_predicted_rank"], marker="o")
    ax.axhline(3, linestyle="--", linewidth=1.0, label="Granica Top 3")
    ax.axhline(5, linestyle=":", linewidth=1.0, label="Granica Top 5")
    ax.set_xticks(data["round"])
    ax.set_ylim(max(8, float(data["actual_winner_predicted_rank"].max()) + 1), 0.5)
    ax.set_xlabel("Redni broj utrke")
    ax.set_ylabel("Pozicija koju je model dodijelio stvarnom pobjedniku")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_remaining_winner_probabilities(probabilities: pd.DataFrame, path: Path, top_per_race: int = 3) -> None:
    data = probabilities.sort_values(["round", "win_probability"], ascending=[True, False]).groupby("round").head(top_per_race).copy()
    data["label"] = data["round"].astype(str) + ". " + data["race_name"] + " - " + data["driver_name"]
    data = data.sort_values(["round", "win_probability"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(9.2, max(7.0, 0.24 * len(data) + 1.5)))
    ax.barh(data["label"], 100 * data["win_probability"])
    ax.set_xlabel("Vjerovatnoća pobjede (%)")
    ax.set_title("Ažurirana prognoza 2026: tri glavna kandidata za pobjedu u preostalim utrkama")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
