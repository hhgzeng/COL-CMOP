"""Compute per-problem rankings and statistical comparisons from NPZ results.

The same script supports the main comparison and the ablation comparison.  The
algorithm profile is inferred from the result directory, or can be supplied
explicitly with ``--algorithms``.

Examples::

    python experiments/analyze_ranks.py \
        --results-dir results-exp --output-dir results-exp
    python experiments/analyze_ranks.py \
        --results-dir results-ablation --output-dir results-ablation \
        --algorithms DSOCOL DSOCOL1 DSOCOL3 DSOCOL4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ranksums

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.analysis_utils import (  # noqa: E402
    ALGORITHM_LABELS,
    DEFAULT_RESULTS_DIR,
    PROBLEMS_INFO,
    infer_stat_algorithms,
)


def load_raw_data(results_dir: Path, algorithms: list[str]) -> pd.DataFrame:
    """Load scalar metrics from every NPZ file for the selected algorithms."""
    rows: list[dict] = []
    for algorithm in algorithms:
        algorithm_path = results_dir / algorithm
        for npz_file in algorithm_path.glob("**/*.npz"):
            problem_name = npz_file.parent.name
            with np.load(npz_file) as data:
                rows.append(
                    {
                        "Algorithm": algorithm,
                        "Problem": problem_name,
                        "Seed": npz_file.stem.split("_")[-1],
                        "N_Feasible": int(data["n_feasible"])
                        if "n_feasible" in data
                        else 0,
                        "IGD": float(data["igd"])
                        if "igd" in data and not np.isnan(data["igd"])
                        else np.nan,
                        "HV": float(data["hv"])
                        if "hv" in data and not np.isnan(data["hv"])
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows, columns=["Algorithm", "Problem", "Seed", "N_Feasible", "IGD", "HV"])


def _problem_info(problem_names: list[str] | None) -> list[dict]:
    if not problem_names:
        return PROBLEMS_INFO.copy()
    info_by_name = {item["name"]: item for item in PROBLEMS_INFO}
    unknown = [name for name in problem_names if name not in info_by_name]
    if unknown:
        raise ValueError(
            f"未知问题: {unknown}。当前统计脚本支持: {list(info_by_name)}"
        )
    return [info_by_name[name] for name in problem_names]


def numeric_metric_values(
    frame: Any,
    column: str,
    fill_value: float | None = None,
) -> np.ndarray:
    """Convert a pandas metric column to a numeric NumPy array."""
    values = np.asarray(pd.to_numeric(frame[column], errors="coerce"), dtype=float).copy()
    if fill_value is None:
        return values[np.isfinite(values)]
    values[~np.isfinite(values)] = fill_value
    return values


def analyze_data(
    results_dir: Path,
    output_dir: Path,
    algorithms: list[str],
    control_algo: str = "DSOCOL",
    problem_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze NPZ metrics and save the two CSV reports."""
    if control_algo not in algorithms:
        raise ValueError("--control-algo 必须包含在 --algorithms 中。")

    problems_info = _problem_info(problem_names)
    raw = load_raw_data(results_dir, algorithms)
    detailed_records: list[dict] = []
    problem_names = [item["name"] for item in problems_info]

    igd_ranks = {algorithm: [] for algorithm in algorithms}
    hv_ranks = {algorithm: [] for algorithm in algorithms}
    wilcoxon_igd = {
        algorithm: {"+": 0, "=": 0, "-": 0}
        for algorithm in algorithms
        if algorithm != control_algo
    }
    wilcoxon_hv = {
        algorithm: {"+": 0, "=": 0, "-": 0}
        for algorithm in algorithms
        if algorithm != control_algo
    }
    friedman_igd = {algorithm: [] for algorithm in algorithms}
    friedman_hv = {algorithm: [] for algorithm in algorithms}

    for problem_info in problems_info:
        problem_name = problem_info["name"]
        problem_frame = raw[raw["Problem"] == problem_name]
        igd_means: dict[str, float] = {}
        igd_stds: dict[str, float] = {}
        hv_means: dict[str, float] = {}
        hv_stds: dict[str, float] = {}

        for algorithm in algorithms:
            subset = problem_frame[problem_frame["Algorithm"] == algorithm]
            igd_values = numeric_metric_values(subset, "IGD")
            hv_values = numeric_metric_values(subset, "HV")
            igd_means[algorithm] = float(np.mean(igd_values)) if len(igd_values) else 9999.0
            igd_stds[algorithm] = float(np.std(igd_values)) if len(igd_values) else 0.0
            hv_means[algorithm] = float(np.mean(hv_values)) if len(hv_values) else 0.0
            hv_stds[algorithm] = float(np.std(hv_values)) if len(hv_values) else 0.0
            friedman_igd[algorithm].append(igd_means[algorithm])
            friedman_hv[algorithm].append(hv_means[algorithm])

        sorted_igd = sorted(algorithms, key=igd_means.__getitem__)
        sorted_hv = sorted(algorithms, key=hv_means.__getitem__, reverse=True)
        for algorithm in algorithms:
            igd_ranks[algorithm].append(sorted_igd.index(algorithm) + 1)
            hv_ranks[algorithm].append(sorted_hv.index(algorithm) + 1)

        control_frame = problem_frame[problem_frame["Algorithm"] == control_algo]
        control_igd = numeric_metric_values(control_frame, "IGD", fill_value=9999.0)
        control_hv = numeric_metric_values(control_frame, "HV", fill_value=0.0)

        for algorithm in algorithms:
            algorithm_frame = problem_frame[problem_frame["Algorithm"] == algorithm]
            algorithm_igd = numeric_metric_values(algorithm_frame, "IGD", fill_value=9999.0)
            algorithm_hv = numeric_metric_values(algorithm_frame, "HV", fill_value=0.0)
            if algorithm == control_algo:
                igd_p_value, igd_symbol = 1.0, "="
                hv_p_value, hv_symbol = 1.0, "="
            else:
                if np.array_equal(control_igd, algorithm_igd):
                    igd_p_value, igd_symbol = 1.0, "="
                else:
                    _, igd_p_value = ranksums(control_igd, algorithm_igd)
                    if igd_p_value < 0.05:
                        igd_symbol = (
                            "+"
                            if igd_means[control_algo] < igd_means[algorithm]
                            else "-"
                        )
                    else:
                        igd_symbol = "="
                wilcoxon_igd[algorithm][igd_symbol] += 1

                if np.array_equal(control_hv, algorithm_hv):
                    hv_p_value, hv_symbol = 1.0, "="
                else:
                    _, hv_p_value = ranksums(control_hv, algorithm_hv)
                    if hv_p_value < 0.05:
                        hv_symbol = (
                            "+"
                            if hv_means[control_algo] > hv_means[algorithm]
                            else "-"
                        )
                    else:
                        hv_symbol = "="
                wilcoxon_hv[algorithm][hv_symbol] += 1

            detailed_records.append(
                {
                    "Category": problem_info["category"],
                    "Problem": problem_name,
                    "Algorithm": algorithm,
                    "IGD_Mean": igd_means[algorithm],
                    "IGD_Std": igd_stds[algorithm],
                    "IGD_Rank": sorted_igd.index(algorithm) + 1,
                    "IGD_PValue": igd_p_value,
                    "IGD_Symbol": igd_symbol,
                    "HV_Mean": hv_means[algorithm],
                    "HV_Std": hv_stds[algorithm],
                    "HV_Rank": sorted_hv.index(algorithm) + 1,
                    "HV_PValue": hv_p_value,
                    "HV_Symbol": hv_symbol,
                }
            )

    detailed = pd.DataFrame(detailed_records)
    average_igd_ranks = {
        algorithm: float(np.mean(igd_ranks[algorithm])) for algorithm in algorithms
    }
    average_hv_ranks = {
        algorithm: float(np.mean(hv_ranks[algorithm])) for algorithm in algorithms
    }

    if len(problem_names) >= 2 and len(algorithms) >= 3:
        friedman_igd_result = friedmanchisquare(
            *[friedman_igd[algorithm] for algorithm in algorithms]
        )
        friedman_hv_result = friedmanchisquare(
            *[friedman_hv[algorithm] for algorithm in algorithms]
        )
    else:
        friedman_igd_result = (float("nan"), float("nan"))
        friedman_hv_result = (float("nan"), float("nan"))

    summary = pd.DataFrame(
        [
            {
                "Algorithm": algorithm,
                "Label": ALGORITHM_LABELS.get(algorithm, algorithm),
                "Avg_IGD_Rank": average_igd_ranks[algorithm],
                "IGD_Wilcoxon_W/T/L": (
                    "N/A"
                    if algorithm == control_algo
                    else "/".join(str(wilcoxon_igd[algorithm][symbol]) for symbol in ["+", "=", "-"])
                ),
                "Avg_HV_Rank": average_hv_ranks[algorithm],
                "HV_Wilcoxon_W/T/L": (
                    "N/A"
                    if algorithm == control_algo
                    else "/".join(str(wilcoxon_hv[algorithm][symbol]) for symbol in ["+", "=", "-"])
                ),
            }
            for algorithm in algorithms
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "per_problem_metrics_detail.csv"
    summary_path = output_dir / "ranking_and_pvalues_summary.csv"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("=== PER-PROBLEM DETAILED METRICS & RANKS ===")
    print(
        detailed[
            [
                "Problem",
                "Algorithm",
                "IGD_Mean",
                "IGD_Rank",
                "IGD_Symbol",
                "HV_Mean",
                "HV_Rank",
                "HV_Symbol",
            ]
        ].to_string(index=False)
    )
    print("\n=== OVERALL RANKING & STATISTICAL SUMMARY ===")
    print(summary.to_string(index=False))
    print(
        f"\nFriedman Test IGD: Stat = {friedman_igd_result[0]:.4f}, "
        f"p-value = {friedman_igd_result[1]:.4e}"
    )
    print(
        f"Friedman Test HV:  Stat = {friedman_hv_result[0]:.4f}, "
        f"p-value = {friedman_hv_result[1]:.4e}"
    )
    print(f"\nCSV 输出: {detailed_path}, {summary_path}")
    return detailed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="计算 NPZ 实验结果的排名与显著性检验。")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="NPZ 结果目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="CSV 输出目录（默认与结果目录相同）")
    parser.add_argument("--algorithms", nargs="+", default=None, help="参与统计的算法列表")
    parser.add_argument("--control-algo", default="DSOCOL", help="显著性比较的基准算法")
    parser.add_argument("--problems", nargs="+", default=None, help="只统计指定的问题")
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser()
    output_dir = (args.output_dir or results_dir).expanduser()
    algorithms = args.algorithms or infer_stat_algorithms(results_dir)
    if not algorithms:
        raise SystemExit(f"结果目录中未找到算法 NPZ 数据: {results_dir}")
    analyze_data(
        results_dir,
        output_dir,
        algorithms,
        control_algo=args.control_algo,
        problem_names=args.problems,
    )


if __name__ == "__main__":
    main()
