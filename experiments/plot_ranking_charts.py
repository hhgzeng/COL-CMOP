"""plot_ranking_charts.py

Generates publication-quality bar charts for algorithm rankings, statistical p-values,
and relative performance scores across 4 constrained multi-objective optimization algorithms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.analysis_utils import (  # noqa: E402
    ALGORITHM_LABELS,
    DEFAULT_RESULTS_DIR,
    MAIN_ALGORITHMS,
    ABLATION_ALGORITHMS,
)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "SimHei", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = DEFAULT_RESULTS_DIR
df_detail = pd.DataFrame()
df_summary = pd.DataFrame()
ALGOS: list[str] = []
ALGO_LABELS: dict[str, str] = {}
COLORS: dict[str, str] = {}
EDGE_COLORS: dict[str, str] = {}
HATCHES: dict[str, str] = {}
PROBLEMS = ["C1DTLZ1", "DC1DTLZ1", "DASCMOP1", "DASCMOP7", "LIRCMOP1", "LIRCMOP13"]


def first_value(frame: Any, column: str) -> Any:
    """Return the first value from a filtered pandas frame."""
    return frame[column].iloc[0]


def plot_overall_ranking_bars():
    """1. 双图对比：Overall Average IGD Rank & Overall Average HV Rank (1是第一名/最优)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    # Sort algorithms by Avg_HV_Rank for aesthetic presentation or keep constant order
    x_indices = np.arange(len(ALGOS))
    bar_width = 0.55

    avg_igd = [first_value(df_summary[df_summary["Algorithm"] == a], "Avg_IGD_Rank") for a in ALGOS]
    avg_hv = [first_value(df_summary[df_summary["Algorithm"] == a], "Avg_HV_Rank") for a in ALGOS]

    # --- Subplot 1: IGD Average Ranking ---
    bars1 = ax1.bar(
        x_indices,
        avg_igd,
        width=bar_width,
        color=[COLORS[a] for a in ALGOS],
        edgecolor=[EDGE_COLORS[a] for a in ALGOS],
        hatch=[HATCHES[a] for a in ALGOS],
        linewidth=1.2,
        zorder=3
    )
    ax1.set_title("Average IGD Ranking across 6 Problems\n(Lower Rank Number is Better ↓)", fontsize=11.5, fontweight='bold', color='#0F172A', pad=12)
    ax1.set_ylabel("Average Rank", fontsize=11, fontweight='bold', color='#334155')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([ALGO_LABELS[a] for a in ALGOS], fontsize=10, fontweight='bold', color='#0F172A')
    ax1.set_ylim(0, 4.2)
    ax1.grid(axis='y', linestyle='--', linewidth=0.8, color='#E2E8F0', zorder=1)
    ax1.set_axisbelow(True)

    for bar, a, r in zip(bars1, ALGOS, avg_igd):
        h = bar.get_height()
        fontweight = 'bold' if a == "DSOCOL" else 'normal'
        color = '#1E3A8A' if a == "DSOCOL" else '#334155'
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.08,
            f"Rank {r:.2f}",
            ha='center', va='bottom',
            fontsize=10, fontweight=fontweight, color=color, zorder=4
        )

    # --- Subplot 2: HV Average Ranking ---
    bars2 = ax2.bar(
        x_indices,
        avg_hv,
        width=bar_width,
        color=[COLORS[a] for a in ALGOS],
        edgecolor=[EDGE_COLORS[a] for a in ALGOS],
        hatch=[HATCHES[a] for a in ALGOS],
        linewidth=1.2,
        zorder=3
    )
    ax2.set_title("Average HV Ranking across 6 Problems\n(Lower Rank Number is Better ↓)", fontsize=11.5, fontweight='bold', color='#0F172A', pad=12)
    ax2.set_ylabel("Average Rank", fontsize=11, fontweight='bold', color='#334155')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([ALGO_LABELS[a] for a in ALGOS], fontsize=10, fontweight='bold', color='#0F172A')
    ax2.set_ylim(0, 4.2)
    ax2.grid(axis='y', linestyle='--', linewidth=0.8, color='#E2E8F0', zorder=1)
    ax2.set_axisbelow(True)

    for bar, a, r in zip(bars2, ALGOS, avg_hv):
        h = bar.get_height()
        fontweight = 'bold' if a == "DSOCOL" else 'normal'
        color = '#1E3A8A' if a == "DSOCOL" else '#334155'
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.08,
            f"Rank {r:.2f}",
            ha='center', va='bottom',
            fontsize=10, fontweight=fontweight, color=color, zorder=4
        )

    # Styling spines
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94A3B8')
        ax.spines['bottom'].set_color('#94A3B8')

    fig.suptitle("Overall Algorithm Ranking Comparison (4 Algorithms across 6 Benchmarks)",
                 fontsize=13.5, fontweight='bold', color='#0F172A', y=0.98)

    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    
    out_file = RESULTS_DIR / "igd_hv_overall_ranking_bars.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_per_problem_ranks():
    """2. 按测试问题的分组 Ranking 柱状图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    n_problems = len(PROBLEMS)
    n_algos = len(ALGOS)
    bar_width = 0.18
    x_positions = np.arange(n_problems)

    # --- Top Subplot: IGD Ranks per Problem ---
    for a_idx, algo in enumerate(ALGOS):
        offset = (a_idx - (n_algos - 1) / 2) * bar_width
        ranks = [first_value(df_detail[(df_detail["Problem"] == p) & (df_detail["Algorithm"] == algo)], "IGD_Rank") for p in PROBLEMS]
        bars = ax1.bar(
            x_positions + offset,
            ranks,
            width=bar_width * 0.9,
            color=COLORS[algo],
            edgecolor=EDGE_COLORS[algo],
            hatch=HATCHES[algo],
            linewidth=1.1,
            zorder=3,
            label=ALGO_LABELS[algo]
        )
        for bar, r in zip(bars, ranks):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.08,
                f"R{r}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E293B', zorder=4
            )

    ax1.set_title("Per-Problem IGD Rankings (1 = Best, 4 = Worst)", fontsize=11, fontweight='bold', color='#0F172A')
    ax1.set_ylabel("IGD Rank", fontsize=10.5, fontweight='bold', color='#334155')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(PROBLEMS, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 4.8)
    ax1.set_yticks([1, 2, 3, 4])
    ax1.grid(axis='y', linestyle='--', linewidth=0.8, color='#E2E8F0', zorder=1)
    ax1.legend(loc='upper right', ncol=4, frameon=False, fontsize=10)

    # --- Bottom Subplot: HV Ranks per Problem ---
    for a_idx, algo in enumerate(ALGOS):
        offset = (a_idx - (n_algos - 1) / 2) * bar_width
        ranks = [first_value(df_detail[(df_detail["Problem"] == p) & (df_detail["Algorithm"] == algo)], "HV_Rank") for p in PROBLEMS]
        bars = ax2.bar(
            x_positions + offset,
            ranks,
            width=bar_width * 0.9,
            color=COLORS[algo],
            edgecolor=EDGE_COLORS[algo],
            hatch=HATCHES[algo],
            linewidth=1.1,
            zorder=3,
            label=ALGO_LABELS[algo]
        )
        for bar, r in zip(bars, ranks):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.08,
                f"R{r}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E293B', zorder=4
            )

    ax2.set_title("Per-Problem HV Rankings (1 = Best, 4 = Worst)", fontsize=11, fontweight='bold', color='#0F172A')
    ax2.set_ylabel("HV Rank", fontsize=10.5, fontweight='bold', color='#334155')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(PROBLEMS, fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 4.8)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.grid(axis='y', linestyle='--', linewidth=0.8, color='#E2E8F0', zorder=1)

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94A3B8')
        ax.spines['bottom'].set_color('#94A3B8')

    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))

    out_file = RESULTS_DIR / "per_problem_rank_bars.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_relative_score_bars():
    """3. 相对百分比性能得分柱状图 (100% = Best)，标注 Wilcoxon p-value 符号 (+/=/--)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.5, 9.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    n_problems = len(PROBLEMS)
    n_algos = len(ALGOS)
    bar_width = 0.18
    gap_between_groups = 0.45

    x_positions = []
    curr_x = 0
    for i in range(n_problems):
        x_positions.append(curr_x)
        curr_x += (n_algos * bar_width) + gap_between_groups
    x_positions = np.array(x_positions)

    # Calculate Relative Scores
    scores_igd = np.zeros((n_problems, n_algos))
    scores_hv = np.zeros((n_problems, n_algos))

    for p_idx, p in enumerate(PROBLEMS):
        p_rows = df_detail[df_detail["Problem"] == p]
        min_igd = p_rows["IGD_Mean"].min()
        max_hv = p_rows["HV_Mean"].max()

        for a_idx, a in enumerate(ALGOS):
            row = p_rows[p_rows["Algorithm"] == a]
            v_i = first_value(row, "IGD_Mean")
            v_h = first_value(row, "HV_Mean")

            scores_igd[p_idx, a_idx] = (min_igd / v_i * 100.0) if v_i > 0 else 0.0
            scores_hv[p_idx, a_idx] = (v_h / max_hv * 100.0) if max_hv > 0 else 0.0

    # Subplot 1: IGD Relative Score (%)
    for a_idx, algo in enumerate(ALGOS):
        offset = (a_idx - (n_algos - 1) / 2) * bar_width
        bars = ax1.bar(
            x_positions + offset,
            scores_igd[:, a_idx],
            width=bar_width * 0.9,
            color=COLORS[algo],
            edgecolor=EDGE_COLORS[algo],
            hatch=HATCHES[algo],
            linewidth=1.1,
            zorder=3,
            label=ALGO_LABELS[algo]
        )
        for p_idx, bar in enumerate(bars):
            p = PROBLEMS[p_idx]
            row = df_detail[(df_detail["Problem"] == p) & (df_detail["Algorithm"] == algo)]
            sym = first_value(row, "IGD_Symbol")
            sc = scores_igd[p_idx, a_idx]

            tag = f"({sym})" if algo != "DSOCOL" else "(Ours)"
            label = f"{sc:.1f}%\n{tag}"
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 2.0,
                label,
                ha='center', va='bottom', fontsize=7.5,
                fontweight='bold' if algo == "DSOCOL" else 'normal',
                color='#1E3A8A' if algo == "DSOCOL" else '#475569', zorder=4
            )

    ax1.set_title("Relative IGD Performance Score (%) [100% = Best Performance] & Wilcoxon p-value Symbols (+/=/--)", fontsize=11, fontweight='bold', color='#0F172A')
    ax1.set_ylabel("Relative IGD Score (%)", fontsize=10, fontweight='bold', color='#334155')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(PROBLEMS, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 125)
    ax1.grid(axis='y', linestyle='--', linewidth=0.8, color='#E2E8F0', zorder=1)
    ax1.legend(loc='upper right', ncol=4, frameon=False, fontsize=10)

    # Subplot 2: HV Relative Score (%)
    for a_idx, algo in enumerate(ALGOS):
        offset = (a_idx - (n_algos - 1) / 2) * bar_width
        bars = ax2.bar(
            x_positions + offset,
            scores_hv[:, a_idx],
            width=bar_width * 0.9,
            color=COLORS[algo],
            edgecolor=EDGE_COLORS[algo],
            hatch=HATCHES[algo],
            linewidth=1.1,
            zorder=3,
            label=ALGO_LABELS[algo]
        )
        for p_idx, bar in enumerate(bars):
            p = PROBLEMS[p_idx]
            row = df_detail[(df_detail["Problem"] == p) & (df_detail["Algorithm"] == algo)]
            sym = first_value(row, "HV_Symbol")
            sc = scores_hv[p_idx, a_idx]

            tag = f"({sym})" if algo != "DSOCOL" else "(Ours)"
            label = f"{sc:.1f}%\n{tag}"
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 2.0,
                label,
                ha='center', va='bottom', fontsize=7.5,
                fontweight='bold' if algo == "DSOCOL" else 'normal',
                color='#1E3A8A' if algo == "DSOCOL" else '#475569', zorder=4
            )

    ax2.set_title("Relative HV Performance Score (%) [100% = Best Performance] & Wilcoxon p-value Symbols (+/=/--)", fontsize=11, fontweight='bold', color='#0F172A')
    ax2.set_ylabel("Relative HV Score (%)", fontsize=10, fontweight='bold', color='#334155')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(PROBLEMS, fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 125)
    ax2.grid(axis='y', linestyle='--', linewidth=0.8, color='#E2E8F0', zorder=1)

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94A3B8')
        ax.spines['bottom'].set_color('#94A3B8')

    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))

    out_file = RESULTS_DIR / "igd_hv_relative_score_bars.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成排名、显著性与相对性能柱状图。")
    parser.add_argument(
        "--input-dir",
        "--results-dir",
        dest="input_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="排名 CSV 所在目录",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="图片输出目录（默认与输入目录相同）")
    parser.add_argument("--algorithms", nargs="+", default=None, help="参与绘图的算法列表")
    parser.add_argument("--detail-file", type=Path, default=None, help="明细 CSV 文件名或路径")
    parser.add_argument("--summary-file", type=Path, default=None, help="汇总 CSV 文件名或路径")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser()
    output_dir = (args.output_dir or input_dir).expanduser()
    detail_path = args.detail_file or Path("per_problem_metrics_detail.csv")
    summary_path = args.summary_file or Path("ranking_and_pvalues_summary.csv")
    if not detail_path.is_absolute():
        detail_path = input_dir / detail_path
    if not summary_path.is_absolute():
        summary_path = input_dir / summary_path

    global RESULTS_DIR, df_detail, df_summary, ALGOS, ALGO_LABELS, COLORS, EDGE_COLORS, HATCHES, PROBLEMS
    RESULTS_DIR = output_dir
    df_detail = pd.read_csv(detail_path)
    df_summary = pd.read_csv(summary_path)
    df_detail.columns = [str(column).strip() for column in df_detail.columns]
    df_summary.columns = [str(column).strip() for column in df_summary.columns]
    for frame in (df_detail, df_summary):
        for column in ["Category", "Problem", "Algorithm", "Label", "IGD_Symbol", "HV_Symbol"]:
            if column in frame:
                frame[column] = frame[column].astype(str).str.strip()

    available = list(dict.fromkeys(df_detail["Algorithm"].tolist()))
    if args.algorithms:
        ALGOS = args.algorithms
    else:
        preferred = ABLATION_ALGORITHMS if set(ABLATION_ALGORITHMS) <= set(available) else MAIN_ALGORITHMS
        ALGOS = [algorithm for algorithm in preferred if algorithm in available] or available
    missing = [algorithm for algorithm in ALGOS if algorithm not in available]
    if missing:
        raise ValueError(f"排名明细 CSV 中不存在算法: {missing}")

    ALGO_LABELS = {algorithm: ALGORITHM_LABELS.get(algorithm, algorithm) for algorithm in ALGOS}
    palette = ["#2563EB", "#64748B", "#0D9488", "#94A3B8", "#F59E0B", "#10B981", "#8B5CF6"]
    edge_palette = ["#1E40AF", "#475569", "#0F766E", "#64748B", "#D97706", "#059669", "#7C3AED"]
    hatch_palette = ["///", "", "..", "", "..", "\\\\\\", "xx"]
    COLORS = {algorithm: palette[index % len(palette)] for index, algorithm in enumerate(ALGOS)}
    EDGE_COLORS = {algorithm: edge_palette[index % len(edge_palette)] for index, algorithm in enumerate(ALGOS)}
    HATCHES = {algorithm: hatch_palette[index % len(hatch_palette)] for index, algorithm in enumerate(ALGOS)}
    PROBLEMS = [problem for problem in PROBLEMS if problem in set(df_detail["Problem"].tolist())]
    if not PROBLEMS:
        raise ValueError("排名明细 CSV 中未找到支持的 Benchmark 问题。")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取明细: {detail_path}")
    print(f"读取汇总: {summary_path}")
    print(f"算法: {ALGOS}")
    plot_overall_ranking_bars()
    plot_per_problem_ranks()
    plot_relative_score_bars()


if __name__ == "__main__":
    main()
