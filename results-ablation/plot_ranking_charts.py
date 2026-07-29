"""plot_ranking_charts.py

Generates publication-quality bar charts for algorithm rankings, statistical p-values,
and relative performance scores across 4 constrained multi-objective optimization algorithms.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = Path(__file__).parent.resolve()
ARTIFACT_DIR = Path('/Users/jingzeng/.gemini/antigravity/brain/e26117dc-3a6f-4c08-910a-e4d6d0dc698d')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df_detail = pd.read_csv(RESULTS_DIR / "per_problem_metrics_detail.csv")
df_summary = pd.read_csv(RESULTS_DIR / "ranking_and_pvalues_summary.csv")

ALGOS = ["DSOCOL", "DSOCOL1", "DSOCOL3", "DSOCOL4"]
ALGO_LABELS = {
    "DSOCOL": "DSOCOL (Full)",
    "DSOCOL1": "w/o NGSS",
    "DSOCOL3": "w/o COL",
    "DSOCOL4": "w/o Trend"
}

COLORS = {
    "DSOCOL": "#2563EB",   # Vibrant Blue (Full Model)
    "DSOCOL1": "#F59E0B",  # Amber/Orange (w/o NGSS)
    "DSOCOL3": "#10B981",  # Emerald/Green (w/o COL)
    "DSOCOL4": "#8B5CF6"   # Purple/Indigo (w/o Trend)
}

EDGE_COLORS = {
    "DSOCOL": "#1E40AF",
    "DSOCOL1": "#D97706",
    "DSOCOL3": "#059669",
    "DSOCOL4": "#7C3AED"
}

HATCHES = {
    "DSOCOL": "///",
    "DSOCOL1": "..",
    "DSOCOL3": "\\\\\\",
    "DSOCOL4": "xx"
}

PROBLEMS = ["C1DTLZ1", "DC1DTLZ1", "DASCMOP1", "DASCMOP7", "LIRCMOP1", "LIRCMOP13"]


def plot_overall_ranking_bars():
    """1. 双图对比：Overall Average IGD Rank & Overall Average HV Rank (1是第一名/最优)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    # Sort algorithms by Avg_HV_Rank for aesthetic presentation or keep constant order
    x_indices = np.arange(len(ALGOS))
    bar_width = 0.55

    avg_igd = [df_summary[df_summary["Algorithm"] == a]["Avg_IGD_Rank"].values[0] for a in ALGOS]
    avg_hv = [df_summary[df_summary["Algorithm"] == a]["Avg_HV_Rank"].values[0] for a in ALGOS]

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

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
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
        ranks = [df_detail[(df_detail["Problem"] == p) & (df_detail["Algorithm"] == algo)]["IGD_Rank"].values[0] for p in PROBLEMS]
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
        ranks = [df_detail[(df_detail["Problem"] == p) & (df_detail["Algorithm"] == algo)]["HV_Rank"].values[0] for p in PROBLEMS]
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

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

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
            v_i = row["IGD_Mean"].values[0]
            v_h = row["HV_Mean"].values[0]

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
            sym = row["IGD_Symbol"].values[0]
            sc = scores_igd[p_idx, a_idx]

            tag = f"({sym})" if algo != "DSOCOL" else "(Full)"
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
            sym = row["HV_Symbol"].values[0]
            sc = scores_hv[p_idx, a_idx]

            tag = f"({sym})" if algo != "DSOCOL" else "(Full)"
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

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_file = RESULTS_DIR / "igd_hv_relative_score_bars.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_overall_ranking_bars()
    plot_per_problem_ranks()
    plot_relative_score_bars()
