"""results/comparison/plot_results.py

读取 results/comparison 目录下的算法 NPZ 实验结果，生成精简的 3 个统计表格和 3 张学术对比图：

【表格 (CSV)】:
1. overall_summary.csv: 各 Benchmark 下每个算法的总汇总表 (Runs, IGD Mean/Std, HV Mean/Std, 可行解数, 耗时)
2. algorithm_scores.csv: 算法在不同 Benchmark 下的得分表 (Mean ± Std)
3. algorithm_rankings.csv: 算法在不同 Benchmark 下的比较排名与 Wilcoxon 显著性检验表 (+/=/-, 平均排名)

【图表 (PNG)】:
1. igd_score_chart.png: 算法在不同 Benchmark 下的 IGD 得分对比图 (各问题 Mean ± Std 分面柱状图)
2. hv_score_chart.png: 算法在不同 Benchmark 下的 HV 得分对比图 (各问题 Mean ± Std 分面柱状图)
3. overall_ranking.png: 算法跨 Benchmark 的总体平均排名对比图 (Friedman Rank 对比)

用法:
    python results/comparison/plot_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ranksums

# 设置项目根目录以便导入模块
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 设置学术图表全局绘图样式
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "SimHei", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

DEFAULT_DIR = Path(__file__).resolve().parent

PROBLEMS_ORDER = [
    {"name": "C1DTLZ1", "category": "C-DTLZs", "note": "Standard Constraints"},
    {"name": "DC1DTLZ1", "category": "DC-DTLZs", "note": "Disconnected PF"},
    {"name": "DASCMOP1", "category": "DAS-CMOP", "note": "Adjustable Difficulty"},
    {"name": "DASCMOP7", "category": "DAS-CMOP", "note": "Complex Boundary"},
    {"name": "LIRCMOP1", "category": "LIR-CMOP", "note": "Large Infeasible Region"},
    {"name": "LIRCMOP13", "category": "LIR-CMOP", "note": "Narrow Traps"},
]

ALGORITHM_ORDER = ["DSOCOL", "APSEA", "CMOCSO", "IMCMOEAD"]
ALGORITHM_LABELS = {
    "DSOCOL": "DSOCOL (Ours)",
    "APSEA": "APSEA",
    "CMOCSO": "CMOCSO",
    "IMCMOEAD": "IM-C-MOEA/D",
}

COLORS = {
    "DSOCOL": "#2563EB",
    "APSEA": "#64748B",
    "CMOCSO": "#0D9488",
    "IMCMOEAD": "#94A3B8",
}

EDGE_COLORS = {
    "DSOCOL": "#1E40AF",
    "APSEA": "#475569",
    "CMOCSO": "#0F766E",
    "IMCMOEAD": "#64748B",
}

HATCHES = {
    "DSOCOL": "///",
    "APSEA": "",
    "CMOCSO": "..",
    "IMCMOEAD": "",
}


def load_all_runs(results_dir: Path, algorithms: list[str]) -> pd.DataFrame:
    """加载所有 NPZ 运行结果。"""
    rows: list[dict] = []
    for algorithm in algorithms:
        algo_dir = results_dir / algorithm
        if not algo_dir.exists():
            continue
        for npz_file in algo_dir.glob("**/*.npz"):
            prob_name = npz_file.parent.name
            with np.load(npz_file) as data:
                n_feas = int(data["n_feasible"]) if "n_feasible" in data else len(data.get("feas_f", []))
                igd_val = float(data["igd"]) if "igd" in data and not np.isnan(data["igd"]) else np.nan
                hv_val = float(data["hv"]) if "hv" in data and not np.isnan(data["hv"]) else np.nan
                t_val = float(data["elapsed_time"]) if "elapsed_time" in data else 0.0
                rows.append({
                    "Algorithm": algorithm,
                    "Problem": prob_name,
                    "Seed": npz_file.stem.split("_")[-1],
                    "N_Feasible": n_feas,
                    "IGD": igd_val,
                    "HV": hv_val,
                    "Time_s": t_val,
                })
    return pd.DataFrame(rows)


def get_numeric_values(df: pd.DataFrame, col: str, fill_val: float | None = None) -> np.ndarray:
    """提取指标数值数组。若 fill_val 为 None 则过滤出有限数值，否则用 fill_val 填充。"""
    vals = np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float).copy()
    if fill_val is None:
        return vals[np.isfinite(vals)]
    vals[~np.isfinite(vals)] = fill_val
    return vals


def format_metric(mean_val: float, std_val: float) -> str:
    """格式化均值与标准差。"""
    if not np.isfinite(mean_val) or mean_val == 0.0:
        return "0.0000 ± 0.0000"
    if mean_val < 0.01:
        return f"{mean_val:.4e} ± {std_val:.4e}"
    elif mean_val < 1.0:
        return f"{mean_val:.4f} ± {std_val:.4f}"
    else:
        return f"{mean_val:.3f} ± {std_val:.3f}"


def compute_statistics_and_reports(
    raw_df: pd.DataFrame,
    problems_info: list[dict],
    algorithms: list[str],
    control_algo: str = "DSOCOL",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """计算 3 个数据表格与相关统计信息。"""
    summary_rows: list[dict] = []
    score_rows: list[dict] = []
    ranking_rows: list[dict] = []

    prob_names = [p["name"] for p in problems_info]

    igd_ranks = {algo: [] for algo in algorithms}
    hv_ranks = {algo: [] for algo in algorithms}
    wilcoxon_igd = {algo: {"+": 0, "=": 0, "-": 0} for algo in algorithms if algo != control_algo}
    wilcoxon_hv = {algo: {"+": 0, "=": 0, "-": 0} for algo in algorithms if algo != control_algo}

    # 存储按问题各算法的 Mean/Std 供画图使用
    plot_data = {
        "problems": problems_info,
        "algorithms": algorithms,
        "igd_means": {p: {} for p in prob_names},
        "igd_stds": {p: {} for p in prob_names},
        "hv_means": {p: {} for p in prob_names},
        "hv_stds": {p: {} for p in prob_names},
        "igd_ranks": {p: {} for p in prob_names},
        "hv_ranks": {p: {} for p in prob_names},
    }

    for p_info in problems_info:
        p_name = p_info["name"]
        cat = p_info["category"]
        p_df = raw_df[raw_df["Problem"] == p_name]

        igd_means = {}
        igd_stds = {}
        hv_means = {}
        hv_stds = {}
        feas_means = {}
        time_means = {}
        runs_counts = {}

        for algo in algorithms:
            sub = p_df[p_df["Algorithm"] == algo]
            igd_vals = get_numeric_values(sub, "IGD", fill_val=None)
            hv_vals = get_numeric_values(sub, "HV", fill_val=None)
            feas_vals = get_numeric_values(sub, "N_Feasible", fill_val=0.0)
            t_vals = get_numeric_values(sub, "Time_s", fill_val=0.0)

            runs_counts[algo] = len(sub)
            igd_means[algo] = float(np.mean(igd_vals)) if len(igd_vals) else 9999.0
            igd_stds[algo] = float(np.std(igd_vals)) if len(igd_vals) else 0.0
            hv_means[algo] = float(np.mean(hv_vals)) if len(hv_vals) else 0.0
            hv_stds[algo] = float(np.std(hv_vals)) if len(hv_vals) else 0.0
            feas_means[algo] = float(np.mean(feas_vals)) if len(feas_vals) else 0.0
            time_means[algo] = float(np.mean(t_vals)) if len(t_vals) else 0.0

            plot_data["igd_means"][p_name][algo] = igd_means[algo]
            plot_data["igd_stds"][p_name][algo] = igd_stds[algo]
            plot_data["hv_means"][p_name][algo] = hv_means[algo]
            plot_data["hv_stds"][p_name][algo] = hv_stds[algo]

            # 1. 汇总表 (overall_summary) 行记录
            summary_rows.append({
                "Category": cat,
                "Problem": p_name,
                "Algorithm": algo,
                "Runs": runs_counts[algo],
                "IGD_Mean": igd_means[algo],
                "IGD_Std": igd_stds[algo],
                "HV_Mean": hv_means[algo],
                "HV_Std": hv_stds[algo],
                "Feasible_Mean": feas_means[algo],
                "Time_Mean_s": time_means[algo],
            })

        # 排序计算 Rank
        sorted_igd = sorted(algorithms, key=igd_means.__getitem__)
        sorted_hv = sorted(algorithms, key=hv_means.__getitem__, reverse=True)

        for algo in algorithms:
            r_igd = sorted_igd.index(algo) + 1
            r_hv = sorted_hv.index(algo) + 1
            igd_ranks[algo].append(r_igd)
            hv_ranks[algo].append(r_hv)
            plot_data["igd_ranks"][p_name][algo] = r_igd
            plot_data["hv_ranks"][p_name][algo] = r_hv

        # Wilcoxon 检验 vs control_algo (缺失用极端值补齐)
        ctrl_sub = p_df[p_df["Algorithm"] == control_algo]
        ctrl_igd = get_numeric_values(ctrl_sub, "IGD", fill_val=9999.0)
        ctrl_hv = get_numeric_values(ctrl_sub, "HV", fill_val=0.0)

        igd_symbols = {}
        hv_symbols = {}
        for algo in algorithms:
            if algo == control_algo:
                igd_symbols[algo] = "="
                hv_symbols[algo] = "="
                continue
            algo_sub = p_df[p_df["Algorithm"] == algo]
            algo_igd = get_numeric_values(algo_sub, "IGD", fill_val=9999.0)
            algo_hv = get_numeric_values(algo_sub, "HV", fill_val=0.0)

            # IGD 检验
            if np.array_equal(ctrl_igd, algo_igd):
                sym_igd = "="
            else:
                _, p_igd = ranksums(ctrl_igd, algo_igd)
                if p_igd < 0.05:
                    sym_igd = "+" if igd_means[control_algo] < igd_means[algo] else "-"
                else:
                    sym_igd = "="
            igd_symbols[algo] = sym_igd
            wilcoxon_igd[algo][sym_igd] += 1

            # HV 检验
            if np.array_equal(ctrl_hv, algo_hv):
                sym_hv = "="
            else:
                _, p_hv = ranksums(ctrl_hv, algo_hv)
                if p_hv < 0.05:
                    sym_hv = "+" if hv_means[control_algo] > hv_means[algo] else "-"
                else:
                    sym_hv = "="
            hv_symbols[algo] = sym_hv
            wilcoxon_hv[algo][sym_hv] += 1

        # 2. 得分表 (algorithm_scores) 行记录
        score_entry = {"Category": cat, "Problem": p_name}
        for algo in algorithms:
            label = ALGORITHM_LABELS.get(algo, algo)
            score_entry[f"{label} (IGD)"] = format_metric(igd_means[algo], igd_stds[algo])
            score_entry[f"{label} (HV)"] = format_metric(hv_means[algo], hv_stds[algo])
        score_rows.append(score_entry)

        # 3. 排名表 (algorithm_rankings) 行记录
        rank_entry = {"Category": cat, "Problem": p_name}
        for algo in algorithms:
            label = ALGORITHM_LABELS.get(algo, algo)
            r_i = sorted_igd.index(algo) + 1
            r_h = sorted_hv.index(algo) + 1
            sym_i = f" ({igd_symbols[algo]})" if algo != control_algo else " (Base)"
            sym_h = f" ({hv_symbols[algo]})" if algo != control_algo else " (Base)"
            rank_entry[f"{label} IGD Rank"] = f"Rank {r_i}{sym_i}"
            rank_entry[f"{label} HV Rank"] = f"Rank {r_h}{sym_h}"
        ranking_rows.append(rank_entry)

    # 排名表添加汇总行：平均排名 (Friedman Average Rank) 和 Wilcoxon 胜/平/负
    avg_rank_row = {"Category": "Summary", "Problem": "Average Rank"}
    wilcoxon_row = {"Category": "Summary", "Problem": "Wilcoxon (+/=/−)"}
    for algo in algorithms:
        label = ALGORITHM_LABELS.get(algo, algo)
        avg_igd = float(np.mean(igd_ranks[algo]))
        avg_hv = float(np.mean(hv_ranks[algo]))
        avg_rank_row[f"{label} IGD Rank"] = f"{avg_igd:.2f}"
        avg_rank_row[f"{label} HV Rank"] = f"{avg_hv:.2f}"

        if algo == control_algo:
            wilcoxon_row[f"{label} IGD Rank"] = "Base (Ours)"
            wilcoxon_row[f"{label} HV Rank"] = "Base (Ours)"
        else:
            w_i = f"{wilcoxon_igd[algo]['+']}/{wilcoxon_igd[algo]['=']}/{wilcoxon_igd[algo]['-']}"
            w_h = f"{wilcoxon_hv[algo]['+']}/{wilcoxon_hv[algo]['=']}/{wilcoxon_hv[algo]['-']}"
            wilcoxon_row[f"{label} IGD Rank"] = w_i
            wilcoxon_row[f"{label} HV Rank"] = w_h

    ranking_rows.append(avg_rank_row)
    ranking_rows.append(wilcoxon_row)

    df_summary = pd.DataFrame(summary_rows)
    df_scores = pd.DataFrame(score_rows)
    df_rankings = pd.DataFrame(ranking_rows)

    plot_data["avg_igd_ranks"] = {algo: float(np.mean(igd_ranks[algo])) for algo in algorithms}
    plot_data["avg_hv_ranks"] = {algo: float(np.mean(hv_ranks[algo])) for algo in algorithms}

    return df_summary, df_scores, df_rankings, plot_data


def plot_metric_chart(
    plot_data: dict,
    metric: str,
    output_path: Path,
) -> None:
    """绘制算法在不同 Benchmark 下的分面柱状图 (IGD 或 HV)。"""
    problems = plot_data["problems"]
    algorithms = plot_data["algorithms"]
    means_dict = plot_data[f"{metric.lower()}_means"]
    stds_dict = plot_data[f"{metric.lower()}_stds"]

    n_problems = len(problems)
    ncols = 3
    nrows = int(np.ceil(n_problems / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    axes_flat = axes.flatten() if n_problems > 1 else [axes]

    x_indices = np.arange(len(algorithms))
    bar_width = 0.55

    direction_note = "(Lower is Better ↓)" if metric == "IGD" else "(Higher is Better ↑)"

    for idx, p_info in enumerate(problems):
        ax = axes_flat[idx]
        ax.set_facecolor("#FFFFFF")
        p_name = p_info["name"]
        cat = p_info["category"]
        note = p_info.get("note", "")

        values = [means_dict[p_name][algo] for algo in algorithms]
        errors = [stds_dict[p_name][algo] for algo in algorithms]

        colors = [COLORS[algo] for algo in algorithms]
        edges = [EDGE_COLORS[algo] for algo in algorithms]
        hatches = [HATCHES[algo] for algo in algorithms]

        bars = ax.bar(
            x_indices,
            values,
            yerr=errors,
            capsize=3.5,
            width=bar_width,
            color=colors,
            edgecolor=edges,
            hatch=hatches,
            linewidth=1.1,
            zorder=3,
            error_kw={"ecolor": "#334155", "linewidth": 1.0},
        )

        # 标注具体数值
        max_val = max(v + e for v, e in zip(values, errors)) if values else 1.0
        for b_idx, (bar, algo, v, err) in enumerate(zip(bars, algorithms, values, errors)):
            y_pos = bar.get_height() + err + (max_val * 0.03)
            txt = f"{v:.4f}" if v < 1.0 else f"{v:.3f}"
            if v < 0.001 and v > 0:
                txt = f"{v:.2e}"
            is_ours = (algo == "DSOCOL")
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                txt,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold" if is_ours else "normal",
                color="#1E3A8A" if is_ours else "#334155",
                zorder=4,
            )

        ax.set_title(f"{p_name} ({cat})\n{note}", fontsize=11, fontweight="bold", color="#0F172A", pad=8)
        ax.set_ylabel(metric, fontsize=10.5, fontweight="bold", color="#334155")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([ALGORITHM_LABELS[a] for a in algorithms], fontsize=9, rotation=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#E2E8F0", zorder=1)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#94A3B8")
        ax.spines["bottom"].set_color("#94A3B8")

        # 适当增加 y 轴上限留出文字空间
        ax.set_ylim(0, max_val * 1.28 if max_val > 0 else 1.0)

    # 隐藏多余的子图网格
    for ax in axes_flat[n_problems:]:
        ax.axis("off")

    fig.suptitle(
        f"Algorithm Performance Comparison: {metric} Scores across Benchmarks {direction_note}",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 已生成图表: {output_path.name}")


def plot_overall_ranking_chart(
    plot_data: dict,
    output_path: Path,
) -> None:
    """绘制各算法跨 Benchmark 的总体平均排名对比图 (1x2 双子图)。"""
    algorithms = plot_data["algorithms"]
    avg_igd = [plot_data["avg_igd_ranks"][algo] for algo in algorithms]
    avg_hv = [plot_data["avg_hv_ranks"][algo] for algo in algorithms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    x_indices = np.arange(len(algorithms))
    bar_width = 0.52

    # --- 左子图: IGD Average Rank ---
    bars1 = ax1.bar(
        x_indices,
        avg_igd,
        width=bar_width,
        color=[COLORS[a] for a in algorithms],
        edgecolor=[EDGE_COLORS[a] for a in algorithms],
        hatch=[HATCHES[a] for a in algorithms],
        linewidth=1.2,
        zorder=3,
    )
    ax1.set_title(
        "Average IGD Ranking across 6 Benchmarks\n(Lower Rank Number is Better ↓)",
        fontsize=11.5,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax1.set_ylabel("Average Friedman Rank", fontsize=11, fontweight="bold", color="#334155")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([ALGORITHM_LABELS[a] for a in algorithms], fontsize=10, fontweight="bold", color="#0F172A")
    ax1.set_ylim(0, 4.2)
    ax1.grid(axis="y", linestyle="--", linewidth=0.8, color="#E2E8F0", zorder=1)
    ax1.set_axisbelow(True)

    for bar, a, r in zip(bars1, algorithms, avg_igd):
        h = bar.get_height()
        is_ours = (a == "DSOCOL")
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.08,
            f"Rank {r:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if is_ours else "normal",
            color="#1E3A8A" if is_ours else "#334155",
            zorder=4,
        )

    # --- 右子图: HV Average Rank ---
    bars2 = ax2.bar(
        x_indices,
        avg_hv,
        width=bar_width,
        color=[COLORS[a] for a in algorithms],
        edgecolor=[EDGE_COLORS[a] for a in algorithms],
        hatch=[HATCHES[a] for a in algorithms],
        linewidth=1.2,
        zorder=3,
    )
    ax2.set_title(
        "Average HV Ranking across 6 Benchmarks\n(Lower Rank Number is Better ↓)",
        fontsize=11.5,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax2.set_ylabel("Average Friedman Rank", fontsize=11, fontweight="bold", color="#334155")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([ALGORITHM_LABELS[a] for a in algorithms], fontsize=10, fontweight="bold", color="#0F172A")
    ax2.set_ylim(0, 4.2)
    ax2.grid(axis="y", linestyle="--", linewidth=0.8, color="#E2E8F0", zorder=1)
    ax2.set_axisbelow(True)

    for bar, a, r in zip(bars2, algorithms, avg_hv):
        h = bar.get_height()
        is_ours = (a == "DSOCOL")
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.08,
            f"Rank {r:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if is_ours else "normal",
            color="#1E3A8A" if is_ours else "#334155",
            zorder=4,
        )

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#94A3B8")
        ax.spines["bottom"].set_color("#94A3B8")

    fig.suptitle(
        "Overall Benchmark Algorithm Ranking Comparison (4 Algorithms across 6 Benchmarks)",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 已生成图表: {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 results-exp 下精简的 3 个表格与 3 张图表。")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="实验结果目录 (默认: 当前脚本所在目录)")
    args = parser.parse_args()

    results_dir = args.dir.resolve()
    print(f"🚀 开始处理实验结果目录: {results_dir}")

    raw_df = load_all_runs(results_dir, ALGORITHM_ORDER)
    if raw_df.empty:
        raise RuntimeError(f"在 {results_dir} 下未找到有效的算法 NPZ 实验数据！")

    df_summary, df_scores, df_rankings, plot_data = compute_statistics_and_reports(
        raw_df=raw_df,
        problems_info=PROBLEMS_ORDER,
        algorithms=ALGORITHM_ORDER,
        control_algo="DSOCOL",
    )

    # 1. 保存 3 个标准表格 (CSV)
    summary_path = results_dir / "overall_summary.csv"
    scores_path = results_dir / "algorithm_scores.csv"
    rankings_path = results_dir / "algorithm_rankings.csv"

    df_summary.to_csv(summary_path, index=False)
    df_scores.to_csv(scores_path, index=False)
    df_rankings.to_csv(rankings_path, index=False)

    print(f"✅ [1/3 表格] 总汇总表已保存: {summary_path.name}")
    print(f"✅ [2/3 表格] 算法得分表已保存: {scores_path.name}")
    print(f"✅ [3/3 表格] 比较排名表已保存: {rankings_path.name}")

    # 2. 绘制并保存 3 张高质量图表 (PNG)
    plot_metric_chart(plot_data, "IGD", results_dir / "igd_score_chart.png")
    plot_metric_chart(plot_data, "HV", results_dir / "hv_score_chart.png")
    plot_overall_ranking_chart(plot_data, results_dir / "overall_ranking.png")

    print("\n🎉 全部 3 个核心表格与 3 张对比图已成功生成！")


if __name__ == "__main__":
    main()

