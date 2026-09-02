"""通用算法单机测试与评估运行脚本。

功能：
1. 结果按算法名称平铺保存在 results 目录下（如 results/dsocol/），benchmark 结果不分二级文件夹；
2. 运行生成每个 Seed 的原始 NPZ 数据文件；
3. 测试完成后生成学术汇总表 overall_summary.csv（对齐 results/comparison/overall_summary.csv）；
4. 为每个 Benchmark 绘制带 True PF 与参数/指标信息面板的 Pareto 图表（对齐 results/ablation/DSOCOL/C-DTLZs/plots/ 风格）。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, cast

# 设置临时 matplotlib 配置目录以避免权限告警
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import pandas as pd

from core.metrics import calculate_hv, calculate_igd
from core.problem import PymooProblemAdapter
from experiments.config import (
    ALGORITHM_REGISTRY,
    BENCHMARK_CATEGORIES,
    PROBLEM_REGISTRY,
)
from experiments.run_experiment import get_reference_front_and_point

# 学术绘图全局样式设置
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
    "SimHei",
    "STHeiti",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"


def classify_problem(prob_name: str) -> str:
    """根据问题名称归类到对应的 Benchmark 大类 (C-DTLZs, DC-DTLZs, DAS-CMOP, LIR-CMOP)。"""
    prob_upper = prob_name.upper()
    if prob_upper.startswith("DC"):
        return "DC-DTLZs"
    elif prob_upper.startswith("C") and "DTLZ" in prob_upper:
        return "C-DTLZs"
    elif prob_upper.startswith("DAS"):
        return "DAS-CMOP"
    elif prob_upper.startswith("LIR"):
        return "LIR-CMOP"
    return "Other-Benchmarks"


def resolve_problems(
    problems: list[str] | None = None, categories: list[str] | None = None
) -> list[str]:
    """解析问题列表，支持通过 Benchmark 分类展开所有问题。"""
    resolved: list[str] = []

    if categories:
        for cat in categories:
            cat_upper = cat.upper().replace("_", "-")
            if cat_upper == "ALL":
                for probs in BENCHMARK_CATEGORIES.values():
                    resolved.extend(probs)
            else:
                matched = False
                for cat_name, probs in BENCHMARK_CATEGORIES.items():
                    if cat_name.upper().replace("-", "") == cat_upper.replace("-", ""):
                        resolved.extend(probs)
                        matched = True
                        break
                if not matched:
                    raise ValueError(
                        f"未知的 Benchmark 分类: '{cat}'。可选分类: {list(BENCHMARK_CATEGORIES.keys())} 或 'ALL'"
                    )

    if problems:
        for item in problems:
            item_upper = item.upper().replace("_", "-")
            is_cat = False
            if item_upper == "ALL":
                for probs in BENCHMARK_CATEGORIES.values():
                    resolved.extend(probs)
                is_cat = True
            else:
                for cat_name, probs in BENCHMARK_CATEGORIES.items():
                    if cat_name.upper().replace("-", "") == item_upper.replace("-", ""):
                        resolved.extend(probs)
                        is_cat = True
                        break

            if not is_cat:
                if item in PROBLEM_REGISTRY:
                    resolved.append(item)
                else:
                    raise ValueError(
                        f"未知测试问题或分类名: '{item}'。可选: {list(PROBLEM_REGISTRY.keys())} 或 {list(BENCHMARK_CATEGORIES.keys())}"
                    )

    if not resolved:
        resolved = ["C1DTLZ1"]

    return list(dict.fromkeys(resolved))


def get_true_pareto_front(prob_name: str, raw_prob: Any = None) -> np.ndarray | None:
    """尝试获取测试问题的真实参考 Pareto Front (True PF)。"""
    if raw_prob is not None:
        try:
            ref_pf = raw_prob.pareto_front()
            if ref_pf is not None and len(ref_pf) > 0:
                return np.asarray(ref_pf)
        except Exception:
            pass

    if prob_name in PROBLEM_REGISTRY:
        try:
            prob_inst = PROBLEM_REGISTRY[prob_name]()
            ref_pf = prob_inst.pareto_front()
            if ref_pf is not None and len(ref_pf) > 0:
                return np.asarray(ref_pf)
        except Exception:
            pass
    return None


def plot_problem_pareto(
    algo_name: str,
    prob_name: str,
    category: str,
    runs_data: list[dict[str, Any]],
    output_file: Path,
    true_pf: np.ndarray | None = None,
) -> None:
    """为指定问题绘制学术标准 Pareto Front 对比图（含真实 PF 与参数统计说明面板）。"""
    if not runs_data:
        return

    first_run = runs_data[0]
    pop_size = first_run.get("pop_size", 100)
    x_dim = first_run.get("x_dim", "N/A")
    f_dim = first_run.get("f_dim", 2)
    max_evals = first_run.get("max_evals", "N/A")

    eval_counts = [r["eval_count"] for r in runs_data]
    feas_counts = [r["n_feasible"] for r in runs_data]
    igds = [r["igd"] for r in runs_data if not np.isnan(r["igd"])]
    hvs = [r["hv"] for r in runs_data if not np.isnan(r["hv"])]
    times = [r["elapsed_time"] for r in runs_data]

    mean_fe = int(np.mean(eval_counts)) if eval_counts else max_evals
    mean_feas = float(np.mean(feas_counts)) if feas_counts else 0.0
    feas_rate = (
        (mean_feas / pop_size * 100)
        if isinstance(pop_size, (int, float)) and pop_size > 0
        else 0.0
    )
    mean_igd = f"{np.mean(igds):.4e}" if igds else "N/A"
    mean_hv = f"{np.mean(hvs):.4f}" if hvs else "N/A"
    mean_time = f"{np.mean(times):.2f}s" if times else "N/A"

    info_text = (
        f"--- Experiment Setup ---\n"
        f"Algorithm: {algo_name}\n"
        f"Problem: {prob_name} ({category})\n"
        f"Max FEs: {mean_fe}\n"
        f"Pop Size (N): {pop_size}\n"
        f"Decision Vars (D): {x_dim}\n"
        f"Objectives (M): {f_dim}\n"
        f"Independent Runs: {len(runs_data)}\n"
        f"------------------------\n"
        f"--- Performance Stats ---\n"
        f"Feasible Count: {mean_feas:.1f} / {pop_size}\n"
        f"Feasible Rate: {feas_rate:.1f}%\n"
        f"Mean IGD: {mean_igd}\n"
        f"Mean HV: {mean_hv}\n"
        f"Mean Time: {mean_time}"
    )

    if true_pf is None:
        true_pf = get_true_pareto_front(prob_name)

    fig = plt.figure(figsize=(11, 6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2])

    n_obj = f_dim if isinstance(f_dim, int) else 2

    if n_obj == 2:
        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor("#FFFFFF")

        # 1. 绘制 True PF 参考前沿
        if true_pf is not None and true_pf.ndim == 2 and true_pf.shape[1] >= 2:
            sort_idx = np.argsort(true_pf[:, 0])
            sorted_pf = true_pf[sort_idx]
            ax.plot(
                sorted_pf[:, 0],
                sorted_pf[:, 1],
                color="#DC2626",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                label="True PF (Reference)",
                zorder=1,
            )
            ax.scatter(
                sorted_pf[:, 0],
                sorted_pf[:, 1],
                color="#DC2626",
                s=30,
                alpha=0.85,
                edgecolors="#7F1D1D",
                linewidths=0.5,
                zorder=1,
            )

        # 2. 绘制每个 Seed 搜索到的可行非支配解
        has_feas = False
        for run in runs_data:
            seed = run["seed"]
            feas_f = run.get("feas_f", np.empty((0, 2)))
            if len(feas_f) > 0:
                has_feas = True
                ax.scatter(
                    feas_f[:, 0],
                    feas_f[:, 1],
                    alpha=0.75,
                    s=28,
                    label=f"Seed {seed}",
                    zorder=2,
                )

        if not has_feas:
            # 若无可行解，绘制最终种群解以供直观调试
            for run in runs_data:
                seed = run["seed"]
                pop_f = run.get("f", np.empty((0, 2)))
                if len(pop_f) > 0:
                    ax.scatter(
                        pop_f[:, 0],
                        pop_f[:, 1],
                        alpha=0.35,
                        s=18,
                        color="gray",
                        label=f"Seed {seed} (Infeasible)",
                        zorder=2,
                    )

        ax.set_xlabel("$f_1$", fontsize=12, fontweight="bold")
        ax.set_ylabel("$f_2$", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Pareto Front Comparison: {algo_name} vs True PF on {prob_name}",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    elif n_obj == 3:
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        ax.set_facecolor("#FFFFFF")

        # 1. 绘制 True PF 参考前沿
        if true_pf is not None and true_pf.ndim == 2 and true_pf.shape[1] >= 3:
            cast(Axes3D, ax).scatter(
                true_pf[:, 0],
                true_pf[:, 1],
                true_pf[:, 2],
                color="#DC2626",
                s=25,
                alpha=0.7,
                edgecolors="#7F1D1D",
                linewidths=0.4,
                label="True PF (Reference)",
                zorder=1,
            )

        # 2. 绘制每个 Seed 解集
        has_feas = False
        for run in runs_data:
            seed = run["seed"]
            feas_f = run.get("feas_f", np.empty((0, 3)))
            if len(feas_f) > 0:
                has_feas = True
                cast(Axes3D, ax).scatter(
                    feas_f[:, 0],
                    feas_f[:, 1],
                    feas_f[:, 2],
                    alpha=0.7,
                    s=22,
                    label=f"Seed {seed}",
                    zorder=2,
                )

        if not has_feas:
            for run in runs_data:
                seed = run["seed"]
                pop_f = run.get("f", np.empty((0, 3)))
                if len(pop_f) > 0:
                    cast(Axes3D, ax).scatter(
                        pop_f[:, 0],
                        pop_f[:, 1],
                        pop_f[:, 2],
                        alpha=0.35,
                        s=16,
                        color="gray",
                        label=f"Seed {seed} (Infeasible)",
                        zorder=2,
                    )

        ax.set_xlabel("$f_1$", fontsize=10, fontweight="bold")
        ax.set_ylabel("$f_2$", fontsize=10, fontweight="bold")
        ax.set_zlabel("$f_3$", fontsize=10, fontweight="bold")
        ax.set_title(
            f"3D Pareto Front: {algo_name} vs True PF on {prob_name}",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.legend(loc="upper right", fontsize=7)

    else:
        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor("#FFFFFF")
        for run in runs_data:
            feas_f = run.get("feas_f", np.empty((0, n_obj)))
            for sol in feas_f:
                ax.plot(range(1, n_obj + 1), sol, alpha=0.35, color="#2563EB")
        ax.set_xlabel("Objective Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{n_obj}-Obj Parallel Coordinates: {algo_name} on {prob_name}",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.grid(True, linestyle="--", alpha=0.5)

    # 4. 右侧增加参数与统计面板说明 (Text Panel)
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    ax_info.text(
        0.05,
        0.95,
        info_text,
        transform=ax_info.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#f8f9fa",
            edgecolor="#ced4da",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 [Plot] Pareto 对比图已保存: {output_file.name}")


def update_overall_summary_csv(
    csv_path: Path, new_summary_rows: list[dict[str, Any]]
) -> pd.DataFrame:
    """生成或更新 overall_summary.csv 文件，对齐 results/comparison/overall_summary.csv 格式。"""
    columns = [
        "Category",
        "Problem",
        "Algorithm",
        "Runs",
        "IGD_Mean",
        "IGD_Std",
        "HV_Mean",
        "HV_Std",
        "Feasible_Mean",
        "Time_Mean_s",
    ]

    new_df = pd.DataFrame(new_summary_rows)

    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            # 移除将被本次新运行更新的记录 (按 Algorithm + Problem 组合键)
            keys_to_update = set(zip(new_df["Algorithm"], new_df["Problem"]))
            kept_rows = [
                row
                for _, row in existing_df.iterrows()
                if (row["Algorithm"], row["Problem"]) not in keys_to_update
            ]
            merged_df = pd.concat([pd.DataFrame(kept_rows), new_df], ignore_index=True)
        except Exception:
            merged_df = new_df
    else:
        merged_df = new_df

    # 确保列顺序
    for col in columns:
        if col not in merged_df.columns:
            merged_df[col] = np.nan
    merged_df = merged_df[columns]

    merged_df.to_csv(csv_path, index=False)
    return merged_df


def run_algorithm_on_problem(
    algo_name: str,
    prob_name: str,
    runs: int,
    pop_size: int,
    max_evals: int,
    base_seed: int,
    algo_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """对特定算法和问题执行 N 次运行，平铺落盘 NPZ 并返回汇总统计与详细运行数据。"""
    category = classify_problem(prob_name)
    prob_cls = PROBLEM_REGISTRY[prob_name]
    algo_cls = ALGORITHM_REGISTRY[algo_name]

    raw_prob = prob_cls()
    adapter = PymooProblemAdapter(raw_prob, max_evals=max_evals)
    ref_front, ref_point = get_reference_front_and_point(raw_prob, adapter)

    runs_data: list[dict[str, Any]] = []

    print(f"\n---> 运行 [{algo_name}] 在 [{prob_name}] ({category}) 上 ({runs} 次独立运行)...")

    for run_idx in range(runs):
        seed = base_seed + run_idx * 100
        algo = algo_cls(population_size=pop_size, seed=seed)

        t0 = time.perf_counter()
        result = algo.run(adapter)
        elapsed = time.perf_counter() - t0

        feas_pop = result.feasible_nondominated
        n_feasible = len(feas_pop.x)

        igd_val = (
            calculate_igd(feas_pop.f, ref_front)
            if ref_front is not None and len(ref_front) > 0
            else float("nan")
        )
        hv_val = calculate_hv(feas_pop.f, ref_point)

        # 1. 结果不分文件夹保存，直接存放在 algo_dir 根目录下
        npz_file = algo_dir / f"{prob_name}_seed_{seed}.npz"
        np.savez_compressed(
            npz_file,
            x=result.population.x,
            f=result.population.f,
            cv=result.population.cv,
            g=result.population.g if result.population.g is not None else np.empty(0),
            h=result.population.h if result.population.h is not None else np.empty(0),
            feas_x=feas_pop.x,
            feas_f=feas_pop.f,
            feas_cv=feas_pop.cv,
            eval_count=result.eval_count,
            elapsed_time=elapsed,
            igd=igd_val,
            hv=hv_val,
            n_feasible=n_feasible,
        )

        run_record = {
            "seed": seed,
            "eval_count": result.eval_count,
            "elapsed_time": elapsed,
            "n_feasible": n_feasible,
            "igd": igd_val,
            "hv": hv_val,
            "feas_f": feas_pop.f,
            "f": result.population.f,
            "pop_size": pop_size,
            "x_dim": result.population.x.shape[1] if result.population.x.ndim == 2 else "N/A",
            "f_dim": adapter.n_obj,
            "max_evals": max_evals,
        }
        runs_data.append(run_record)

        igd_str = f"{igd_val:.4e}" if not np.isnan(igd_val) else "N/A"
        print(
            f"  [Run {run_idx + 1:02d}/{runs:02d} | Seed {seed}] "
            f"FE={result.eval_count}/{max_evals} | 可行解={n_feasible} | "
            f"IGD={igd_str} | HV={hv_val:.4f} | 时间={elapsed:.2f}s"
        )

    # 2. 生成单个问题的 Pareto Plot 图表，直接保存在 algo_dir 目录下
    plot_file = algo_dir / f"{prob_name}_pareto.png"
    plot_problem_pareto(
        algo_name=algo_name,
        prob_name=prob_name,
        category=category,
        runs_data=runs_data,
        output_file=plot_file,
        true_pf=ref_front,
    )

    # 3. 计算汇总统计指标
    igds = [r["igd"] for r in runs_data if not np.isnan(r["igd"])]
    hvs = [r["hv"] for r in runs_data if not np.isnan(r["hv"])]
    feas_counts = [r["n_feasible"] for r in runs_data]
    times = [r["elapsed_time"] for r in runs_data]

    summary_row = {
        "Category": category,
        "Problem": prob_name,
        "Algorithm": algo_name,
        "Runs": runs,
        "IGD_Mean": float(np.mean(igds)) if igds else float("nan"),
        "IGD_Std": float(np.std(igds)) if igds else 0.0,
        "HV_Mean": float(np.mean(hvs)) if hvs else 0.0,
        "HV_Std": float(np.std(hvs)) if hvs else 0.0,
        "Feasible_Mean": float(np.mean(feas_counts)) if feas_counts else 0.0,
        "Time_Mean_s": float(np.mean(times)) if times else 0.0,
    }

    return summary_row, runs_data


def main() -> None:
    """通用算法单机测试 CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="通用算法单机测试与结果生成脚本 (自动落盘 NPZ、生成 overall_summary.csv 与 Pareto 图表)"
    )
    parser.add_argument(
        "--algorithms",
        "--algorithm",
        nargs="+",
        default=["DSOCOL"],
        help=f"选择测试算法，支持: {list(ALGORITHM_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--problems",
        "--problem",
        nargs="+",
        default=["C1DTLZ1"],
        help=f"选择测试问题或分类名 (如 C1DTLZ1 或 C-DTLZs 或 ALL)，支持: {list(PROBLEM_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--categories",
        "--category",
        nargs="+",
        default=None,
        help=f"选择 Benchmark 分类批量运行，支持: {list(BENCHMARK_CATEGORIES.keys())} 或 ALL",
    )
    parser.add_argument(
        "--runs",
        "--n-runs",
        type=int,
        default=1,
        help="独立重复运行次数 (默认 1 次快速验证，基准测试可指定 5 或 30)",
    )
    parser.add_argument(
        "--population-size", "--pop-size", type=int, default=100, help="种群规模"
    )
    parser.add_argument(
        "--max-evals", type=int, default=30000, help="最大函数评估次数 FEmax"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子基数")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="结果输出根目录 (默认: results)",
    )
    args = parser.parse_args()

    # 校验算法输入
    for algo_name in args.algorithms:
        if algo_name not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"未知算法 '{algo_name}'，可选算法: {list(ALGORITHM_REGISTRY.keys())}"
            )

    # 展开并解析所有测试问题
    resolved_problems = resolve_problems(
        problems=args.problems, categories=args.categories
    )

    results_base = Path(args.results_dir)

    print("==================================================")
    print(f"🚀 开始算法实验测试: 算法={args.algorithms}")
    print(f"📋 包含测试问题 ({len(resolved_problems)} 个): {resolved_problems}")
    print(
        f"⚙️ 参数: 运行次数={args.runs}, 种群规模={args.population_size}, FEmax={args.max_evals}, 基准 Seed={args.seed}"
    )
    print("==================================================")

    for algo_name in args.algorithms:
        # 结果保存在 results 目录下按算法小写命名的文件夹中 (如 results/dsocol)
        algo_dir = results_base / algo_name.lower()
        algo_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 算法 [{algo_name}] 结果输出目录: {algo_dir}")

        new_summary_rows: list[dict[str, Any]] = []

        for prob_name in resolved_problems:
            summary_row, _ = run_algorithm_on_problem(
                algo_name=algo_name,
                prob_name=prob_name,
                runs=args.runs,
                pop_size=args.population_size,
                max_evals=args.max_evals,
                base_seed=args.seed,
                algo_dir=algo_dir,
            )
            new_summary_rows.append(summary_row)

        # 在该算法目录下更新/生成 overall_summary.csv
        csv_file = algo_dir / "overall_summary.csv"
        df_summary = update_overall_summary_csv(csv_file, new_summary_rows)

        print("\n--------------------------------------------------")
        print(f"📄 [{algo_name}] 汇总统计表已生成: {csv_file}")
        print(df_summary.to_string(index=False))
        print("--------------------------------------------------")

    print("\n==================================================")
    print("🎉 所有算法与 Benchmark 测试运行及图表、CSV 生成完成！")
    print("==================================================")


if __name__ == "__main__":
    main()

