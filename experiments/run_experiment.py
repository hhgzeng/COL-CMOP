"""实验主程序：支持单次运行、单问题批量、按 Benchmark 分类批量运行及结果落盘 (CSV/NPZ)。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.metrics import calculate_hv, calculate_igd
from core.problem import PymooProblemAdapter
from experiments.config import (
    ALGORITHM_REGISTRY,
    BENCHMARK_CATEGORIES,
    PROBLEM_REGISTRY,
    ExperimentConfig,
)


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
    """解析并扩展问题列表，支持通过 Benchmark 分类名称展开该分类下的所有测试问题。

    例如:
        categories=["C-DTLZs"] -> ["C1DTLZ1", "C1DTLZ3", "C2DTLZ2", "C3DTLZ4"]
        problems=["C-DTLZs", "DC1DTLZ1"] -> ["C1DTLZ1", ..., "DC1DTLZ1"]
        categories=["ALL"] -> 展开所有 4 大类 Benchmark 的所有问题
    """
    resolved: list[str] = []

    # 1. 处理 --categories 参数
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

    # 2. 处理 --problems 参数 (如果指定了问题名称或分类名)
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
                        f"未知的测试问题或分类名称: '{item}'。请在 PROBLEM_REGISTRY 或 BENCHMARK_CATEGORIES 中确认。"
                    )

    # 若二者均未指定，默认返回 C1DTLZ1
    if not resolved:
        resolved = ["C1DTLZ1"]

    # 去重保持原有顺序
    return list(dict.fromkeys(resolved))


def clear_old_problem_results(
    algo_name: str, prob_name: str, results_dir: str | Path = "results"
) -> None:
    """在运行新的实验批次前，自动删除该算法与测试问题下的旧 NPZ 运行结果文件。"""
    category = classify_problem(prob_name)
    results_path = Path(results_dir)
    target_dirs = [
        results_path / algo_name / category / prob_name,
        results_path / algo_name / prob_name,
    ]
    for target_dir in target_dirs:
        if target_dir.exists():
            old_npzs = list(target_dir.glob("*.npz"))
            if old_npzs:
                for npz_file in old_npzs:
                    npz_file.unlink()
                print(
                    f"🧹 [自动清理] 已删除 [{algo_name} -> {prob_name}] 下的 {len(old_npzs)} 个旧 NPZ 文件。"
                )


def get_reference_front_and_point(
    problem_instance, adapter: PymooProblemAdapter
) -> tuple[np.ndarray | None, np.ndarray]:
    """获取问题对应的参考 Pareto Front 和超体积 Reference Point。"""
    ref_front = None
    try:
        ref_front = problem_instance.pareto_front()
    except Exception:
        ref_front = None

    if ref_front is not None and len(ref_front) > 0:
        nadir = np.max(ref_front, axis=0)
        ref_point = nadir * 1.1 + 0.1
    else:
        ref_point = np.full(adapter.n_obj, 2.0)

    return ref_front, ref_point


def run_single_run(
    algo_name: str,
    prob_name: str,
    seed: int,
    max_evals: int = 100000,
    population_size: int = 100,
    save_dir: str | Path = "results",
) -> dict[str, float | int | str]:
    """执行单个 Seed 的算法运行，记录耗时与指标并落盘 NPZ 文件。"""
    category = classify_problem(prob_name)
    save_path_dir = Path(save_dir) / algo_name / category / prob_name
    save_path_dir.mkdir(parents=True, exist_ok=True)

    prob_cls = PROBLEM_REGISTRY[prob_name]
    algo_cls = ALGORITHM_REGISTRY[algo_name]

    raw_prob = prob_cls()
    adapter = PymooProblemAdapter(raw_prob, max_evals=max_evals)
    algo = algo_cls(population_size=population_size, seed=seed)

    ref_front, ref_point = get_reference_front_and_point(raw_prob, adapter)

    start_time = time.perf_counter()
    result = algo.run(adapter)
    elapsed_time = time.perf_counter() - start_time

    feas_pop = result.feasible_nondominated
    n_feasible = len(feas_pop.x)

    igd_val = calculate_igd(feas_pop.f, ref_front) if ref_front is not None else float("nan")
    hv_val = calculate_hv(feas_pop.f, ref_point)

    # 保存单次原始数据 NPZ
    npz_path = save_path_dir / f"run_seed_{seed}.npz"
    np.savez_compressed(
        npz_path,
        x=result.population.x,
        f=result.population.f,
        cv=result.population.cv,
        g=result.population.g if result.population.g is not None else np.empty(0),
        h=result.population.h if result.population.h is not None else np.empty(0),
        feas_x=feas_pop.x,
        feas_f=feas_pop.f,
        feas_cv=feas_pop.cv,
        eval_count=result.eval_count,
        elapsed_time=elapsed_time,
        igd=igd_val,
        hv=hv_val,
        n_feasible=n_feasible,
    )

    return {
        "algorithm": algo_name,
        "problem": prob_name,
        "seed": seed,
        "eval_count": result.eval_count,
        "elapsed_time": elapsed_time,
        "n_feasible_nd": n_feasible,
        "igd": igd_val,
        "hv": hv_val,
    }


def run_batch_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """按配置进行多算法、多问题、N 次独立重复实验并生成汇总 CSV/NPZ 报告。"""
    all_metrics: list[dict[str, float | int | str]] = []

    print("==================================================")
    print(f"开始执行实验批次: 算法={config.algorithms}")
    print(f"测试问题列表 ({len(config.problems)} 个): {config.problems}")
    print(
        f"独立重复次数: {config.n_runs}, 最大 FE: {config.max_evals}, 种群规模: {config.population_size}"
    )
    print("==================================================")

    for prob_name in config.problems:
        for algo_name in config.algorithms:
            # 运行新测试前，自动清理该算法与问题下的旧 NPZ 文件
            clear_old_problem_results(algo_name, prob_name, config.results_dir)

            print(
                f"\n---> 运行 [{algo_name}] 在 [{prob_name}] 上 ({config.n_runs} 次独立运行)..."
            )
            for run_idx in range(config.n_runs):
                seed = config.base_seed + run_idx * 100
                metrics = run_single_run(
                    algo_name=algo_name,
                    prob_name=prob_name,
                    seed=seed,
                    max_evals=config.max_evals,
                    population_size=config.population_size,
                    save_dir=config.results_dir,
                )
                all_metrics.append(metrics)
                print(
                    f"  [Run {run_idx + 1:02d}/{config.n_runs:02d} | Seed {seed}] "
                    f"FE={metrics['eval_count']} | 可行解={metrics['n_feasible_nd']} | "
                    f"IGD={metrics['igd']:.4e} | HV={metrics['hv']:.4f} | 时间={metrics['elapsed_time']:.2f}s"
                )

    df_detail = pd.DataFrame(all_metrics)

    # 保存明细与汇总 CSV 文件
    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    grouped = df_detail.groupby(["algorithm", "problem"])
    for key, group in grouped:
        assert isinstance(key, tuple)
        algo, prob = key
        summary_rows.append(
            {
                "Algorithm": algo,
                "Problem": prob,
                "Runs": len(group),
                "Feasible_Ratio": f"{np.mean(group['n_feasible_nd'] > 0):.2%}",
                "Mean_Feasible_ND": f"{group['n_feasible_nd'].mean():.2f} ± {group['n_feasible_nd'].std():.2f}",
                "IGD_Mean": group["igd"].mean(),
                "IGD_Std": group["igd"].std(),
                "HV_Mean": group["hv"].mean(),
                "HV_Std": group["hv"].std(),
                "Time_Mean_Sec": group["elapsed_time"].mean(),
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    print("\n==================================================")
    print(f"实验全部完成！统计汇总已输出到控制台，结果目录为 {out_dir}")
    print(df_summary.to_string(index=False))
    print("==================================================")

    return df_summary


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="CMOP 批量实验主程序 (支持按单个问题或 Benchmark 分类批量运行)")
    parser.add_argument(
        "--algorithms", nargs="+", default=["DSOCOL", "APSEA"], help="包含的算法列表 (如: DSOCOL APSEA)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="按 Benchmark 分类批量运行 (如: C-DTLZs, DC-DTLZs, DAS-CMOP, LIR-CMOP 或 ALL)",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help="包含的问题列表或分类名称 (如: C1DTLZ1 C1DTLZ3 或 C-DTLZs)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="独立重复运行次数 (默认 5 次演示，论文标准 30 次)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=20000,
        help="最大 FE 预算 (默认 20000 演示，论文标准 100000)",
    )
    parser.add_argument("--pop-size", type=int, default=100, help="种群规模")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="结果输出目录"
    )
    args = parser.parse_args()

    # 自动解析将 Benchmark 分类名展开为具体问题列表
    problems = resolve_problems(problems=args.problems, categories=args.categories)

    config = ExperimentConfig(
        population_size=args.pop_size,
        max_evals=args.max_evals,
        n_runs=args.n_runs,
        algorithms=args.algorithms,
        problems=problems,
        results_dir=args.results_dir,
    )
    run_batch_experiment(config)


if __name__ == "__main__":
    main()
