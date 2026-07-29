"""消融实验对比运行脚本 (experiments/run_ablation_comparison.py)。

与 experiments/run_experiment.py 的 CLI 参数与运行方式一致，支持通过 --algorithms 和 --problems / --categories 指定要运行的算法和测试问题。

使用示例：
    # 默认模式：自动运行论文三组消融对比
    python experiments/run_ablation_comparison.py

    # 方式一：自定义算法与测试问题
    python experiments/run_ablation_comparison.py --algorithms DSOCOL1 DSOCOL --problems LIRCMOP3

    # 方式二：运行第二组对比
    python experiments/run_ablation_comparison.py --algorithms DSOCOL3 DSOCOL --problems DC1DTLZ3 LIRCMOP11

    # 方式三：设置评估次数与独立重复次数
    python experiments/run_ablation_comparison.py --algorithms DSOCOL4 DSOCOL --problems LIRCMOP10 DC3DTLZ1 --max-evals 100000 --n-runs 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 将项目根目录添加到 sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.metrics import calculate_hv, calculate_igd
from core.problem import PymooProblemAdapter
from experiments.config import (
    ALGORITHM_REGISTRY,
    BENCHMARK_CATEGORIES,
    PROBLEM_REGISTRY,
)


def classify_problem(prob_name: str) -> str:
    """按 Benchmark 名称归类。"""
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
    """解析并扩展问题列表，与 run_experiment.py 保持完全一致。"""
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
                        f"未知的测试问题或分类名称: '{item}'。请在 PROBLEM_REGISTRY 或 BENCHMARK_CATEGORIES 中确认。"
                    )

    return list(dict.fromkeys(resolved))


def get_reference_front_and_point(
    problem_instance, adapter: PymooProblemAdapter
) -> tuple[np.ndarray | None, np.ndarray]:
    """获取问题对应的参考 Pareto Front 和参考点。"""
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


def run_single_ablation_run(
    algo_name: str,
    prob_name: str,
    seed: int = 42,
    max_evals: int = 100000,
    population_size: int = 100,
    dc_n_var: int | None = None,
    results_dir: str | Path = "results-compare",
) -> dict[str, str | float | int]:
    """执行单次实验并落盘 NPZ 轨迹。"""
    category = classify_problem(prob_name)
    save_path_dir = Path(results_dir) / algo_name / category / prob_name
    save_path_dir.mkdir(parents=True, exist_ok=True)

    prob_cls = PROBLEM_REGISTRY[prob_name]
    algo_cls = ALGORITHM_REGISTRY[algo_name]

    if dc_n_var is not None and prob_name.upper().startswith("DC"):
        try:
            raw_prob = prob_cls(n_var=dc_n_var)
        except TypeError:
            raw_prob = prob_cls()
    else:
        raw_prob = prob_cls()

    adapter = PymooProblemAdapter(raw_prob, max_evals=max_evals)
    algo = algo_cls(population_size=population_size, seed=seed)

    ref_front, ref_point = get_reference_front_and_point(raw_prob, adapter)
    adapter.ref_front = ref_front

    print(
        f"🚀 运行 [{algo_name}] 在 [{prob_name}] (Seed={seed}, FE_max={max_evals}, D={adapter.n_var})..."
    )
    start_time = time.perf_counter()
    result = algo.run(adapter)
    elapsed_time = time.perf_counter() - start_time

    feas_pop = result.feasible_nondominated
    n_feasible = len(feas_pop.x)

    igd_val = calculate_igd(feas_pop.f, ref_front) if ref_front is not None else float("nan")
    hv_val = calculate_hv(feas_pop.f, ref_point)

    eval_history = np.array(result.history.get("fe", []), dtype=float)
    igd_history = np.array(result.history.get("igd", []), dtype=float)

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
        eval_history=eval_history,
        igd_history=igd_history,
        eval_count=result.eval_count,
        elapsed_time=elapsed_time,
        igd=igd_val,
        hv=hv_val,
        n_feasible=n_feasible,
    )

    print(
        f"   └─ 完成: FE={result.eval_count} | 可行解={n_feasible} | "
        f"IGD={igd_val:.4e} | HV={hv_val:.4f} | 耗时={elapsed_time:.2f}s"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="消融对比实验运行程序 (与 run_experiment.py 参数一致，支持 --algorithms 与 --problems)"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="包含的算法列表 (如: --algorithms DSOCOL1 DSOCOL)",
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
        help="包含的问题列表或分类名称 (如: --problems LIRCMOP3 或 DC1DTLZ3 LIRCMOP11)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="独立重复运行次数 (消融对比默认 1 次)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=100000,
        help="最大 FE 预算 (默认 100000)",
    )
    parser.add_argument("--pop-size", type=int, default=100, help="种群规模 (默认 100)")
    parser.add_argument(
        "--dc-n-var",
        type=int,
        default=None,
        help="DC-DTLZ 问题决策变量维数 (默认使用问题标准参数)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42, help="基础随机种子 (默认 42)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results-compare",
        help="结果保存目录 (默认 results-compare)",
    )
    args = parser.parse_args()

    results_dir = ROOT_DIR / args.results_dir

    # 如果用户显式传入了 --algorithms 或 --problems / --categories，按用户传入组组合
    user_problems = resolve_problems(problems=args.problems, categories=args.categories)

    if args.algorithms or user_problems:
        algorithms = args.algorithms if args.algorithms else ["DSOCOL", "DSOCOL1", "DSOCOL3", "DSOCOL4"]
        problems = user_problems if user_problems else ["LIRCMOP3"]
        experiments_plan = []
        for prob in problems:
            for algo in algorithms:
                experiments_plan.append((algo, prob))
    else:
        # 默认模式：自动运行论文三组消融对比
        experiments_plan = [
            # 第一组：DSOCOL1 vs DSOCOL on LIR-CMOP3
            ("DSOCOL1", "LIRCMOP3"),
            ("DSOCOL", "LIRCMOP3"),
            # 第二组：DSOCOL3 vs DSOCOL on DC1-DTLZ3 & LIR-CMOP11
            ("DSOCOL3", "DC1DTLZ3"),
            ("DSOCOL", "DC1DTLZ3"),
            ("DSOCOL3", "LIRCMOP11"),
            ("DSOCOL", "LIRCMOP11"),
            # 第三组：DSOCOL4 vs DSOCOL on LIR-CMOP10 & DC3-DTLZ1
            ("DSOCOL4", "LIRCMOP10"),
            ("DSOCOL", "LIRCMOP10"),
            ("DSOCOL4", "DC3DTLZ1"),
            ("DSOCOL", "DC3DTLZ1"),
        ]

    print("==================================================")
    print(f"开始运行消融实验对比任务 (共 {len(experiments_plan) * args.n_runs} 次运行)...")
    print(
        f"实验配置: max_evals={args.max_evals}, pop_size={args.pop_size}, n_runs={args.n_runs}, base_seed={args.base_seed}"
    )
    print(f"输出目录: {results_dir}")
    print("==================================================")

    summary_list = []
    for algo_name, prob_name in experiments_plan:
        for run_idx in range(args.n_runs):
            seed = args.base_seed + run_idx * 100
            res = run_single_ablation_run(
                algo_name=algo_name,
                prob_name=prob_name,
                seed=seed,
                max_evals=args.max_evals,
                population_size=args.pop_size,
                dc_n_var=args.dc_n_var,
                results_dir=results_dir,
            )
            summary_list.append(res)

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(results_dir / "ablation_comparison_summary.csv", index=False)
    print("\n==================================================")
    print("所有消融对比实验完成！汇总信息如下：")
    print(df_summary.to_string(index=False))
    print(f"已生成报告: {results_dir / 'ablation_comparison_summary.csv'}")
    print("==================================================")


if __name__ == "__main__":
    main()
