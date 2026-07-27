"""方法三：批量查看并绘制每个算法在各 Benchmark 分类下的 NPZ 文件中 Pareto 前沿（feas_f 目标空间分布）。
已增加功能：
1. 自动叠加真实 Pareto 前沿参考点（True PF）。
2. 在图表中自动增加实验参数说明（FE 次数、种群规模 N、变量维数 D、可行解数量、均值 IGD/HV 等）。

用法示例：
    # 绘制 APSEA 在 C-DTLZs 分类下所有 NPZ 文件的 Pareto 前沿并保存图像
    python batch_plot_pareto.py --algo APSEA --category C-DTLZs

    # 批量绘制所有算法和所有 Benchmark 分类的 Pareto 前沿图像
    python batch_plot_pareto.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到 sys.path 以便导入 PROBLEM_REGISTRY
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from experiments.config import PROBLEM_REGISTRY
except ImportError:
    PROBLEM_REGISTRY = {}

BENCHMARK_CATEGORIES = {
    "C-DTLZs": ["C1DTLZ1", "C1DTLZ3", "C2DTLZ2", "C3DTLZ4"],
    "DC-DTLZs": ["DC1DTLZ1", "DC1DTLZ3", "DC2DTLZ1", "DC2DTLZ3", "DC3DTLZ1", "DC3DTLZ3"],
    "DAS-CMOP": [f"DASCMOP{i}" for i in range(1, 10)],
    "LIR-CMOP": [f"LIRCMOP{i}" for i in range(1, 15)],
}

RESULTS_DIR = Path(__file__).parent.resolve()


def classify_problem(prob_name: str) -> str:
    """根据问题名称归类到对应的 Benchmark 大类。"""
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


def ensure_category_structure(results_dir: Path = RESULTS_DIR):
    """确保每个算法目录下都建立了 4 类 Benchmark 的文件夹，并将未归类的测试问题移入对应分类。"""
    if not results_dir.exists():
        return

    algo_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for algo_dir in algo_dirs:
        for cat in BENCHMARK_CATEGORIES.keys():
            cat_dir = algo_dir / cat
            cat_dir.mkdir(exist_ok=True)

        for item in list(algo_dir.iterdir()):
            if item.is_dir() and item.name not in BENCHMARK_CATEGORIES.keys() and not item.name.startswith("."):
                cat = classify_problem(item.name)
                target_dir = algo_dir / cat / item.name
                if not target_dir.exists():
                    item.rename(target_dir)
                    print(f"[整理] 移动 {algo_dir.name}/{item.name} -> {algo_dir.name}/{cat}/{item.name}")


def get_true_pareto_front(prob_name: str) -> np.ndarray | None:
    """尝试获取该测试问题的真实/标准 Pareto 前沿 (True PF)。"""
    if prob_name not in PROBLEM_REGISTRY:
        return None
    try:
        prob_cls = PROBLEM_REGISTRY[prob_name]
        prob_inst = prob_cls()
        ref_pf = prob_inst.pareto_front()
        if ref_pf is not None and len(ref_pf) > 0:
            return ref_pf
    except Exception:
        pass
    return None


def plot_pareto_for_problem(
    algo_name: str,
    category: str,
    prob_name: str,
    npz_files: list[Path],
    output_dir: Path,
    show: bool = False
):
    """读取某个问题的所有 NPZ 文件，包含真实 PF 绘制与详细参数说明面板。"""
    if not npz_files:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. 搜集与解析统计数据信息
    eval_counts, pop_sizes, x_dims, f_dims = [], [], [], []
    feas_counts, igds, hvs, times = [], [], [], []

    for npz_file in npz_files:
        with np.load(npz_file) as data:
            files = data.files
            if "eval_count" in files:
                eval_counts.append(int(data["eval_count"]))
            if "x" in files and data["x"].ndim == 2:
                pop_sizes.append(data["x"].shape[0])
                x_dims.append(data["x"].shape[1])
            if "f" in files and data["f"].ndim == 2:
                f_dims.append(data["f"].shape[1])
            if "n_feasible" in files:
                feas_counts.append(int(data["n_feasible"]))
            elif "feas_f" in files:
                feas_counts.append(len(data["feas_f"]))
            if "igd" in files and not np.isnan(data["igd"]):
                igds.append(float(data["igd"]))
            if "hv" in files and not np.isnan(data["hv"]):
                hvs.append(float(data["hv"]))
            if "elapsed_time" in files:
                times.append(float(data["elapsed_time"]))

    # 汇总计算说明文本
    mean_fe = int(np.mean(eval_counts)) if eval_counts else "N/A"
    pop_size = int(pop_sizes[0]) if pop_sizes else "N/A"
    x_dim = int(x_dims[0]) if x_dims else "N/A"
    f_dim = int(f_dims[0]) if f_dims else "N/A"
    mean_feas = np.mean(feas_counts) if feas_counts else 0
    feas_rate = (mean_feas / pop_size * 100) if (pop_sizes and isinstance(pop_size, int) and pop_size > 0) else 0
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
        f"Independent Runs: {len(npz_files)}\n"
        f"------------------------\n"
        f"--- Performance Stats ---\n"
        f"Feasible Count: {mean_feas:.1f} / {pop_size}\n"
        f"Feasible Rate: {feas_rate:.1f}%\n"
        f"Mean IGD: {mean_igd}\n"
        f"Mean HV: {mean_hv}\n"
        f"Mean Time: {mean_time}"
    )

    # 2. 获取真实 Pareto Front (True PF)
    true_pf = get_true_pareto_front(prob_name)

    # 3. 创建画布 (用右侧布局提供参数面板)
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2])

    n_obj = f_dim if isinstance(f_dim, int) else 2

    # 绘制图形区域
    if n_obj == 2:
        ax = fig.add_subplot(gs[0, 0])

        # A. 绘制 True PF 参考线/点 (使用高对比度鲜艳红色 #DC2626 + 明显边框与高透明度)
        if true_pf is not None and true_pf.shape[1] >= 2:
            sort_idx = np.argsort(true_pf[:, 0])
            sorted_pf = true_pf[sort_idx]
            ax.plot(sorted_pf[:, 0], sorted_pf[:, 1], color="#DC2626", linestyle="--", linewidth=2.0, alpha=0.9, label="True PF (Reference)", zorder=1)
            ax.scatter(sorted_pf[:, 0], sorted_pf[:, 1], color="#DC2626", s=30, alpha=0.85, edgecolors="#7F1D1D", linewidths=0.5, zorder=1)

        # B. 绘制各个 Seed 求得的可行解
        for npz_file in npz_files:
            seed = npz_file.stem.split("_")[-1]
            with np.load(npz_file) as data:
                feas_f = data.get("feas_f", data.get("f", np.empty((0, 2))))
                if len(feas_f) > 0:
                    ax.scatter(feas_f[:, 0], feas_f[:, 1], alpha=0.7, s=25, label=f"Seed {seed}", zorder=2)

        ax.set_xlabel("$f_1$", fontsize=12)
        ax.set_ylabel("$f_2$", fontsize=12)
        ax.set_title(f"Pareto Front Comparison: {algo_name} vs True PF on {prob_name}", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)

    elif n_obj == 3:
        ax = fig.add_subplot(gs[0, 0], projection='3d')

        # A. 绘制 True PF 参考散点 (使用高对比度鲜艳红色 #DC2626)
        if true_pf is not None and true_pf.shape[1] >= 3:
            ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], color="#DC2626", s=25, alpha=0.7, edgecolors="#7F1D1D", linewidths=0.4, label="True PF (Reference)", zorder=1)

        # B. 绘制各 Seed 解集
        for npz_file in npz_files:
            seed = npz_file.stem.split("_")[-1]
            with np.load(npz_file) as data:
                feas_f = data.get("feas_f", data.get("f", np.empty((0, 3))))
                if len(feas_f) > 0:
                    ax.scatter(feas_f[:, 0], feas_f[:, 1], feas_f[:, 2], alpha=0.7, s=20, label=f"Seed {seed}", zorder=2)

        ax.set_xlabel("$f_1$", fontsize=10)
        ax.set_ylabel("$f_2$", fontsize=10)
        ax.set_zlabel("$f_3$", fontsize=10)
        ax.set_title(f"3D Pareto Front: {algo_name} vs True PF on {prob_name}", fontsize=13)
        ax.legend(loc='upper right', fontsize=7)

    else:
        ax = fig.add_subplot(gs[0, 0])
        for npz_file in npz_files:
            with np.load(npz_file) as data:
                feas_f = data.get("feas_f", data.get("f", np.empty((0, n_obj))))
                for sol in feas_f:
                    ax.plot(range(1, n_obj + 1), sol, alpha=0.3, color='blue')
        ax.set_xlabel("Objective Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(f"{n_obj}-Obj Parallel Coordinates: {algo_name} on {prob_name}", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.5)

    # 4. 右侧增加参数与统计面板说明 (Text Panel)
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    ax_info.text(
        0.05, 0.95, info_text,
        transform=ax_info.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
    )

    plt.tight_layout()
    save_path = plots_dir / f"{prob_name}_pareto.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ 已生成高信息量图像 (包含 True PF 与参数说明): {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def batch_plot_category(algo_name: str, category: str, results_dir: Path = RESULTS_DIR, show: bool = False):
    """批量绘制某算法下某 Benchmark 分类下的所有 NPZ 文件 Pareto 图形。"""
    cat_dir = results_dir / algo_name / category
    if not cat_dir.exists():
        return

    prob_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir() and d.name != "plots"])
    if not prob_dirs:
        return

    print(f"\n🎨 批量绘制 Pareto 前沿: 算法 [{algo_name}] | Benchmark 分类 [{category}]")
    for prob_dir in prob_dirs:
        npz_files = sorted(list(prob_dir.glob("*.npz")))
        if npz_files:
            plot_pareto_for_problem(algo_name, category, prob_dir.name, npz_files, cat_dir, show=show)


def main():
    parser = argparse.ArgumentParser(description="【方法三】批量查看并绘制每个算法在各 Benchmark 分类下的 NPZ 文件 Pareto 前沿图 (含 True PF 与参数面板)。")
    parser.add_argument("--algo", type=str, default=None, help="指定要查看绘图的算法名称 (如 APSEA, DSOCOL)。若不指定则扫描所有算法。")
    parser.add_argument("--category", type=str, default=None, choices=list(BENCHMARK_CATEGORIES.keys()), help="指定 Benchmark 分类 (C-DTLZs, DC-DTLZs, DAS-CMOP, LIR-CMOP)。")
    parser.add_argument("--show", action="store_true", help="是否在屏幕上弹出显示图形窗口。")

    args = parser.parse_args()

    ensure_category_structure()

    algo_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    algo_list = [args.algo] if args.algo else [d.name for d in algo_dirs]
    cat_list = [args.category] if args.category else list(BENCHMARK_CATEGORIES.keys())

    for algo in algo_list:
        for cat in cat_list:
            batch_plot_category(algo, cat, show=args.show)


if __name__ == "__main__":
    main()
