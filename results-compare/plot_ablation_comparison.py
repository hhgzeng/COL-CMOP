"""消融实验图表绘制脚本 (results-compare/plot_ablation_comparison.py)。

读取 results-compare 目录下的 NPZ 实验数据，绘制三组消融对比图：
- 不输出 PDF 文件，仅保存高清 PNG 图片。
- 将对比算法画在同一个图中进行直观对比。

- Fig 1: DSOCOL1 vs. DSOCOL on LIR-CMOP3 (同图种群散点分布对比)
- Fig 2: DSOCOL3 vs. DSOCOL on DC1-DTLZ3 & LIR-CMOP11 (同图 IGD 收敛对比)
- Fig 3: DSOCOL4 vs. DSOCOL on LIR-CMOP10 & DC3-DTLZ1 (同图 IGD 收敛对比)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录添加到 sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from experiments.config import PROBLEM_REGISTRY
except ImportError:
    PROBLEM_REGISTRY = {}

# 设置学术论文绘图全局参数
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

RESULTS_DIR = Path(__file__).parent.resolve()


def find_npz_file(algo_name: str, prob_name: str, seed: int = 42) -> Path | None:
    """寻找指定算法、问题与 Seed 的 NPZ 结果文件。"""
    candidates = list(RESULTS_DIR.glob(f"**/{algo_name}/**/{prob_name}/run_seed_{seed}.npz"))
    if not candidates:
        candidates = list(RESULTS_DIR.glob(f"**/{algo_name}/**/{prob_name}/*.npz"))

    return candidates[0] if candidates else None


def get_true_pareto_front(prob_name: str) -> np.ndarray | None:
    """获取测试问题的真实 Pareto Front 参考数据。"""
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


# ==============================================================================
# 第一组绘图: DSOCOL1 vs DSOCOL on LIR-CMOP3 (分图左右对比, 对应论文 Fig. 3)
# ==============================================================================
def plot_group1_population(seed: int = 42):
    """绘制第一组：DSOCOL1 与 DSOCOL 在 LIR-CMOP3 上分图左右并列对比。"""
    prob_name = "LIRCMOP3"
    npz_1 = find_npz_file("DSOCOL1", prob_name, seed)
    npz_full = find_npz_file("DSOCOL", prob_name, seed)

    if not npz_1 or not npz_full:
        print(f"⚠️ [Group 1 跳过] 未能找齐 DSOCOL1 或 DSOCOL 在 {prob_name} 上的 NPZ 数据。")
        return

    data_1 = np.load(npz_1)
    data_full = np.load(npz_full)

    feas_f1 = data_1.get("feas_f", data_1["f"])
    feas_ffull = data_full.get("feas_f", data_full["f"])

    ref_pf = get_true_pareto_front(prob_name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    for ax, feas_f, title in zip(
        axes,
        [feas_f1, feas_ffull],
        ["DSOCOL1 on LIR-CMOP3", "DSOCOL on LIR-CMOP3"],
    ):
        ax.set_facecolor("#FFFFFF")
        # 绘制 True PF 虚线
        if ref_pf is not None and len(ref_pf) > 0:
            idx_sort = np.argsort(ref_pf[:, 0])
            ax.plot(
                ref_pf[idx_sort, 0],
                ref_pf[idx_sort, 1],
                "k--",
                linewidth=1.2,
                label="True PF",
                zorder=1,
            )

        # 绘制算法获取的解集散点
        if len(feas_f) > 0:
            ax.scatter(
                feas_f[:, 0],
                feas_f[:, 1],
                c="#808080",
                edgecolors="k",
                s=35,
                alpha=0.85,
                zorder=2,
                label="Populations",
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(r"$f_1$", fontsize=12)
        ax.set_ylabel(r"$f_2$", fontsize=12)
        ax.tick_params(direction="in", top=True, right=True, labelsize=10)

    plt.tight_layout()
    png_path = RESULTS_DIR / "Fig1_DSOCOL1_vs_DSOCOL_LIRCMOP3.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 第一组左右分图消融对比图已生成: {png_path.name}")


# ==============================================================================
# 第二组绘图: DSOCOL3 vs DSOCOL on DC1-DTLZ3 & LIR-CMOP11 (同图收敛对比)
# ==============================================================================
def plot_group2_convergence(seed: int = 42):
    """绘制第二组：DSOCOL3 与 DSOCOL 在 DC1-DTLZ3 和 LIR-CMOP11 上的 IGD 同图收敛对比。"""
    problems = [
        ("DC1DTLZ3", "DC1-DTLZ3"),
        ("LIRCMOP11", "LIR-CMOP11"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    for ax, (prob_name, display_title) in zip(axes, problems):
        ax.set_facecolor("#FFFFFF")
        npz_3 = find_npz_file("DSOCOL3", prob_name, seed)
        npz_full = find_npz_file("DSOCOL", prob_name, seed)

        if not npz_3 or not npz_full:
            print(f"⚠️ [Group 2 部分跳过] 未找齐 {prob_name} 上的 NPZ 数据。")
            continue

        d3 = np.load(npz_3)
        dfull = np.load(npz_full)

        fe_3, igd_3 = d3.get("eval_history"), d3.get("igd_history")
        fe_full, igd_full = dfull.get("eval_history"), dfull.get("igd_history")

        if fe_3 is not None and len(fe_3) > 0 and igd_3 is not None:
            step3 = max(1, len(fe_3) // 15)
            ax.plot(
                fe_3 / 1e5,
                igd_3,
                "s-",
                color="#D97706",
                markerfacecolor="none",
                markevery=step3,
                label="DSOCOL3 (w/o COL)",
                linewidth=1.5,
            )

        if fe_full is not None and len(fe_full) > 0 and igd_full is not None:
            step_full = max(1, len(fe_full) // 15)
            ax.plot(
                fe_full / 1e5,
                igd_full,
                "*-",
                color="#2563EB",
                markevery=step_full,
                label="DSOCOL (Full)",
                linewidth=1.5,
            )

        ax.set_title(f"IGD convergence on {display_title}", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Evaluations $\times 10^5$", fontsize=11)
        ax.set_ylabel("IGD", fontsize=11)
        ax.legend(loc="upper right", frameon=True, edgecolor="gray", fontsize=9)
        ax.tick_params(direction="in", top=True, right=True, labelsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    png_path = RESULTS_DIR / "Fig2_DSOCOL3_vs_DSOCOL_Convergence.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 第二组同图收敛图已生成: {png_path.name}")


# ==============================================================================
# 第三组绘图: DSOCOL4 vs DSOCOL on LIR-CMOP10 & DC3-DTLZ1 (同图收敛对比)
# ==============================================================================
def plot_group3_convergence(seed: int = 42):
    """绘制第三组：DSOCOL4 与 DSOCOL 在 LIR-CMOP10 和 DC3-DTLZ1 上的 IGD 同图收敛对比。"""
    problems = [
        ("LIRCMOP10", "LIR-CMOP10"),
        ("DC3DTLZ1", "DC3-DTLZ1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    for ax, (prob_name, display_title) in zip(axes, problems):
        ax.set_facecolor("#FFFFFF")
        npz_4 = find_npz_file("DSOCOL4", prob_name, seed)
        npz_full = find_npz_file("DSOCOL", prob_name, seed)

        if not npz_4 or not npz_full:
            print(f"⚠️ [Group 3 部分跳过] 未找齐 {prob_name} 上的 NPZ 数据。")
            continue

        d4 = np.load(npz_4)
        dfull = np.load(npz_full)

        fe_4, igd_4 = d4.get("eval_history"), d4.get("igd_history")
        fe_full, igd_full = dfull.get("eval_history"), dfull.get("igd_history")

        if fe_4 is not None and len(fe_4) > 0 and igd_4 is not None:
            step4 = max(1, len(fe_4) // 15)
            ax.plot(
                fe_4 / 1e5,
                igd_4,
                "o-",
                color="#8B5CF6",
                markerfacecolor="none",
                markevery=step4,
                label="DSOCOL4 (w/o Trend)",
                linewidth=1.5,
            )

        if fe_full is not None and len(fe_full) > 0 and igd_full is not None:
            step_full = max(1, len(fe_full) // 15)
            ax.plot(
                fe_full / 1e5,
                igd_full,
                "s-",
                color="#2563EB",
                markevery=step_full,
                label="DSOCOL (Full)",
                linewidth=1.5,
            )

        ax.set_title(f"IGD convergence on {display_title}", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Evaluations $\times 10^5$", fontsize=11)
        ax.set_ylabel("IGD", fontsize=11)
        ax.legend(loc="upper right", frameon=True, edgecolor="gray", fontsize=9)
        ax.tick_params(direction="in", top=True, right=True, labelsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    png_path = RESULTS_DIR / "Fig3_DSOCOL4_vs_DSOCOL_Convergence.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 第三组同图收敛图已生成: {png_path.name}")


def main():
    parser = argparse.ArgumentParser(description="消融图表自动绘制脚本 (仅保存 PNG)")
    parser.add_argument("--seed", type=int, default=42, help="要绘制数据的随机 Seed (默认 42)")
    args = parser.parse_args()

    print("==================================================")
    print("开始生成三组消融实验同图对比图表 (仅 PNG 格式)...")
    print(f"搜索目录: {RESULTS_DIR}")
    print("==================================================")

    plot_group1_population(seed=args.seed)
    plot_group2_convergence(seed=args.seed)
    plot_group3_convergence(seed=args.seed)

    print("==================================================")
    print("图表绘制完成！生成的 PNG 图像已存入 results-compare 目录。")


if __name__ == "__main__":
    main()
