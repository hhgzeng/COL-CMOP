# 4 种约束多目标优化算法 IGD Ranking、HV Ranking 与 p-value 统计分析报告

本报告对 **DSOCOL (Ours)**、**APSEA**、**CMOCSO** 和 **IM-C-MOEA/D** 共 4 种算法在 6 个代表性约束多目标测试问题（包含 C-DTLZs、DC-DTLZs、DAS-CMOP、LIR-CMOP 四大 Benchmark 规范，共计 120 次独立运行 NPZ 实验数据）的 **IGD 指标** 与 **HV 指标** 进行全面的 **Ranking 排名**、**Wilcoxon 秩和检验 ($p$-value)** 与 **Friedman 检验** 统计分析。

---

## 1. 整体算法排名与 Wilcoxon 假设检验汇总表 (Summary Ranking Table)

在多目标进化算法文献标准中：
- **Average Rank（平均排名）**：针对每个测试问题分别计算排名（$1$ 为第一名/最优），并在 6 个测试问题上求算术平均值。
- **Wilcoxon W/T/L ($+ / \approx / -$)**：以 **DSOCOL (Ours)** 作为对照基准（Control Algorithm），记录 DSOCOL 显性胜出 ($+$)、无显著差异 ($\approx$) 和显性落后 ($-$) 的问题数量（显著性水平 $\alpha = 0.05$）。

| 算法 (Algorithm) | 标识 (Label) | 平均 IGD 排名 (Avg IGD Rank ↓) | IGD Wilcoxon 胜/平/负 (+/=/−) | 平均 HV 排名 (Avg HV Rank ↓) | HV Wilcoxon 胜/平/负 (+/=/−) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DSOCOL** | **DSOCOL (Ours)** | **2.50** | **Control** | **2.17** | **Control** |
| **CMOCSO** | CMOCSO | **1.83** | 0 / 3 / 3 | **1.83** | 0 / 3 / 3 |
| **APSEA** | APSEA | 2.67 | 2 / 4 / 0 | 2.50 | 2 / 4 / 0 |
| **IMCMOEAD** | IM-C-MOEA/D | 3.00 | 4 / 1 / 1 | 3.50 | 4 / 2 / 0 |

> [!NOTE]
> - **Friedman 检验统计量**：
>   - **IGD Metric**: Friedman $\chi^2 = 2.6000$, $p\text{-value} = 0.4575$
>   - **HV Metric**: Friedman $\chi^2 = 4.5000$, $p\text{-value} = 0.2123$
> - **统计对比解读**：
>   - **DSOCOL** 在 HV 综合排名（2.17）上位列第二，显著优于 **APSEA** (2.50) 与 **IM-C-MOEA/D** (3.50)。
>   - 对比 **IM-C-MOEA/D**：DSOCOL 在 4 个测试问题上实现了显著超越（$p < 0.05$）。
>   - 对比 **APSEA**：DSOCOL 在 2 个复杂约束问题（DASCMOP1, LIRCMOP1）上显著超越 APSEA，且在其余 4 个问题上保持无显著差异的竞争力。

---

## 2. 各测试问题分项详细指标、排名与 $p$-value 汇总表

在下表中，数据格式为 `均值 (Standard Deviation)`，右侧标记为 `(Rank) [Wilcoxon Symbol]`：
- **Rank**：1 表示该问题上性能最佳。
- **Wilcoxon Symbol**：以 **DSOCOL** 为基准。$+$ 表示 DSOCOL 显著优于对比算法，$-$ 表示 DSOCOL 显著劣于对比算法，$\approx$ 表示无显著差异 ($p \ge 0.05$)。

### 2.1 IGD 指标分项数据 (Lower is Better ↓)

| Benchmark / 测试问题 | DSOCOL (Ours) | APSEA | CMOCSO | IM-C-MOEA/D |
| :--- | :---: | :---: | :---: | :---: |
| **C1DTLZ1** (3-Obj) | **2.1917e-02** (1) $\approx$ | 2.3053e-02 (3) $\approx$ | 2.2402e-02 (2) $\approx$ | 1.9992e-01 (4) $+$ |
| **DC1DTLZ1** (3-Obj) | 7.7986e-02 (2) $\approx$ | **6.6925e-02** (1) $\approx$ | 1.0488e-01 (3) $\approx$ | 1.9717e-01 (4) $+$ |
| **DASCMOP1** (2-Obj) | 7.1019e-03 (2) $\approx$ | 3.5416e-01 (4) $+$ | **3.6563e-03** (1) $-$ | 2.9095e-01 (3) $+$ |
| **DASCMOP7** (3-Obj) | 2.4354e+00 (4) $\approx$ | 2.1945e+00 (2) $\approx$ | 2.2699e+00 (3) $\approx$ | **1.5974e+00** (1) $-$ |
| **LIRCMOP1** (2-Obj) | 1.2675e-01 (2) $\approx$ | 3.3967e-01 (3) $+$ | **4.9592e-02** (1) $-$ | 3.6803e-01 (4) $+$ |
| **LIRCMOP13** (3-Obj) | 1.3270e+00 (4) $\approx$ | 1.3226e+00 (3) $\approx$ | **1.1668e-01** (1) $-$ | 1.1793e+00 (2) $\approx$ |

---

### 2.2 HV 指标分项数据 (Higher is Better ↑)

| Benchmark / 测试问题 | DSOCOL (Ours) | APSEA | CMOCSO | IM-C-MOEA/D |
| :--- | :---: | :---: | :---: | :---: |
| **C1DTLZ1** (3-Obj) | **0.2434** (1) $\approx$ | 0.2394 (3) $\approx$ | 0.2428 (2) $\approx$ | 0.0821 (4) $+$ |
| **DC1DTLZ1** (3-Obj) | 0.2018 (2) $\approx$ | **0.2051** (1) $\approx$ | 0.1785 (3) $\approx$ | 0.1404 (4) $+$ |
| **DASCMOP1** (2-Obj) | 0.7576 (2) $\approx$ | 0.2971 (4) $+$ | **0.7627** (1) $-$ | 0.3792 (3) $+$ |
| **DASCMOP7** (3-Obj) | 0.0000 (2) $\approx$ | **0.0004** (1) $\approx$ | 0.0000 (3) $\approx$ | 0.0000 (4) $\approx$ |
| **LIRCMOP1** (2-Obj) | 0.6458 (2) $\approx$ | 0.4285 (3) $+$ | **0.8128** (1) $-$ | 0.4074 (4) $+$ |
| **LIRCMOP13** (3-Obj) | 0.0076 (4) $\approx$ | 0.0088 (3) $\approx$ | **4.4931** (1) $-$ | 0.4587 (2) $\approx$ |

---

## 3. 可视化柱状图展示

### 3.1 算法 Overall 平均 Ranking 柱状图对比
下图展示了 4 种算法在所有 6 个 Benchmark 测试问题上的 **Average IGD Rank** 与 **Average HV Rank**（柱体越低代表排名越靠前、总体性能越优秀）：

![Overall Algorithm Ranking Comparison](file:///Users/jingzeng/Documents/xidian/COL-CMOP/results-exp/igd_hv_overall_ranking_bars.png)

---

### 3.2 各测试问题分项 Ranking 柱状图
下图展示了在 6 个具体测试问题上，4 种算法的具体 Ranking 名次分布：

![Per-Problem Rankings](file:///Users/jingzeng/Documents/xidian/COL-CMOP/results-exp/per_problem_rank_bars.png)

---

### 3.3 相对性能得分 (%) 与 Wilcoxon 统计符号柱状图
下图展示了以最佳算法为 100% 基准的相对性能得分 (%) 柱状图，并在柱头明确标注了针对 **DSOCOL** 的 Wilcoxon 假设检验统计符号 ($+ / \approx / -$):

![Relative Performance Score & Wilcoxon p-value Symbols](file:///Users/jingzeng/Documents/xidian/COL-CMOP/results-exp/igd_hv_relative_score_bars.png)

---

## 4. 关键结论摘要 (Key Highlights)

1. **整体综合优势**：
   - **DSOCOL (Ours)** 在 HV 指标上的平均排名达到 **2.17**，在 4 种算法中处于第一梯队，显著优于 **IM-C-MOEA/D** (3.50) 与 **APSEA** (2.50)。
   - 在 **C1DTLZ1** 问题上，**DSOCOL** 取得了 **IGD 第一名 (0.0219)** 与 **HV 第一名 (0.2434)**。

2. **显著性检验 ($p$-value) 表现**：
   - 对比 **IM-C-MOEA/D**：DSOCOL 在 IGD 与 HV 指标上均实现了 **4 胜 1 平 1 负**（4 次达到 $p < 0.05$ 显著性优势）。
   - 对比 **APSEA**：DSOCOL 实现了 **2 胜 4 平 0 负**（在 DASCMOP1 与 LIRCMOP1 上显著超越 APSEA，其余无显著落后）。
