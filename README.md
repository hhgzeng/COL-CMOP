# COL-CMOP 算法复现与统一基准实验框架

论文 *Collaborative Orthogonal Learning for Constrained Multi-Objective Optimization* (Wang et al., IEEE TEVC 2026) 的 Python/Numpy 高效复现与对比实验框架。

项目采用标准的**“公共核心层 / 算法层 / 问题层 / 实验层 / 测试层”**五层架构设计。每个算法独立位于 `algorithms/<算法名>/` 子包中，通过统一的问题适配器 `PymooProblemAdapter` 进行精确的函数评估次数 (FE) 预算控制、约束违反度计算与评估指标统计。

---

## 目录结构说明

```text
COL-CMOP/
├── core/                       # [公共基础设施] 统一接口、数据类型、算子与指标
│   ├── __init__.py
│   ├── schema.py               # EvaluationResult, Population, Result, CMOP Protocol
│   ├── problem.py              # PymooProblemAdapter (封装 pymoo 问题并精确累计 FE)
│   ├── operators.py            # 二元锦标赛选择 (tournament_selection), OperatorGA (SBX+PM)
│   └── metrics.py              # 评价指标计算 (IGD, HV)
│
├── algorithms/                 # [算法实现库] (支持 10 种经典与前沿 CMOEA)
│   ├── __init__.py
│   ├── dsocol/                 # DSOCOL (双群体协同正交学习) 算法包 [核心复现]
│   │   ├── __init__.py
│   │   ├── formulas.py         # 论文公式文件: 汇集公式 (2)--(9) 的所有核心数学逻辑与算子
│   │   └── algorithms.py       # 论文算法流程文件: 汇集 Algorithm 1--5 的所有伪代码流程
│   ├── apsea/                  # APSEA (自适应种群规模进化算法)
│   ├── c3m/                    # C3M (多阶段多种群约束多目标进化算法)
│   ├── cmocso/                 # CMOCSO (约束多目标竞争粒子群优化算法)
│   ├── cmoemt/                 # CMOEMT (约束多目标多任务进化算法)
│   ├── drlos_emcmo/            # DRLOS-EMCMO (动态资源分配与正交搜索算法)
│   ├── dvcea/                  # DVCEA (双向量约束进化算法)
│   ├── im_c_moea_d/            # IM-C-MOEA/D (改进型约束 MOEA/D)
│   ├── lcmea/                  # LCMEA (分层约束多目标进化算法)
│   └── pocea/                  # POCEA (成对偏好约束进化算法)
│
├── problems/                   # [Benchmark 问题集 (Python/pymoo)]
│   ├── cdtlz.py                # 校准后的 C-DTLZ 问题集 (C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4)
│   ├── dascmop.py              # DAS-CMOP 问题集 (DASCMOP1 ~ DASCMOP9)
│   ├── dcdtlz.py               # DC-DTLZ 问题集 (DC1DTLZ1 ~ DC3DTLZ3)
│   └── lircmop.py              # LIR-CMOP 问题集 (LIRCMOP1 ~ LIRCMOP14)
│
├── experiments/                # [实验管理与驱动入口]
│   ├── config.py               # 实验配置对象 (ExperimentConfig) 与算法/问题注册表
│   └── run_experiment.py       # 单次/批量实验主脚本 (生成 NPZ 原始数据与 CSV 统计报表)
│
├── tests/                      # [单元与集成测试套件] (31 个测试用例)
│   ├── test_adapter.py         # PymooProblemAdapter 维度提取与 FE 精准累计测试
│   ├── test_cdtlz.py           # C-DTLZ 问题目标/约束公式数值校准测试
│   ├── test_lircmop.py         # LIR-CMOP 问题定义与求解测试
│   ├── test_dsocol.py          # DSOCOL 运行、FE 终止与 Seed 可重复性测试
│   ├── test_apsea.py           # APSEA 运行、适应度/选择算子测试与 Seed 可重复性测试
│   ├── test_converted_algorithms.py # 8 个转换基线算法运行与 Seed 可重复性测试
│   ├── test_experiments.py     # 实验指标计算、NPZ 数据落盘与 CSV 表格输出测试
│   └── run_test.py             # 通用算法与 Benchmark 测试/演示入口 (支持多算法与多问题, 不保存结果)
│
├── results/                    # [实验结果持久化存储]
│   ├── detailed_runs.csv       # 多 Seed 运行明细表
│   └── summary_metrics.csv     # 算法对比统计汇总表 (mean ± std)
│
├── problems-matlab/            # PlatEMO Benchmark MATLAB 参考源码
├── algorithms-matlab/          # PlatEMO CMOEAs MATLAB 参考源码
├── implementation_plan.md      # 阶段 1 设计方案与对齐规约
└── walkthrough.md              # 项目阶段复现总结与 Walkthrough
```

---

## 算法与 Benchmark 支持清单

### 支持算法 (10 种)
| 算法标识 | 算法全称 / 描述 | 来源 |
| :--- | :--- | :--- |
| **DSOCOL** | Dual-Swarm Orthogonal Collaborative Learning | Wang et al., IEEE TEVC 2026 (核心复现) |
| **APSEA** | Adaptive Population Size Evolutionary Algorithm | PlatEMO / CMOEA Baseline |
| **C3M** | Constrained Multi-Objective Evolutionary Algorithm with Multi-Stage and Multi-Population | PlatEMO / CMOEA Baseline |
| **CMOCSO** | Constrained Multi-Objective Competitive Swarm Optimizer | PlatEMO / CMOEA Baseline |
| **CMOEMT** | Constrained Multi-Objective Evolutionary Multitasking | PlatEMO / CMOEA Baseline |
| **DRLOS-EMCMO**| Dynamic Resource Allocation & Learning-based Orthogonal Search | PlatEMO / CMOEA Baseline |
| **DVCEA** | Dual-Vector Constrained Evolutionary Algorithm | PlatEMO / CMOEA Baseline |
| **IM-C-MOEA/D**| Improved Constrained MOEA/D | PlatEMO / CMOEA Baseline |
| **LCMEA** | Layered Constrained Multi-objective Evolutionary Algorithm | PlatEMO / CMOEA Baseline |
| **POCEA** | Pairwise Preference Constrained Evolutionary Algorithm | PlatEMO / CMOEA Baseline |

### Benchmark 测试集 (4 大系列)
- **C-DTLZ**: C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4
- **DC-DTLZ**: DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3
- **DAS-CMOP**: DASCMOP1 ~ DASCMOP9 (支持自定义 `difficulty` 参数)
- **LIR-CMOP**: LIRCMOP1 ~ LIRCMOP14 (包含复杂/大面积不可行区域约束)

---

## 核心设计与通用协议

### 1. 数据解耦与 Population 规范
- **`Population`**: 仅包含解的公共属性：`x` (决策变量), `f` (目标矩阵), `cv` (约束违反度向量), `g` (不等式约束), `h` (等式约束)。
- **算法私有状态**: 算法独有变量（如 DSOCOL 的速度 $V_1, V_2$、APSEA 的理想点轨迹）全部在算法类内部独立维护，绝不污染公共数据结构。

### 2. 统一问题适配器 `PymooProblemAdapter`
- 封装 pymoo Benchmark，自动提供变量上下界 `lower`/`upper` 与维度 `n_var`/`n_obj`。
- **自动 FE 累加**：每次调用 `evaluate(x)` 时自动且精确记录 `eval_count` (FE)。
- **约束违反度统一**：一元化计算 $CV = \sum \max(0, G) + \sum \max(0, |H| - \text{tol})$。

---

## 快速开始与使用指南

### 环境依赖
项目推荐使用 Python 3.10+，可通过 `uv` 或标准 `pip` 管理依赖：
```bash
uv sync
```

### 1. 运行单元与集成测试 (31 个测试用例)
运行全套自动化测试，校验系统基础设施与算法可重复性：
```bash
uv run python -m unittest discover tests
```

### 2. 运行算法与 Benchmark 演示测试 (不保存文件)
- **单算法测试**：
  ```bash
  uv run python -m tests.run_test --algorithm DSOCOL --problem C1DTLZ1 --max-evals 30000 --seed 42
  ```
- **多算法与多 Benchmark 灵活组合测试**（支持 10 种算法与 4 大 Benchmark 任意组合）：
  ```bash
  uv run python -m tests.run_test --algorithms DSOCOL APSEA C3M CMOCSO --problems C1DTLZ1 LIRCMOP1 --max-evals 30000
  ```

### 3. 运行批量对比实验与导出报表
使用 `run_experiment.py` 驱动多算法、多 Seed 独立重复实验：
```bash
PYTHONPATH=. uv run python experiments/run_experiment.py \
  --algorithms DSOCOL APSEA C3M CMOCSO \
  --problems C1DTLZ1 LIRCMOP1 \
  --n-runs 30 \
  --max-evals 100000 \
  --pop-size 100 \
  --results-dir results
```
运行完成后，将在 `results/` 目录下自动生成：
- `summary_metrics.csv`：包含 `Feasible_Ratio`, `Mean_Feasible_ND`, `IGD (mean ± std)`, `HV (mean ± std)`, `Time_Mean_Sec` 的统计对照表。
- `detailed_runs.csv`：包含每次 Seed 运行的明细记录。
- `<Algorithm>/<Problem>/run_seed_<seed>.npz`：保存每次运行原始解集与目标空间的压缩二进制数据。
