"""评价指标模块，包含 IGD (反向世代距离) 与 HV (超体积) 计算。"""
from __future__ import annotations

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from core.schema import Array


def calculate_igd(points: Array | None, ref_front: Array) -> float:
    """计算非支配解集 points 相对于参考 Pareto 前沿 ref_front 的 IGD 值。

    Args:
        points: 算法求得的可行非支配解目标矩阵 (N, M)。
        ref_front: 真实 Pareto 前沿点集矩阵 (K, M)。

    Returns:
        IGD 值 (越小越好)。若 points 为空则返回 nan。
    """
    if points is None or len(points) == 0:
        return float("nan")
    indicator = IGD(ref_front)
    val = indicator(points)
    return float("nan") if val is None else float(val)


def calculate_hv(points: Array | None, ref_point: Array) -> float:
    """计算非支配解集 points 相对于参考点 ref_point 的超体积 HV。

    Args:
        points: 算法求得的可行非支配解目标矩阵 (N, M)。
        ref_point: 超体积计算参考点向量 (M,)。

    Returns:
        HV 值 (越大越好)。若 points 为空则返回 0.0。
    """
    if points is None or len(points) == 0:
        return 0.0
    indicator = HV(ref_point=ref_point)
    val = indicator(points)
    return 0.0 if val is None else float(val)
