"""pymoo 问题的统一适配器模块。"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem as PymooProblem

from core.schema import Array, EvaluationResult


class PymooProblemAdapter:
    """包装 pymoo Problem，使其适配统一 CMOP 接口并精准控制 FE。"""

    def __init__(
        self,
        problem: PymooProblem,
        max_evals: int = 100000,
        equality_tolerance: float = 1e-4,
    ) -> None:
        """初始化 pymoo 问题适配器。

        Args:
            problem: 待适配的 pymoo Problem 实例。
            max_evals: 最大函数评估次数 FEmax。
            equality_tolerance: 等式约束的容许误差界限。
        """
        self.raw_problem = problem
        self.max_evals = max_evals
        self.equality_tolerance = equality_tolerance
        self._eval_count = 0

        self.lower = np.asarray(problem.xl, dtype=float)
        self.upper = np.asarray(problem.xu, dtype=float)
        self.n_var = int(problem.n_var)
        self.n_obj = int(problem.n_obj)

    @property
    def eval_count(self) -> int:
        """当前累计函数评估次数 (FE)。"""
        return self._eval_count

    def reset_eval_count(self) -> None:
        """重置评估次数计数器。"""
        self._eval_count = 0

    def evaluate(self, x: Array) -> EvaluationResult:
        """评估决策变量矩阵 x。

        Args:
            x: 决策变量矩阵 (N, D) 或单解向量 (D,)。

        Returns:
            EvaluationResult 包含 f (N, M), g (N, K_g), h (N, K_h), cv (N,)。
        """
        x_arr = np.asarray(x, dtype=float)
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr[None, :]

        n_samples = len(x_arr)

        out = self.raw_problem.evaluate(x_arr, return_values_of=["F", "G", "H"])
        if isinstance(out, tuple):
            # 极少数 pymoo 版本可能返回 tuple (F, G, H)
            f = out[0]
            g = out[1] if len(out) > 1 else None
            h = out[2] if len(out) > 2 else None
        elif isinstance(out, dict):
            f = np.asarray(out["F"], dtype=float)
            g = (
                np.asarray(out["G"], dtype=float).reshape(n_samples, -1)
                if "G" in out and out["G"] is not None and out["G"].size > 0
                else None
            )
            h = (
                np.asarray(out["H"], dtype=float).reshape(n_samples, -1)
                if "H" in out and out["H"] is not None and out["H"].size > 0
                else None
            )
        else:
            f = np.asarray(out, dtype=float)
            g, h = None, None

        # 计算约束违反度 CV
        cv = np.zeros(n_samples, dtype=float)
        if g is not None:
            cv += np.maximum(g, 0.0).sum(axis=1)
        if h is not None:
            cv += np.maximum(np.abs(h) - self.equality_tolerance, 0.0).sum(axis=1)

        # 更新 FE 计数器
        self._eval_count += n_samples

        return EvaluationResult(f=f, g=g, h=h, cv=cv)
