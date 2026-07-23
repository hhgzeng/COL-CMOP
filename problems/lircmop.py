"""LIR-CMOP benchmark problem suite for pymoo.

Reference:
Z. Fan, W. Li, X. Cai, H. Huang, Y. Fang, Y. You, J. Mo, C. Wei, and E. Goodman.
An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions.
Soft Computing, 2019, 23: 12491-12510.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions


class LIRCMOP(Problem):
    """Base class for LIR-CMOP problems."""

    def __init__(self, n_var: int = 30, n_obj: int = 2, n_ieq_constr: int = 2, **kwargs) -> None:
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            vtype=float,
            xl=0.0,
            xu=1.0,
            **kwargs,
        )


class LIRCMOP1(LIRCMOP):
    """LIR-CMOP1 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=2, **kwargs)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        X1 = X[:, 0:1]
        x_odd = X[:, 2::2]
        x_even = X[:, 1::2]

        g1 = np.sum((x_odd - np.sin(0.5 * np.pi * X1)) ** 2, axis=1, keepdims=True)
        g2 = np.sum((x_even - np.cos(0.5 * np.pi * X1)) ** 2, axis=1, keepdims=True)

        f1 = X1 + g1
        f2 = 1.0 - X1**2 + g2

        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.5 - g2) * (0.51 - g2)

        out["F"] = np.hstack([f1, f2])
        out["G"] = np.hstack([c1, c2])

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        f1 = x1 + 0.5
        f2 = 1.0 - x1**2 + 0.5
        return np.column_stack([f1, f2])


class LIRCMOP2(LIRCMOP):
    """LIR-CMOP2 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=2, **kwargs)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        X1 = X[:, 0:1]
        x_odd = X[:, 2::2]
        x_even = X[:, 1::2]

        g1 = np.sum((x_odd - X1) ** 2, axis=1, keepdims=True)
        g2 = np.sum((x_even - X1) ** 2, axis=1, keepdims=True)

        f1 = X1 + g1
        f2 = 1.0 - np.sqrt(X1) + g2

        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.5 - g2) * (0.51 - g2)

        out["F"] = np.hstack([f1, f2])
        out["G"] = np.hstack([c1, c2])

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        f1 = x1 + 0.5
        f2 = 1.0 - np.sqrt(x1) + 0.5
        return np.column_stack([f1, f2])


class LIRCMOP3(LIRCMOP):
    """LIR-CMOP3 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=3, **kwargs)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        X1 = X[:, 0:1]
        x_odd = X[:, 2::2]
        x_even = X[:, 1::2]

        g1 = np.sum((x_odd - X1) ** 2, axis=1, keepdims=True)
        g2 = np.sum((x_even - X1) ** 2, axis=1, keepdims=True)

        f1 = X1 + g1
        f2 = 1.0 - X1**2 + g2

        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.5 - g2) * (0.51 - g2)
        c3 = 0.5 - np.sin(20.0 * np.pi * X1)

        out["F"] = np.hstack([f1, f2])
        out["G"] = np.hstack([c1, c2, c3])

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        mask = np.sin(20.0 * np.pi * x1) >= 0.5
        x1 = x1[mask]
        f1 = x1 + 0.5
        f2 = 1.0 - x1**2 + 0.5
        return np.column_stack([f1, f2])


class LIRCMOP4(LIRCMOP):
    """LIR-CMOP4 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=3, **kwargs)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        X1 = X[:, 0:1]
        x_odd = X[:, 2::2]
        x_even = X[:, 1::2]

        g1 = np.sum((x_odd - X1) ** 2, axis=1, keepdims=True)
        g2 = np.sum((x_even - X1) ** 2, axis=1, keepdims=True)

        f1 = X1 + g1
        f2 = 1.0 - np.sqrt(X1) + g2

        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.5 - g2) * (0.51 - g2)
        c3 = 0.5 - np.sin(20.0 * np.pi * X1)

        out["F"] = np.hstack([f1, f2])
        out["G"] = np.hstack([c1, c2, c3])

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        mask = np.sin(20.0 * np.pi * x1) >= 0.5
        x1 = x1[mask]
        f1 = x1 + 0.5
        f2 = 1.0 - np.sqrt(x1) + 0.5
        return np.column_stack([f1, f2])


def _calc_lircmop_sums(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_var = X.shape[1]
    j_indices = np.arange(2, n_var + 1)
    angles = (0.5 * j_indices / n_var * np.pi) * X[:, 0:1]
    X_sub = X[:, 1:]
    is_odd = j_indices % 2 == 1
    is_even = ~is_odd

    diff_odd = X_sub[:, is_odd] - np.sin(angles[:, is_odd])
    diff_even = X_sub[:, is_even] - np.cos(angles[:, is_even])
    sum1 = np.sum(diff_odd**2, axis=1, keepdims=True)
    sum2 = np.sum(diff_even**2, axis=1, keepdims=True)
    return sum1, sum2


class LIRCMOP5(LIRCMOP):
    """LIR-CMOP5 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=2, **kwargs)

    @staticmethod
    def _constraint(F: np.ndarray) -> np.ndarray:
        p = np.array([1.6, 2.5])
        q = np.array([1.6, 2.5])
        a = np.array([2.0, 2.0])
        b = np.array([4.0, 8.0])
        r = 0.1
        theta = -0.25 * np.pi
        c = np.zeros((F.shape[0], 2))
        for k in range(2):
            term1 = ((F[:, 0] - p[k]) * np.cos(theta) - (F[:, 1] - q[k]) * np.sin(theta)) ** 2 / (a[k] ** 2)
            term2 = ((F[:, 0] - p[k]) * np.sin(theta) + (F[:, 1] - q[k]) * np.cos(theta)) ** 2 / (b[k] ** 2)
            c[:, k] = r - term1 - term2
        return c

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1, sum2 = _calc_lircmop_sums(X)
        gx = 0.7057
        f1 = X[:, 0:1] + 10.0 * sum1 + gx
        f2 = 1.0 - np.sqrt(X[:, 0:1]) + 10.0 * sum2 + gx
        F = np.hstack([f1, f2])

        out["F"] = F
        out["G"] = self._constraint(F)

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        f1 = x1 + 0.7057
        f2 = 1.0 - np.sqrt(x1) + 0.7057
        F = np.column_stack([f1, f2])
        con = self._constraint(F)
        feasible = np.all(con <= 0, axis=1)
        return F[feasible]


class LIRCMOP6(LIRCMOP):
    """LIR-CMOP6 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=2, **kwargs)

    @staticmethod
    def _constraint(F: np.ndarray) -> np.ndarray:
        p = np.array([1.8, 2.8])
        q = np.array([1.8, 2.8])
        a = np.array([2.0, 2.0])
        b = np.array([8.0, 8.0])
        r = 0.1
        theta = -0.25 * np.pi
        c = np.zeros((F.shape[0], 2))
        for k in range(2):
            term1 = ((F[:, 0] - p[k]) * np.cos(theta) - (F[:, 1] - q[k]) * np.sin(theta)) ** 2 / (a[k] ** 2)
            term2 = ((F[:, 0] - p[k]) * np.sin(theta) + (F[:, 1] - q[k]) * np.cos(theta)) ** 2 / (b[k] ** 2)
            c[:, k] = r - term1 - term2
        return c

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1, sum2 = _calc_lircmop_sums(X)
        gx = 0.7057
        f1 = X[:, 0:1] + 10.0 * sum1 + gx
        f2 = 1.0 - X[:, 0:1] ** 2 + 10.0 * sum2 + gx
        F = np.hstack([f1, f2])

        out["F"] = F
        out["G"] = self._constraint(F)

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        f1 = x1 + 0.7057
        f2 = 1.0 - x1**2 + 0.7057
        F = np.column_stack([f1, f2])
        con = self._constraint(F)
        feasible = np.all(con <= 0, axis=1)
        return F[feasible]


class LIRCMOP7(LIRCMOP):
    """LIR-CMOP7 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=3, **kwargs)

    @staticmethod
    def _constraint(F: np.ndarray) -> np.ndarray:
        p = np.array([1.2, 2.25, 3.5])
        q = np.array([1.2, 2.25, 3.5])
        a = np.array([2.0, 2.5, 2.5])
        b = np.array([6.0, 12.0, 10.0])
        r = 0.1
        theta = -0.25 * np.pi
        c = np.zeros((F.shape[0], 3))
        for k in range(3):
            term1 = ((F[:, 0] - p[k]) * np.cos(theta) - (F[:, 1] - q[k]) * np.sin(theta)) ** 2 / (a[k] ** 2)
            term2 = ((F[:, 0] - p[k]) * np.sin(theta) + (F[:, 1] - q[k]) * np.cos(theta)) ** 2 / (b[k] ** 2)
            c[:, k] = r - term1 - term2
        return c

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1, sum2 = _calc_lircmop_sums(X)
        gx = 0.7057
        f1 = X[:, 0:1] + 10.0 * sum1 + gx
        f2 = 1.0 - np.sqrt(X[:, 0:1]) + 10.0 * sum2 + gx
        F = np.hstack([f1, f2])

        out["F"] = F
        out["G"] = self._constraint(F)

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        r1 = x1 + 0.7057
        r2 = 1.0 - np.sqrt(x1) + 0.7057
        R = np.column_stack([r1, r2])
        theta = -0.25 * np.pi
        c1 = 0.1 - ((R[:, 0] - 1.2) * np.cos(theta) - (R[:, 1] - 1.2) * np.sin(theta)) ** 2 / 4.0 - \
             ((R[:, 0] - 1.2) * np.sin(theta) + (R[:, 1] - 1.2) * np.cos(theta)) ** 2 / 36.0
        invalid = c1 > 0
        while np.any(invalid):
            R[invalid] = (R[invalid] - 0.7057) * 1.001 + 0.7057
            c1 = 0.1 - ((R[:, 0] - 1.2) * np.cos(theta) - (R[:, 1] - 1.2) * np.sin(theta)) ** 2 / 4.0 - \
                 ((R[:, 0] - 1.2) * np.sin(theta) + (R[:, 1] - 1.2) * np.cos(theta)) ** 2 / 36.0
            invalid = c1 > 0
        return R


class LIRCMOP8(LIRCMOP):
    """LIR-CMOP8 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=3, **kwargs)

    @staticmethod
    def _constraint(F: np.ndarray) -> np.ndarray:
        p = np.array([1.2, 2.25, 3.5])
        q = np.array([1.2, 2.25, 3.5])
        a = np.array([2.0, 2.5, 2.5])
        b = np.array([6.0, 12.0, 10.0])
        r = 0.1
        theta = -0.25 * np.pi
        c = np.zeros((F.shape[0], 3))
        for k in range(3):
            term1 = ((F[:, 0] - p[k]) * np.cos(theta) - (F[:, 1] - q[k]) * np.sin(theta)) ** 2 / (a[k] ** 2)
            term2 = ((F[:, 0] - p[k]) * np.sin(theta) + (F[:, 1] - q[k]) * np.cos(theta)) ** 2 / (b[k] ** 2)
            c[:, k] = r - term1 - term2
        return c

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1, sum2 = _calc_lircmop_sums(X)
        gx = 0.7057
        f1 = X[:, 0:1] + 10.0 * sum1 + gx
        f2 = 1.0 - X[:, 0:1] ** 2 + 10.0 * sum2 + gx
        F = np.hstack([f1, f2])

        out["F"] = F
        out["G"] = self._constraint(F)

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        r1 = x1 + 0.7057
        r2 = 1.0 - np.sqrt(x1) + 0.7057
        R = np.column_stack([r1, r2])
        theta = -0.25 * np.pi
        c1 = 0.1 - ((R[:, 0] - 1.2) * np.cos(theta) - (R[:, 1] - 1.2) * np.sin(theta)) ** 2 / 4.0 - \
             ((R[:, 0] - 1.2) * np.sin(theta) + (R[:, 1] - 1.2) * np.cos(theta)) ** 2 / 36.0
        invalid = c1 > 0
        while np.any(invalid):
            R[invalid] = (R[invalid] - 0.7057) * 1.001 + 0.7057
            c1 = 0.1 - ((R[:, 0] - 1.2) * np.cos(theta) - (R[:, 1] - 1.2) * np.sin(theta)) ** 2 / 4.0 - \
                 ((R[:, 0] - 1.2) * np.sin(theta) + (R[:, 1] - 1.2) * np.cos(theta)) ** 2 / 36.0
            invalid = c1 > 0
        return R


class LIRCMOP9(LIRCMOP):
    """LIR-CMOP9 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=2, **kwargs)

    @staticmethod
    def _constraint(F: np.ndarray) -> np.ndarray:
        p, q, a, b, r = 1.4, 1.4, 1.5, 6.0, 0.1
        theta = -0.25 * np.pi
        alpha = 0.25 * np.pi
        c1 = r - (((F[:, 0] - p) * np.cos(theta) - (F[:, 1] - q) * np.sin(theta)) ** 2) / (a**2) - \
             (((F[:, 0] - p) * np.sin(theta) + (F[:, 1] - q) * np.cos(theta)) ** 2) / (b**2)
        c2 = 2.0 - F[:, 0] * np.sin(alpha) - F[:, 1] * np.cos(alpha) + \
             np.sin(4.0 * np.pi * (F[:, 0] * np.cos(alpha) - F[:, 1] * np.sin(alpha)))
        return np.column_stack([c1, c2])

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1, sum2 = _calc_lircmop_sums(X)
        f1 = 1.7057 * X[:, 0:1] * (10.0 * sum1 + 1.0)
        f2 = 1.7057 * (1.0 - X[:, 0:1] ** 2) * (10.0 * sum2 + 1.0)
        F = np.hstack([f1, f2])

        out["F"] = F
        out["G"] = self._constraint(F)

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        r1 = x1 * 1.7057
        r2 = (1.0 - x1**2) * 1.7057
        R = np.column_stack([r1, r2])
        con = self._constraint(R)
        feasible = np.all(con <= 0, axis=1)
        R = R[feasible]
        return np.vstack([R, [0.0, 2.182], [1.856, 0.0]])


class LIRCMOP10(LIRCMOP):
    """LIR-CMOP10 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=2, **kwargs)

    @staticmethod
    def _constraint(F: np.ndarray) -> np.ndarray:
        p, q, a, b, r = 1.1, 1.2, 2.0, 4.0, 0.1
        theta = -0.25 * np.pi
        alpha = 0.25 * np.pi
        c1 = r - (((F[:, 0] - p) * np.cos(theta) - (F[:, 1] - q) * np.sin(theta)) ** 2) / (a**2) - \
             (((F[:, 0] - p) * np.sin(theta) + (F[:, 1] - q) * np.cos(theta)) ** 2) / (b**2)
        c2 = 1.0 - F[:, 0] * np.sin(alpha) - F[:, 1] * np.cos(alpha) + \
             np.sin(4.0 * np.pi * (F[:, 0] * np.cos(alpha) - F[:, 1] * np.sin(alpha)))
        return np.column_stack([c1, c2])

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1, sum2 = _calc_lircmop_sums(X)
        f1 = 1.7057 * X[:, 0:1] * (10.0 * sum1 + 1.0)
        f2 = 1.7057 * (1.0 - np.sqrt(X[:, 0:1])) * (10.0 * sum2 + 1.0)
        F = np.hstack([f1, f2])

        out["F"] = F
        out["G"] = self._constraint(F)

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        x1 = np.linspace(0, 1, n_points)
        r1 = x1 * 1.7057
        r2 = (1.0 - np.sqrt(x1)) * 1.7057
        R = np.column_stack([r1, r2])
        con = self._constraint(R)
        feasible = np.all(con <= 0, axis=1)
        R = R[feasible]
        return np.vstack([R, [1.747, 0.0]])


class LIRCMOP11(LIRCMOP):
    """LIR-CMOP11 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=2, **kwargs)

    @staticmethod
    def _constraint(F: np.ndarray) -> np.ndarray:
        p, q, a, b, r = 1.2, 1.2, 1.5, 5.0, 0.1
        theta = -0.25 * np.pi
        alpha = 0.25 * np.pi
        c1 = r - (((F[:, 0] - p) * np.cos(theta) - (F[:, 1] - q) * np.sin(theta)) ** 2) / (a**2) - \
             (((F[:, 0] - p) * np.sin(theta) + (F[:, 1] - q) * np.cos(theta)) ** 2) / (b**2)
        c2 = 2.1 - F[:, 0] * np.sin(alpha) - F[:, 1] * np.cos(alpha) + \
             np.sin(4.0 * np.pi * (F[:, 0] * np.cos(alpha) - F[:, 1] * np.sin(alpha)))
        return np.column_stack([c1, c2])

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1, sum2 = _calc_lircmop_sums(X)
        f1 = 1.7057 * X[:, 0:1] * (10.0 * sum1 + 1.0)
        f2 = 1.7057 * (1.0 - np.sqrt(X[:, 0:1])) * (10.0 * sum2 + 1.0)
        F = np.hstack([f1, f2])

        out["F"] = F
        out["G"] = self._constraint(F)

    def _calc_pareto_front(self, *args, **kwargs) -> np.ndarray:
        return np.array([
            [1.3965, 0.1591],
            [1.0430, 0.5127],
            [0.6894, 0.8662],
            [0.3359, 1.2198],
            [0.0106, 1.6016],
            [0.0, 2.1910],
            [1.8730, 0.0],
        ])


class LIRCMOP12(LIRCMOP):
    """LIR-CMOP12 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=2, **kwargs)

    @staticmethod
    def _constraint(F: np.ndarray) -> np.ndarray:
        p, q, a, b, r = 1.6, 1.6, 1.5, 6.0, 0.1
        theta = -0.25 * np.pi
        alpha = 0.25 * np.pi
        c1 = r - (((F[:, 0] - p) * np.cos(theta) - (F[:, 1] - q) * np.sin(theta)) ** 2) / (a**2) - \
             (((F[:, 0] - p) * np.sin(theta) + (F[:, 1] - q) * np.cos(theta)) ** 2) / (b**2)
        c2 = 2.5 - F[:, 0] * np.sin(alpha) - F[:, 1] * np.cos(alpha) + \
             np.sin(4.0 * np.pi * (F[:, 0] * np.cos(alpha) - F[:, 1] * np.sin(alpha)))
        return np.column_stack([c1, c2])

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1, sum2 = _calc_lircmop_sums(X)
        f1 = 1.7057 * X[:, 0:1] * (10.0 * sum1 + 1.0)
        f2 = 1.7057 * (1.0 - X[:, 0:1] ** 2) * (10.0 * sum2 + 1.0)
        F = np.hstack([f1, f2])

        out["F"] = F
        out["G"] = self._constraint(F)

    def _calc_pareto_front(self, *args, **kwargs) -> np.ndarray:
        return np.array([
            [1.6794, 0.4419],
            [1.3258, 0.7955],
            [0.9723, 1.1490],
            [2.0320, 0.0990],
            [0.6187, 1.5026],
            [0.2652, 1.8562],
            [0.0, 2.2580],
            [2.5690, 0.0],
        ])


class LIRCMOP13(LIRCMOP):
    """LIR-CMOP13 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=2, **kwargs)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1 = np.sum(10.0 * (X[:, 2:] - 0.5) ** 2, axis=1, keepdims=True)

        f1 = (1.7057 + sum1) * np.cos(0.5 * np.pi * X[:, 0:1]) * np.cos(0.5 * np.pi * X[:, 1:2])
        f2 = (1.7057 + sum1) * np.cos(0.5 * np.pi * X[:, 0:1]) * np.sin(0.5 * np.pi * X[:, 1:2])
        f3 = (1.7057 + sum1) * np.sin(0.5 * np.pi * X[:, 0:1])
        F = np.hstack([f1, f2, f3])

        gx = f1**2 + f2**2 + f3**2
        c1 = (gx - 9.0) * (4.0 - gx)
        c2 = (gx - 3.61) * (3.24 - gx)
        G = np.hstack([c1, c2])

        out["F"] = F
        out["G"] = G

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        try:
            ref_dirs = get_reference_directions("energy", 3, n_points=n_points)
        except Exception:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        norm = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        return 1.7057 * (ref_dirs / norm)


class LIRCMOP14(LIRCMOP):
    """LIR-CMOP14 problem."""

    def __init__(self, n_var: int = 30, **kwargs) -> None:
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=3, **kwargs)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        sum1 = np.sum(10.0 * (X[:, 2:] - 0.5) ** 2, axis=1, keepdims=True)

        f1 = (1.7057 + sum1) * np.cos(0.5 * np.pi * X[:, 0:1]) * np.cos(0.5 * np.pi * X[:, 1:2])
        f2 = (1.7057 + sum1) * np.cos(0.5 * np.pi * X[:, 0:1]) * np.sin(0.5 * np.pi * X[:, 1:2])
        f3 = (1.7057 + sum1) * np.sin(0.5 * np.pi * X[:, 0:1])
        F = np.hstack([f1, f2, f3])

        gx = f1**2 + f2**2 + f3**2
        c1 = (gx - 9.0) * (4.0 - gx)
        c2 = (gx - 3.61) * (3.24 - gx)
        c3 = (gx - 3.0625) * (2.56 - gx)
        G = np.hstack([c1, c2, c3])

        out["F"] = F
        out["G"] = G

    def _calc_pareto_front(self, n_points: int = 100, *args, **kwargs) -> np.ndarray:
        try:
            ref_dirs = get_reference_directions("energy", 3, n_points=n_points)
        except Exception:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        norm = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        return np.sqrt(3.0625) * (ref_dirs / norm)
