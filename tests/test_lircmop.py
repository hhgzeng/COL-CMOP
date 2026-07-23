"""单元测试：验证 problems/lircmop.py 中 LIR-CMOP 问题 (1-14) 的定义、计算与适配器兼容性。"""

import unittest

import numpy as np

from core.problem import PymooProblemAdapter
from problems.lircmop import (
    LIRCMOP1,
    LIRCMOP2,
    LIRCMOP3,
    LIRCMOP4,
    LIRCMOP5,
    LIRCMOP6,
    LIRCMOP7,
    LIRCMOP8,
    LIRCMOP9,
    LIRCMOP10,
    LIRCMOP11,
    LIRCMOP12,
    LIRCMOP13,
    LIRCMOP14,
)


class TestLIRCMOPProblems(unittest.TestCase):
    """测试 LIR-CMOP1 到 LIR-CMOP14 问题维度、目标值、约束值及 PF 计算。"""

    def setUp(self):
        self.problems = [
            LIRCMOP1,
            LIRCMOP2,
            LIRCMOP3,
            LIRCMOP4,
            LIRCMOP5,
            LIRCMOP6,
            LIRCMOP7,
            LIRCMOP8,
            LIRCMOP9,
            LIRCMOP10,
            LIRCMOP11,
            LIRCMOP12,
            LIRCMOP13,
            LIRCMOP14,
        ]
        # 对应问题的 (n_obj, n_ieq_constr) 预期
        self.expected_specs = [
            (2, 2),  # LIRCMOP1
            (2, 2),  # LIRCMOP2
            (2, 3),  # LIRCMOP3
            (2, 3),  # LIRCMOP4
            (2, 2),  # LIRCMOP5
            (2, 2),  # LIRCMOP6
            (2, 3),  # LIRCMOP7
            (2, 3),  # LIRCMOP8
            (2, 2),  # LIRCMOP9
            (2, 2),  # LIRCMOP10
            (2, 2),  # LIRCMOP11
            (2, 2),  # LIRCMOP12
            (3, 2),  # LIRCMOP13
            (3, 3),  # LIRCMOP14
        ]

    def test_dimensions_and_shapes(self):
        """测试所有 14 个问题的维度设置及 evaluate 输出 shape。"""
        for cls, (exp_n_obj, exp_n_con) in zip(self.problems, self.expected_specs):
            with self.subTest(problem=cls.__name__):
                prob = cls(n_var=30)
                adapter = PymooProblemAdapter(prob)

                self.assertEqual(adapter.n_var, 30)
                self.assertEqual(adapter.n_obj, exp_n_obj)
                self.assertEqual(prob.n_ieq_constr, exp_n_con)

                x = np.full((5, 30), 0.5)
                res = adapter.evaluate(x)

                self.assertEqual(res.f.shape, (5, exp_n_obj))
                self.assertEqual(res.g.shape, (5, exp_n_con))
                self.assertFalse(np.isnan(res.f).any())
                self.assertFalse(np.isnan(res.g).any())

    def test_lircmop1_fixed_input(self):
        """测试 LIRCMOP1 在固定输入 x=0.5 时的精确定量数值。"""
        prob = LIRCMOP1(n_var=30)
        adapter = PymooProblemAdapter(prob)
        x = np.full((1, 30), 0.5)
        res = adapter.evaluate(x)

        x1 = 0.5
        sin_val = np.sin(0.5 * np.pi * x1)
        cos_val = np.cos(0.5 * np.pi * x1)

        # 30维中，odd 变量 14 个 (indices 3,5,..,29 -> python 2,4,..,28)
        # even 变量 15 个 (indices 2,4,..,30 -> python 1,3,..,29)
        expected_g1 = 14 * ((0.5 - sin_val) ** 2)
        expected_g2 = 15 * ((0.5 - cos_val) ** 2)

        expected_f1 = x1 + expected_g1
        expected_f2 = 1.0 - x1**2 + expected_g2

        expected_c1 = (0.5 - expected_g1) * (0.51 - expected_g1)
        expected_c2 = (0.5 - expected_g2) * (0.51 - expected_g2)

        np.testing.assert_allclose(res.f[0, 0], expected_f1, atol=1e-7)
        np.testing.assert_allclose(res.f[0, 1], expected_f2, atol=1e-7)
        np.testing.assert_allclose(res.g[0, 0], expected_c1, atol=1e-7)
        np.testing.assert_allclose(res.g[0, 1], expected_c2, atol=1e-7)

    def test_lircmop13_fixed_input(self):
        """测试 3 目标问题 LIRCMOP13 在固定输入 x=0.5 时的精确定量数值。"""
        prob = LIRCMOP13(n_var=30)
        adapter = PymooProblemAdapter(prob)
        x = np.full((1, 30), 0.5)
        res = adapter.evaluate(x)

        # x3..x30 均为 0.5，sum1 = 0
        f1 = 1.7057 * np.cos(0.25 * np.pi) * np.cos(0.25 * np.pi)
        f2 = 1.7057 * np.cos(0.25 * np.pi) * np.sin(0.25 * np.pi)
        f3 = 1.7057 * np.sin(0.25 * np.pi)

        np.testing.assert_allclose(res.f[0, 0], f1, atol=1e-7)
        np.testing.assert_allclose(res.f[0, 1], f2, atol=1e-7)
        np.testing.assert_allclose(res.f[0, 2], f3, atol=1e-7)

        gx = f1**2 + f2**2 + f3**2
        expected_c1 = (gx - 9.0) * (4.0 - gx)
        expected_c2 = (gx - 3.61) * (3.24 - gx)

        np.testing.assert_allclose(res.g[0, 0], expected_c1, atol=1e-7)
        np.testing.assert_allclose(res.g[0, 1], expected_c2, atol=1e-7)

    def test_pareto_front(self):
        """测试所有 14 个问题的 Pareto Front 计算。"""
        for cls in self.problems:
            with self.subTest(problem=cls.__name__):
                prob = cls(n_var=30)
                pf = prob.pareto_front()
                self.assertIsNotNone(pf)
                self.assertGreater(len(pf), 0)
                self.assertEqual(pf.shape[1], prob.n_obj)
                self.assertFalse(np.isnan(pf).any())


if __name__ == "__main__":
    unittest.main()
