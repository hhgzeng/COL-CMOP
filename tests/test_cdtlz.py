"""单元测试：验证 problems/cdtlz.py 中 C-DTLZ 问题在固定输入 X 下的目标与约束计算。"""
import unittest
import numpy as np

from core.problem import PymooProblemAdapter
from problems.cdtlz import C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4


class TestCDTLZProblems(unittest.TestCase):

    def test_c1dtlz1_fixed_input(self):
        """测试 C1DTLZ1 在固定 X 输入下的维度、F 和 G 计算。"""
        prob = C1DTLZ1()
        adapter = PymooProblemAdapter(prob)

        self.assertEqual(adapter.n_var, 7)
        self.assertEqual(adapter.n_obj, 3)

        # 固定输入点：x = 0.5 全向量
        x = np.full((1, 7), 0.5)
        res = adapter.evaluate(x)

        # 当 x[M-1:] = 0.5 时，g(x) = 0，DTLZ1 目标值分布在 [0, 0.5]
        self.assertEqual(res.f.shape, (1, 3))
        self.assertEqual(res.g.shape, (1, 1))

        # C1DTLZ1 线性约束公式: G = -(1 - f3/0.6 - f1/0.5 - f2/0.5)
        f = res.f[0]
        expected_g = -(1.0 - f[2] / 0.6 - (f[0] + f[1]) / 0.5)
        np.testing.assert_allclose(res.g[0, 0], expected_g, atol=1e-7)

    def test_c1dtlz3_fixed_input(self):
        """测试 C1DTLZ3 在固定 X 输入下的维度、F 和 G 计算。"""
        prob = C1DTLZ3()
        adapter = PymooProblemAdapter(prob)

        self.assertEqual(adapter.n_var, 12)
        self.assertEqual(adapter.n_obj, 3)

        x = np.full((1, 12), 0.5)
        res = adapter.evaluate(x)

        self.assertEqual(res.f.shape, (1, 3))
        self.assertEqual(res.g.shape, (1, 1))

        # C1DTLZ3 球面约束公式: G = -(radius - 16) * (radius - 9^2)
        f = res.f[0]
        radius = np.sum(f**2)
        expected_g = -(radius - 16.0) * (radius - 81.0)
        np.testing.assert_allclose(res.g[0, 0], expected_g, atol=1e-6)

    def test_c2dtlz2_fixed_input(self):
        """测试 C2DTLZ2 在固定 X 输入下的维度、F 和 G 计算。"""
        prob = C2DTLZ2()
        adapter = PymooProblemAdapter(prob)

        self.assertEqual(adapter.n_var, 12)
        self.assertEqual(adapter.n_obj, 3)

        x = np.full((1, 12), 0.5)
        res = adapter.evaluate(x)

        self.assertEqual(res.f.shape, (1, 3))
        self.assertEqual(res.g.shape, (1, 1))

    def test_c3dtlz1_fixed_input(self):
        """测试 C3DTLZ1 在固定 X 输入下的维度及多线性约束计算。"""
        prob = C3DTLZ1()
        adapter = PymooProblemAdapter(prob)

        self.assertEqual(adapter.n_var, 7)
        self.assertEqual(adapter.n_obj, 3)

        x = np.full((1, 7), 0.5)
        res = adapter.evaluate(x)

        self.assertEqual(res.f.shape, (1, 3))
        self.assertEqual(res.g.shape, (1, 3))  # 3 个线性约束

    def test_c3dtlz4_fixed_input(self):
        """测试 C3DTLZ4 在固定 X 输入下的维度及多球面约束计算。"""
        prob = C3DTLZ4()
        adapter = PymooProblemAdapter(prob)

        self.assertEqual(adapter.n_var, 12)
        self.assertEqual(adapter.n_obj, 3)

        x = np.full((1, 12), 0.5)
        res = adapter.evaluate(x)

        self.assertEqual(res.f.shape, (1, 3))
        self.assertEqual(res.g.shape, (1, 3))  # 3 个球面约束


if __name__ == "__main__":
    unittest.main()
