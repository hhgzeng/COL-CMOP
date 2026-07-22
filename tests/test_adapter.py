"""单元测试：验证 PymooProblemAdapter 对 C1DTLZ1 的适配、评估及 FE 准确累计。"""
import unittest
import numpy as np

from core.problem import PymooProblemAdapter
from problems.cdtlz import C1DTLZ1


class TestPymooProblemAdapter(unittest.TestCase):

    def test_adapter_c1dtlz1_properties(self):
        """测试适配器能正确提取 C1DTLZ1 属性。"""
        raw_prob = C1DTLZ1()
        adapter = PymooProblemAdapter(raw_prob, max_evals=1000)

        self.assertEqual(adapter.n_var, raw_prob.n_var)
        self.assertEqual(adapter.n_obj, raw_prob.n_obj)
        self.assertEqual(len(adapter.lower), adapter.n_var)
        self.assertEqual(len(adapter.upper), adapter.n_var)
        self.assertTrue(np.all(adapter.lower == 0.0))
        self.assertTrue(np.all(adapter.upper == 1.0))
        self.assertEqual(adapter.eval_count, 0)

    def test_adapter_evaluation_and_fe_accumulation(self):
        """测试评估输出形状、约束计算及 FE 的准确累计。"""
        raw_prob = C1DTLZ1()
        adapter = PymooProblemAdapter(raw_prob, max_evals=1000)

        # 生成 10 个随机样本
        rng = np.random.default_rng(42)
        x = rng.uniform(adapter.lower, adapter.upper, size=(10, adapter.n_var))

        res = adapter.evaluate(x)

        # 校验输出数据维度
        self.assertEqual(res.f.shape, (10, adapter.n_obj))
        self.assertIsNotNone(res.g)
        self.assertEqual(res.g.shape, (10, 1))  # C1DTLZ1 有 1 个不等式约束
        self.assertEqual(res.cv.shape, (10,))
        self.assertEqual(adapter.eval_count, 10)

        # 再评估 5 个样本
        x2 = rng.uniform(adapter.lower, adapter.upper, size=(5, adapter.n_var))
        res2 = adapter.evaluate(x2)

        self.assertEqual(res2.f.shape, (5, adapter.n_obj))
        self.assertEqual(adapter.eval_count, 15)

        # 验证单向量 1D 输入自动提升为 (1, D)
        x_single = adapter.lower.copy()
        res_single = adapter.evaluate(x_single)
        self.assertEqual(res_single.f.shape, (1, adapter.n_obj))
        self.assertEqual(adapter.eval_count, 16)

    def test_constraint_violation_calculation(self):
        """测试约束违反度 CV 的计算规则 CV = sum(max(0, G))."""
        raw_prob = C1DTLZ1()
        adapter = PymooProblemAdapter(raw_prob)

        x = np.ones((2, adapter.n_var)) * 0.5
        res = adapter.evaluate(x)

        expected_cv = np.maximum(res.g, 0.0).sum(axis=1)
        np.testing.assert_allclose(res.cv, expected_cv)


if __name__ == "__main__":
    unittest.main()

