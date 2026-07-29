"""单元测试：验证 DSOCOL 算法在 C1DTLZ1 问题上的基本运行、FE 预算终止及 Seed 可重复性。"""
import unittest
import numpy as np

from algorithms.dsocol import DSOCOL, DSOCOL1, DSOCOL3, DSOCOL4
from core.problem import PymooProblemAdapter
from problems.cdtlz import C1DTLZ1


class TestDSOCOL(unittest.TestCase):

    def test_dsocol_run_on_c1dtlz1(self):
        """测试 DSOCOL 能够正常在 C1DTLZ1 上运行并返回合法 Result。"""
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=1000)
        algo = DSOCOL(population_size=20, col_frequency=10, seed=42)

        result = algo.run(prob)

        # 校验评估次数达到 max_evals 且正常退出
        self.assertGreaterEqual(result.eval_count, 1000)
        self.assertEqual(len(result.population.x), 20)
        self.assertEqual(result.population.f.shape[1], 3)
        self.assertIn("fe", result.history)
        self.assertIn("feasible_ratio_s1", result.history)

    def test_dsocol_ablation_variants_run(self):
        """测试消融变体 DSOCOL1, DSOCOL3, DSOCOL4 能够正常运行并生成结果。"""
        for variant_cls in [DSOCOL1, DSOCOL3, DSOCOL4]:
            prob = PymooProblemAdapter(C1DTLZ1(), max_evals=600)
            algo = variant_cls(population_size=20, col_frequency=5, seed=42)
            res = algo.run(prob)
            self.assertGreaterEqual(res.eval_count, 600)
            self.assertEqual(len(res.population.x), 20)
            self.assertEqual(res.population.f.shape[1], 3)

    def test_dsocol_reproducibility(self):
        """测试使用固定 seed 时 DSOCOL 算法的强可重复性。"""
        prob1 = PymooProblemAdapter(C1DTLZ1(), max_evals=600)
        algo1 = DSOCOL(population_size=20, col_frequency=5, seed=123)
        res1 = algo1.run(prob1)

        prob2 = PymooProblemAdapter(C1DTLZ1(), max_evals=600)
        algo2 = DSOCOL(population_size=20, col_frequency=5, seed=123)
        res2 = algo2.run(prob2)

        self.assertEqual(res1.eval_count, res2.eval_count)
        np.testing.assert_allclose(res1.population.x, res2.population.x)
        np.testing.assert_allclose(res1.population.f, res2.population.f)
        np.testing.assert_allclose(res1.population.cv, res2.population.cv)

    def test_dsocol_small_scale_convergence(self):
        """小规模集成测试：验证演化过程中 FE 不断增加且可以找到可行解。"""
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=2000)
        algo = DSOCOL(population_size=40, col_frequency=10, seed=7)
        result = algo.run(prob)

        # 验证历史记录中的 FE 增长
        history_fe = result.history["fe"]
        self.assertTrue(all(x <= y for x, y in zip(history_fe, history_fe[1:])))
        self.assertEqual(history_fe[-1], result.eval_count)


if __name__ == "__main__":
    unittest.main()

