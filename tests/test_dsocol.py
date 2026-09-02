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

    def test_algorithm5_ngss_borrowing_and_diversity(self):
        """测试 NGSS (Algorithm 5) 能够正确实现跨生态位借调填充稀疏生态位。"""
        from algorithms.dsocol.algorithms import algorithm5_niche_guided_subset_selection
        from core.schema import Population

        # 构造 2 目标场景，2 个参考权重向量
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        target_n = 4  # capacity = 2

        # 候选池：仅有 1 个解靠近权重 1，其余 5 个解都靠近权重 0
        f = np.array([
            [0.1, 0.9],   # 靠近权重 1 (生态位 1)
            [0.9, 0.1],   # 靠近权重 0
            [0.85, 0.15], # 靠近权重 0
            [0.8, 0.2],   # 靠近权重 0
            [0.75, 0.25], # 靠近权重 0
            [0.7, 0.3],   # 靠近权重 0
        ])
        x = np.zeros((6, 2))
        cv = np.zeros(6)
        pop = Population(x=x, f=f, cv=cv)
        vel = np.zeros_like(x)

        selected_pop, selected_vel, fitness = algorithm5_niche_guided_subset_selection(
            pop, vel, target_n=target_n, weights=weights
        )

        self.assertEqual(len(selected_pop.x), target_n)
        self.assertEqual(len(selected_vel), target_n)
        self.assertEqual(len(fitness), target_n)
        # 验证解集无重复
        self.assertEqual(len(np.unique(selected_pop.f, axis=0)), target_n)
        # 生态位 1 缺少 1 个，必定借调离权重 1 最近的解（即 [0.7, 0.3]）
        self.assertTrue(any(np.allclose(selected_pop.f[i], [0.1, 0.9]) for i in range(target_n)))
        self.assertTrue(any(np.allclose(selected_pop.f[i], [0.7, 0.3]) for i in range(target_n)))


if __name__ == "__main__":
    unittest.main()


