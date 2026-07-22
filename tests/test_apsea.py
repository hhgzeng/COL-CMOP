"""单元测试：验证 APSEA 算法模块的适应度、环境选择及在 C1DTLZ1 上的小预算运行与可重复性。"""
import unittest
import numpy as np

from algorithms.apsea.algorithm import APSEA
from algorithms.apsea.fitness import cal_fitness
from algorithms.apsea.selection import (
    environmental_selection_cdp,
    environmental_selection_epsilon,
    environmental_selection_no_constrained,
)
from core.problem import PymooProblemAdapter
from core.schema import Population
from problems.cdtlz import C1DTLZ1


class TestAPSEA(unittest.TestCase):

    def test_cal_fitness(self):
        """测试 SPEA2-like 适应度算子计算结果。"""
        objs = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        cons = np.array([[0.0], [0.0], [1.0]])

        fit = cal_fitness(objs, cons)
        self.assertEqual(len(fit), 3)
        # 解 0 和 解 1 可行，且互相非支配；解 2 不可行且支配关系最差
        self.assertLess(fit[0], fit[2])
        self.assertLess(fit[1], fit[2])

    def test_environmental_selections(self):
        """测试三种环境选择函数均能精准截断并返回 N 个个体。"""
        n_samples = 20
        x = np.random.uniform(0, 1, size=(n_samples, 7))
        f = np.random.uniform(0, 5, size=(n_samples, 3))
        cv = np.random.uniform(0, 2, size=n_samples)
        # 让前 5 个解可行
        cv[:5] = 0.0
        pop = Population(x=x, f=f, cv=cv, g=cv[:, None])

        # 测试 CDP 选择
        pop_cdp, fit_cdp = environmental_selection_cdp(pop, target_n=10)
        self.assertEqual(len(pop_cdp.x), 10)
        self.assertEqual(len(fit_cdp), 10)

        # 测试 无约束选择
        pop_nc, fit_nc = environmental_selection_no_constrained(pop, target_n=10)
        self.assertEqual(len(pop_nc.x), 10)

        # 测试 Epsilon 选择
        pop_eps, fit_eps = environmental_selection_epsilon(pop, target_n=10, var_epsilon=0.5)
        self.assertEqual(len(pop_eps.x), 10)

    def test_apsea_small_scale_run(self):
        """测试 APSEA 在 C1DTLZ1 小预算下正常演化。"""
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=2000)
        algo = APSEA(population_size=20, seed=42)

        result = algo.run(prob)

        self.assertGreaterEqual(result.eval_count, 2000)
        self.assertEqual(len(result.population.x), 20)
        self.assertIn("sub_pop2_size", result.history)

    def test_apsea_reproducibility(self):
        """测试使用固定 seed 时 APSEA 算法的可重复性。"""
        prob1 = PymooProblemAdapter(C1DTLZ1(), max_evals=1000)
        algo1 = APSEA(population_size=20, seed=123)
        res1 = algo1.run(prob1)

        prob2 = PymooProblemAdapter(C1DTLZ1(), max_evals=1000)
        algo2 = APSEA(population_size=20, seed=123)
        res2 = algo2.run(prob2)

        self.assertEqual(res1.eval_count, res2.eval_count)
        np.testing.assert_allclose(res1.population.x, res2.population.x)
        np.testing.assert_allclose(res1.population.f, res2.population.f)


if __name__ == "__main__":
    unittest.main()
