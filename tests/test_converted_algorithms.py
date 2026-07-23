"""单元测试：验证从 MATLAB 转换而来的 8 个算法在 C1DTLZ1 上的微型预算运行与可重复性。"""

import unittest
import numpy as np

from algorithms import (
    C3M,
    CMOCSO,
    CMOEMT,
    DRLOSEMCMO,
    DVCEA,
    IMCMOEAD,
    LCMEA,
    POCEA,
)
from core.problem import PymooProblemAdapter
from problems.cdtlz import C1DTLZ1


class TestConvertedAlgorithms(unittest.TestCase):

    def setUp(self):
        self.max_evals = 300
        self.pop_size = 20

    def test_c3m(self):
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=self.max_evals)
        algo = C3M(population_size=self.pop_size, seed=42)
        res = algo.run(prob)
        self.assertGreaterEqual(res.eval_count, self.max_evals)
        self.assertEqual(len(res.population.x), self.pop_size)

    def test_cmocso(self):
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=self.max_evals)
        algo = CMOCSO(population_size=self.pop_size, seed=42)
        res = algo.run(prob)
        self.assertGreaterEqual(res.eval_count, self.max_evals)
        self.assertGreater(len(res.population.x), 0)

    def test_cmoemt(self):
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=self.max_evals)
        algo = CMOEMT(population_size=self.pop_size, seed=42)
        res = algo.run(prob)
        self.assertGreaterEqual(res.eval_count, self.max_evals)
        self.assertGreater(len(res.population.x), 0)

    def test_drlos_emcmo(self):
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=self.max_evals)
        algo = DRLOSEMCMO(population_size=self.pop_size, seed=42)
        res = algo.run(prob)
        self.assertGreaterEqual(res.eval_count, self.max_evals)
        self.assertEqual(len(res.population.x), self.pop_size)

    def test_dvcea(self):
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=self.max_evals)
        algo = DVCEA(population_size=self.pop_size, seed=42)
        res = algo.run(prob)
        self.assertGreaterEqual(res.eval_count, self.max_evals)
        self.assertEqual(len(res.population.x), self.pop_size)

    def test_im_c_moea_d(self):
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=self.max_evals)
        algo = IMCMOEAD(population_size=self.pop_size, seed=42)
        res = algo.run(prob)
        self.assertGreaterEqual(res.eval_count, self.max_evals)
        self.assertGreater(len(res.population.x), 0)

    def test_lcmea(self):
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=self.max_evals)
        algo = LCMEA(population_size=self.pop_size, seed=42)
        res = algo.run(prob)
        self.assertGreaterEqual(res.eval_count, self.max_evals)
        self.assertEqual(len(res.population.x), self.pop_size)

    def test_pocea(self):
        prob = PymooProblemAdapter(C1DTLZ1(), max_evals=self.max_evals)
        algo = POCEA(population_size=self.pop_size, seed=42)
        res = algo.run(prob)
        self.assertGreaterEqual(res.eval_count, self.max_evals)
        self.assertGreater(len(res.population.x), 0)

    def test_reproducibility(self):
        prob1 = PymooProblemAdapter(C1DTLZ1(), max_evals=200)
        algo1 = C3M(population_size=20, seed=123)
        res1 = algo1.run(prob1)

        prob2 = PymooProblemAdapter(C1DTLZ1(), max_evals=200)
        algo2 = C3M(population_size=20, seed=123)
        res2 = algo2.run(prob2)

        self.assertEqual(res1.eval_count, res2.eval_count)
        np.testing.assert_allclose(res1.population.x, res2.population.x)
        np.testing.assert_allclose(res1.population.f, res2.population.f)


if __name__ == "__main__":
    unittest.main()
