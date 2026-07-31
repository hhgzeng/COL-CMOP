"""单元与集成测试：验证实验框架、指标计算、结果文件 (NPZ/CSV) 自动持久化落盘、自动旧数据清理及 Benchmark 分类扩展。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from core.metrics import calculate_hv, calculate_igd
from experiments.config import ExperimentConfig
from experiments.run_experiment import resolve_problems, run_batch_experiment, run_single_run


class TestExperiments(unittest.TestCase):

    def test_metrics_calculation(self):
        """测试 IGD 和 HV 指标函数边界逻辑。"""
        ref_front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        points = np.array([[0.1, 0.9], [0.6, 0.4]])

        igd_val = calculate_igd(points, ref_front)
        self.assertGreater(igd_val, 0.0)

        # 空解集测试
        self.assertTrue(np.isnan(calculate_igd(None, ref_front)))
        self.assertEqual(calculate_hv(None, np.array([2.0, 2.0])), 0.0)

        # 正常 HV 计算
        hv_val = calculate_hv(points, np.array([2.0, 2.0]))
        self.assertGreater(hv_val, 0.0)

    def test_resolve_problems_by_category(self):
        """测试按 Benchmark 分类解析展开测试问题。"""
        # 测试单一分类展开
        cdtlzs = resolve_problems(categories=["C-DTLZs"])
        self.assertEqual(len(cdtlzs), 4)
        self.assertIn("C1DTLZ1", cdtlzs)
        self.assertIn("C3DTLZ4", cdtlzs)

        # 测试 --problems 中传入分类名称
        probs = resolve_problems(problems=["C-DTLZs"])
        self.assertEqual(probs, cdtlzs)

        # 测试 ALL 分类展开
        all_probs = resolve_problems(categories=["ALL"])
        self.assertEqual(len(all_probs), 4 + 6 + 9 + 14)

    def test_run_single_run_and_npz_saving(self):
        """测试单问题运行并验证在 Benchmark 分类目录下生成 NPZ 文件。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_single_run(
                algo_name="DSOCOL",
                prob_name="C1DTLZ1",
                seed=42,
                max_evals=600,
                population_size=20,
                save_dir=tmp_dir,
            )

            self.assertEqual(res["algorithm"], "DSOCOL")
            self.assertEqual(res["problem"], "C1DTLZ1")
            self.assertIn("igd", res)
            self.assertIn("hv", res)

            # 验证在 C-DTLZs 分类下生成了 NPZ 文件
            npz_files = list(Path(tmp_dir).rglob("run_seed_42.npz"))
            self.assertTrue(len(npz_files) == 1)

            data = np.load(npz_files[0])
            self.assertIn("x", data)
            self.assertIn("f", data)
            self.assertIn("cv", data)
            self.assertIn("elapsed_time", data)
            self.assertIn("igd", data)
            self.assertIn("hv", data)

    def test_clear_old_results_on_new_run(self):
        """测试重新运行实验时自动清除旧的 NPZ 运行结果文件。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. 第一次运行：n_runs=5
            config1 = ExperimentConfig(
                population_size=20,
                max_evals=400,
                n_runs=5,
                algorithms=["APSEA"],
                problems=["C1DTLZ1"],
                results_dir=tmp_dir,
            )
            run_batch_experiment(config1)
            npz_files_after_run1 = list(Path(tmp_dir).rglob("*.npz"))
            self.assertEqual(len(npz_files_after_run1), 5)

            # 2. 第二次运行：n_runs=3 (应该先清空之前的 5 个 NPZ，最终只留 3 个)
            config2 = ExperimentConfig(
                population_size=20,
                max_evals=400,
                n_runs=3,
                algorithms=["APSEA"],
                problems=["C1DTLZ1"],
                results_dir=tmp_dir,
            )
            run_batch_experiment(config2)
            npz_files_after_run2 = list(Path(tmp_dir).rglob("*.npz"))
            self.assertEqual(len(npz_files_after_run2), 3)

    def test_run_batch_experiment_summary(self):
        """测试小规模批量运行并验证返回控制台汇总 DataFrame。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                population_size=20,
                max_evals=600,
                n_runs=2,
                algorithms=["DSOCOL", "APSEA"],
                problems=["C1DTLZ1"],
                results_dir=tmp_dir,
            )

            df_summary = run_batch_experiment(config)

            self.assertEqual(len(df_summary), 2)  # 2 个算法
            self.assertIn("Algorithm", df_summary.columns)
            self.assertIn("HV_Mean", df_summary.columns)


if __name__ == "__main__":
    unittest.main()
