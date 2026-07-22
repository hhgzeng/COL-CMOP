"""单元与集成测试：验证实验框架、指标计算及结果文件 (NPZ/CSV) 自动持久化落盘。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from core.metrics import calculate_hv, calculate_igd
from experiments.config import ExperimentConfig
from experiments.run_experiment import run_batch_experiment, run_single_run


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

    def test_run_single_run_and_npz_saving(self):
        """测试单问题运行并验证 NPZ 保存的完整字段。"""
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

            # 验证生成了 NPZ 文件
            npz_file = Path(tmp_dir) / "DSOCOL" / "C1DTLZ1" / "run_seed_42.npz"
            self.assertTrue(npz_file.exists())

            data = np.load(npz_file)
            self.assertIn("x", data)
            self.assertIn("f", data)
            self.assertIn("cv", data)
            self.assertIn("elapsed_time", data)
            self.assertIn("igd", data)
            self.assertIn("hv", data)

    def test_run_batch_experiment_and_csv_generation(self):
        """测试小规模批量运行并验证导出 CSV 汇总表格。"""
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
            detail_csv = Path(tmp_dir) / "detailed_runs.csv"
            summary_csv = Path(tmp_dir) / "summary_metrics.csv"

            self.assertTrue(detail_csv.exists())
            self.assertTrue(summary_csv.exists())


if __name__ == "__main__":
    unittest.main()
