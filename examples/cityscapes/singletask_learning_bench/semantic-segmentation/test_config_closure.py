import os
import unittest
import yaml


class TestCityscapesConfigClosure(unittest.TestCase):
    """
    Tests configuration closure for cityscapes singletask_learning_bench example.
    Verifies that benchmarkingjob.yaml can be parsed from repository root,
    resolves testenv.yaml, testalgorithms (rfnet_algorithm.yaml), Python module URLs,
    and metric URLs cleanly without requiring external datasets or model checkpoints.
    """

    def setUp(self):
        # Determine repo root directory (assumed 4 levels up from this test file, or current directory)
        self.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        self.benchmarkingjob_path = os.path.join(
            self.repo_root,
            "examples",
            "cityscapes",
            "singletask_learning_bench",
            "semantic-segmentation",
            "benchmarkingjob.yaml",
        )

    def test_benchmarkingjob_exists_and_parses(self):
        self.assertTrue(
            os.path.exists(self.benchmarkingjob_path),
            f"benchmarkingjob.yaml not found at {self.benchmarkingjob_path}",
        )
        with open(self.benchmarkingjob_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertIn("benchmarkingjob", config)
        job_cfg = config["benchmarkingjob"]

        # 1. Verify testenv file exists
        testenv_rel = job_cfg.get("testenv")
        self.assertIsNotNone(testenv_rel, "testenv path missing in benchmarkingjob.yaml")
        testenv_abs = os.path.abspath(os.path.join(self.repo_root, testenv_rel))
        self.assertTrue(
            os.path.exists(testenv_abs),
            f"Referenced testenv file not found: {testenv_abs}",
        )

        # 2. Parse testenv.yaml and verify referenced metric modules
        with open(testenv_abs, "r", encoding="utf-8") as f:
            testenv_cfg = yaml.safe_load(f)

        self.assertIn("testenv", testenv_cfg)
        testenv_body = testenv_cfg["testenv"]

        # Verify dataset index key fields match Dataset._check_fields() expectation
        dataset_cfg = testenv_body.get("dataset", {})
        self.assertIn("train_index", dataset_cfg, "train_index missing in dataset config")
        self.assertIn("test_index", dataset_cfg, "test_index missing in dataset config")

        # Verify metric Python URLs exist
        metrics = testenv_body.get("metrics", [])
        for metric in metrics:
            metric_url = metric.get("url")
            self.assertIsNotNone(metric_url, f"Metric {metric.get('name')} missing url")
            metric_abs = os.path.abspath(os.path.join(self.repo_root, metric_url))
            self.assertTrue(
                os.path.exists(metric_abs),
                f"Referenced metric Python module not found: {metric_abs}",
            )

        # 3. Verify algorithm config files & module URLs exist
        test_obj = job_cfg.get("test_object", {})
        algorithms = test_obj.get("algorithms", [])
        self.assertTrue(len(algorithms) > 0, "No algorithms configured in benchmarkingjob.yaml")

        for algo in algorithms:
            algo_url = algo.get("url")
            self.assertIsNotNone(algo_url, f"Algorithm {algo.get('name')} missing url")
            algo_abs = os.path.abspath(os.path.join(self.repo_root, algo_url))
            self.assertTrue(
                os.path.exists(algo_abs),
                f"Referenced algorithm YAML file not found: {algo_abs}",
            )

            # Parse rfnet_algorithm.yaml and check module code URL
            with open(algo_abs, "r", encoding="utf-8") as f:
                algo_cfg = yaml.safe_load(f)

            self.assertIn("algorithm", algo_cfg)
            modules = algo_cfg["algorithm"].get("modules", [])
            for mod in modules:
                mod_url = mod.get("url")
                self.assertIsNotNone(mod_url, f"Module {mod.get('name')} missing url")
                mod_abs = os.path.abspath(os.path.join(self.repo_root, mod_url))
                self.assertTrue(
                    os.path.exists(mod_abs),
                    f"Referenced module Python file not found: {mod_abs}",
                )


if __name__ == "__main__":
    unittest.main()
