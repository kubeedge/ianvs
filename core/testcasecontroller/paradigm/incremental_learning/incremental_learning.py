import copy
import os

from core.testcasecontroller.paradigm.base import ParadigmBase
from core.testenvmanager.testenv import TestEnv
from core.testcasecontroller.algorithm import Algorithm
from core.testcasecontroller.metrics import get_metric_func


class IncrementalLearning(ParadigmBase):
    """ IncrementalLearning pipeline """

    def __init__(self, test_env: TestEnv, algorithm: Algorithm, workspace: str):
        super(IncrementalLearning, self).__init__(test_env, algorithm, workspace)

    def run(self):
        rounds = self.test_env.incremental_rounds

        try:
            dataset_files = self.preprocess_dataset(splitting_times=rounds)
        except Exception as err:
            raise Exception(f"preprocess dataset failed, error: {err}.")

        current_model_url = self.algorithm.initial_model_url
        for r in range(1, rounds + 1):
            train_dataset_file, eval_dataset_file = dataset_files[r - 1]

            train_output_dir = os.path.join(self.workspace, f"output/train/{r}")
            os.environ["MODEL_URL"] = train_output_dir
            os.environ["BASE_MODEL_URL"] = current_model_url
            job, feature_process = self.algorithm.build()
            train_dataset = self.load_data(train_dataset_file, "train", feature_process=feature_process)
            new_model_path = job.train(train_dataset)

            os.environ["MODEL_URLS"] = f"{new_model_path};{current_model_url}"
            eval_dataset = self.load_data(eval_dataset_file, "eval", feature_process=feature_process)
            model_eval_info = copy.deepcopy(self.test_env.model_eval)
            model_metric = model_eval_info.get("model_metric")
            metric_name = model_metric.get("name")
            eval_results = job.evaluate(eval_dataset, metric=get_metric_func(model_metric))

            operator_info = model_eval_info
            if self._trigger_deploy(eval_results, metric_name, operator_info):
                current_model_url = new_model_path
            current_model_url = "/home/yj/core/examples/pcb-aoi/workspace/pcb-algorithm-test/test-algorithm/5856ba58-ebdc-11ec-83c3-53ead20896e4/output/train/1/model.zip"
            inference_dataset = self.load_data(self.dataset.eval_dataset, "inference", feature_process=feature_process)
            inference_output_dir = os.path.join(self.workspace, f"output/inference/{r}")
            os.environ["INFERENCE_OUTPUT_DIR"] = inference_output_dir
            os.environ["MODEL_URL"] = current_model_url
            infer_res, _, _ = job.inference(inference_dataset.x)

        return self.eval_overall(infer_res)

    def _trigger_deploy(self, eval_results, metric_name, operator_info):
        operator = operator_info.get("operator")
        threshold = operator_info.get("threshold")

        operator_map = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            "=": lambda x, y: x == y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }

        if operator not in operator_map:
            raise ValueError(f"operator {operator} use to compare is not allow, set to <")

        operator_func = operator_map[operator]

        if len(eval_results) != 2:
            raise Exception(f"two models of evaluation should have two results. the eval results: {eval_results}")

        metric_values = [0, 0]
        for i, result in enumerate(eval_results):
            metrics = result.get("metrics")
            metric_values[i] = metrics.get(metric_name)

        metric_delta = metric_values[0] - metric_values[1]
        return operator_func(metric_delta, threshold)
