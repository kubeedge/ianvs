import os

from core.testcasecontroller.paradigm.base import ParadigmBase
from core.testenvmanager.testenv import TestEnv
from core.testcasecontroller.algorithm import Algorithm


class SingleTaskLearning(ParadigmBase):
    """ SingleTaskLearning pipeline """

    def __init__(self, test_env: TestEnv, algorithm: Algorithm, workspace: str):
        super(SingleTaskLearning, self).__init__(test_env, algorithm, workspace)

    def run(self):
        current_model_url = self.algorithm.initial_model_url
        train_output_dir = os.path.join(self.workspace, f"output/train/")
        os.environ["BASE_MODEL_URL"] = current_model_url

        job, feature_process = self.algorithm.build()
        train_dataset = self.load_data(self.dataset.train_dataset, "train", feature_process=feature_process)
        job.train(train_dataset)
        trained_model_path = job.save(train_output_dir)

        inference_dataset = self.load_data(self.dataset.eval_dataset, "inference", feature_process=feature_process)
        inference_output_dir = os.path.join(self.workspace, f"output/inference/")
        os.environ["INFERENCE_OUTPUT_DIR"] = inference_output_dir
        job.load(trained_model_path)
        infer_res = job.predict(inference_dataset.x)

        return self.eval_overall(infer_res)
