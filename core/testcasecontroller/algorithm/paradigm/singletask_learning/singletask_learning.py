# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single Task Learning Paradigm"""

import os

from core.common.constant import ParadigmType
from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase


class SingleTaskLearning(ParadigmBase):
    """
    SingleTaskLearning:
    provide the flow of single task learning paradigm.
    Notes:
          1. Ianvs serves as testing tools for test objects, e.g., algorithms.
          2. Ianvs does NOT include code directly on test object.
          3. Algorithms serve as typical test objects in Ianvs
          and detailed algorithms are thus NOT included in this Ianvs python file.
          4. As for the details of example test objects, e.g., algorithms,
          please refer to third party packages in Ianvs example.
          For example, AI workflow and interface pls refer to sedna
          (sedna docs: https://sedna.readthedocs.io/en/latest/api/lib/index.html),
          and module implementation pls refer to `examples' test algorithms`,
          e.g., basemodel.py, hard_example_mining.py.

    Parameters
    ---------
    workspace: string
        the output required for single task learning paradigm.
    kwargs: dict
        config required for the test process of single task learning paradigm,
        e.g.: algorithm modules, dataset, initial model, etc.

    """

    def __init__(self, workspace, **kwargs):
        ParadigmBase.__init__(self, workspace, **kwargs)
        self.initial_model = kwargs.get("initial_model_url")

    def run(self):
        """
        run the test flow of single task learning paradigm.

        Returns
        ------
        test result: numpy.ndarray
        system metric info: dict
            information needed to compute system metrics.

        """

        job = self.build_paradigm_job(ParadigmType.SINGLE_TASK_LEARNING.value)

        trained_model = self._train(job, self.initial_model)

        inference_result = self._inference(job, trained_model)

        return inference_result, self.system_metric_info

    def _train(self, job, initial_model):
        train_output_dir = os.path.join(self.workspace, "output/train/")
        os.environ["BASE_MODEL_URL"] = initial_model

        train_dataset = self.dataset.load_data(self.dataset.train_url, "train")
        job.train(train_dataset)
        trained_model_path = job.save(train_output_dir)
        return trained_model_path

    def _inference(self, job, trained_model):
        inference_dataset = self.dataset.load_data(self.dataset.test_url, "inference")
        inference_output_dir = os.path.join(self.workspace, "output/inference/")
        os.environ["RESULT_SAVED_URL"] = inference_output_dir
        job.load(trained_model)
        infer_res = job.predict(inference_dataset.x)
        return infer_res
