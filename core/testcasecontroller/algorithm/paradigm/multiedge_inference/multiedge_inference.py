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

"""Multiedge Inference"""

import os

import onnx

from core.common.log import LOGGER
from core.common.constant import ParadigmType
from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase


class MultiedgeInference(ParadigmBase):
    """
    MultiedgeInference:
    provide the flow of multi-edge inference paradigm.
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
        the output required for multi-edge inference paradigm.
    kwargs: dict
        config required for the test process of multi-edge inference paradigm,
        e.g.: algorithm modules, dataset, initial model, etc.

    """

    def __init__(self, workspace, **kwargs):
        ParadigmBase.__init__(self, workspace, **kwargs)
        self.initial_model = kwargs.get("initial_model_url")

    def run(self):
        """
        run the test flow of multi-edge inference paradigm.

        Returns
        ------
        test result: numpy.ndarray
        system metric info: dict
            information needed to compute system metrics.

        """

        job = self.build_paradigm_job(ParadigmType.MULTIEDGE_INFERENCE.value)
        if not job.__dict__.get('model_parallel'):
            inference_result = self._inference(job, self.initial_model)
        else:
            if 'partition' in dir(job):
                models_dir, map_info = job.partition(self.initial_model)
            else:
                models_dir, map_info = self._partition(job.__dict__.get('partition_point_list'), self.initial_model, os.path.dirname(self.initial_model))
            inference_result = self._inference_mp(job, models_dir, map_info)

        return inference_result, self.system_metric_info

    def _inference(self, job, trained_model):
        train_dataset = self.dataset.load_data(self.dataset.train_url, "train")
        os.environ["BASE_MODEL_URL"] = trained_model
        inference_dataset = self.dataset.load_data(self.dataset.test_url, "inference")
        inference_output_dir = os.path.join(self.workspace, "output/inference/")
        os.environ["RESULT_SAVED_URL"] = inference_output_dir
        job.load(trained_model)
        infer_res = job.predict(inference_dataset.x, train_dataset=train_dataset)
        return infer_res
    
    def _inference_mp(self, job, models_dir, map_info):
        inference_dataset = self.dataset.load_data(self.dataset.test_url, "inference")
        inference_output_dir = os.path.join(self.workspace, "output/inference/")
        os.environ["RESULT_SAVED_URL"] = inference_output_dir
        job.load(models_dir, map_info)
        infer_res = job.predict(inference_dataset.x)
        return infer_res

    def _partition(self, partition_point_list, initial_model_path, sub_model_dir):
        cnt = 0
        map_info = dict({})
        for point in partition_point_list:
            cnt += 1
            input_names = point['input_names']
            output_names = point['output_names']
            sub_model_path = sub_model_dir + '/' + 'sub_model_' + str(cnt) + '.onnx'
            try:
                onnx.utils.extract_model(initial_model_path, sub_model_path, input_names, output_names)
            except Exception as e:
                LOGGER.info(str(e))
            map_info[sub_model_path.split('/')[-1]] = point['device_name']
        return sub_model_dir, map_info
