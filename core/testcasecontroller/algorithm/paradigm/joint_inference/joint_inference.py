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

"""Cloud-Edge Joint Inference"""

# Ianvs imports
import os
from copy import deepcopy
from core.common.constant import ParadigmType
from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase
from sedna.common.class_factory import ClassFactory, ClassType
from sedna_joint_inference import JointInference as SednaJointInference

class JointInference(ParadigmBase):
    """
    Cloud-Edge-JointInference:
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
        self.kwargs = kwargs
        self.initial_model = kwargs.get("initial_model_url")
        self.hard_example_mining_mode = kwargs.get("hard_example_mining_mode", "mining_edge_or_cloud")

    def run(self):
        """
        run the test flow of multi-edge inference paradigm.

        Returns
        ------
        test result: numpy.ndarray
        system metric info: dict
            information needed to compute system metrics.

        """

        job = self.build_paradigm_job(ParadigmType.JOINT_INFERENCE.value)

        inference_result = self._inference(job, self.initial_model)

        return inference_result, self.system_metric_info

    def _inference(self, job : SednaJointInference, trained_model):
        # Ianvs API
        inference_dataset = self.dataset.load_data(self.dataset.test_url, "inference")
        inference_output_dir = os.path.join(self.workspace, "output/inference/")
        os.environ["RESULT_SAVED_URL"] = inference_output_dir

        # inference via sedna JointInference API 
        infer_res = job.inference(inference_dataset.x, mining_mode=self.hard_example_mining_mode)
        
        return infer_res