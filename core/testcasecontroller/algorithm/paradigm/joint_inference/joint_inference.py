# Copyright 2024 The KubeEdge Authors.
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

import os
from tqdm import tqdm

from core.common.log import LOGGER
from core.common.constant import ParadigmType
from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase

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
        config required for the test process of joint inference paradigm,
        e.g.: hard_example_mining_mode

    """

    def __init__(self, workspace, **kwargs):
        ParadigmBase.__init__(self, workspace, **kwargs)
        self.inference_dataset = None
        self.kwargs = kwargs
        self.hard_example_mining_mode = kwargs.get(
            "hard_example_mining_mode",
            "mining-then-inference"
        )

    def set_config(self):
        """ Set the configuration for the joint inference paradigm.

        Raises
        ------
        KeyError
            If required modules are not provided.
        """


        inference_output_dir = os.path.dirname(self.workspace)
        os.environ["RESULT_SAVED_URL"] = inference_output_dir
        os.makedirs(inference_output_dir, exist_ok=True)

        LOGGER.info("Loading dataset")

        self.inference_dataset = self.dataset.load_data(
            self.dataset.test_data_info,
            "inference"
        )

        dataset_processor = self.module_instances.get("dataset_processor", None)
        if callable(dataset_processor):
            self.inference_dataset = dataset_processor(self.inference_dataset)

        # validate module instances
        required_modules = {"edgemodel", "cloudmodel", "hard_example_mining"}

        if not required_modules.issubset(set(self.module_instances.keys())):
            raise KeyError(
                f"Required modules: {required_modules}, "
                f"but got: {self.module_instances.keys()}"
            )

        # if hard example mining is OracleRouter,
        # add the edgemodel and cloudmodel object to its kwargs so that it can use them.
        mining = self.module_instances["hard_example_mining"]
        param = mining.get("param")
        if mining.get("method", None) == "OracleRouter":
            param["edgemodel"] = self.module_instances["edgemodel"]
            param["cloudmodel"] = self.module_instances["cloudmodel"]

    def run(self):
        """
        run the test flow of joint inference paradigm.

        Returns
        ------
        inference_result: list
        system_metric_info: dict
            information needed to compute system metrics.

        """
        self.set_config()

        job = self.build_paradigm_job(ParadigmType.JOINT_INFERENCE.value)

        inference_result = self._inference(job)

        self._cleanup(job)

        return inference_result, self.system_metric_info

    def _cleanup(self, job):
        """Call module's cleanup method to release resources

        Parameters
        ----------
        job : Sedna JointInference
            Sedna JointInference API
        """

        LOGGER.info("Release models")
        # release module resources
        for module in self.module_instances.values():
            if hasattr(module, "cleanup"):
                module.cleanup()

        # Special call is required for hard example mining module
        # since it is instantiated within the job.
        mining_instance = job.hard_example_mining_algorithm
        if hasattr(mining_instance, "cleanup"):
            mining_instance.cleanup()

        del job

    def _inference(self, job):
        """Inference each data in Inference Dataset

        Parameters
        ----------
        job : Sedna JointInference
            Sedna JointInference API

        Returns
        -------
        tuple
            Inference Result with the format of `(is_hard_example, res, edge_result, cloud_result)`
        """
        results = []

        cloud_count, edge_count = 0,0

        LOGGER.info("Inference Start")

        pbar = tqdm(
            self.inference_dataset.x,
            total=len(self.inference_dataset.x),
            ncols=100
        )

        for data in pbar:
            # inference via sedna JointInference API
            infer_res = job.inference(
                data,
                mining_mode=self.hard_example_mining_mode
            )

            if infer_res[2]:
                edge_count += 1
            elif infer_res[3]:
                cloud_count += 1

            pbar.set_postfix({"Edge": edge_count, "Cloud": cloud_count})

            results.append(infer_res)

        LOGGER.info("Inference Finished")

        return results
