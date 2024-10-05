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
        config required for the test process of multi-edge inference paradigm,
        e.g.: algorithm modules, dataset, initial model, etc.

    """

    def __init__(self, workspace, **kwargs):
        ParadigmBase.__init__(self, workspace, **kwargs)
        self.kwargs = kwargs
        self.hard_example_mining_mode = kwargs.get(
            "hard_example_mining_mode",
            "mining-then-inference"
        )

    def set_config(self):
        shot_nums = self.kwargs.get("shot_nums", 0)

        inference_output_dir = os.path.join(os.path.dirname(self.workspace), f"{shot_nums}-shot/")
        os.environ["RESULT_SAVED_URL"] = inference_output_dir
        os.makedirs(inference_output_dir, exist_ok=True)

        LOGGER.info("Loading dataset")

        self.inference_dataset = self.dataset.load_data(
            self.dataset.test_data_info, 
            "inference", 
            shot_nums = self.kwargs.get("shot_nums", 0)
        )

        # validate module instances
        required_modules = {"edgemodel", "cloudmodel", "hard_example_mining"}
        if self.module_instances.keys() != required_modules:
            raise ValueError(
                f"Required modules: {required_modules}, "
                f"but got: {self.module_instances.keys()}"
            )
        
        # if hard example mining is OracleRouter, add the edgemodel and cloudmodel object to its kwargs so that it can use them.
        mining = self.module_instances["hard_example_mining"]
        param = mining.get("param")
        if mining.get("method", None) == "OracleRouter":
            param["edgemodel"] = self.module_instances["edgemodel"]
            param["cloudmodel"] = self.module_instances["cloudmodel"]

    def run(self):
        """
        run the test flow of multi-edge inference paradigm.

        Returns
        ------
        test result: numpy.ndarray
        system metric info: dict
            information needed to compute system metrics.

        """
        self.set_config()

        job = self.build_paradigm_job(ParadigmType.JOINT_INFERENCE.value)

        inference_result = self._inference(job)

        self._cleanup(job)

        return inference_result, self.system_metric_info

    def _cleanup(self, job):
        LOGGER.info("Release models")
        # release module resources
        for module in self.module_instances.values():
            if hasattr(module, "cleanup"):
                module.cleanup()
    
        # Since the hard example mining module is instantiated within the job, special handling is required.
        mining_instance = job.hard_example_mining_algorithm
        if hasattr(mining_instance, "cleanup"):
            mining_instance.cleanup()
        
        del job

    def _inference(self, job):
        results = []

        cloud_count, edge_count = 0,0

        LOGGER.info("Inference Start")

        pbar = tqdm(
            zip(self.inference_dataset.x, self.inference_dataset.y), 
            total=len(self.inference_dataset.x),
            ncols=100
        )

        for data in pbar:
            # inference via sedna JointInference API
            infer_res = job.inference(
                {"messages": data[0], "gold": data[1]},
                mining_mode=self.hard_example_mining_mode
            )

            if infer_res[2]:
                edge_count += 1
            elif infer_res[3]:
                cloud_count += 1

            pbar.set_postfix({"Edge": edge_count, "Cloud": cloud_count})

            results.append(infer_res)

        LOGGER.info("Inference Finished")

        return results # (is_hard_example, res, edge_result, cloud_result)
