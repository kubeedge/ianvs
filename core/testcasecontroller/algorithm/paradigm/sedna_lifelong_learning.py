# Copyright 2026 The KubeEdge Authors.
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

"""Ianvs-side extension of Sedna's LifelongLearning job.

Ianvs runs training, evaluation and inference of lifelong learning in a single
process, with no edge deployment and no knowledge-base server. This subclass
adds what that standalone flow needs on top of Sedna's LifelongLearning,
without changing any of Sedna's existing methods.
"""

import inspect
import os
import tempfile

from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.config import BaseConfig, Context
from sedna.common.file_ops import FileOps
from sedna.core.lifelong_learning import LifelongLearning
from sedna.algorithms.seen_task_learning.task_update_decision.base_task_update_decision \
    import BaseTaskUpdateDecision


class IanvsLifelongLearning(LifelongLearning):
    """Sedna's LifelongLearning with the additions Ianvs's standalone runs need."""

    def __init__(self, *args, **kwargs):
        # Sedna creates its edge knowledge-base directory on construction,
        # defaulting to /var/lib/sedna/kb, which a non-root user cannot
        # create. Standalone runs do not use the edge knowledge base, so
        # default it to a temporary directory unless one is configured.
        if not Context.get_parameters("edge_output_url"):
            os.environ["EDGE_OUTPUT_URL"] = tempfile.mkdtemp(prefix="ianvs-edge-kb-")
        super().__init__(*args, **kwargs)
        # Sedna keeps its local copy of the knowledge-base index at "index.pkl"
        # in the current directory and reuses it if present, so a later round
        # can pick up an earlier round's index. Each job gets its own copy.
        self.cloud_knowledge_management.local_task_index_url = os.path.join(
            tempfile.mkdtemp(prefix="ianvs-kb-"), "index.pkl")
        self._unseen_recognizer = None

    def update(self, train_data, valid_data=None, post_process=None, **kwargs):
        """Sedna's update(), with the previous index resolved as given.

        update() joins DATA_PATH_PREFIX (default /home/data) onto
        CLOUD_KB_INDEX. Ianvs passes an absolute local path, so for that case
        the prefix is treated as "/" for the duration of the call only.
        """
        index = Context.get_parameters("CLOUD_KB_INDEX", "")
        if not (os.path.isabs(index) and os.path.isfile(index)):
            return super().update(train_data, valid_data=valid_data,
                                  post_process=post_process, **kwargs)
        prefix = BaseConfig.data_path_prefix
        BaseConfig.data_path_prefix = os.sep
        try:
            return super().update(train_data, valid_data=valid_data,
                                  post_process=post_process, **kwargs)
        finally:
            BaseConfig.data_path_prefix = prefix

    @property
    def cloud_task_index(self):
        """Cloud knowledge-base index, as left by train() or evaluate()."""
        return self.cloud_knowledge_management.task_index

    def inference_lifelong(self, data=None, post_process=None,
                           detect_unseen=True, **kwargs):
        """Lifelong-learning inference for standalone (Ianvs) runs.

        Unlike inference(), it uses the cloud knowledge-base index and starts
        no edge inference service.

        Parameters
        ----------
        data : BaseDataSource
            inference samples.
        post_process : function or str
            applied to each task model's prediction, as in inference().
        detect_unseen : bool
            when False, skip unseen-sample detection and predict every sample
            with the seen-task models.

        Returns
        -------
        (res, is_unseen_task, tasks)
            the same shape as inference(): is_unseen_task is a bool for one
            sample and a dict keyed by sample for several.
        """
        task_index = self._lifelong_task_index()
        res, tasks = self.cloud_knowledge_management.seen_estimator.predict(
            data=data, post_process=post_process, task_index=task_index,
            task_type="seen_task", **kwargs)
        if not detect_unseen:
            return res, False, tasks

        _, unseen_samples = self._recognize_unseen_samples(
            data, task_index, **kwargs)
        return res, _unseen_flags(data, unseen_samples), tasks

    def _lifelong_task_index(self):
        """Path of the cloud index that core passed in MODEL_URLS."""
        task_index_url = Context.get_parameters(
            "MODEL_URLS", self.cloud_knowledge_management.task_index)
        if os.path.isfile(task_index_url):
            return task_index_url
        local_index = os.path.join(tempfile.mkdtemp(prefix="ianvs-kb-"), "index.pkl")
        FileOps.download(task_index_url, local_index)
        return local_index

    def _recognize_unseen_samples(self, data, task_index, **kwargs):
        if self._unseen_recognizer is None:
            # Built the same way as in LifelongLearning._start_inference_service,
            # but from the cloud knowledge base.
            self._unseen_recognizer = ClassFactory.get_cls(
                ClassType.UTD, self.unseen_sample_recognition["method"])(
                    task_index,
                    estimator=self.cloud_knowledge_management.seen_estimator
                    .estimator.base_model,
                    **self.unseen_sample_recognition_param)

        params = inspect.signature(self._unseen_recognizer.__call__).parameters
        if "scores" in params:
            raise NotImplementedError(
                "unseen-sample recognizers that need seen-task scores are not "
                "supported by inference_lifelong() yet.")
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return self._unseen_recognizer(data, **kwargs)
        return self._unseen_recognizer(data)


def _unseen_flags(data, unseen_samples):
    """is_unseen_task in the form LifelongLearning.inference() returns it."""
    unseen_x = unseen_samples.x if unseen_samples.x is not None else []
    if data.num_examples() == 1:
        return len(unseen_x) > 0
    unseen_names = {x[0] for x in unseen_x}
    return {x[0]: x[0] in unseen_names for x in data.x}


if not ClassFactory.is_exists(ClassType.KM, "UpdateStrategyDefault"):
    # Sedna 0.6.0.1 defaults task_update_decision to "UpdateStrategyDefault"
    # but never defines or registers it. Same behaviour as the class added in
    # #488's sedna-0.6.0.2; registered only when the installed Sedna lacks it.
    # pylint: disable=too-few-public-methods
    @ClassFactory.register(ClassType.KM, alias="UpdateStrategyDefault")
    class UpdateStrategyDefault(BaseTaskUpdateDecision):
        """Retrain every existing task model on the new samples."""

        def __init__(self, task_index, **kwargs):
            super().__init__(task_index)

        def __call__(self, samples, task_type="seen_task"):
            if task_type == self.seen_task_key:
                task_index = self.task_index.get(self.seen_task_key, {})
            else:
                task_index = self.task_index.get(self.unseen_task_key, {})

            tasks = []
            task_update_strategies = {}
            for task_group in task_index.get(self.task_group_key, []):
                if task_group is None or not task_group.tasks:
                    continue
                task = task_group.tasks[0]
                task.samples = samples
                tasks.append(task)
                task_update_strategies[task.entry] = {
                    "raw_data_update": samples,
                    "target_model_update": True,
                    "task_attr_update": None,
                }
            return tasks, task_update_strategies
