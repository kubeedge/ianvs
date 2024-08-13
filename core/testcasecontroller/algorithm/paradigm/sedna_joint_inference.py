
# Sena imports
import os
from copy import deepcopy

from sedna.common.utils import get_host_ip
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.server import InferenceServer
from sedna.service.client import ModelClient, LCReporter
from sedna.common.constant import K8sResourceKind
from sedna.core.base import JobBase
import re

HUGGINGFACE_PATH_PATTERN = r'^[a-zA-Z0-9][\w\-]*/[a-zA-Z0-9][\w\-\.]*$'

# Currently rename "JointInference" to "SednaJointInference"
class JointInference(JobBase):
    """
    Sedna provide a framework make sure under the condition of limited
    resources on the edge, difficult inference tasks are offloaded to the
    cloud to improve the overall performance, keeping the throughput.

    Parameters
    ----------
    estimator : Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    hard_example_mining : Dict
        HEM algorithms with parameters which has registered to ClassFactory,
        see `sedna.algorithms.hard_example_mining` for more detail.

    Examples
    --------
    >>> Estimator = keras.models.Sequential()
    >>> ji_service = JointInference(
            estimator=Estimator,
            hard_example_mining={
                "method": "IBT",
                "param": {
                    "threshold_img": 0.9
                }
            }
        )

    Notes
    -----
    Sedna provide an interface call `get_hem_algorithm_from_config` to build
    the `hard_example_mining` parameter from CRD definition.
    """

    def __init__(self, estimator=None, cloud=None, hard_example_mining: dict = None):
        super(JointInference, self).__init__(estimator=estimator)


        self.job_kind = K8sResourceKind.JOINT_INFERENCE_SERVICE.value
        self.local_ip = get_host_ip()
    
        report_msg = {
            "name": self.worker_name,
            "namespace": self.config.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": "inference",
            "results": []
        }
        period_interval = int(self.get_parameters("LC_PERIOD", "30"))
        self.lc_reporter = LCReporter(lc_server=self.config.lc_server,
                                      message=report_msg,
                                      period_interval=period_interval)
        self.lc_reporter.setDaemon(True)
        self.lc_reporter.start()

        if callable(self.estimator):
            self.estimator = self.estimator()

        check_huggingface_repo = lambda x: bool(re.match(HUGGINGFACE_PATH_PATTERN, x))

        if not os.path.exists(self.model_path) and not check_huggingface_repo(self.model_path):
            raise FileExistsError(f"{self.model_path} miss")
        else:
            self.estimator.load(model_url=self.model_path)

        # If cloud is None, then initialize ModelClint as cloud.
        # Otherwise, we will regard the cloud as a client implemented by the user.
        if cloud is None:
            self.remote_ip = self.get_parameters(
            "BIG_MODEL_IP", self.local_ip)
            self.port = int(self.get_parameters("BIG_MODEL_PORT", "5000"))

            self.cloud = ModelClient(
                service_name=self.job_name,
                host=self.remote_ip, 
                port=self.port
            )
        else:
            self.cloud = cloud

        self.hard_example_mining_algorithm = None
        if not hard_example_mining:
            hard_example_mining = self.get_hem_algorithm_from_config()
        if hard_example_mining:
            # hem = hard_example_mining.get("method", "IBT")
            hem = "BERT"
            hem_parameters = {}
            # hem_parameters = hard_example_mining.get("param", {})
            self.hard_example_mining_algorithm = ClassFactory.get_cls(
                ClassType.HEM, hem
            )(**hem_parameters)

    @classmethod
    def get_hem_algorithm_from_config(cls, **param):
        """
        get the `algorithm` name and `param` of hard_example_mining from crd

        Parameters
        ----------
        param : Dict
            update value in parameters of hard_example_mining

        Returns
        -------
        dict
            e.g.: {"method": "IBT", "param": {"threshold_img": 0.5}}

        Examples
        --------
        >>> JointInference.get_hem_algorithm_from_config(
                threshold_img=0.9
            )
        {"method": "IBT", "param": {"threshold_img": 0.9}}
        """
        return cls.parameters.get_algorithm_from_api(
            algorithm="HEM",
            **param
        )
    
    
    def _get_edge_result(self, data, callback_func, **kwargs):
        edge_result = self.estimator.predict(data, **kwargs)
        res = deepcopy(edge_result)

        if callback_func:
            res = callback_func(res)

        self.lc_reporter.update_for_edge_inference()

        return res, edge_result
    
    def _get_cloud_result(self, data, post_process, **kwargs):
        
        try:
            cloud_result = self.cloud.inference(
                data.tolist(), post_process=post_process, **kwargs)
        except Exception as err:
            self.log.error(f"get cloud result error: {err}")

        res = deepcopy(cloud_result)

        self.lc_reporter.update_for_collaboration_inference()

        return res, cloud_result


    def inference(self, data=None, post_process=None, **kwargs):
        """
        Inference task with JointInference

        Parameters
        ----------
        data: BaseDataSource
            datasource use for inference, see
            `sedna.datasources.BaseDataSource` for more detail.
        post_process: function or a registered method
            effected after `estimator` inference.
        kwargs: Dict
            parameters for `estimator` inference,
            Like:  `ntree_limit` in Xgboost.XGBClassifier

        Returns
        -------
        if is hard sample : bool
        inference result : object
        result from little-model : object
        result from big-model: object
        """

        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        # Try to obtain mining_mode, with the default value being edge_mining_cloud,
        # which means performing edge inference first, then mining difficult cases, 
        # and uploading to the cloud if necessary.
        mining_mode = kwargs.get("mining_mode", "inference-then-mining")

        is_hard_example = False
        sepeculative_decoding = False

        edge_result, cloud_result = None, None

        if mining_mode == "inference-then-mining":
            res, edge_result = self._get_edge_result(data, callback_func, **kwargs)
            
            if self.hard_example_mining_algorithm is None:
                raise ValueError("Hard example mining algorithm is not set.")
            
            is_hard_example = self.hard_example_mining_algorithm(res)
            if is_hard_example:
                res, cloud_result = self._get_cloud_result(data, post_process=post_process, **kwargs)

        elif mining_mode == "mining-then-inference":
            # First conduct hard example mining, and then decide whether to execute on the edge or in the cloud.
            if self.hard_example_mining_algorithm is None:
                raise ValueError("Hard example mining algorithm is not set.")

            is_hard_example = self.hard_example_mining_algorithm(data)
            if is_hard_example:
                if not sepeculative_decoding:
                    res, cloud_result = self._get_cloud_result(data, post_process=post_process, **kwargs)
                else:
                    # do speculative_decoding
                    pass
            else:
                res, edge_result = self._get_edge_result(data, callback_func, **kwargs)
        
        else:
            raise ValueError(
                "Mining Mode must be in ['mining-then-inference', 'inference-then-mining']"
            )

        return [is_hard_example, res, edge_result, cloud_result]
    
