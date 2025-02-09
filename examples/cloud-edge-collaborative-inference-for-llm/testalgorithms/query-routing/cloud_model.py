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

from __future__ import absolute_import, division, print_function

import os

from core.common.log import LOGGER
from sedna.common.class_factory import ClassType, ClassFactory
from models import APIBasedLLM

os.environ['BACKEND_TYPE'] = 'TORCH'

__all__ = ["BaseModel"]

@ClassFactory.register(ClassType.GENERAL, alias="CloudModel")
class CloudModel:
    """Models being deployed on the Cloud
    """
    def __init__(self, **kwargs):
        """Initialize the CloudModel.  See `APIBasedLLM` for details about `kwargs`.
        """
        LOGGER.info(kwargs)
        self.model = APIBasedLLM(**kwargs)
        self.load(kwargs.get("model", "gpt-4o-mini"))

    def load(self, model):
        """Set the model.

        Parameters
        ----------
        model : str
            Existing model from your OpenAI provider. Example: `gpt-4o-mini`
        """
        self.model._load(model = model)

    def inference(self, data, **kwargs):
        """Inference the model with the given data.

        Parameters
        ----------
        data : dict
            The data to be used for inference. See format at BaseLLM's `inference()`.
        kwargs : dict
            To Align with Sedna's JointInference interface.

        Returns
        -------
        dict
            Formatted Response. See `model._format_response()` for more details.
        """

        return self.model.inference(data)

    def cleanup(self):
        """Save the cache and cleanup the model.
        """

        self.model.save_cache()
        self.model.cleanup()