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


from sedna.common.class_factory import ClassFactory, ClassType
from sedna.datasources import BaseDataSource

@ClassFactory.register(ClassType.GENERAL, alias="OracleRouterDatasetProcessor")
class OracleRouterDatasetProcessor:
    """ A Customized Dataset Processor for Oracle Router"""
    def __init__(self, **kwargs):
        pass

    def __call__(self, dataset):
        """Transform the dataset to another format for Oracle Router

        Parameters
        ----------
        dataset : sedna.datasources.BaseDataSource
            The dataset loaded by Sedna

        Returns
        -------
        sedna.datasources.BaseDataSource
            Transformed dataset
        """
        try:
            dataset.x = [{"query": x, "gold": y} for x, y in zip(dataset.x, dataset.y)]
        except Exception as e:
            raise RuntimeError("Failed to transform dataset for Oracle Router.") from e

        return dataset