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

from __future__ import absolute_import, division, print_function

import os
import sys

# Ensure RFNet vendored package is on path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_rfnet_dir = os.path.join(_current_dir, "rfnet")
if _rfnet_dir not in sys.path:
    sys.path.insert(0, _rfnet_dir)

from rfnet.basemodel import BaseModel

__all__ = ["BaseModel"]