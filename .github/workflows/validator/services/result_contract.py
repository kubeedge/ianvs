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

"""Compatibility contract for validator result artifacts."""

from pathlib import Path
from typing import Mapping


RESULT_SCHEMA_VERSION = 1


def validate_result_payload(payload: object, source_path: Path) -> Mapping[str, object]:
    """Reject malformed or explicitly incompatible validator artifacts.

    Artifacts without a schema version remain readable for compatibility with
    results produced before the contract was versioned.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("{} does not contain a JSON object".format(source_path))

    schema_version = payload.get("schema_version", RESULT_SCHEMA_VERSION)
    if type(schema_version) is not int or schema_version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            (
                "{} uses unsupported validator result schema version {}; "
                "expected {}"
            ).format(
                source_path,
                schema_version,
                RESULT_SCHEMA_VERSION,
            )
        )
    if not isinstance(payload.get("examples"), list):
        raise ValueError("{} does not contain an examples list".format(source_path))
    return payload
