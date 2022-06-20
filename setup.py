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

"""Setuptools of Ianvs"""
import sys
import os

from setuptools import setup, find_packages

assert sys.version_info >= (3, 6), "Sorry, Python < 3.6 is not supported."


class InstallPrepare:
    """
    Parsing dependencies
    """

    def __init__(self):
        self.project = os.path.join(os.path.dirname(__file__), "core")
        self._long_desc = os.path.join(self.project, "..", "README.md")
        self._owner = os.path.join(self.project, "..", "OWNERS")
        self._requirements = os.path.join(self.project, "..", "requirements.txt")

    @property
    def long_desc(self):
        if not os.path.isfile(self._long_desc):
            return ""
        with open(self._long_desc, "r", encoding="utf-8") as fh:
            long_desc = fh.read()
        return long_desc

    @property
    def version(self):
        default_version = "0.1.0"
        return default_version

    @property
    def owners(self):
        default_owner = "ianvs"
        if not os.path.isfile(self._owner):
            return default_owner
        with open(self._owner, "r", encoding="utf-8") as fh:
            check, approvers = False, set()
            for line in fh:
                if not line.strip():
                    continue
                if check:
                    approvers.add(line.strip().split()[-1])
                check = (line.startswith("approvers:") or
                         (line.startswith(" -") and check))
        return ",".join(approvers) or default_owner

    @property
    def basic_dependencies(self):
        return self._read_requirements(self._requirements)

    @staticmethod
    def _read_requirements(file_path, section="all"):
        print(f"Start to install requirements of {section} "
              f"in ianvs from {file_path}")
        if not os.path.isfile(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            install_requires = [p.strip() for p in f.readlines() if p.strip()]
        if section == "all":
            return list(filter(lambda x: not x.startswith("#"),
                               install_requires))
        section_start = False
        section_requires = []
        for p in install_requires:
            if section_start:
                if p.startswith("#"):
                    return section_requires
                section_requires.append(p)
            elif p.startswith(f"# {section}"):
                section_start = True
        return section_requires


_infos = InstallPrepare()

setup(
    name='ianvs',
    version=_infos.version,
    description="The ianvs package is designed to help algorithm developers \
                better do algorithm test.",
    packages=find_packages(exclude=["tests", "*.tests",
                                    "*.tests.*", "tests.*"]),
    author=_infos.owners,
    author_email="yangjin39@huawei.com",
    maintainer=_infos.owners,
    maintainer_email="",
    include_package_data=True,
    entry_points={
        "console_scripts": ["ianvs = core.cmd.benchmarking:main"]
    },
    python_requires=">=3.6",
    long_description=_infos.long_desc,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/kubeedge/ianvs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
)
