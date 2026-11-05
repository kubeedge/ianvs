# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap only the example package and repository path, not a cluster."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from infra_repair.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
