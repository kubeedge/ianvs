# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator-only scenario definitions; never pass this catalog to an agent."""

SCENARIOS = {
    "S1": {"name": "model_path", "category": "service_configuration",
           "cause": "config.model_path", "backend": "wsl-process", "implemented": True},
    "S2": {"name": "gpu_allocator_budget", "category": "service_configuration",
           "cause": "config.gpu_memory", "backend": "isolated-gpu", "implemented": True},
    "S3": {"name": "truncated_weights", "category": "model_artifact",
           "cause": "artifact.integrity", "backend": "wsl-process", "implemented": True},
    "S4": {"name": "image_platform", "category": "heterogeneous_environment",
           "cause": "image.platform", "backend": "dedicated-kubeedge", "implemented": True},
    "S5": {"name": "stale_edge_release", "category": "cloud_edge_consistency",
           "cause": "control_link.disconnected", "backend": "dedicated-kubeedge", "implemented": True},
}
