# PIPL-Compliant Cloud-Edge Collaborative Privacy-Preserving Prompt Processing Framework on KubeEdge-Ianvs(#203)



## Summary

With the widespread adoption of Large Language Models (LLMs), the traditional direct-to-cloud inference model requires users to upload prompts containing potentially sensitive information to remote servers, posing privacy and compliance risks (especially for cross-border personal information transfers). Pure edge-only lightweight models often cannot meet complex inference requirements. This proposal implements a Cloud-Edge Collaborative, PIPL-compliant privacy-preserving LLM inference framework on the KubeEdge-Ianvs platform. The Edge performs irreversible privacy transformation on user prompts into anonymized feature vectors; the Cloud performs inference solely based on the anonymized vector and minimal necessary tags. This achieves Zero Raw Text Cross-Border and Minimal Tags Cross-Border while leveraging cloud models.

## Table of Contents

- Overview and Value
- Scenario Example
- Models & Datasets
- Architecture & End-to-End Flow
- Core Algorithm Design
- Evaluation Methods & Report
- Execution & Reproduction
- Acceptance Criteria (DoD)
- Limitations & Future Plans


## Overview and Value

- New Compliance Scenario & Baseline: Introduces a runnable Privacy-Preserving Cloud-Edge LLM scenario and baseline for Ianvs.
- Reusable Privacy Components: Edge privacy detector, irreversible transform engine, compliance gate/budget manager, and privacy attack suite.
- Lightweight Reproduction: Colab-based edge simulation, lightweight dataset, and one-click scripts.
- PIPL Compliance Template: Engineering template for minimal necessity, budget auditing, and zero raw text cross-border.

## Scenario Example (Cross-Border E-commerce, Multi-turn)

- Jurisdiction & Compute: Edge in mainland China (Colab simulated), Cloud overseas (Ianvs inference service).
- Privacy Policy: high-sensitivity ε�?.8, general ε�?.2, default δ=1e-5; session-level budget accumulation and rate limiting.
- Zero Raw Text Cross-Border: Only anonymized vector + minimal necessary tags are transmitted.

Turn 1

- User: The iPhone 15 Pro I bought last week overheats. How can I return it?
- Edge: PIPL classifier �?General Sensitivity. Apply ε=1.0, mask_ratio=0.4, projection 768�?4. Payload: 64-dim vector + {"intent": "return_request"}.
- Cloud: Generates return policy guidance based on anonymized context.

Turn 2

- User: Order 2024-09-01-3309, recipient John Doe, phone 138****2567. Do I pay shipping?
- Edge: High-Sensitivity entities detected; apply ε=0.8; local validation and sanitization. Payload: 64-dim vector + {"order_valid": true, "return_period": "in_range"}.
- Cloud: Explains shipping policy without any PII.

Auditing: PIPL classification, parameters (ε, mask_ratio, proj_dims), budget accumulation, and payload hashes logged each turn.

## Models & Datasets

- Edge Model: Llama-3-8B-Instruct (4-bit/INT4), for local PIPL detection, auxiliary semantic classification, and hidden-state extraction.
- Cloud Model: GPT-4o-mini, receives anonymized vectors and minimal tags.
- Dataset: ChnSentiCorp-Lite (2k train/500 val/500 test) plus locally synthesized PII templates; synthesized PII never leaves the edge.

## Architecture & End-to-End Flow

The system strictly separates cloud and edge to ensure compliant data flow.

#### System Component Architecture

<p align="center">
  <img src="images/System Component Architecture.png" alt="Component Architecture" width="900">
</p>

#### End-to-End Workflow

<p align="center">
  <img src="images/End-to-End Workflow.png" alt="End-to-End Workflow" width="900">
</p>

## Core Algorithm Design

### 1. Privacy Detection (PIPL Classification)

- Detection Channels: Regex/rules, lightweight NER (e.g., BiLSTM-CRF), and semantic classification executed locally on edge.
- Risk Scoring & Policy Mapping: Aggregate detection results into risk∈[0,1], map to privacy_policy.yaml for ε, mask_ratio, proj_dims; assign pipl_level (general/high).

### 2. Irreversible Transformation (Privacy "Encryption")

Integrates adaptive differential privacy, saliency-guided masking, and random/multi-layer projection into a pipeline:
- Adaptive DP: L2 clipping and calibrated Gaussian noise per (ε, δ) to bound leakage risk.
- Saliency Masking: mask top-k salient embedding dimensions to suppress sensitive cues.
- Projection: random/MLP projection to a lower-dimensional space to reduce invertibility.
- Dynamic Policy: ε and mask_ratio updated per pipl_level; payload is anonymized vector only.

- Budget Management & Compliance Gate: Session-level PrivacyBudgetManager accumulates ε usage and enforces rate limits; ComplianceGate validates payload format before cross-border transmission.

## Evaluation Methods & Report

- Utility: Task Accuracy on ChnSentiCorp-Lite; End-to-End Latency.
- Privacy: Neighbourhood MIA vs LOSS/LiRA; report TPR@FPR �?{1%, 0.1%, 0.01%} and AUC; NMI; inversion alert baseline.
- Compliance: Minimal necessity, budget compliance, and audit integrity checks.

## Execution & Reproduction

- One-Click Scripts: scripts/setup_colab_edge.sh, scripts/run_edge_pipeline.sh, scripts/run_cloud_infer.sh, scripts/run_eval_all.sh
- Hardware: Edge with NVIDIA T4 (16GB) or higher for 4-bit Llama-3-8B-Instruct.
- Key Config Examples:

```yaml
# configs/model_edge.yaml
model_name: "meta-llama/Llama-3-8B-Instruct"
quantization: "4bit"
hidden_layer_index: -2
pooling_strategy: "mean"
projection_dim: 768
```

```yaml
# configs/model_cloud.yaml
model_name: "gpt-4o-mini"
adapter_config:
  input_dim: 64
  hidden_dim: 512
  output_dim: 4096
```

```yaml
# configs/privacy_policy.yaml
pipl_levels:
  - level: "general"
    epsilon: 1.2
    delta: 0.00001
    mask_ratio: 0.4
    proj_dims: 64
  - level: "high_sensitivity"
    epsilon: 0.8
    delta: 0.00001
    mask_ratio: 0.6
    proj_dims: 64
```

## Acceptance Criteria (DoD)

- Engineering: Unique model/dataset declared; diagrams + PlantUML; run_eval_all.sh completes end-to-end and all evaluations.
- Algorithm/Evaluation: LOSS, LiRA, and Neighbourhood MIA implemented with TPR@FPR and AUC comparison.
- Compliance: Packet inspection confirms payload format; unit tests ensure no raw text/PII cross-border; audit logs complete.

## Limitations & Future Plans

- Static TransformEngine parameters; explore adaptive strategies.
- Preliminary inversion defense baseline; integrate stronger attacks.
- Text-only modality; extend to speech/images.


