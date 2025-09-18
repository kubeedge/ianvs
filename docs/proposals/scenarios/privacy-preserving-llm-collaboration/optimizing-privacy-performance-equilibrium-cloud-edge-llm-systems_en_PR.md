# **PIPL-Compliant Cloud-Edge Collaborative Privacy-Preserving Prompt Processing Framework on KubeEdge-Ianvs (#203)**

**Associated Issue:** Closes #203

## **Description**

With the widespread adoption of Large Language Models (LLMs), the traditional "direct-to-cloud" inference model requires users to upload prompts containing potentially sensitive information to remote servers. This poses significant privacy and compliance risks, especially in scenarios involving the cross-border transfer of personal information. Meanwhile, lightweight models deployed purely on the edge often fail to meet complex inference performance requirements.  
This proposal implements a "Cloud-Edge Collaborative, PIPL-Compliant" privacy-preserving LLM inference framework on the KubeEdge-Ianvs platform. The core idea is:
 The **Edge** performs an irreversible privacy transformation on the user's sensitive input prompt, converting it into an anonymized feature vector. 
 The **Cloud** then performs inference based solely on this anonymized vector, without ever accessing or storing the original raw text. This design aims to achieve "Zero Raw Text Cross-Border" and "Minimal Tags Cross-Border," ensuring strict adherence to the "minimal necessity" and "security assurance" principles of China's Personal Information Protection Law (PIPL) while leveraging the powerful capabilities of cloud-based models.

## **Models & Datasets**

To ensure the reproducibility of the scenario and the determinism of the evaluation, this solution uses the following fixed model and dataset configuration:

* **Edge Model:** Llama-3-8B-Instruct (4-bit/INT4 quantized version).  
  * **Purpose:** Runs on the edge device, responsible for local PIPL entity/intent recognition, auxiliary semantic classification, and extracting text hidden states to generate the anonymized vector. The 4-bit quantization is intended to fit edge computing environments with limited VRAM, such as NVIDIA T4 or consumer-grade GPUs.  
* **Cloud Model:** GPT-4o-mini.  
  * **Purpose:** Serves as the large language model in the cloud, receiving the anonymized vector and minimal necessary tags from the edge to perform core inference and generation tasks.  
* **Dataset:** ChnSentiCorp-Lite.  
  * **Source:** A sampled subset of the ChnSentiCorp Chinese sentiment analysis dataset, containing 2,000 training, 500 validation, and 500 test samples.  
  * **Purpose:** Used to evaluate the framework's utility on general text. Additionally, the framework includes a built-in set of PII (Personally Identifiable Information) synthesis templates to dynamically generate samples containing highly sensitive information locally (e.g., inserting names, phone numbers into ChnSentiCorp-Lite samples). This is specifically for evaluating high-sensitivity policies, and **the synthesized PII data is guaranteed not to leave the edge device**.

## **Value to Ianvs**

* **New Compliance Scenario & Baseline:** Introduces a runnable "Privacy-Preserving Cloud-Edge Collaborative LLM" scenario and evaluation baseline to Ianvs for the first time, expanding the platform's application in AI security and compliance.  
* **Reusable Privacy Components:** Contributes a series of pluggable core privacy components, including an edge privacy detector, an irreversible transformation engine, a compliance gate/budget manager, and a comprehensive privacy attack evaluation suite.  
* **Lightweight Reproduction Scheme:** Significantly lowers the hardware barrier and evaluation cost for community members by using Colab to simulate the edge environment, combined with a lightweight dataset and one-click scripts.  
* **PIPL Compliance Practice Template:** Provides an engineering template based on PIPL that covers the core requirements of "minimal necessity," "budget auditing," and "zero raw text cross-border," serving as a reference for other compliance-sensitive industries like finance and healthcare.

## **Scenario Example (Cross-Border E-commerce Customer Service, Multi-turn)**

This is a simulated multi-turn conversation in a cross-border e-commerce scenario, demonstrating how the framework dynamically adjusts its privacy strategy based on the conversation content.

* **Jurisdiction & Computing Power:** The edge side is located within mainland China (simulated via Colab), while the cloud side is located overseas (Ianvs inference service).  
* **Privacy Policy Parameters:** High-sensitivity ε≤0.8, general ε≤1.2, default δ=1e-5; session-level budget accumulation and rate limiting.  
* **Zero Raw Text Cross-Border:** Only the "anonymized vector \+ minimal necessary tags" are transmitted throughout the entire process, ensuring the original text never leaves the local jurisdiction.

**Turn 1**

* **User:** The iPhone 15 Pro I bought from your store last week is overheating badly. How can I return it?  
* **Edge Side:**  
  * The PIPL classifier identifies this prompt as **General Sensitivity** (product issue, no PII).  
  * Applies policy: ε=1.0, saliency mask ratio 0.4, projection dimension 768→64.  
  * **Cross-Border Payload:** Anonymized Vector (64-dim) + {"intent": "return_request"}.  
* **Cloud Side:** Based on the anonymized vector and intent tag, it generates a response with the general return policy and procedure.

**Turn 2**

* **User:** The order number is 2024-09-01-3309, recipient is John Doe, phone is 138****2567. Do I need to pay for the return shipping?  
* **Edge Side:**  
  * The PIPL classifier identifies **High-Sensitivity Entities** (order number/name/phone).  
  * Triggers high-sensitivity policy: ε=0.8; performs local order number validation and entity sanitization.  
  * **Cross-Border Payload:** Anonymized Vector (64-dim) + {"order_valid": true, "return_period": "in_range"}.  
* **Cloud Side:** Based on the anonymized context and boolean tags, it generates an explanation of the shipping fee policy without including any PII.

**Auditing:** The PIPL classification, sanitization/transformation parameters (ε, mask_ratio, proj_dims), budget accumulation, and cross-border transmission hash are logged for each turn, fulfilling the requirements of minimal necessity and traceability for compliance.

## **Directory & Artifacts**

Project artifacts are organized as follows, covering documentation, code, evaluations, and configurations:

```
privacy_llm_cross_border/
├── README.md
├── configs/
│ ├── model_config.yaml
│ └── privacy_policy.yaml
├── data/
│ └── chnsenticorp_lite/
│ ├── test.jsonl
│ └── train.jsonl
├── scripts/
│ └── run_benchmark.sh
└── src/
├── cloud_inference.py
├── edge_privacy.py
└── evaluation/
├── attacks.py
└── metrics.py

```

## **Architecture & End-to-End Flow**

The system adopts a strict separation of duties between the cloud and the edge to ensure the compliance of the data processing workflow.

#### **System Component Architecture**

![System Component Architecture](<./images/System Component Architecture.png>)

#### **End-to-End Workflow**

![End-to-End Workflow](<./images/End-to-End Workflow.png>)

## **Core Algorithm Design**

### **1. Privacy Detection (PIPL Classification)**

* **Detection Channels:** Edge combines regex/rule dictionaries (e.g., ID, phone), a lightweight NER, and a semantic classifier to tag PII locally.  
* **Risk Scoring & Policy Mapping:** Aggregate results into risk∈[0,1], then map to configs/privacy_policy.yaml to obtain ε, mask_ratio, proj_dims.  
* **Output:** {entities, pipl_level ∈ {general, high}, risk, policy}

### **2. Irreversible Transformation (Privacy "Encryption")**

This module integrates adaptive differential privacy, saliency-guided masking, and random/multi-layer projection to achieve efficient and irreversible privacy protection.  
Key steps:  
* L2 clipping and calibrated Gaussian noise for (ε, δ)-DP.  
* Saliency-guided masking with top-k suppression controlled by mask_ratio.  
* Johnson–Lindenstrauss random projection or MLP-based projection to lower dimensions.  
* Parameters dynamically set by policy: {epsilon, mask_ratio, proj_dims}.

* **Budget Management & Compliance Gate:** At the session level, the PrivacyBudgetManager is responsible for accumulating the consumed privacy budget and enforcing rate limits. The ComplianceGate performs a final check before cross-border data transmission to ensure the payload conforms to the "anonymized vector + minimal tags" format.

## **Evaluation Methods & Report**

A one-click script, run_eval_all.sh, is provided to execute three categories of evaluations and generate a report.

### **1. Utility Evaluation**

* **Purpose:** To measure the system's performance on downstream tasks and the associated overhead after introducing the privacy-preserving transformations.  
* **Metrics:**  
  * **Task Accuracy:** For the text sentiment classification task on the ChnSentiCorp-Lite dataset, compare the accuracy change of the cloud-side LLM inference before and after enabling privacy transformations.  
  * **End-to-End Latency:** Measure the total time from user prompt input to receiving the final response to assess the additional performance overhead introduced by the edge-side privacy processing.

### **2. Privacy Evaluation**

* **Purpose:** To quantitatively assess the risk of information leakage.  
* **Core Method:** Employs the **Neighbourhood Membership Inference Attack (MIA)** as the primary evaluation method, compared against two baseline methods.  
  * **Core Idea:** A model typically exhibits lower loss (i.e., higher confidence) for samples it was trained on (members) compared to very similar, unseen samples (non-members). The neighborhood attack leverages this by creating "neighbor" samples through small, meaning-preserving modifications to a target sample, without needing a reference model. It then compares the model's loss on the target sample to the average loss on its neighbors. If the target's loss is significantly lower, it is classified as a member.  
  * **Neighborhood Generation:** We use a word substitution method based on a BERT-like Masked Language Model (MLM) to generate neighbors. Specifically, for a word in the original sentence, we apply a high dropout rate to its input embedding and then have the MLM predict the most suitable replacement words, thus generating a set of semantically and grammatically similar neighbor samples.  
* **Metrics:**  
  * **MIA Comparison:** Report the attack's True Positive Rate (TPR) at very low False Positive Rates (FPR), i.e., **TPR @ FPR ∈ {1%, 0.1%, 0.01%}**, and the **AUC (Area Under the Curve)**. Compare the results of the neighborhood attack with the **LOSS Attack** and **LiRA (Likelihood Ratio Attack)** baselines.  
  * **Mutual Information Proxy:** Calculate the Normalized Mutual Information (NMI) between the anonymized vector and the original PII categories as a proxy metric for leakage risk.  
  * **Inversion Alert Baseline:** Implement a basic embedding inversion model and report its BLEU score on the anonymized vectors as a baseline alert for defense effectiveness.

### **3. Compliance Evaluation**

* **Purpose:** To verify that the system meets PIPL engineering requirements.  
* **Metrics:** Minimal Necessity Check (payload structure validation), Budget Compliance Check (ε accumulation validation), Audit Integrity Check (log coverage).

## **Execution & Reproduction**

* **One-Click Scripts:**  
  * scripts/setup_colab_edge.sh: Quickly configures the Colab edge environment.  
  * scripts/run_edge_pipeline.sh: Starts the edge-side service.  
  * scripts/run_cloud_infer.sh: Starts the cloud-side inference service.  
  * scripts/run_eval_all.sh: Runs all utility, privacy, and compliance evaluations.  
* **Hardware Recommendations:** For the edge side, a GPU with NVIDIA T4 (16GB) or higher VRAM is recommended to smoothly run the 4-bit quantized Llama-3-8B-Instruct.  
* **Key Configuration Examples:**

```yaml
# configs/model_edge.yaml
model_name: "meta-llama/Llama-3-8B-Instruct"
quantization: "4bit"
hidden_layer_index: -2 # Second to last layer
pooling_strategy: "mean"
projection_dim: 768 # Project down to 768
```

```yaml
# configs/model_cloud.yaml
model_name: "gpt-4o-mini"
adapter_config:
  input_dim: 64
  hidden_dim: 512
  output_dim: 4096 # Align with GPT-4o-mini's embedding dim
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



## **Acceptance Criteria (DoD)**

* **Engineering:**  
  * The single model/dataset combination is clearly declared in docs and configs and can be verified by CI scripts.  
  * The core architecture and workflow diagrams (PNGs) and their corresponding PlantUML source code are in place.  
  * The one-click script run_eval_all.sh can successfully run the end-to-end flow and all three evaluation categories.  
* **Algorithm/Evaluation:**  
  * A comparative evaluation of the LOSS, LiRA, and Neighbourhood MIA attacks is successfully implemented.  
  * The evaluation report clearly presents a comparison table of **TPR@FPR** and **AUC**.  
  * The privacy budget and transformation parameters are dynamically applied based on the pipl_level.  
* **Compliance:**  
  * Packet inspection verifies that the cross-border payload strictly adheres to the "anonymized vector + minimal tags" structure.  
  * Unit tests cover scenarios ensuring no raw text or PII is transmitted cross-border.  
  * Edge/cloud audit logs contain all critical parameters and verification hashes.

## **Limitations & Future Plans**

* The current TransformEngine parameters are statically configured. Future work could explore adaptive strategies that dynamically adjust the transformation strength based on the input content.  
* The embedding inversion defense baseline is preliminary. Stronger inversion attack models could be integrated for stress testing in the future.  
* The current solution only supports the text modality. It could be extended to protect privacy for multi-modal inputs like speech and images in the future.

## **Commit Message**

feat(privacy_llm_cross_border): PIPL-compliant cloud-edge LLMs with Edge=Llama-3-8B-Instruct and MIA eval (#203)

- End-to-end with irreversible edge transforms and minimal-tag cross-border.  
- Unique models/dataset: Edge Llama-3-8B-Instruct 4bit; Cloud GPT-4o-mini; ChnSentiCorp-Lite.  
- Privacy eval: LOSS/LiRA/Neighbour with low-FPR TPR/AUC.  

