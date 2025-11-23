<<<<<<< HEAD
# PIPL-Compliant Cloud-Edge Collaborative Privacy-Preserving Prompt Processing Framework

This example implements a PIPL-compliant cloud-edge collaborative privacy-preserving LLM inference workflow validated with the ChnSentiCorp-Lite dataset, including:

- Edge-first inference with hard sample mining
- Adaptive privacy desensitization (regex, NER masking, differential privacy)
- Privacy and performance metrics visualization
- Zero raw text cross-border transmission
- Real-time PIPL compliance verification and audit logging

## Directory Structure

```
edge-cloud_collaborative_learning_bench/
├── benchmarkingjob.yaml                    # Benchmarking job configuration
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── test_algorithms/                        # Test algorithms directory
│   ├── algorithm.yaml                      # Algorithm configuration
│   ├── privacy_preserving_llm/            # Privacy-preserving LLM main module
│   │   ├── __init__.py
│   │   └── privacy_preserving_llm.py
│   ├── privacy_detection/                  # Privacy detection module
│   │   ├── __init__.py
│   │   ├── pipl_classifier.py
│   │   ├── pii_detector.py
│   │   └── risk_evaluator.py
│   └── privacy_encryption/                 # Privacy encryption module
│       ├── __init__.py
│       ├── differential_privacy.py
│       ├── saliency_masking.py
│       ├── dimensionality_reduction.py
│       └── compliance_monitor.py
└── testenv/                               # Test environment directory
    ├── testenv.yaml                       # Test environment configuration
    ├── privacy_metrics.py                 # Privacy evaluation metrics
    └── performance_metrics.py             # Performance evaluation metrics
```

## Project Background

The rapid advancement of large language models (LLMs) has driven the adoption of cloud-edge collaborative inference architectures, where computationally intensive inference tasks are distributed between edge devices and cloud servers to optimize performance and resource utilization. However, this paradigm introduces critical privacy challenges, particularly when processing user prompts containing sensitive personal information. With the enactment of China's Personal Information Protection Law (PIPL), which mandates strict requirements for cross-border data transmission including "minimal necessity" and "security assurance" principles, organizations face an urgent need to develop privacy-preserving solutions that comply with regulatory requirements while maintaining inference quality. Traditional approaches often require transmitting raw text across borders, creating significant privacy risks and regulatory compliance challenges. This project addresses the fundamental tension between privacy protection and inference utility in cloud-edge collaborative LLM systems, particularly in scenarios requiring cross-border data processing.

## Problems Solved

This project addresses three critical problems in cloud-edge collaborative LLM inference systems. First, it eliminates the privacy leakage risks associated with raw text cross-border transmission by implementing a zero raw text transmission architecture that converts sensitive prompts into anonymized feature vectors before any cross-border transfer occurs. Second, it ensures PIPL compliance by implementing strict adherence to Articles 38-40 of PIPL, including minimal necessity checks, privacy budget management, and real-time compliance verification mechanisms. Third, it resolves the privacy-utility trade-off challenge by developing adaptive privacy desensitization techniques—including differential privacy, saliency-guided masking, and dimensionality reduction—that preserve inference quality while providing strong privacy guarantees. The framework prevents unauthorized reconstruction of original user data from transmitted anonymized vectors, ensuring that sensitive personal information such as names, identification numbers, and locations cannot be recovered by cloud-side adversaries or through membership inference attacks.

## Project Results

The project has achieved comprehensive results across multiple dimensions. Technically, it delivers a complete PIPL-compliant cloud-edge collaborative privacy-preserving prompt processing framework integrated into KubeEdge-Ianvs, featuring edge-first inference with hard sample mining, adaptive privacy desensitization (regex patterns, NER masking, and differential privacy), and real-time compliance monitoring with audit logging. The framework achieves zero raw text cross-border transmission while maintaining inference accuracy comparable to non-privacy-preserving baselines. Academically, it introduces ChnSentiCorp-Lite, the first PIPL-compliant cross-border LLM inference benchmark dataset with 3,000 samples, multi-layer privacy annotations, synthetic PII templates, and dedicated attack evaluation subsets for comprehensive privacy assessment. The project provides a complete evaluation methodology covering utility metrics (task accuracy, end-to-end latency), privacy metrics (Neighbourhood MIA, LOSS Attack, LiRA), and compliance metrics (minimal necessity validation, budget compliance checks, audit integrity verification). Practically, the framework demonstrates successful deployment with Llama-3-8B-Instruct on edge devices and GPT-4o-mini on cloud servers, showcasing production-ready capabilities for privacy-preserving LLM inference in regulated environments.

## Core Features

### 1. Edge-side Privacy Protection
- Perform irreversible privacy transformation on user's sensitive input prompts
- Convert raw text into anonymized feature vectors
- Complete PII detection, entity recognition, and privacy classification locally

### 2. Cloud-side Inference Processing
- Perform inference based solely on anonymized vectors, never accessing raw text
- Receive minimal necessary tags to execute core inference tasks
- Ensure "Zero Raw Text Cross-Border" and "Minimal Tags Cross-Border"

### 3. PIPL Compliance Assurance
- Strictly adhere to "minimal necessity" and "security assurance" principles
- Real-time privacy budget management and audit logging
- Compliance verification before cross-border transmission

## Model Configuration

### Edge Model
- **Model**: Llama-3-8B-Instruct (4-bit quantized)
- **Function**: Local PIPL entity recognition, semantic classification, and anonymized vector generation
- **Deployment**: Adapted for edge computing environments (e.g., NVIDIA T4)

### Cloud Model
- **Model**: GPT-4o-mini (API access)
- **Function**: Receive anonymized vectors to perform core inference tasks
- **Deployment**: OpenAI API format, ensuring scalable deployment

## Dataset

**ChnSentiCorp-Lite** - First PIPL-compliant cross-border LLM inference benchmark dataset

- **Total Samples**: 3,000 (2,000 train, 500 validation, 500 test)
- **Data Source**: Carefully curated subset of ChnSentiCorp Chinese sentiment analysis dataset
- **Format**: JSONL with comprehensive privacy annotations
- **Size**: ~15MB (lightweight for rapid evaluation)

### Key Dataset Contributions
- **Multi-layer Privacy Annotations**: Each sample tagged with privacy sensitivity levels and PII entity types
- **Synthetic PII Templates**: 50+ built-in templates for dynamic generation of realistic Chinese personal information
- **PIPL Compliance Mapping**: Granular annotations indicating cross-border transfer permissions under PIPL Articles 38-40
- **Attack Evaluation Subsets**: Dedicated samples for Neighbourhood MIA, LOSS, and LiRA attack testing

## Quick Start

### Requirements
- Python 3.8+
- NVIDIA GPU (recommended T4 or higher)
- API access keys (OpenAI or compatible services)

### Installation Steps

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys:
```bash
export EDGE_API_KEY="your_edge_model_api_key"
export CLOUD_API_KEY="your_cloud_model_api_key"
```

3. Run benchmark:
```bash
ianvs -f benchmarkingjob.yaml
```

## Evaluation Methods

### 1. Utility Evaluation
- **Task Accuracy**: Compare accuracy changes before and after enabling privacy transformations
- **End-to-End Latency**: Measure total time from user prompt input to receiving final response

### 2. Privacy Evaluation
- **Neighbourhood MIA**: Model-agnostic approach using semantically similar neighbor samples
- **LOSS Attack**: Traditional loss-based membership inference baseline
- **LiRA**: Advanced likelihood ratio test with theoretical optimality properties

### 3. Compliance Evaluation
- **Minimal Necessity Check**: Payload structure validation
- **Budget Compliance Check**: ε accumulation validation
- **Audit Integrity Check**: Log coverage verification

## Technical Architecture

The system adopts strict separation of duties between cloud and edge to ensure compliance of data processing workflows:

1. **Privacy Detection Module**: Identifies and classifies privacy-sensitive information in user prompts
2. **Privacy Encryption Module**: Performs irreversible transformation of sensitive prompts into anonymized vectors
3. **Edge Inference**: Local privacy processing and preliminary inference
4. **Cloud Collaboration**: Advanced inference based on anonymized data
5. **Compliance Monitoring**: Real-time monitoring and audit logging

## Privacy Protection Technologies

- **Differential Privacy**: L2-norm clipping, Gaussian noise injection, budget tracking
- **Saliency-Guided Masking**: Attention-based token importance with configurable suppression
- **Dimensionality Reduction**: Johnson-Lindenstrauss projection with semantic preservation
- **Compliance Verification**: Real-time monitoring and audit logging

## Contributing

We welcome contributions, bug reports, and improvement suggestions. Please follow these steps:

1. Fork this repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project uses the same license as KubeEdge-Ianvs.

## Contact

For questions or suggestions, please contact us through GitHub Issues.
=======
version https://git-lfs.github.com/spec/v1
oid sha256:37331a137fa091fff553f8815fa4988124acdb305c376a6a62533ed7635d73b0
size 10309
>>>>>>> 9676c3e (ya toh aar ya toh par)
