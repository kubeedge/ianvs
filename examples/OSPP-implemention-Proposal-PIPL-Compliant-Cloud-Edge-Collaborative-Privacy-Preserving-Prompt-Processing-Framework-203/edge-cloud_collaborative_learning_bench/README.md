# PIPL-Compliant Cloud-Edge Collaborative Privacy-Preserving Prompt Processing Framework

This example implements a comprehensive PIPL-compliant cloud-edge collaborative privacy-preserving LLM inference workflow validated with the ChnSentiCorp-Lite dataset, including:

- Edge-first inference with hard sample mining
- Adaptive privacy desensitization (regex, NER masking, differential privacy)
- Privacy and performance metrics visualization
- Zero raw text cross-border transmission
- Real-time PIPL compliance verification and audit logging
- IANVS StoryManager integration for comprehensive reporting
- Membership Inference Attack (MIA) evaluation framework

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
