<<<<<<< HEAD
# New Government Poster Agent System

## Table of Contents

- [Introduction](#introduction)
- [System Highlights](#system-highlights)
- [Supported Government Document Types](#supported-government-document-types)
- [Technical Features](#technical-features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
  - [Step 1: Configure API Key](#step-1-configure-api-key)
  - [Step 2: Enter Ianvs Directory](#step-2-enter-ianvs-directory)
  - [Step 3: Run the System](#step-3-run-the-system)
- [Installation and Configuration](#installation-and-configuration)
- [VLM Evaluation System](#vlm-evaluation-system)
- [Output Structure and Reports](#output-structure-and-reports)
- [Project Structure](#project-structure)
- [Performance Optimization](#performance-optimization)

---

## Introduction

The **New Government Poster Agent System** is an intelligent document-to-visual poster generation platform built upon the **Ianvs** architecture. It automatically converts government reports (PDF format) into high-quality visual posters through a **four-component architecture**, including:

- **Parser**
- **Planner**
- **Painter**
- **Evaluator (VLM-based)**

The system performs an end-to-end transformation from PDF to visual poster, using a **Visual-Language Model (VLM) score (0–10 scale)** as the sole evaluation metric. A poster with a score above **8.0** is considered optimized and complete.

---

## System Highlights

### 🚀 Core Features

- **Intelligent Document Parsing:**  
  Extracts structured information (text, tables, images) from PDF government documents using PyPDF2 and PyMuPDF.
- **Smart Layout Planning:**  
  Automatically designs poster layouts based on document content, supporting six document categories.
- **High-Quality Poster Rendering:**  
  Generates PowerPoint-based posters with official government design styles.
- **VLM-Based Quality Evaluation:**  
  Uses a 5-dimensional scoring model (layout, color, typography, hierarchy, compliance).
- **Self-Optimization Loop:**  
  Iteratively improves poster design until it achieves ≥8.0 score.
- **Parallel Processing:**  
  Supports multi-document concurrent processing to boost throughput.

---

## Supported Government Document Types

| Type | Description | Identifier |
|------|--------------|-------------|
| Policy Guidance | Policies, plans, resolutions | `guiding_policy` |
| Implementation Methods | Regulations, guidelines, operation measures | `implementation` |
| Service Guides | User guides, service procedures | `service_guide` |
| Notices | Announcements, statements, bulletins | `notices` |
| FAQs | Question–answer documents, consultation replies | `faq` |
| Legal Documents | Laws, regulations, articles | `legal_documents` |

---

## Technical Features

- **Four-Component Collaboration:** Parser, Planner, Painter, and Evaluator work as independent yet coordinated modules.
- **Comprehensive VLM Evaluation:** 5 dimensions of assessment, each weighted equally (2 points × 5 = 10).
- **Intelligent Optimization Mechanism:** Automatic layout tuning and refinement based on feedback.
- **Government Style Adaptation:** Built-in palette and typography standard for government themes.
- **Extensibility:** Modularized design for easy customization or integration of new agents.
- **Performance-Oriented Design:** Supports parallel processing and asynchronous optimization.

---

## System Architecture

```
New Government Poster Agent System
├── Parser (gov_parser.py)
│   ├── PDF content extraction
│   ├── Text and image parsing
│   ├── Document classification
│   └── Key information structuring
├── Planner (gov_planner.py)
│   ├── Layout generation
│   ├── Content planning
│   ├── Theme configuration
│   └── Feedback-based layout optimization
├── Painter (gov_painter.py)
│   ├── PowerPoint poster rendering
│   ├── Government-style theme application
│   ├── Image enhancement
│   └── Multi-format export
└── Evaluator (gov_evaluator.py)
    ├── 5-dimensional VLM scoring
    ├── Detailed evaluation report
    ├── Improvement suggestion generation
    └── Optimization feedback propagation
```

---

## Quick Start

### Step 1: Configure API Key

Edit the 64th line in `basemodel.py`:

```bash
vim singletask_learning_bench/testalgorithms/gen/basemodel.py
```

Locate:

```python
self.api_key = ''
```

Replace with your **DashScope API key**:

```python
self.api_key = 'your_api_key_here'
```

---

### Step 2: Enter Ianvs Directory

```bash
cd /home/linux/Desktop/ianvs
```

---

### Step 3: Run the System

```bash
ianvs -f examples/GovDoc2Poster/singletask_learning_bench/testalgorithms/gen/government_data_source.yaml
```

✅ Done! The system will automatically parse the government documents from the dataset and generate visual posters.

---

## Installation and Configuration

### Environment Requirements

- **Python:** 3.9+
- **Memory:** ≥8GB
- **Storage:** ≥10GB free space
- **GPU:** Optional (recommended for faster inference)

---

### Installation Steps

1. **Navigate to the project directory**
   ```bash
   cd ./ianvs/examples/new_government_agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```python
   # singletask_learning_bench/testalgorithms/gen/basemodel.py
   self.api_key = 'your_api_key_here'
   ```

   > 💡 Obtain your API key from [Alibaba Cloud DashScope Console](https://dashscope.console.aliyun.com/)

4. **Prepare test data**
   Place test PDF files in:
   ```
   resources/datasets/test/
   ```

---

## VLM Evaluation System

### Evaluation Dimensions (Total: 10 points)

| Dimension | Description | Weight |
|------------|-------------|---------|
| Layout Rationality | Overall layout and spatial distribution | 2 |
| Color Harmony | Color coordination and visual consistency | 2 |
| Font Readability | Text clarity and accessibility | 2 |
| Information Hierarchy | Logical structure and information layering | 2 |
| Government Compliance | Formality and policy-style alignment | 2 |

### Optimization Mechanism

- **VLM Threshold:** 8.0/10  
- **Trigger Condition:** Score < 8.0  
- **Stop Condition:** Score ≥ 8.0 or max iterations reached  
- **Quality Levels:**  
  - Excellent (9–10)  
  - Good (8–9)  
  - Average (6–8)  
  - Poor (4–6)  
  - Very Poor (0–4)

---

## Output Structure and Reports

### Output Directory

```
new_government_agent_output/
├── posters/                # Generated posters (.png/.pptx)
├── images/                 # Extracted image elements
├── logs/                   # Processing logs
├── reports/                # Evaluation reports
│   ├── evaluation_stats_*.json
│   └── evaluation_report_*.json
└── temp/                   # Temporary cache files
```

### Report Contents

Each evaluation report includes:
- Overall VLM score and five sub-scores  
- Optimization history and iteration record  
- Improvement suggestions  
- Performance statistics  

---

## Project Structure

```
new_government_agent/
└── singletask_learning_bench/
    ├── benchmarkingjob.yaml
    ├── testalgorithms/
    │   └── gen/
    │       ├── basemodel.py          # Main algorithm (API key required)
    │       ├── gen_algorithm.yaml    # Algorithm configuration
    │       ├── gov_parser.py         # Parser component
    │       ├── gov_planner.py        # Planner component
    │       ├── gov_painter.py        # Painter component
    │       └── gov_evaluator.py      # Evaluator component
    ├── testenv/
    │     ├── acc.py
    │     └── testenv.yaml
    ├── requirements.txt
    └── README.md
```

### Key Files

| File Path | Description | Modification |
|------------|-------------|---------------|
| `basemodel.py` | Main algorithm, includes API key config | ⚠️ Must edit line 64 |
| `gen_algorithm.yaml` | Algorithm and model hyperparameters | Optional |
| `benchmarkingjob.yaml` | Ianvs benchmarking configuration | Optional |
| `resources/datasets/test/*.pdf` | Test dataset | Required |

---

=======
version https://git-lfs.github.com/spec/v1
oid sha256:232276de521aa3c9761659a35ac749fc2fe9f7bd7d483cc857310d50f8f69747
size 8294
>>>>>>> 9676c3e (ya toh aar ya toh par)
