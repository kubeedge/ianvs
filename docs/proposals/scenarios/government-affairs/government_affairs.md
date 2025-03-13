# Project Proposal: Domain-Specific Large Model Benchmarks for Edge-Oriented E-Government Services

## 1. Introduction
With the rapid adoption of AI in public sectors, domain-specific large language models (LLMs) are increasingly deployed at the edge for real-time, localized decision-making in e-government services. However, existing benchmarks predominantly focus on cloud-centric scenarios, lacking tailored evaluation frameworks for edge environments where data privacy, regional specificity, and resource constraints are critical. This project aims to address this gap by developing a **province-specific benchmark** for Chinese e-government LLMs using KubeEdge-Ianvs, integrating Retrieval-Augmented Generation (RAG) techniques to enhance contextual accuracy.

## 2. Objectives
1. Build a **multi-province knowledge repository** of Chinese e-government data for RAG-enhanced LLM benchmarking.
2. Design **two test modes**:
   - *Province-specific*: Answers generated using only local provincial data
   - *Cross-province*: Responses leveraging nationwide data
3. Implement and compare popular RAG architectures in Ianvs

## 3. Methodology
### 3.1 Data Collection & Processing
- **Sources**: 
  - Provincial government portals (e.g., Zhejiang "Zhejiang Ban")
  - Policy documents from 34 provincial-level regions
  - Localized service catalogs (social security, tax, etc.)

### 3.2 Benchmark Design
| Test Scenario | Knowledge Scope |
|---------------|-----------------|
| Local Policy QA | Single province data |
| Cross-region Service | All provinces |

### 3.3 RAG Implementation
Integrate the following components into Ianvs:
- **Knowledge Indexing Module**
  - Hierarchical storage for provincial/national data
  - FAISS-based vector databases
- **Retrieval Strategies**
  - Sparse-dense hybrid search (BM25 + DPR)
  - Location-aware semantic routing
- **Response Generator**
  - Model-agnostic interface supporting LLaMA/GPT/DeepSeek

## 4. Technical Implementation
### 4.1 System Architecture
<img src="./assets/rag.png" width="200">

### 4.2 Key Innovations
1. **Context-Aware Benchmarking**:  
   Aligns with the Contextual Benchmark Method by evaluating models under three governance scenarios:
   - Local policy compliance (province-specific)
   - National standard interpretation (cross-province)

2. **Edge-Optimized RAG**:
   - smaller index size through province-based sharding
   - Conduct effectiveness tests using different knowledge bases for various edge nodes.


## 5. Expected Outcomes
1. **Ianvs Integration**:
   - New `llm_rag_benchmark` module
   - Comparative analysis dashboard for RAG strategies

2. **Performance Guidelines**:
   - Model selection matrix based on provincial needs
   - Optimal RAG configuration templates

## 6. Timeline
| Phase | Dates | Deliverables |
|-------|-------|--------------|
| Data Collection | Mar 3-21 | Provincial knowledge corpus |
| RAG Integration | Mar 24-Apr 11 | 5 working prototypes |
| Benchmark Tests | Apr 14-May 2 | Cross-province evaluation |
| Optimization | May 5-23 | Performance tuning |
| Finalization | May 26-30 | Documentation & reports |
