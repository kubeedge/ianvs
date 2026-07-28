# Background

With the rapid development of cloud-edge collaborative large models, their application potential in governmental scenarios is becoming increasingly prominent. Currently, the intelligent upgrade of government services urgently requires support for three core application domains: **internal government collaboration**, **public services**, and **enterprise services**. However, existing LLM evaluation systems generally lack standardized methods tailored for government-specific tasks, making it difficult to effectively assess model performance in real-world governmental environments.

Specifically, general-purpose LLMs often struggle with **low accuracy**, **poor adaptability**, and **weak compliance** when applied to tasks such as governmental knowledge Q&A, official document generation, and city event perception—significantly hindering practical deployment. The root cause lies in the **lack of evaluation pipelines, datasets, and benchmarks adapted to governmental task characteristics**.

# Goals

1. Introduce domain-specific governmental datasets and categorize them into three standard task types:  
   - **Government Services** (e.g., public service guides, business facilitation, hotline responses)  
   - **Internal Governmental Affairs** (e.g., policy Q&A, document extraction, official document drafting)  
   - **Urban Governance** (e.g., urban data analysis, event detection and dispatch)

2. Build standardized evaluation suites for at least one of the above scenarios within the **KubeEdge-Ianvs** platform, including datasets, evaluation environments, and metrics, while unifying dataset formats.

3. Implement a baseline evaluation algorithm for government agents within the Ianvs platform, based on the standardized evaluation suite.

# Proposal

This project focuses on evaluating government service agents in the scenario of **"transforming official policy files or service guides into public-facing visual posters."**

![Flow Diagram](./images/liucheng.png)

## Dataset Construction

### 1. Data Sources

- **Official Government Portals**: Policy announcements, interpretations, and guides from national and local government websites (e.g., State Council of China).
- **Open Government Document Datasets**: Publicly available datasets from GitHub, HuggingFace, etc.
- **Social Media Platforms**: Government WeChat or Weibo accounts, collecting screenshots of visual posters for evaluation alignment.

### 2. Data Modalities

| Modality   | Format          | Examples                                      |
|------------|------------------|-----------------------------------------------|
| Text       | `.txt`           | Policy body text, FAQs, service procedure text |
| PDF        | `.pdf`           | Policy notices, regulations, implementation guides |
| Image      | `.jpg/.png`      | Scans, photos, infographic posters, long web posters |
| Poster     | `.jpg/.png`      | Booklets, banners, visual leaflets             |

This dataset builds upon the existing KubeEdge-Ianvs benchmarking corpus, which evaluates large model capabilities in the government sector through cloud-edge collaborative AI applications. The original dataset emphasizes both objective and subjective benchmarks to measure general competitiveness and trustworthiness, with the goal of bridging large model assessment and real-world government applications. By focusing on industry-specific use cases, it provides a more practical reflection of the value of large models in public service scenarios.（https://www.kaggle.com/datasets/kubeedgeianvs/the-government-affairs-dataset-govaff?resource=download）

In addition, we extend the dataset with newly constructed government-related data sources to further enrich evaluation diversity and realism. These include official government portals (policy announcements, interpretations, guides), open government datasets (from GitHub, HuggingFace), and social media platforms (government WeChat/Weibo accounts, particularly visual poster content). The new dataset covers multiple modalities—text (.txt), PDF (.pdf), image (.jpg/.png), and poster (.jpg/.png)—representing policy documents, regulations, FAQs, scanned forms, and infographic-style visual leaflets. This expansion ensures that both textual and multimodal government data are incorporated for more comprehensive benchmarking of large models.

### 3. Data Classification by Rule Type

To address structural and stylistic variation among documents, all collected data is classified by **`rule_type`**, which governs layout templates and evaluation criteria. Classification is automated using **LLM + Prompt** techniques. Each rule_type corresponds to a different structured template and evaluation rubric.

| Rule Type     | Example Keywords                 | Suggested Use Case                |
|---------------|----------------------------------|-----------------------------------|
| Guiding Policy | “opinions”, “plans”, “resolutions” | Policy direction infographics     |
| Implementation | “methods”, “regulations”, “details” | Bullet-pointed posters            |
| Service Guide  | “guide”, “steps”, “process”        | Procedure diagrams/booklets       |
| Notices        | “announcements”, “notices”, “statements” | Public service notices/post copy |
| FAQ            | “frequently asked questions”, “consultations” | Question-based poster cards       |
| Legal Documents| “laws”, “articles”, “regulations” | Structural tree or simplification |

**Structured JSON example**:
```json
{
  "input": {
    "file_type": "pdf",
    "rule_type": "notice",
    "file_path": "recruitment_notice_2025.pdf"
  },
  "reference_poster": "recruitment_poster.jpg",
  "structured_output": {
    "title": "Notice on the 2025 Military Recruitment Campaign",
    "sections": [...]
  }
}
```
### 4.Dataparse in sedna
This code provides a unified data processing framework that converts various raw data formats into a standardized structure of inputs (`x`) and outputs (`y`). Each data source has a dedicated parser class: `TxtDataParse` reads line-based text files, `CSVDataParse` handles tabular data, `JSONDataParse` parses COCO-style JSON annotations, `JsonlDataParse` processes JSON lines files, and `JSONMetaDataParse` combines metadata with data entries. The extended functionality includes `ImageDataParse`, which loads images from a directory into numerical arrays, and `PDFDataParse`, which extracts text from PDF files. Through this modular design, diverse data types (text, structured data, images, and documents) are normalized into a consistent format, making them ready for downstream tasks such as training, inference, or evaluation.


## work flow
![Ianvs Diagram](./images/flow.png)

The work of AI agents can be primarily broken down into three key components

### 1.parser
This is the first step, responsible for "understanding" and structuring the raw PDF documents. It uses PDF parsing tools (such as MARKER, DOCLING) to convert the PDF into Markdown format, retaining the document's chapter structure, titles, and other relevant information.
The goal is to achieve coarse-grained information compression and organization, turning what was originally a scattered long document into a well-organized, easy-to-handle structured data.
Next, an LLM (Large Language Model) processes the Markdown text and generates a structured JSON format "Asset Library". This library contains the titles of each chapter, corresponding content summaries (textual assets), and extracted tables and their titles (visual assets).

### 2.planner
 Local Organization and Layout
The Planner receives the Asset Library generated by the Parser and begins planning the overall structure of the poster and the specific content for each part.
### 3.painter-commenter
 Local Organization and Layout
The Planner receives the Asset Library generated by the Parser and begins planning the overall structure of the poster and the specific content for each part.Commenter:
A VLM (Visual Language Model) receives the panel sketch image generated by the Painter.
It acts like a “visual quality inspector,” checking if the panel has issues, such as:
Text Overflow: Is the text too long and spilling out of the text box?
Too Blank: Is there too little text, leaving excessive empty space in the panel?
Alignment: Are the text and charts aligned properly?
To help the VLM better understand these concepts, the Commenter will reference example images: one showing a typical overflow case (negative example) and another showing a perfect layout case (positive example). This is known as In-context Reference.
The Commenter provides structured feedback based on the review, such as “overflow,” “too blank,” or “good to go.”
Refinement Loop:
If the Commenter points out an issue (e.g., overflow), the feedback is sent back to the Painter. The Painter will attempt to modify the generated code (e.g., by compressing text, adjusting font size, or re-layout), and then render again. This "draw-check-modify" cycle will repeat until the Commenter deems the panel's quality satisfactory or until the maximum iteration limit is reached.
## Evaluation Metrics

The project adopts **LLM as Judge** as the core evaluation method.

Each `rule_type` is paired with custom **evaluation dimensions**. Example criteria:

| Evaluation Dimension | Sub-Indicator      | Description                                                                 |
|----------------------|-------------------|-----------------------------------------------------------------------------|
| **Visual Quality**   | Visual Similarity | The degree of similarity between generated posters and manual posters in style and layout |
|                      | Chart Relevance   | Semantic relevance between charts and corresponding section text, ensuring reasonable chart placement |
| **Overall Evaluation** | VLM-as-Judge      | Automatic scoring by Qwen2.5 across six granular dimensions (1-5 points)    |
|                      | Aesthetic Dimension | Element quality (e.g., chart clarity), layout balance, and attractiveness |
|                      | Information Dimension | Text readability, content completeness, and logical fluency          |

**Example LLM Prompt**:
```
You are a compliance officer reviewing a government promotional poster.

[Original Document Type]: Notice  
[Original Document Content]: …  
[Generated Poster Content]: … (text or structured JSON)

Evaluate on the following 10-point scale:

1. Are core policy points preserved?
2. Is the language formal and public-friendly?
3. Does it conform to the expected style of a notice?
4. Is the visual consistent, clear, and attractive?
5. Is overall readability and policy alignment adequate?

Return the result in this format:
{
  "core_policy_coverage": 8,
  "style_appropriateness": 9,
  "rule_alignment": 8,
  "visual_consistency": 7,
  "overall_readability": 9,
  "comments": "Accurate overall, but some parts are overly complex. Visuals could be better aligned."
}
```

LLMs with strong long-text comprehension and structured output ability (e.g., from HuggingFace) will be used. Evaluation results are compiled into formal scoring reports.

**Overall Evaluation Flow**:

```
Government Document + Generated Poster
               ↓
     LLM Evaluation Module (Judge)
               ↓
    Output: Score, Comments, Explanation (JSON)
```

## Project Structure

Directory: `examples/GovDoc2Poster`

```
GovDoc2Poster
└── singletask_learning_bench
    ├── benchmarkingjob.yaml
    └── testalgorithms
    │   └── gen
    │       ├── basemodel.py
    │       ├── gen_algorithm.yaml
    |       ├── gov_metrics.py
    └── testenv
            ├── acc.py
            └── testenv.yaml 
```
Directory: `examples/resources`
```
sedna
└──datasources
    ├──PDFDataParse()
    └──imgDataParse()
```
Modified Ianvs architecture:

![Ianvs Diagram](./images/ianvs_arch.png)


## Development Phases

### Early Phase (July Early - July Late)
Collect various formats of government documents, classify by `rule_type`, and define associated compliance rules and prompts. Create a structured dataset labeled with `rule_type`, key elements, and content segments. This serves as the foundation for evaluation and model training.

### Mid Phase (August Early - August Mid)
Develop a fully automated evaluation pipeline under Ianvs: multimodal input processing, structured extraction, poster generation, and multi-metric evaluation. Integrate traditional metrics (CLIP similarity, QA accuracy, perplexity) and LLM-based subjective evaluation (“LLM as Judge”) aligned to `rule_type`-specific rubrics.

### Late Phase (August Late - September Mid)
Test and refine the multimodal pipeline, optimize model outputs, and tune LLM scoring prompts. Complete integration into Ianvs and demonstrate full functionality with visualized results, model outputs, and scoring reports.

## Project Timeline (Roadmap)

| Time Period            | Phase                          | Key Tasks |
|------------------------|--------------------------------|-----------|
| **Early July – Late July** | **Early Phase: Document & Rule Design** | • Collect government policy documents (PDF, Word, images)<br>• Extract and standardize key content<br>• Define common `rule_type` categories (e.g., Notice, Report, Speech)<br>• Construct compliance rules and prompts per rule type<br>• Build a structured dataset with rule-type labels |
| **Early August – Mid August** | **Mid Phase: Pipeline Implementation** | • Develop evaluation workflow under Ianvs<br>• Implement structured content extraction and poster generation<br>• Integrate metrics: CLIP (visual), QA (info), PPL (language)<br>• Add LLM-as-Judge scoring with rule-specific prompts |
| **Late August – Mid September** | **Late Phase: Optimization & Deployment** | • Debug multimodal processing pipeline<br>• Tune extraction and poster modules for consistency<br>• Optimize evaluation prompts and scoring criteria<br>• Deploy under Ianvs and validate system performance<br>• Deliver outputs: samples, scores, final evaluation report |
