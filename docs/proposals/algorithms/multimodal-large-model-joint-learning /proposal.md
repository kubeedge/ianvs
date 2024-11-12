
### Multimodal Large Model Joint Learning Algorithm: Reproduction Based on KubeEdge-Ianvs

---

**Motivation**  
KubeEdge-Ianvs currently focuses on edge-cloud collaborative learning (training and inference) for a single modality of data. However, edge devices, such as those in autonomous vehicles, often capture multimodal data, including GPS, LIDAR, and Camera data. Single-modal learning can no longer meet the precise inference requirements of edge devices. Therefore, this project aims to integrate mainstream multimodal large model joint learning algorithms into KubeEdge-Ianvs edge-cloud collaborative learning, providing multimodal learning capabilities.

---

**Goals**  
1. **Modify and adapt the existing edge-cloud data collection interface** to meet the requirements of multimodal data collection.
2. **Implement a Multimodal Large Language Model (MLLM) benchmark suite** based on Ianvs.
3. **Reproduce mainstream multimodal joint learning (training and inference) algorithms** and integrate them into Ianvs single-task learning.
4. **(Advanced)** Test the effectiveness of multimodal joint learning in at least one of Ianvs' advanced paradigms (lifelong learning, incremental learning, federated learning, etc.).

---

**Proposal**  
1. **Enhance the Edge-Cloud Data Collection Interface:**  
   Modify and adapt the existing edge-cloud data collection interface to support the requirements for multimodal data collection, ensuring compatibility with diverse data sources.
   
2. **Develop a Multimodal Large Language Model (MLLM) Benchmark Suite:**  
   Implement a benchmark suite specifically for Multimodal Large Language Models based on the Ianvs framework, facilitating robust evaluation and comparison of model performance.

3. **Reproduce and Integrate Mainstream Multimodal Joint Learning Algorithms:**  
   Reproduce widely recognized multimodal joint learning algorithms for both training and inference, integrating them into the single-task learning capabilities of Ianvs to enhance its versatility.

---

**Design Details**  

**Overall Architecture**  
1. **Dataset:**  
   Enhance this component by incorporating support for multimodal data types, including text, audio, and Camera data. This will enable the system to efficiently manage and preprocess diverse data streams necessary for effective multimodal learning.
   
2. **ModelEvaluationController:**  
   Extend this component to include benchmarks specifically for Multimodal Large Language Models (MLLMs). Implement evaluation metrics such as F1 Score, Precision, Recall, and Transfer Ratio, tailored for assessing the performance of multimodal models in edge-cloud scenarios.

---

**Implementation Detail**  
```plaintext
├── testcasecontroller
│   ├── algorithm
│   │   └── paradigm
│   │       ├── __init__.py
│   │       ├── base.py                      # Base class for algorithms
│   │       └── single_task_learning.py      # Single-task learning algorithms
│   │           └── clip_model.py            # Implementation of the CLIP model
│   ├── data_collection
│   │   ├── __init__.py
│   │   ├── multimodal_interface.py          # Interface for multimodal data collection
│   │   └── preprocess.py                     # Preprocessing for text, audio, and images
│   ├── benchmark
│   │   ├── __init__.py
│   │   ├── mllm_benchmark.py                # Logic for MLLM benchmarking
│   │   └── metrics.py                       # BLEU, FID, WER, VQA accuracy metrics
│   ├── tests
│   │   ├── __init__.py
│   │   └── test_benchmark.py                # Unit tests for benchmarking
│   └── main.py                              # Entry point for running the benchmark suite
```

---


<img width="828" alt="Screenshot 2024-10-12 at 3 59 55 PM" src="https://github.com/user-attachments/assets/d8a7006e-8bb8-46ff-9950-e86a193ac48b">



 **Multimodal Learning: Demonstrated Improvements in Inference Accuracy Over Single-Modal Data**
 

- Image-based classification can effectively distinguish between broad document categories (e.g., passports vs. banking documents). However, it 
  struggles with documents that share similar layouts or templates, where fine distinctions often depend on the text content. This limitation 
  highlights the necessity for multimodal learning, which combines both visual and textual data for more accurate classification.**Change of 
  Ianvs**  

- The introduction of this multimodal network improves classification **accuracy by 3% on two specific datasets (Tobacco3482 and RVL-CDIP)** 
  when compared to models that rely on only visual features. This improvement occurs even when the text extracted by OCR is not completely 
  clean, proving the robustness of the multimodal approach.


<img width="690" alt="Screenshot 2024-10-16 at 12 36 01 PM" src="https://github.com/user-attachments/assets/db44a52a-be7c-4d68-a416-4d0491c8a9b9">

---

### OpenAI CLIP: Overview and Functionality

**Introduction to CLIP**  
OpenAI's CLIP (Contrastive Language-Image Pre-training) is a powerful model designed to understand images and their corresponding textual descriptions. By leveraging a large dataset of images paired with their textual descriptions, CLIP learns to associate visual concepts with language. This allows it to perform a variety of tasks without task-specific training data, making it highly versatile.

**How CLIP Works**  
CLIP operates on a dual-encoder architecture, consisting of two primary components:

1. **Image Encoder:** This component processes images using a convolutional neural network (CNN) or Vision Transformer (ViT), transforming them into high-dimensional feature vectors that capture visual representations.

2. **Text Encoder:** This component processes text inputs using a transformer architecture, converting them into feature vectors that encapsulate the semantic meanings of the text.

During training, CLIP uses a contrastive learning approach, where it pairs images and their corresponding text descriptions. The model learns to minimize the distance between the feature vectors of correctly paired image-text pairs while maximizing the distance for incorrectly paired pairs. This allows CLIP to develop a rich understanding of the relationships between visual content and textual descriptions.

**Supported Tasks**  
CLIP can support various tasks across multiple domains, including:

1. **Zero-Shot Classification:** CLIP can classify images into categories without requiring fine-tuning on specific datasets. For instance, it can identify whether an image depicts a cat or a dog based on a natural language description.

2. **Image Search:** By inputting textual queries, CLIP can retrieve relevant images from a database. This capability is particularly useful in applications like stock photo search or content recommendation.

3. **Image Captioning:** CLIP can generate textual descriptions for images by ranking potential captions based on their relevance to the visual content.

4. **Visual Question Answering (VQA):** The model can answer questions about images by understanding both the visual context and the textual query.

5. **Image Manipulation and Synthesis:** CLIP can assist in generating new images by interpreting textual descriptions, thus bridging the gap between language and image generation tasks.

6. **Cross-Modal Retrieval:** CLIP can perform retrieval tasks across modalities, such as finding images that correspond to a given textual description and vice versa.

**Applications**  
The capabilities of CLIP make it applicable in various fields, including:

- **E-commerce:** Enhancing product searches by allowing customers to find items based on textual descriptions.
- **Social Media:** Improving content moderation by automatically categorizing or filtering images based on their content.
- **Creative Industries:** Assisting artists and designers in generating and refining visual concepts from text prompts.
- **Healthcare:** Analyzing medical images by matching them with patient records or textual descriptions of symptoms.

---


<img width="854" alt="Screenshot 2024-10-16 at 12 13 54 PM" src="https://github.com/user-attachments/assets/70c44adf-f517-4c27-b7d8-8fcf33e5f76e">




1. **Dataset Handling**  
   The `Dataset` class has been updated to handle multiple types of data such as text, images, and audio.

   **Adding New Enums in `DatasetFormat`:**
   ```python
   class DatasetFormat(Enum):
       TXT = "txt"
       CSV = "csv"
       JSON = "json"
       IMAGE = "image"
       AUDIO = "audio"
   ```

   **Updating `_check_dataset_url` Method:**
   ```python
   @classmethod
   def _check_dataset_url(cls, url):
       if not utils.is_local_file(url) and not os.path.isabs(url):
           raise ValueError(f"dataset file({url}) is not a local file and not a absolute path.")
       
       file_format = utils.get_file_format(url)
       if file_format not in [v.value for v in DatasetFormat.__members__.values()]:
           raise ValueError(f"dataset file({url})'s format({file_format}) is not supported.")
   ```

   **Implementing `ImageDataParse` Class:**
   ```python
   class ImageDataParse:
       def __init__(self, data_type, func=None):
           self.data_type = data_type
           self.func = func

       def parse(self, file, label=None):
           # Implement image parsing logic here
           pass
   ```

   **Implementing `AudioDataParse` Class:**
   ```python
   class AudioDataParse:
       def __init__(self, data_type, func=None):
           self.data_type = data_type
           self.func = func

       def parse(self, file, label=None):
           # Implement audio parsing logic here
           pass
   ```

   **Updating `process_dataset` Method:**
   ```python
   def process_dataset(self):
       self.train_url = self._process_index_file(self.train_url)
       self.test_url = self._process_index_file(self.test_url)
       # Add any additional processing for image and audio datasets if necessary
   ```

   **Updating `split_dataset` Method:**
   ```python
   def split_dataset(self, dataset_url, dataset_format, ratio, method="default",
                     dataset_types=None, output_dir=None, times=1):
       if method == "default":
           return self._splitting_more_times(dataset_url, dataset_format, ratio,
                                             data_types=dataset_types,
                                             output_dir=output_dir,
                                             times=times)
       # Add new splitting methods for image and audio datasets if necessary
       raise ValueError(f"dataset splitting method({method}) is not supported,"
                        f"currently, method supports 'default'.")
   ```

2. **Test Environment Configuration**  
   The `TestEnv` class has been updated to support configurations for multimodal datasets and specific metrics for large language models.

   ```python
   class TestEnv:
       def __init__(self, config):
           # Initializing model evaluation parameters
           self.model_eval = {
               "model_metric": {
                   "mode": "",
                   "name": "",
                   "url": "",
               },
               "threshold": 0.9,  # Threshold for evaluation
               "operator": ">"    # Operator to compare metrics against the threshold
           }
           self.metrics = []                # List to store evaluation metrics
           self.incremental_rounds = 2      # Number of incremental rounds (minimum 2)
           self.datasets = []                # List of datasets for testing
           self.modalities = []              # List of modalities for multimodal support
           self._parse_config(config)        # Parse the configuration provided

       def _check_fields(self):
           # Check if required fields are populated
           if not self.metrics:
               raise ValueError(f"Metrics not found: {self.metrics}.")
           if not isinstance(self.incremental_rounds, int) or self.incremental_rounds < 2:
               raise ValueError(f"Incremental rounds (value={self.incremental_rounds}) "
                                "must be an integer and not less than 2.")

       def _parse_config(self, config):
           # Parse the configuration dictionary for TestEnv settings
           config_dict = config[str.lower(TestEnv.__name__)]
           for k, v in config_dict.items():
               if k == str.lower(Dataset.__name__):
                   self.datasets.append(Dataset(v))  # Initialize datasets
               elif k == "modalities":
                   self.modalities = v  # Set modalities
               else:
                   if k in self.__dict__:
                       self.__dict__[k] = v  # Update attributes dynamically

           self._check_fields()  # Validate the parsed configuration

       def prepare(self):
           # Prepare datasets for testing
           try:
               for dataset in self.datasets:
                   dataset.process_datasets()  # Process each dataset
           except Exception as err:
               raise RuntimeError(f"Dataset preparation failed, error: {err}.") from err
   ```

Great! Here’s a comprehensive summary that includes the new benchmarking job configuration, algorithm configuration, and the updated metrics functions for evaluating multimodal models.

---

### New Benchmarking Configuration

A new benchmarking job configuration file specific to MLLM has been created.

```yaml
benchmarkingjob:
  name: "mllm_benchmarking_job"  # Name of the benchmarking job
  workspace: "/ianvs/multimodal_language_model_bench/workspace"  # Path to the workspace
  testenv: "./examples/mllm_benchmark/testenv.yaml"  # Path to the test environment configuration
  test_object:
    type: "algorithms"  # Specifies that the test object is algorithms
    algorithms:
      - name: "mllm_evaluation"  # Name of the MLLM evaluation algorithm
        url: "./examples/mllm_benchmark/algorithms/mllm_algorithm.yaml"  # Path to the algorithm configuration
  rank:
    sort_by: 
      - { "accuracy": "descend" }  # Primary sorting criterion by accuracy
      - { "f1_score": "descend" }   # Secondary sorting criterion by F1 score
      - { "bleu": "descend" }        # Tertiary sorting criterion by BLEU score
      - { "fid": "ascend" }          # Quaternary sorting criterion by FID (lower is better)
      - { "wer": "ascend" }          # Quinary sorting criterion by WER (lower is better)
      - { "vqa_accuracy": "descend" }  # Senary sorting criterion by VQA accuracy
    visualization:
      mode: "selected_only"  # Visualization mode, showing only selected items
      method: "print_table"   # Method to visualize results
    selected_dataitem:
      paradigms: [ "all" ]  # Selects all paradigms
      modules: [ "all" ]     # Selects all modules
      hyperparameters: [ "all" ]  # Selects all hyperparameters
      metrics: 
        - "accuracy"          # Include accuracy metric
        - "f1_score"         # Include F1 score metric
        - "bleu"             # Include BLEU score metric
        - "fid"              # Include Fréchet Inception Distance metric
        - "wer"              # Include Word Error Rate metric
        - "vqa_accuracy"     # Include Visual Question Answering Accuracy metric
```

---

### Algorithm Configuration

A new algorithm configuration file for the MLLM benchmark has been created.

```yaml
algorithm:
  paradigm_type: "multimodal_learning"
  initial_model_url: ""
  modules:
    - type: "basemodel"
      name: "MLLM_base"
      url: "./examples/mllm_benchmark/testalgorithms/mllm_base_model.py"
      hyperparameters:
        - config:
            values:
              - "./examples/mllm_benchmark/resource/MLLM_config.py"
        - work_dir:
            values:
              - "./examples/mllm_benchmark/work_dir"
        - resource_dir:
            values:
              - "./examples/mllm_benchmark/resource"
```

---

### Metrics Update

The `metrics.py` file has been updated to include new metric functions for evaluating multimodal models.

```python
def multimodal_accuracy_func(system_metric_info: dict):
    """Calculate the multimodal accuracy."""
    info = system_metric_info.get("multimodal_accuracy")
    correct_predictions = info.get("correct_predictions", 0)
    total_predictions = info.get("total_predictions", 1)
    return round(correct_predictions / total_predictions, 4)

def cross_modal_retrieval_func(system_metric_info: dict):
    """Calculate the cross-modal retrieval score."""
    info = system_metric_info.get("cross_modal_retrieval")
    retrieval_score = info.get("retrieval_score", 0)
    return retrieval_score

def bleu_score_func(system_metric_info: dict):
    """Calculate the BLEU score."""
    info = system_metric_info.get("bleu")
    return info.get("bleu_score", 0.0)

def fid_score_func(system_metric_info: dict):
    """Calculate the Fréchet Inception Distance (FID)."""
    info = system_metric_info.get("fid")
    return info.get("fid_score", 0.0)

def wer_score_func(system_metric_info: dict):
    """Calculate the Word Error Rate (WER)."""
    info = system_metric_info.get("wer")
    return info.get("wer_score", 0.0)

def vqa_accuracy_func(system_metric_info: dict):
    """Calculate Visual Question Answering (VQA) accuracy."""
    info = system_metric_info.get("vqa_accuracy")
    correct_answers = info.get("correct_answers", 0)
    total_questions = info.get("total_questions", 1)
    return round(correct_answers / total_questions, 4)

def get_metric_func(metric_dict: dict):
    """Retrieve the metric function based on the metric name."""
    name = metric_dict.get("name")
    url = metric_dict.get("url")
    if url:
        try:
            load_module(url)
            metric_func = ClassFactory.get_cls(
                type_name=ClassType.GENERAL, t_cls_name=name)
            return name, metric_func
        except Exception as err:
            raise RuntimeError(
                f"get metric func(url={url}) failed, error: {err}.") from err

    metric_func_map = {
        'multimodal_accuracy': multimodal_accuracy_func,
        'cross_modal_retrieval': cross_modal_retrieval_func,
        'bleu': bleu_score_func,
        'fid': fid_score_func,
        'wer': wer_score_func,
        'vqa_accuracy': vqa_accuracy_func,
    }

    return name, metric_func_map.get(name, getattr(sys.modules[__name__], str.lower(name) + "_func"))
```

---


**Expected Output**  
Upon successful integration and testing, the project aims to produce a detailed report encompassing:
- Performance metrics for various multimodal learning models.
- Visual representations (e.g., graphs, charts) illustrating the effectiveness of joint learning.
- Documentation highlighting modifications made to KubeEdge-Ianvs and the rationale behind design decisions.

---


**Conclusion**  
This project seeks to enhance KubeEdge-Ianvs by enabling multimodal data processing and joint learning capabilities. By incorporating advanced multimodal algorithms and a robust benchmark suite, it will pave the way for future developments in edge-cloud collaborative learning, ensuring that edge devices can leverage the full spectrum of data types for improved inference and decision-making.

---

