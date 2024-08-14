# Background
Large Language Models (LLMs) have demonstrated powerful capabilities in tasks such as code generation, automatic programming, and code analysis. However, these models are typically trained on generic code data and often fail to fully leverage the collaboration and feedback from software engineers in real-world scenarios. To construct a more intelligent and efficient code ecosystem, it is necessary to establish a collaborative code dataset and evaluation benchmark to facilitate tight collaboration between LLMs and software engineers. This project aims to build a collaborative code intelligent agent alignment dataset and evaluation benchmark for LLMs based on the open-source edge computing framework KubeEdge-Ianvs. This dataset will include behavioral trajectories, feedback, and iterative processes of software engineers during development, as well as relevant code versions and annotation information. Through this data, we will design evaluation metrics and benchmarks to measure the performance of LLMs in tasks such as code generation, recommendation, and analysis, fostering collaboration between LLMs and software engineers.

In today's software development practice, large language models (LLMs) show great potential in areas such as code generation, recommendation, and analysis. However, existing models are usually trained on general code bases and lack optimization for specific software engineering tasks. Therefore, creating a specific dataset and evaluation benchmark that integrates the actual work experience and feedback of software engineers is crucial to improving the application effect of these models in actual programming environments.
# Goals
1. Build a collaborative code intelligent agent alignment dataset for LLMs
2. Design a code intelligent agent collaborative evaluation benchmark for LLMs
3. Integrate the dataset and evaluation benchmark into the KubeEdge-Ianvs framework
# Proposal
## Building a large code language model dataset

1. **Behavioral Trajectory During Development**：
Record the operations performed by software engineers during the development process. These operations may include code writing, code submission, code merging, code review, code refactoring, etc.
Specific behavioral data may include the development tools used, the code snippets written, submission records, review comments, etc.
2. **Feedback and Iteration Process**：
Collect feedback and iteration records of the code from R&D engineers during the development process. These feedbacks may include code review comments, test results, error reports, improvement suggestions, etc.
Record the time of feedback, feedback content, corresponding code modifications, and final solutions.
3. **Code version and comment information**：
Record each version of the code and the differences between each version, including new, modified, and deleted code.
Include detailed code comments and documentation to understand the function, purpose, and design ideas of the code.

## Code Large Language Model Evaluation Benchmark
1. The benchmark should include common code agent tasks such as code generation, recommendation and analysis.
2. The evaluation indicators should cover multiple dimensions such as functionality, reliability, and interpretability, and match the feedback and needs of software engineers.
3. The benchmark should be able to evaluate the performance of LLMs on collaborative code agent tasks and provide a basis for further algorithm optimization.
### Integrate datasets and benchmarks into the KubeEdge-Ianvs framework

1. The dataset and benchmark are included as part of the Ianvs framework, and provide good scalability and integration.
2. Ensure that the datasets and benchmarks can run efficiently on edge devices of the Ianvs framework and work seamlessly with other functional modules of Ianvs.


`examples/smart_coding` directory structure：
```
smart_coding
└── smart_coding_learning_bench
    └── smart_co
        ├── benchmarkingjob.yaml
        ├── testalgorithms
        │   └── gen
        │       ├── basemodel.py
        │       ├── gen_algorithm.yaml
        │       ├── op_eval.py
        └── testenv
            ├── acc.py
            └── testenv.yaml
```
The content format of the comment test set is as follows:
```
{"description": "Add detailed comments to a given code/function.","code_snippet": "def calculate_area(length, width):\n    return length * width",}
{"description": "Add detailed comments to a given Python function.",
  "code_snippet": "def calculate_area(length, width):\n    return length * width",
  "annotations": [
    {
      "line_number": 1,
      "annotation": "Define a function calculate_area that accepts two parameters: length and width."
    },
    {
      "line_number": 2,
      "annotation": "Returns the product of length and width, which is the area of ​​the rectangle."
    }
  ]}
```

In this project, I am mainly responsible for the test suite of the code model. For the code model, the main evaluation criteria are comments and issues in the task requirements. Different projects use different fields for the evaluation criteria of comments. The scoring part is based on the logic, accuracy, and format of the overall code.

The data set part and the interface definition part adopt a question-and-answer method. Question is a line of code/a function, and Answer is also Comment, which is a comment on the code or function.

The format of the issue test set is as follows:
```
{
  "question": "title",
  "user_login": "name",
  "created_at":"time",
  "updated_at": "time",
  "body":"This is not possible right now afaik :/\r\n\r\nMaybe we could have something like this ? wdyt ?\r\n\r\n```python\r\nds = interleave_datasets(\r\n    [shuffled_dataset_a, dataset_b],\r\n    probabilities=probabilities,\r\n    stopping_strategy='all_exhausted',\r\n    reshuffle_each_iteration=True,\r\n)",
  "answer_1": {
    "user_login": "name",
    "created_at":"time"
    "updated_at": "time",
    "body":"This is not possible right now afaik :/\r\n\r\nMaybe we could have something like this ? wdyt ?\r\n\r\n```python\r\nds = interleave_datasets(\r\n    [shuffled_dataset_a, dataset_b],\r\n    probabilities=probabilities,\r\n    stopping_strategy='all_exhausted',\r\n    reshuffle_each_iteration=True,\r\n)",
  },
  "answer_2": {
    "user_login": "name",
    "created_at":"time"
    "updated_at": "time",
    "body":"XXXX"
  },
  
}
```
The format of the issue test set refers to a simple QA question-answering task. LLM is used to answer the solution to the problem. Manual judgment is used in the evaluation stage, and the acc accuracy is calculated in the end.

The following is the operation flow of the benchmark system based on user input configuration. Because the interface part is written by Meng Zhuo, the general structure is basically consistent with Meng Zhuo. The flowchart shows the data verification, parsing, initialization and other operations. The difference lies in the reading of the issue data set. In the issue data set, there is only one Question, but there may be multiple Comments, so in the training part, the data reading needs to be adjusted.

<img src="E:\2024\第一次开源之夏\ianvs\docs\proposals\scenarios\Smart Coding\image\data_process_change.png"/>

<img src="E:\2024\第一次开源之夏\ianvs\docs\proposals\scenarios\Smart Coding\image\change_part.png"/>

It is worth noting that this design is also compatible with the old version of index data. You only need to change the old version's `train_url` and `test_url` fields to train_index and test_index.

In previous projects, it was necessary to configure the paths of the `train_url` and `test_url` index files in the `testenv.yaml` file. The index file would contain the file path of (input x, expected output y). This design has some limitations.

Dataset file format A data.json/jsonl/ file contains both data and labels. The data set format of the government affairs model is defined as follows
```
{"code_snippet": xxx, "comment": xxx}
{"code_snippet": xxx, "comment": xxx}
{"code_snippet": xxx, "comment": xxx}
```

In the code model, the data format definition for adding comments to code and functions is consistent with the government affairs model, but the attributes of the answer also need to be modified in the issue dataset, for example
```
{ 
    "question": xxx, 
    "user_login": "XXXX",
    "created_at":"XXXX-XX-XX",
    "updated_at": "XXXX-XX-XX",
    "body":"This is not possible right now afaik :/\r\n\r\nMaybe we could have something like this ? wdyt ?\r\n\r\n```python\r\nds = interleave_datasets(\r\n    [shuffled_dataset_a, dataset_b],\r\n    probabilities=probabilities,\r\n    stopping_strategy='all_exhausted',\r\n    reshuffle_each_iteration=True,\r\n)",
    "answer_1":{"user_login": "name","created_at":"time""updated_at": "time","body":"XXXX"}
    "answer_2":{"user_login": "name","created_at":"time""updated_at": "time","body":"XXXX"}
    "answer_3":{"user_login": "name","created_at":"time""updated_at": "time","body":"XXXX"}
  }
{
   "question": xxx, 
   "user_login": "XXXX-XX-XX",
   "created_at":"XXXX-XX-XX",
   "updated_at": "XXXX-XX-XX",
   "body":"XXXX....",
   "answer_1":{"user_login": "name","created_at":"time""updated_at": "time","body":"XXXX"}
   "answer_2":{"user_login": "name","created_at":"time""updated_at": "time","body":"XXXX"}
   "answer_3":{"user_login": "name","created_at":"time""updated_at": "time","body":"XXXX"}
  }
.......
```

The format of the issue test set refers to a simple QA question-answering task. The solution to the problem is answered through LLM. Manual judgment is used in the evaluation stage, and the acc accuracy is calculated in the end.

The following is the operation flow of the benchmark system based on user input configuration. Because the interface part is written by Meng Zhuo, the general structure is basically consistent with Meng Zhuo. The flowchart shows the data verification, parsing, initialization and other operations.
The difference lies in the reading of the issue data set. In the issue data set, there is only one Question, but there may be multiple Comments, so in the training part, the data reading needs to be adjusted.



# Design Details
## Data collection
1. GitHub: Collect open source project code in various programming languages ​​from GitHub. Use GitHub API or manual retrieval.
2. GitHub Issues: collects problem reports submitted by developers, including bug reports, feature requests, and discussions。
3. Pull Requests: Collect pull requests submitted by developers, including the review history of function implementation and code modifications.
4. Commit Logs: Extract the project's commit logs, including every change to the code, the committer information, and the commit time.
5. Branches and Merges: Consider branch and merge information to understand the development and merge history of the code.

### Specific steps
In the early stage, by adding comments to Python files (with key lines and segments as the granularity), collecting classic Python projects, annotating the core code lines or functions in the projects, and organizing the data set, and reading related papers at the same time.

In the mid-term stage, through the related large models read in the early stage, collecting papers on the issue data set, we began to organize data mainly based on issues, as well as App projects with Python as the main development language.

In the final stage, after the data set is organized, we began to design test evaluation indicators, test the data set, and write unit tests/integration tests. Test-driven to ensure code correctness.


## Project development time plan
| Time plan  | Task                                                                       |
|------------|--------------------------------------------------------------------------|
| July 13th - early August | Read relevant papers; collect data sets, read and understand open source projects, write corresponding project product documents and requirement documents. The requirement documents must include key points such as command, issue, PR, etc. of the corresponding project |
| Mid-August to early September  | Organize the collected data sets, expand the collection scope, expand the data sets, and write test evaluation indicators for the data sets and large models                         |
| Mid-September to the end of September   | Write unit tests/integration tests and test-driven tests to ensure code correctness.                                                  |















