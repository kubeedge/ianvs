# 背景
大型语言模型（LLM）在代码生成、自动编程、代码分析等任务中展现出了强大的能力，但这些模型通常是在通用代码数据上训练的，往往不能充分利用实际场景中软件工程师的协作和反馈。为了构建更加智能高效的代码生态，需要建立协作代码数据集和评测基准，促进LLM与软件工程师的紧密协作。本项目旨在基于开源边缘计算框架KubeEdge-Ianvs构建LLM协作代码智能体对齐数据集和评测基准。该数据集将包括软件工程师在开发过程中的行为轨迹、反馈和迭代过程，以及相关的代码版本和注释信息。通过这些数据，我们将设计评测指标和基准来衡量LLM在代码生成、推荐和分析等任务中的表现，促进LLM与软件工程师之间的协作。

在当今的软件开发实践中，大型语言模型（LLM）在代码生成、推荐和分析等领域展现出巨大的潜力。但现有模型通常是在通用代码库上训练的，缺乏针对特定软件工程任务的优化，因此建立融合软件工程师实际工作经验与反馈的特定数据集与评估基准，对提升这些模型在实际编程环境中的应用效果至关重要。
# Goals
1. 为大模型构建协作代码智能数据集
2. 为大模型构建代码协同智能评估基准测试
3. 将数据集和智能评估基准集成到KubeEdge-Ianvs框架中
# Proposal
## 构建数据集

1. **开发过程中的行为轨迹**：
记录软件工程师在开发过程中执行的操作。这些操作可能包括代码编写、代码提交、代码合并、代码审查、代码重构等。
具体的行为数据可能包括使用的开发工具、编写的代码片段、提交记录、审查意见等。
2. **反馈及迭代**：
收集研发工程师在开发过程中对代码的反馈和迭代记录，这些反馈可能包括代码审查意见、测试结果、错误报告、改进建议等。
记录反馈时间、反馈内容、对应的代码修改、最终解决方案。
3. **代码版本及注释**：
记录每个版本的代码，以及各个版本之间的差异，包括新增、修改、删除的代码。
包括详细的代码注释和文档，以了解代码的功能、用途、设计思想。

## 代码大模型语言评估基准
1. 评测基准应包括代码生成、推荐和分析等常见的代码智能体任务。
2. 评测指标应涵盖功能性、可靠性、可解释性等多个维度,并与软件工程师的反馈和需求相匹配。
3. 评测基准应能够评估LLMs在协作式代码智能体任务上的性能,并为进一步的算法优化提供依据。
## 将数据集和评测基准集成到KubeEdge-Ianvs框架中

1. 将数据集和评测基准作为Ianvs框架的一部分,并提供良好的可扩展性和可集成性。
2. 确保数据集和评测基准能够在Ianvs框架的边缘设备上高效运行,并与Ianvs的其他功能模块无缝协作.

# Design Details
## Data collection
1. GitHub: 从GitHub上收集各种编程语言的开源项目代码。通过GitHub API或手动检索。
2. GitHub Issues: 收集开发者提交的问题报告，包括Bug报告、功能请求和讨论。
3. Pull Requests: 收集开发者提交的拉取请求，包括功能实现和代码修改的审查历史。
3. Commit Logs: 提取项目的提交日志，包括代码的每次变更、提交者信息和提交时间。
4. Branches and Merges: 考虑分支和合并的信息，以理解代码的开发和合并历史。

`examples/smart_coding` 目录结构：
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
comment测试集部分内容格式如下：
```
{"description": "为给定的代码/函数添加详细注释。","code_snippet": "def calculate_area(length, width):\n    return length * width",}
{"description": "为给定的Python函数添加详细注释。",
  "code_snippet": "def calculate_area(length, width):\n    return length * width",
  "annotations": [
    {
      "line_number": 1,
      "annotation": "定义一个函数calculate_area，接受两个参数：length和width。"
    },
    {
      "line_number": 2,
      "annotation": "返回length和width的乘积，即矩形的面积。"
    }
  ]}
```

在本项目中，负责的部分主要是代码大模型的测试套件，对于代码大模型来说，主要就是由任务要求中的comment和issue
而对于comment的评测标准，不同的项目，使用不同的字段，打分的部分，由通过代码整体部分的逻辑性，准确性，以及格式等部分来分别进行打分。

数据集部分，接口的定义部分，采用一问一答的方式，Question为某一行代码/某一个函数，Answer也是Comment则是对这个代码或函数的注释。

issue测试集部分内容格式如下：
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
issue部分测试集格式，参照简单的QA问答任务，通过LLM来回答问题的解决办法，在评估阶段使用人工判决，最后来计算acc准确率。

如下是基于用户输入配置的基准测试系统的操作流程，这部分因为接口部分是由孟卓同学来写的，所以大体结构基本是与孟卓一致的。流程图展示了数据的验证、解析、初始化等操作。
其中不同的点在于对issue数据集的读取，issue数据集中，Question只有一个，但Comment可能会有多个，所以在训练部分，数据的读取还有调整。

<img src="E:\2024\第一次开源之夏\ianvs\docs\proposals\scenarios\Smart Coding\image\data_process_change.png"/>

<img src="E:\2024\第一次开源之夏\ianvs\docs\proposals\scenarios\Smart Coding\image\change_part.png"/>

值得注意的是，该设计同时也兼容对旧版的index数据的支持。仅仅只需要将旧版的train_url和test_url字段改成train_index和test_index即可。

在之前的项目中，需要在`testenv.yaml`文件中配置`train_url`和`test_url`索引文件的路径，索引文件中会放 (输入x, 期望输出y) 的文件路径，这个设计是存在一些局限性的。

数据集文件格式一个data.json/jsonl/文件就把数据和标签都写进去了，其中政务大模型的数据集格式定义如下
```
{"code_snippet": xxx, "comment": xxx}
{"code_snippet": xxx, "comment": xxx}
{"code_snippet": xxx, "comment": xxx}
```

在代码大模型中，对于代码和函数添加Comment数据格式定义与政务大模型一致，但issue数据集中还需要对answer的属性进行修改，例如
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


### 具体步骤
前期阶段，通过给python文件加comments (以关键行、段为粒度)，搜集经典python项目，对项目中的核心代码行或者函数，进行注释，以此来整理数据集，并同步阅读相关论文。

中期阶段，通过前期阅读的相关大模型搜集issue数据集的论文，在此开始着手整理以issue为主的数据，还有以python为主要开发语言的App项目。

最终阶段，数据集整理完毕，开始着手设计测试评估指标，针对数据集进行测试，并编写单元测试/集成测试，测试驱动保证代码正确性。

## 时间规划
| 时间规划       | 任务                                                                       |
|------------|--------------------------------------------------------------------------|
| 7月13号-8月上旬 | 阅读相关论文；搜集数据集，阅读并了解开源项目，写出相应的项目产品文档，需求文档，需求文档需包含对应项目的command、issue、PR等关键点 |
| 8月中旬-9月上旬  | 对于已搜集的数据集进行整理，并且扩大搜集范围，扩充数据集，编写针对该数据集，大模型的测试评价指标                         |
| 9月中旬-9月底   | 编写单元测试/集成测试，测试驱动保证代码正确性                                                  |















