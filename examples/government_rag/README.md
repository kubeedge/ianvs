# Government RAG 示例

这个示例展示了如何在政务领域使用检索增强生成（RAG）技术来提升大语言模型的性能。

## 特点

1. 支持多种文档格式（txt, docx）
2. 使用 LangChain 进行文档处理和检索
3. 实现了四种测试模式：
   - Type 1: 无 RAG 模式
   - Type 2: 只用和被测边缘节点相关的数据作为 RAG 知识库
   - Type 3: 使用所有边缘节点数据作为 RAG 知识库
   - Type 4: 使用所有和被测边缘节点不相关的数据作为 RAG 知识库
4. 提供详细的评估指标：
   - 准确率
   - RAG 效果评估

## 目录结构

```
government_rag/
├── singletask_learning_bench/
│   └── objective/
│       ├── benchmarkingjob.yaml
│       ├── testalgorithms/
│       │   └── government_rag_algorithm.py
│       └── testenv/
│           └── metrics/
│               ├── accuracy.py
│               └── rag_effectiveness.py
└── README.md
```

## 使用方法

1. 准备数据：
   - 将 `all_questions.jsonl` 放在 `testenv/dataset/test/` 目录下
   - 将知识库文件放在 `testenv/dataset/knowledge_base/` 目录下

2. 安装依赖：
   ```bash
   pip install langchain faiss-cpu transformers torch
   ```

3. 运行测试：
   ```bash
   ianvs -f examples/government_rag/singletask_learning_bench/benchmarkingjob.yaml
   ```

## 配置说明

在 `benchmarkingjob.yaml` 中，您可以配置：

1. 算法参数：
   - 模型选择
   - 向量数据库参数
   - RAG 检索参数

2. 数据集路径：
   - 测试数据路径
   - 知识库路径

3. 评估指标：
   - 准确率
   - RAG 效果评估

## 注意事项

1. 确保有足够的内存来加载和处理文档
2. 建议使用 GPU 来加速模型推理
3. 知识库文档的质量会直接影响 RAG 的效果
4. 不同地区的政策差异需要考虑在评估中 