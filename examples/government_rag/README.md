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


## 使用方法

1. 准备数据：
   
   把数据放到 `dataset/gov_rag` 目录下，目录结构如下：
   ```
   .
   ├── data.jsonl
   ├── dataset
   │   ├── 上海市
   │   │   ├── 上海市数据交易场所管理实施暂行办法.docx
   │   │   ├── 立足数字经济新赛道推动数据要素产业创新发展行动方案.docx
   ...
   │   └── 黑龙江省
   │       ├── 黑龙江省促进大数据发展应用条例.docx
   │       ├── 黑龙江省促进大数据发展应用条例.txt
   │       └── 黑龙江省.docx
   └── metadata.json
   ```
   数据示例：
   ```json
   {"query": "在上海市关于数据资产管理的通知中，哪种方式被提倡以促进数据资产的合规高效流通？{\"A\": \"完全依靠政府监管\", \"B\": \"市场主导与政府引导相结合\", \"C\": \"企业自主开发\", \"D\": \"无条件的数据共享\"}\n请直接回答 A/B/C/D，不要解释。", "response": "B", "level_1_dim": "single-modal", "level_2_dim": "text", "level_3_dim": "government", "level_4_dim": "上海市"}
   ```



2. 运行测试：
   ```bash
   ianvs -f examples/government_rag/singletask_learning_bench/benchmarkingjob.yaml
   ```
