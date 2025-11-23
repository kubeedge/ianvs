<<<<<<< HEAD

# 标题

**KubeEdge-Ianvs 隐私保护云边协同提示处理框架（PIPL合规）（#203）**

## 概述

本文提出一套面向大型语言模型（LLM）推理的隐私保护云边协同框架，符合PIPL合规要求。在保障跨境合规与隐私安全的同时，兼顾系统性能与可用性，提供完整的评估指标与工程化实现路径。

# 背景

随着大语言模型（LLM）的广泛应用，传统的"云端直连"推理模式要求用户将包含潜在敏感信息的提示语上传至远端服务器，这带来了显著的隐私与合规风险，尤其在涉及个人信息跨境传输的场景下。同时，纯边缘侧的轻量模型往往难以满足复杂的推理性能需求。

在当今的AI应用实践中，大型语言模型在自然语言处理和智能推理领域展现出巨大潜力。但现有模型通常缺乏针对隐私保护和跨境合规的专门优化，因此建立符合PIPL（个人信息保护法）要求的隐私保护LLM推理框架，对提升这些模型在实际应用环境中的合规性和安全性至关重要。

# 目标

1. 构建首个PIPL合规的云边协同隐私保护LLM推理框架
2. 设计零原始文本跨境传输的隐私保护评估基准
3. 将隐私保护LLM框架集成到KubeEdge-Ianvs平台中

# 提案

## 构建隐私保护LLM推理框架

本提案在 KubeEdge-Ianvs 平台上实现一套"云边协同、PIPL合规"的隐私保护LLM推理框架。其核心思想是：

1. **边缘侧隐私保护**：
   - 对用户输入的敏感提示语执行不可逆的隐私变换
   - 将原始文本转换为匿名化的特征向量
   - 本地完成PII检测、实体识别和隐私分级

2. **云端推理处理**：
   - 仅基于匿名向量进行推理，全程不接触原始文本
   - 接收最小必要标签执行核心推理任务
   - 确保"零原文跨境、最小标签跨境"

3. **PIPL合规保障**：
   - 严格遵循"最小必要"和"安全保障"原则
   - 实时隐私预算管理和审计日志
   - 跨境传输前合规性验证

## 模型与数据集配置

为保证场景的可复现性与评测的确定性，本方案固定采用以下模型与数据集配置：

### 模型配置
* **边缘模型**: Llama-3-8B-Instruct (4bit量化，API接入)
  - 负责本地PIPL实体识别、语义分类和匿名向量生成
  - 4bit量化适配边缘计算环境（如NVIDIA T4）
  - API端点：兼容OpenAI风格API格式

* **云端模型**: GPT-4o-mini (API接入)
  - 接收匿名向量执行核心推理任务
  - API端点：OpenAI API格式
  - 确保可扩展部署，无本地资源约束

### 数据集
* **ChnSentiCorp-Lite**（3,000样本，JSONL，含隐私标注与攻击子集）。
* **要点**：多层隐私标注、合成PII模板、PIPL合规映射、MIA评测子集、DP预算基线。
* **可用性**：提供完整文档、示例与评估脚本；隐私数据本地生成且不出边界。

## 价值与验收（精简）

- PIPL合规的云边协同LLM场景与基准
- 可复用隐私模块与轻量复现方案
- IANVS标准结构与一键执行
- 合规确认：零原文跨境、可审计、预算跟踪

## 集成到KubeEdge-Ianvs框架

1. 作为Ianvs框架的标准组件，提供良好的可扩展性
2. 确保在边缘设备上高效运行，与其他功能模块无缝协作
3. 遵循Ianvs标准的目录结构和接口规范

# 设计细节

## Ianvs集成架构

本框架完全集成KubeEdge Ianvs标准组件：
- **测试环境管理器**: 管理隐私保护LLM测试环境，包括数据集加载、模型初始化和隐私配置
- **测试案例控制器**: 控制隐私合规测试案例执行
- **故事管理器**: 管理云边协同推理场景、结果聚合和综合评估报告

**IANVS合规特性**:
- **标准方法接口**: 实现必需的`train()`, `predict()`, `evaluate()`方法
- **配置集成**: 与`benchmarkingjob.yaml`, `testenv.yaml`, `algorithm.yaml`无缝协作
- **指标框架**: 隐私、性能和合规指标集成至IANVS评估系统

### 系统组件架构

![系统组件架构](<./images/System Component Architecture.png>)

### 端到端工作流程

![端到端工作流程](<./images/End-to-End Workflow.png>)

## 项目结构

该框架遵循 IANVS 标准结构以支持隐私保护的云边协同 LLM：

```
examples/privacy_llm_cross_border/
└── edge-cloud_collaborative_learning_bench/
    ├── test_algorithm/
    │   └── privacy-routing/
    │       ├── privacy_routing_algorithm.py
    │       ├── edge_privacy.py
    │       ├── cloud_inference.py
    │       └── algorithm.yaml
    ├── testenv/
    │   ├── testenv.yaml
    │   └── metrics.py
    ├── benchmarkingjob.yaml
    ├── run_benchmark.sh
    ├── model_config.yaml
    ├── DATASET_README.md
    ├── requirements.txt
    └── README.md
```

数据集文件不包含在代码仓库中。

## 核心算法实现

### 集成算法结构

```python
class PrivacyPreservingLLM:
    def __init__(self, **kwargs):
        self.privacy_detector = self._init_detection_module(**kwargs)
        self.privacy_encryptor = self._init_encryption_module(**kwargs)
        
    def train(self, train_data, valid_data=None, **kwargs):
        # Ianvs标准训练接口
        return self._setup_collaborative_inference()
        
    def predict(self, data, **kwargs):
        # Ianvs标准预测接口
        return self._privacy_preserving_inference(data)
        
    def evaluate(self, data, **kwargs):
        # Ianvs标准评估接口
        return self._compliance_evaluation(data)
```

### 隐私保护模块

1. **隐私检测模块**:
   - 多通道PII检测（正则、NER、语义分类）
   - PIPL风险评估和策略映射
   - 实体净化和最小标签生成

2. **隐私加密模块**:
   - 差分隐私保护（L2截断、高斯噪声）
   - 显著性引导掩码
   - Johnson-Lindenstrauss降维投影
   - 实时合规验证

## 数据格式示例

### 通用敏感度样本格式

```json
{
  "sample_id": "chnsc_001234",
  "text": "这家餐厅的服务真的很不错",
  "label": "positive",
  "privacy_level": "general",
  "pii_entities": [],
  "pipl_cross_border": true,
  "privacy_budget_cost": 0.0,
  "metadata": {
    "source": "ChnSentiCorp",
    "domain": "restaurant_review"
  }
}
```

### 高敏感度样本格式

```json
{
  "sample_id": "chnsc_005678", 
  "text": "请联系张三，电话是138****2567处理订单问题",
  "label": "negative",
  "privacy_level": "high_sensitivity",
  "pii_entities": ["PERSON", "PHONE"],
  "pipl_cross_border": false,
  "synthetic_pii": {
    "person_name": "张三",
    "phone_masked": "138****2567"
  },
  "privacy_budget_cost": 1.2
}
```

## 评估基准

### 评估维度
1. **效用评估**: 准确率、F1分数、端到端延迟
2. **隐私评估**: MIA攻击成功率、隐私预算消耗、PII泄露风险  
3. **合规评估**: PIPL合规评分、跨境传输验证、审计完整性

### 隐私攻击测试
采用邻域成员推理攻击（Neighbourhood MIA）作为核心评测方法：
- 与LOSS Attack和LiRA基线对比
- 报告TPR@FPR指标和AUC值
- 评估信息泄露风险

## 场景示例

### 跨境电商客服多轮对话

**轮次1 - 通用问题**:
- 用户: "我买的iPhone 15 Pro发热严重，怎么退货？"
- 边缘: 识别为通用敏感度，应用ε=1.0策略
- 跨境: 匿名向量(64维) + {"intent": "return_request"}

**轮次2 - 高敏信息**:
- 用户: "订单号2024-09-01-3309，收件人张三，电话138****2567"
- 边缘: 识别为高敏感度，应用ε=0.8策略，本地实体脱敏
- 跨境: 匿名向量(64维) + {"order_valid": true, "return_period": "in_range"}

## 运行与部署

### 快速启动
```bash
# 安装依赖
pip install -r requirements.txt

# 运行基准测试
./run_benchmark.sh

# 或使用Ianvs命令
ianvs -f ./benchmarkingjob.yaml
```

### 关键配置示例

```yaml
test_algorithm/privacy-routing/algorithm.yaml:
algorithm:
  name: "privacy-preserving-llm-collaboration"
  type: "privacy_preserving_llm"
  url: "./privacy_routing_algorithm.py"
  
  # 边缘模型配置
  edge_model:
    name: "meta-llama/Llama-3-8B-Instruct"
    quantization: "4bit"
    api_base: "https://api.openai.com/v1"
    api_key: "${EDGE_API_KEY}"
    hidden_layer_index: -2
    pooling_strategy: "mean"
  
  # 云端模型配置
  cloud_model:
    name: "gpt-4o-mini"
    api_base: "https://api.openai.com/v1"
    api_key: "${CLOUD_API_KEY}"
    vector_adapter:
      input_dim: 64
      hidden_dim: 512
      output_dim: 4096
  
  # 隐私检测配置
  privacy_detection:
    detection_methods:
      regex_patterns: ["phone", "id_card", "email", "address"]
      ner_model: "hfl/chinese-bert-wwm-ext"
      entity_types: ["PERSON", "ORG", "LOC"]
    risk_weights:
      structured_pii: 0.8
      named_entities: 0.6
      semantic_context: 0.4
  
  # 隐私加密配置
  privacy_encryption:
    differential_privacy:
      general:
        epsilon: 1.2
        delta: 0.00001
        clipping_norm: 1.0
      high_sensitivity:
        epsilon: 0.8
        delta: 0.00001
        clipping_norm: 0.5
    anonymization:
      general_mask_ratio: 0.4
      high_sensitivity_mask_ratio: 0.6
      projection_method: "johnson_lindenstrauss"
      target_dims: 64
    budget_management:
      session_limit: 10.0
      rate_limit: 5
```

```yaml
testenv/testenv.yaml:
testenv:
  name: "privacy-preserving-llm-collaboration"
  
  dataset:
    name: "ChnSentiCorp-Lite"
    train_data: "./data/chnsenticorp_lite/train.jsonl"
    test_data: "./data/chnsenticorp_lite/test.jsonl"
    val_data: "./data/chnsenticorp_lite/val.jsonl"
  
  metrics:
    utility:
      - name: "accuracy"
        type: "classification_accuracy"
      - name: "f1_score"
        type: "f1_score"
    privacy:
      - name: "mia_attack_success"
        type: "membership_inference_attack"
      - name: "privacy_budget_consumption"
        type: "privacy_budget"
    performance:
      - name: "end_to_end_latency"
        type: "latency"
      - name: "throughput"
        type: "throughput"
    compliance:
      - name: "pipl_compliance_score"
        type: "compliance_score"
```

## 测试与验证

- ✅ 零原文跨境：仅传输匿名向量与最小必要标签
- ✅ 合规与预算：最小必要性校验与会话级隐私预算跟踪
- ✅ 隐私鲁棒性：MIA攻击成功率降至接近随机
- ✅ 效用与延迟：保持任务准确率，端到端开销可接受
- ✅ 集成与复现：兼容IANVS流水线；一键 `ianvs -f benchmarkingjob.yaml`

## DoD（精简）

- ✅ IANVS目录结构；算法实现train/predict/evaluate
- ✅ 通过 `ianvs -f benchmarkingjob.yaml` 端到端运行
- ✅ 实现效用/隐私/性能/合规等评测指标
- ✅ 零原文跨境；完整审计与预算跟踪

# 价值贡献

1. **首个PIPL合规LLM基准**: 为Ianvs引入隐私保护云边协同LLM场景
2. **创新数据集贡献**: ChnSentiCorp-Lite开创性的PIPL合规基准数据集
3. **可复用隐私组件**: 完整的隐私检测、变换和评测组件套件
4. **轻量复现方案**: 4bit量化和轻量数据集降低硬件门槛
5. **行业合规模板**: 为金融、医疗等合规敏感行业提供参考

该框架完全遵循KubeEdge-Ianvs标准，为AI隐私保护和跨境合规提供了完整的工程化解决方案。

## 限制与未来计划

* 目前Transform引擎的参数为静态配置，后续可探索基于输入内容自适应调整的策略。
* 嵌入反演防御基线仍较初步，未来可引入更强的反演攻击模型进行压力测试。
* 当前仅支持文本模态，未来可扩展至语音、图像等多模态的隐私保护。
=======
version https://git-lfs.github.com/spec/v1
oid sha256:926d044d387822f6e671590d74dcaeafc046a2ac6ca78df82a3fe2e05792742c
size 11756
>>>>>>> 9676c3e (ya toh aar ya toh par)
