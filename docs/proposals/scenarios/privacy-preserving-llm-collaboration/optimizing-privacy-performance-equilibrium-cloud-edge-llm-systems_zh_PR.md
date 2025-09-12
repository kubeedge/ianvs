# **基于 KubeEdge-Ianvs 的云边协同PIPL隐私合规提示处理框架（#203）**

**关联 Issue:** Closes #203

## **摘要 (Description)**

随着大语言模型（LLM）的广泛应用，传统的“云端直连”推理模式要求用户将包含潜在敏感信息的提示语上传至远端服务器，这带来了显著的隐私与合规风险，尤其在涉及个人信息跨境传输的场景下。纯边缘侧的轻量模型则难以满足复杂的推理性能需求。  
本提案在 KubeEdge-Ianvs 平台上，实现了一套“云边协同、PIPL 合规”的隐私保护 LLM 推理框架。其核心思想是：
**边缘侧（Edge）** 对用户输入的敏感提示语执行不可逆的隐私变换，将其转换为匿名化的特征向量；
**云侧（Cloud）** 仅基于此匿名向量进行推理，全程不接触、不存储任何原始文本。这一设计旨在实现“零原文跨境、最小标签跨境”，确保在利用云端强大模型能力的同时，严格遵循 PIPL（个人信息保护法）的“最小必要”和“安全保障”原则。

## **模型与数据集**

为保证场景的可复现性与评测的确定性，本方案固定采用以下唯一的模型与数据集配置：

* **边侧模型 (Edge Model):** Llama-3-8B-Instruct (4bit/INT4 量化版)。  
  * **用途:** 在边缘设备上运行，负责本地的 PIPL 实体/意图识别、辅助语义分类，并提取用于生成匿名向量的文本隐藏态。4bit 量化旨在适配如 NVIDIA T4 或消费级低显存 GPU 的边缘算力环境。  
* **云侧模型 (Cloud Model):** GPT-4o-mini。  
  * **用途:** 作为云端的大语言模型，接收从边缘侧传输的匿名向量及最小必要标签，执行核心的推理与生成任务。  
* **数据集 (Dataset):** ChnSentiCorp-Lite。  
  * **来源:** 从中文情感分析数据集 ChnSentiCorp 中抽样构建，包含 2000条 训练、500条 验证与 500条 测试样本。  
  * **用途:** 用于评测框架在通用文本上的效用。此外，框架将内置一套 PII (个人可识别信息) 合成模板，用于在本地动态生成包含高敏信息的样本（例如，在 ChnSentiCorp-Lite 样本中插入姓名、电话等），专门用于高敏感度策略的评测，**合成的含 PII 数据确保不出境**。

## **为 Ianvs 带来的价值**

* **新增合规场景与基线:** 首次在 Ianvs 中提供“隐私保护型云边协同 LLM”的可运行场景与评测基线，拓展了平台在 AI 安全与合规领域的应用。  
* **可复用隐私组件:** 贡献一系列可插拔的核心隐私处理组件，包括边缘隐私检测器、不可逆变换引擎、合规闸门/预算管理器，以及全面的隐私攻击评测套件。  
* **轻量复现方案:** 通过 Colab 模拟边缘环境，结合轻量数据集与一键化脚本，显著降低社区成员的硬件门槛和评测成本。  
* **PIPL 合规实践模板:** 以 PIPL 为口径，提供了一个覆盖“最小必要、预算审计、零原文跨境”核心要求的工程化模板，为其他合规敏感行业（如金融、医疗）的应用提供了参考。

## **场景示例（跨境电商客服，多轮）**

这是一个模拟跨境电商场景的多轮对话，展示框架如何根据对话内容动态调整隐私策略。

* **法域与算力:** 边缘侧在中国境内（Colab 模拟），云侧位于境外（Ianvs 推理服务）。  
* **隐私策略参数:** 高敏 ε≤0.8，通用 ε≤1.2，默认 δ=1e-5；会话级预算累计与速率限制。  
* **零原文跨境:** 全程仅传输“匿名向量 + 最小必要标签”，确保原始文本不离境。

**轮 1**

* **用户:** 我上周在你们店买的 iPhone 15 Pro 发热严重，怎么退货？  
* **边缘侧:**  
  * PIPL 分类器将此提示语识别为 **通用敏感度** (商品问题，无 PII)。  
  * 应用策略：ε=1.0，显著性掩码比例 0.4，投影维度 768→64。  
  * **跨境 Payload:** Anonymized Vector (64-dim) + {"intent": "return_request"}。  
* **云侧:** 基于匿名向量和意图标签，生成通用的退货政策与流程并返回。

**轮 2**

* **用户:** 订单号 2024-09-01-3309，收件人张三，电话 138****2567，需要我付邮费吗？  
* **边缘侧:**  
  * PIPL 分类器识别出 **高敏实体** (订单号/姓名/电话)。  
  * 触发高敏策略：ε=0.8；本地执行订单号校验与实体脱敏。  
  * **跨境 Payload:** Anonymized Vector (64-dim) + {"order_valid": true, "return_period": "in_range"}。  
* **云侧:** 根据匿名上下文和布尔标签，生成关于邮费规则的解释，回复不含任何 PII。

**审计:** 每一轮的 PIPL 分类、脱敏/变换参数 (ε, mask_ratio, proj_dims)、预算累计与跨境传输哈希均被记录，满足最小必要与可追溯的合规要求。

## **目录与产物**

项目产物组织如下，覆盖文档、代码、评测与配置：

```
privacy_llm_cross_border/
├── README.md
├── configs/
│ ├── model_config.yaml
│ └── privacy_policy.yaml
├── data/
│ └── chnsenticorp_lite/
│ ├── test.jsonl
│ └── train.jsonl
├── scripts/
│ └── run_benchmark.sh
└── src/
├── cloud_inference.py
├── edge_privacy.py
└── evaluation/
├── attacks.py
└── metrics.py

```

## **架构与端到端流程**

系统采用严格的云边职责分离架构，确保数据处理流程的合规性。

#### **系统组件架构图 (System Component Architecture)**

![System Component Architecture](<./images/System Component Architecture.png>)

#### **端到端流程图 (End-to-End Workflow)**

![End-to-End Workflow](<./images/End-to-End Workflow.png>)

## **核心算法设计**

### **1. 隐私检测 (PIPL 分级)**

* **检测通道:** 结合规则/字典（身份证、手机号等）、轻量 NER 与语义分类，在边侧本地标注 PII。  
* **风险评分与策略映射:** 将结果聚合为 risk∈[0,1]，映射至 configs/privacy_policy.yaml，得到 ε、mask_ratio、proj_dims。  
* **输出:** {entities, pipl_level ∈ {general, high}, risk, policy}

### **2. 不可逆变换 (隐私“加密”)**

该模块集成自适应差分隐私、显著性引导掩码和随机/多层投影，实现高效且不可逆的隐私保护。  
关键步骤：  
* 基于 (ε, δ)-DP 的 L2 截断与高斯噪声注入。  
* 由 mask_ratio 控制的 top-k 显著性维度抑制。  
* 采用 Johnson–Lindenstrauss 随机投影或 MLP 投影进行降维。  
* 策略动态设参：{epsilon, mask_ratio, proj_dims}。

* **预算管理与合规闸门:** 会话级预算累计与限流；跨境前由 ComplianceGate 校验“匿名向量 + 最小标签”结构。

## **评测方法与报告**

提供一键化脚本 run_eval_all.sh，执行三类评测并生成报告。

### **1. 效用 (Utility) 评测**

* **目的:** 衡量在引入隐私保护变换后，系统在下游任务上的性能表现与开销。  
* **指标:**  
  * **任务准确率 (Task Accuracy):** 针对 ChnSentiCorp-Lite 数据集的文本情感分类任务，对比启用隐私变换前后，云侧 LLM 推理结果的准确率变化。  
  * **端到端时延 (End-to-End Latency):** 测量从用户输入提示到接收最终答复的完整耗时，评估边缘侧隐私处理带来的额外性能开销。

### **2. 隐私 (Privacy) 评测**

* **目的:** 定量评估信息泄露风险。  
* **核心方法:** 采用**邻域比较成员推断攻击 (Neighbourhood Membership Inference Attack)** 作为核心评测主线，并与两种基线方法进行对比。  
  * **核心思想:** 一个模型对其训练集中的样本（成员）通常会比对非常相似的、未见过的样本（非成员）表现出更低的损失（即更高的置信度）。邻域比较攻击正是利用了这一点，它不依赖参考模型，而是通过对目标样本进行微小的、保义的修改来合成“邻居”样本，然后比较目标样本与邻居们在模型下的损失差异。如果目标样本的损失显著低于其邻居的平均损失，则判定其为成员。  
  * **邻域生成:** 我们采用基于 BERT 类似 Masked Language Model (MLM) 的词汇替换方法生成邻居。具体来说，对原句中的某个词，我们在其输入嵌入上施加一个较高的 dropout 概率，然后让 MLM 预测最适合替换的词语，从而生成一系列语义和语法上都高度相似的邻居样本。  
* **指标:**  
  * **MIA 对比:** 在极低的假阳性率 (False Positive Rate, FPR) 下，报告攻击的真阳性率 (True Positive Rate, TPR)，即 **TPR @ FPR ∈ {1%, 0.1%, 0.01%}**，并报告 **AUC (Area Under the Curve)**。将邻域比较攻击的结果与 **LOSS Attack** 和 **LiRA (Likelihood Ratio Attack)** 基线进行对比。  
  * **互信息代理:** 计算匿名向量与原始 PII 类别之间的归一化互信息 (NMI) 作为泄露风险的代理指标。  
  * **反演告警基线:** 实现一个基础的嵌入反演模型，报告其在匿名向量上的 BLEU 分数，作为防御有效性的基线告警。

### **3. 合规 (Compliance) 评测**

* **目的:** 验证系统是否满足 PIPL 工程化要求。  
* **指标:** 最小必要性检查 (Payload 结构校验)、预算合规性检查 (ε 累计校验)、审计完整性检查 (日志覆盖率)。

## **运行与复现**

* **一键化脚本:**  
  * scripts/setup_colab_edge.sh: 快速配置 Colab 边缘环境。  
  * scripts/run_edge_pipeline.sh: 启动边缘侧服务。  
  * scripts/run_cloud_infer.sh: 启动云侧推理服务。  
  * scripts/run_eval_all.sh: 运行所有效用、隐私、合规评测。  
* **硬件建议:** 边缘侧建议使用配备 NVIDIA T4 (16GB) 或更高显存的 GPU，以流畅运行 4bit 量化的 Llama-3-8B-Instruct。  
* **重要配置样例:**

```yaml
# configs/model_edge.yaml
model_name: "meta-llama/Llama-3-8B-Instruct"
quantization: "4bit"
hidden_layer_index: -2 # 倒数第二层
pooling_strategy: "mean"
projection_dim: 768 # 降维至 768
```

```yaml
# configs/model_cloud.yaml
model_name: "gpt-4o-mini"
adapter_config:
  input_dim: 64
  hidden_dim: 512
  output_dim: 4096 # 对齐 GPT-4o-mini 的 embedding dim
```

```yaml
# configs/privacy_policy.yaml
pipl_levels:
  - level: "general"
    epsilon: 1.2
    delta: 0.00001
    mask_ratio: 0.4
    proj_dims: 64
  - level: "high_sensitivity"
    epsilon: 0.8
    delta: 0.00001
    mask_ratio: 0.6
    proj_dims: 64
```


## **验收标准 (DoD)**

* **工程侧:**  
  * 唯一的模型/数据集在 docs 与 configs 中明确声明，并可通过 CI 脚本校验。  
  * 核心架构与流程图表及其对应的 PlantUML 源码均已就位。  
  * 一键化脚本 run_eval_all.sh 能成功跑通端到端流程及三类评测。  
* **算法/评测侧:**  
  * 成功实现 LOSS / LiRA / Neighbourhood 三种 MIA 攻击的对比评测。  
  * 评测报告中清晰展示 **TPR@FPR** 与 **AUC** 的对比表格。  
  * 隐私预算和变换参数能根据 pipl_level 动态生效。  
* **合规侧:**  
  * 经抓包验证，跨境 payload 严格为“匿名向量 + 最小标签”结构。  
  * 单元测试覆盖无原文/PII 出境的场景。  
  * 边/云审计日志包含了所有关键参数与校验哈希。

## **限制与后续计划**

* 当前方案的 TransformEngine 参数为静态配置，未来可探索自适应的、基于输入内容动态调整变换强度的策略。  
* 嵌入反演的防御基线较为初步，后续可集成更强的反演攻击模型进行压力测试。  
* 当前仅支持文本模态，未来可扩展至语音、图像等多模态输入的隐私保护。

## **提交说明 (Commit Message)**

feat(privacy_llm_cross_border): PIPL-compliant cloud-edge LLMs with Edge=Llama-3-8B-Instruct and MIA eval (#203)

- End-to-end with irreversible edge transforms and minimal-tag cross-border.  
- Unique models/dataset: Edge Llama-3-8B-Instruct 4bit; Cloud GPT-4o-mini; ChnSentiCorp-Lite.  
- Privacy eval: LOSS/LiRA/Neighbour with low-FPR TPR/AUC.  

