# 基于 KubeEdge-Ianvs 的云边协同PIPL隐私合规提示处理框架（PIPL 合规）(#203)



## 摘要

随着大语言模型（LLM）的广泛应用，传统“云端直连”的推理模式要求用户将包含潜在敏感信息的提示语上传至远端服务器，带来隐私与合规风险（尤其涉及个人信息跨境传输）。纯边缘轻量模型往往难以满足复杂推理需求。本提案在 KubeEdge-Ianvs 实现“云边协同、PIPL 合规”的隐私保护 LLM 推理框架：边缘侧对用户提示执行不可逆的隐私变换为匿名特征向量；云侧仅基于匿名向量与最小必要标签进行推理，达到“零原文跨境、最小标签跨境”，同时利用云端模型能力。

## 目录

- 概述与价值
- 场景示例
- 模型与数据集
- 架构与端到端流程
- 核心算法设计
- 评测方法与报告
- 运行与复现
- 验收标准（DoD）
- 限制与后续计划


## 概述与价值

- 新增合规场景与基线：提供“隐私保护型云边协同 LLM”的可运行场景与评测基线。
- 可复用隐私组件：边缘隐私检测器、不可逆变换引擎、合规闸门/预算管理器与隐私攻击评测套件。
- 轻量复现方案：基于 Colab 的边缘模拟、轻量数据集与一键脚本。
- PIPL 合规模板：覆盖“最小必要、预算审计、零原文跨境”的工程化模板。

## 场景示例（跨境电商客服，多轮）

- 法域与算力：边缘在中国境内（Colab 模拟），云侧在境外（Ianvs 推理服务）。
- 隐私策略：高敏 ε≤0.8，通用 ε≤1.2，默认 δ=1e-5；会话级预算累计与限速。
- 零原文跨境：全程仅传输“匿名向量 + 最小必要标签”。

轮次 1

- 用户：我上周买的 iPhone 15 Pro 发热严重，怎么退货？
- 边缘：判为通用敏感度；应用 ε=1.0、mask_ratio=0.4、投影 768→64。载荷：64 维向量 + {"intent": "return_request"}。
- 云侧：基于匿名上下文生成退货政策指引。

轮次 2

- 用户：订单 2024-09-01-3309，收件人张三，电话 138****2567。需要我付邮费吗？
- 边缘：检测高敏实体；应用 ε=0.8；本地校验与脱敏。载荷：64 维向量 + {"order_valid": true, "return_period": "in_range"}。
- 云侧：不含任何 PII 的邮费规则解释。

审计：记录每轮 PIPL 分级、参数（ε、mask_ratio、proj_dims）、预算累计与载荷哈希。

## 模型与数据集

- 边侧模型：Llama-3-8B-Instruct（4bit/INT4），用于本地 PIPL 检测、辅助语义分类与隐藏态提取。
- 云侧模型：GPT-4o-mini，接收匿名向量与最小标签。
- 数据集：ChnSentiCorp-Lite（2k 训练/500 验证/500 测试）＋本地合成 PII 模板；合成 PII 不出境。

## 架构与端到端流程

系统采用严格的云边职责分离，保障数据处理流程的合规性。

#### 系统组件架构图

<p align="center">
  <img src="images/System Component Architecture.png" alt="组件架构图" width="900">
</p>

#### 端到端流程图

<p align="center">
  <img src="images/End-to-End Workflow.png" alt="端到端流程图" width="900">
</p>

## 核心算法设计

### 1. 隐私检测（PIPL 分级）

- 检测通道：正则/规则、轻量 NER（如 BiLSTM-CRF）与语义分类在边缘侧本地执行。
- 风险评分与策略映射：将检测结果汇总为 risk∈[0,1]，映射到 privacy_policy.yaml 确定 ε、mask_ratio、proj_dims，并据此设定 pipl_level（general/high）。

### 2. 不可逆变换（隐私“加密”）

将自适应差分隐私、显著性引导掩码与随机/多层投影整合为流水线：
- 自适应差分隐私（DP）：L2 裁剪与按（ε, δ）校准的高斯噪声，约束泄露风险。
- 显著性掩码：屏蔽嵌入向量中 top-k 显著维度，抑制敏感线索。
- 投影：随机/MLP 降维至低维空间，降低可逆性。
- 动态策略：依据 pipl_level 动态更新 ε 与 mask_ratio；跨境仅传输匿名向量。

- 预算管理与合规闸门：会话级 PrivacyBudgetManager 累计 ε 并限速；ComplianceGate 在跨境前校验载荷格式。

## 评测方法与报告

- 效用：ChnSentiCorp-Lite 的任务准确率；端到端时延。
- 隐私：邻域 MIA 对比 LOSS/LiRA；报告 TPR@FPR ∈ {1%, 0.1%, 0.01%} 与 AUC；NMI；反演告警基线。
- 合规：最小必要性、预算合规与审计完整性检查。

## 运行与复现

- 一键脚本：scripts/setup_colab_edge.sh、scripts/run_edge_pipeline.sh、scripts/run_cloud_infer.sh、scripts/run_eval_all.sh
- 硬件：边缘建议 NVIDIA T4（16GB）或更高，以运行 4bit Llama-3-8B-Instruct。
- 关键配置示例：

```yaml
# configs/model_edge.yaml
model_name: "meta-llama/Llama-3-8B-Instruct"
quantization: "4bit"
hidden_layer_index: -2
pooling_strategy: "mean"
projection_dim: 768
```

```yaml
# configs/model_cloud.yaml
model_name: "gpt-4o-mini"
adapter_config:
  input_dim: 64
  hidden_dim: 512
  output_dim: 4096
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

## 验收标准（DoD）

- 工程：唯一模型/数据集已声明；图表与 PlantUML；run_eval_all.sh 跑通端到端与全部评测。
- 算法/评测：实现 LOSS、LiRA 与邻域 MIA，并产出 TPR@FPR 与 AUC 对比。
- 合规：抓包确认载荷格式；单测确保无原文/PII 跨境；审计日志完备。

## 限制与后续计划

- TransformEngine 参数为静态；探索自适应策略。
- 反演防御基线初步；引入更强攻击模型。
- 仅支持文本；扩展到语音/图像多模态。


