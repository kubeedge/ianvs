# ChnSentiCorp-Lite数据集构建和使用指南

## 概述

ChnSentiCorp-Lite是首个专门为PIPL合规的云边协同LLM系统设计的基准数据集。该数据集包含3,000个样本，具有多层隐私标注、合成PII模板、PIPL合规映射、MIA评测子集和差分隐私预算基线。

## 数据集特性

### 核心特性
- **总样本数**: 3,000 (2,000训练集, 500验证集, 500测试集)
- **数据格式**: JSONL格式，包含完整的隐私标注
- **数据大小**: ~15MB (轻量级，适合快速评估)
- **隐私保证**: 所有合成PII数据本地生成，保证不出边界

### 隐私标注特性
- **多层隐私标注**: 每个样本标注隐私敏感度级别 (`general`, `high_sensitivity`)
- **PII实体类型**: 支持姓名、电话、邮箱、身份证等多种PII类型
- **PIPL合规映射**: 细粒度标注PIPL第38-40条跨境传输权限
- **攻击评测子集**: 专门用于Neighbourhood MIA、LOSS和LiRA攻击测试
- **差分隐私基线**: 预计算各种ε/δ配置的隐私预算消耗

## 数据模式

### 通用敏感度样本格式
```json
{
  "sample_id": "chnsc_001234",
  "text": "这家餐厅的服务真的很不错",
  "label": "positive",
  "privacy_level": "general",
  "pii_entities": [],
  "pipl_cross_border": true,
  "synthetic_pii": null,
  "privacy_budget_cost": 0.0,
  "metadata": {
    "source": "ChnSentiCorp",
    "domain": "restaurant_review",
    "length": 12,
    "mia_test_subset": false
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
    "phone_masked": "138****2567",
    "generation_template": "complaint_with_contact"
  },
  "privacy_budget_cost": 1.2,
  "metadata": {
    "source": "synthetic_generation",
    "domain": "customer_service",
    "length": 18,
    "mia_test_subset": true
  }
}
```

## 构建数据集

### 1. 运行数据集构建器

```bash
cd examples/OSPP-implemention-Proposal-PIPL-Compliant-Cloud-Edge-Collaborative-Privacy-Preserving-Prompt-Processing-Framework-203/edge-cloud_collaborative_learning_bench
python dataset_builder.py
```

### 2. 验证数据集质量

```bash
python dataset_validator.py
```

### 3. 查看数据集使用示例

```bash
python dataset_usage_example.py
```

## 数据集结构

```
data/chnsenticorp_lite/
├── train.jsonl          # 训练集 (2,000样本)
├── val.jsonl            # 验证集 (500样本)
├── test.jsonl            # 测试集 (500样本)
├── statistics.json       # 数据集统计信息
└── validation_report.json # 验证报告
```

## 字段说明

### 必需字段
- **sample_id**: 样本唯一标识符
- **text**: 文本内容
- **label**: 情感标签 (`positive`, `negative`)
- **privacy_level**: 隐私敏感度级别 (`general`, `high_sensitivity`)
- **pii_entities**: PII实体类型列表
- **pipl_cross_border**: 是否允许跨境传输
- **privacy_budget_cost**: 隐私预算成本
- **metadata**: 元数据信息

### 可选字段
- **synthetic_pii**: 合成PII数据 (仅高敏感度样本)

## 隐私保护特性

### 1. 隐私级别分类
- **general**: 通用敏感度，允许跨境传输
- **high_sensitivity**: 高敏感度，禁止跨境传输

### 2. PII实体类型
- **PERSON**: 人名
- **PHONE**: 电话号码
- **EMAIL**: 邮箱地址
- **ID_CARD**: 身份证号码

### 3. 隐私预算管理
- 根据隐私级别和PII实体类型计算预算成本
- 支持不同ε/δ配置的预算消耗计算

### 4. 合规性验证
- 高敏感度数据禁止跨境传输
- 实时隐私预算管理和审计日志
- 跨境传输前合规性验证

## 使用示例

### 1. 加载数据集

```python
import json
from pathlib import Path

def load_dataset(data_dir="./data/chnsenticorp_lite"):
    """加载数据集"""
    dataset = {}
    for split in ["train", "val", "test"]:
        file_path = Path(data_dir) / f"{split}.jsonl"
        with open(file_path, "r", encoding="utf-8") as f:
            dataset[split] = [json.loads(line) for line in f]
    return dataset
```

### 2. 过滤高敏感度样本

```python
def filter_high_sensitivity_samples(dataset):
    """过滤高敏感度样本"""
    high_sensitivity_samples = []
    for split, data in dataset.items():
        for sample in data:
            if sample["privacy_level"] == "high_sensitivity":
                high_sensitivity_samples.append(sample)
    return high_sensitivity_samples
```

### 3. 检查跨境传输合规性

```python
def check_cross_border_compliance(dataset):
    """检查跨境传输合规性"""
    violations = []
    for split, data in dataset.items():
        for sample in data:
            if (sample["privacy_level"] == "high_sensitivity" and 
                sample["pipl_cross_border"]):
                violations.append(sample)
    return violations
```

### 4. 计算隐私预算统计

```python
def calculate_privacy_budget_stats(dataset):
    """计算隐私预算统计"""
    budget_costs = []
    for split, data in dataset.items():
        for sample in data:
            budget_costs.append(sample["privacy_budget_cost"])
    
    return {
        "mean": np.mean(budget_costs),
        "std": np.std(budget_costs),
        "min": np.min(budget_costs),
        "max": np.max(budget_costs)
    }
```

## 质量保证

### 1. 数据验证
- 自动PII检测准确率 >95%
- 隐私泄露测试：零原始PII在跨境传输样本中
- 标注一致性：标注者间一致性κ>0.85

### 2. 隐私保证
- 所有合成PII数据本地生成
- 保证原始个人信息不出边界
- 零原始文本跨境传输

### 3. 可用性
- 包含完整文档、使用示例和评估脚本
- 支持IANVS标准结构
- 一键执行部署

## 评估基准

### 1. 效用评估
- 准确率、F1分数
- 端到端延迟
- 吞吐量

### 2. 隐私评估
- MIA攻击成功率
- 隐私预算消耗
- PII泄露风险

### 3. 合规评估
- PIPL合规评分
- 跨境传输验证
- 审计完整性

### 4. 性能评估
- 边缘设备延迟
- 云端处理时间
- 网络传输开销

## 注意事项

1. **数据隐私**: 所有合成PII数据保证本地生成，不出边界
2. **合规性**: 严格遵循PIPL法规，确保跨境传输合规
3. **质量保证**: 定期验证数据集质量和隐私保护效果
4. **版本控制**: 使用版本控制管理数据集更新

## 技术支持

如有问题或建议，请参考：
- 数据集构建器: `dataset_builder.py`
- 数据集验证器: `dataset_validator.py`
- 使用示例: `dataset_usage_example.py`
- 完整文档: `DATASET_GUIDE.md`

## 更新日志

- **v1.0.0**: 初始版本，包含3,000个样本的完整数据集
- 支持多层隐私标注和PIPL合规映射
- 集成MIA攻击评测子集和差分隐私基线
