# PIPL隐私保护LLM框架 - 完整API文档

## 📋 目录

1. [框架概述](#框架概述)
2. [快速开始](#快速开始)
3. [核心模块API](#核心模块api)
4. [配置管理API](#配置管理api)
5. [错误处理API](#错误处理api)
6. [工作流程API](#工作流程api)
7. [部署和运维API](#部署和运维api)
8. [示例代码](#示例代码)
9. [故障排除](#故障排除)

## 框架概述

PIPL隐私保护LLM框架是一个符合《个人信息保护法》(PIPL)的云边协同隐私保护大语言模型处理框架，提供完整的隐私检测、保护、合规性监控和审计功能。

### 核心特性

- 🔒 **隐私检测**: 自动识别和分类个人敏感信息
- 🛡️ **隐私保护**: 差分隐私、数据脱敏、加密传输
- 📊 **合规监控**: PIPL合规性实时监控和审计
- ⚡ **云边协同**: 边缘和云端模型协同处理
- 📈 **性能优化**: 高效的数据处理和资源管理

### 架构组件

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   边缘设备      │    │   云端服务      │    │   合规监控      │
│                │    │                │    │                │
│ • PII检测      │◄──►│ • 差分隐私      │◄──►│ • 审计日志      │
│ • 数据脱敏     │    │ • 模型推理      │    │ • 合规检查      │
│ • 本地处理     │    │ • 结果聚合      │    │ • 风险评估      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from test_algorithms.privacy_detection.pii_detector import PIIDetector
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor

# 初始化模块
config = {
    'detection_methods': {
        'regex_patterns': ['phone', 'email', 'id_card'],
        'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD']
    }
}

detector = PIIDetector(config)
dp = DifferentialPrivacy(config)
monitor = ComplianceMonitor(config)

# 检测PII
text = "用户姓名：张三，电话：13812345678"
entities = detector.detect(text)

# 应用差分隐私
import numpy as np
features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
dp_params = {'epsilon': 1.0, 'delta': 0.00001, 'clipping_norm': 1.0}
protected_data = dp.add_noise(features, dp_params)

# 合规性检查
compliance_result = monitor.check_compliance({
    'type': 'personal_info',
    'content': text,
    'cross_border': False
})
```

## 核心模块API

### 1. PII检测器 (PIIDetector)

#### 初始化

```python
from test_algorithms.privacy_detection.pii_detector import PIIDetector

config = {
    'detection_methods': {
        'regex_patterns': ['phone', 'email', 'id_card', 'address'],
        'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD', 'ADDRESS'],
        'ner_model': 'hfl/chinese-bert-wwm-ext'
    }
}

detector = PIIDetector(config)
```

#### 主要方法

##### `detect(text: str) -> List[Dict[str, Any]]`

检测文本中的个人敏感信息。

**参数**:
- `text` (str): 待检测的文本

**返回**:
- `List[Dict[str, Any]]`: 检测到的PII实体列表

**返回格式**:
```python
[
    {
        'type': 'PHONE',
        'text': '13812345678',
        'start': 19,
        'end': 30,
        'method': 'regex',
        'confidence': 0.9,
        'risk_level': 'critical',
        'context': '用户姓名：张三，电话：[PHONE]13812345678[/PHONE]',
        'sensitive': True,
        'requires_protection': True
    }
]
```

**示例**:
```python
text = "用户姓名：张三，电话：13812345678，邮箱：zhangsan@example.com"
entities = detector.detect(text)

for entity in entities:
    print(f"检测到 {entity['type']}: {entity['text']}")
    print(f"风险级别: {entity['risk_level']}")
    print(f"需要保护: {entity['requires_protection']}")
```

### 2. 差分隐私模块 (DifferentialPrivacy)

#### 初始化

```python
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy

config = {
    'differential_privacy': {
        'general': {
            'epsilon': 1.2,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
    },
    'budget_management': {
        'session_limit': 10.0,
        'rate_limit': 5
    }
}

dp = DifferentialPrivacy(config)
```

#### 主要方法

##### `add_noise(data: np.ndarray, dp_params: Dict[str, Any]) -> Dict[str, Any]`

为数据添加差分隐私噪声。

**参数**:
- `data` (np.ndarray): 原始数据
- `dp_params` (Dict[str, Any]): 差分隐私参数

**返回**:
- `Dict[str, Any]`: 处理结果

**返回格式**:
```python
{
    'noisy_data': array([...]),
    'original_shape': torch.Size([5]),
    'noise_scale': 5.3293,
    'epsilon_used': 1.0,
    'delta_used': 1e-05,
    'clipping_norm': 1.0,
    'privacy_budget_remaining': 9.0
}
```

**示例**:
```python
import numpy as np

# 准备数据
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# 设置差分隐私参数
dp_params = {
    'epsilon': 1.0,
    'delta': 0.00001,
    'clipping_norm': 1.0
}

# 添加噪声
result = dp.add_noise(data, dp_params)
print(f"原始数据: {data}")
print(f"噪声数据: {result['noisy_data']}")
print(f"剩余隐私预算: {result['privacy_budget_remaining']}")
```

##### `get_privacy_parameters(sensitivity_level: str = 'general') -> Dict[str, Any]`

获取隐私参数。

**参数**:
- `sensitivity_level` (str): 敏感度级别 ('general', 'high', 'critical')

**返回**:
- `Dict[str, Any]`: 隐私参数

**示例**:
```python
params = dp.get_privacy_parameters('general')
print(f"Epsilon: {params['epsilon']}")
print(f"Delta: {params['delta']}")
print(f"Clipping Norm: {params['clipping_norm']}")
```

##### `get_privacy_accountant_report() -> Dict[str, Any]`

获取隐私预算使用报告。

**返回**:
- `Dict[str, Any]`: 隐私预算报告

**示例**:
```python
report = dp.get_privacy_accountant_report()
print(f"总预算: {report['total_budget']}")
print(f"已使用: {report['consumed_budget']}")
print(f"剩余预算: {report['remaining_budget']}")
```

### 3. 合规性监控器 (ComplianceMonitor)

#### 初始化

```python
from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor

config = {
    'compliance_monitoring': {
        'audit_level': 'detailed',
        'cross_border_policy': 'strict',
        'pipl_version': '2021'
    }
}

monitor = ComplianceMonitor(config)
```

#### 主要方法

##### `check_compliance(data: Dict[str, Any]) -> Dict[str, Any]`

检查数据合规性。

**参数**:
- `data` (Dict[str, Any]): 数据字典

**返回**:
- `Dict[str, Any]`: 合规性检查结果

**返回格式**:
```python
{
    'status': 'compliant',
    'violations': [],
    'recommendations': [],
    'risk_level': 'low',
    'timestamp': '2025-10-23T02:00:34.182980'
}
```

**示例**:
```python
data = {
    'type': 'personal_info',
    'content': '用户信息',
    'cross_border': False,
    'risk_level': 'low'
}

result = monitor.check_compliance(data)
print(f"合规状态: {result['status']}")
print(f"风险级别: {result['risk_level']}")
if result['recommendations']:
    print(f"建议: {result['recommendations']}")
```

##### `get_audit_log() -> List[Dict[str, Any]]`

获取审计日志。

**返回**:
- `List[Dict[str, Any]]`: 审计日志条目

**示例**:
```python
audit_logs = monitor.get_audit_log()
for entry in audit_logs:
    print(f"时间: {entry['timestamp']}")
    print(f"操作: {entry['operation_type']}")
    print(f"状态: {entry['status']}")
```

##### `log_operation(operation: Dict[str, Any]) -> bool`

记录操作到审计轨迹。

**参数**:
- `operation` (Dict[str, Any]): 操作详情

**返回**:
- `bool`: 记录是否成功

**示例**:
```python
operation = {
    'operation_id': 'op_001',
    'operation_type': 'data_processing',
    'user_id': 'user_001',
    'data_type': 'personal_info',
    'details': {'processing_method': 'differential_privacy'}
}

success = monitor.log_operation(operation)
print(f"操作记录成功: {success}")
```

##### `get_audit_report() -> Dict[str, Any]`

生成审计报告。

**返回**:
- `Dict[str, Any]`: 审计报告

**示例**:
```python
report = monitor.get_audit_report()
print(f"总条目数: {report['total_entries']}")
print(f"时间范围: {report['period']}")
print(f"操作类型分布: {report['operation_types']}")
```

##### `get_compliance_statistics() -> Dict[str, Any]`

获取合规性统计。

**返回**:
- `Dict[str, Any]`: 合规性统计

**示例**:
```python
stats = monitor.get_compliance_statistics()
print(f"总操作数: {stats['total_operations']}")
print(f"合规率: {stats['compliance_rate']:.2%}")
print(f"违规数量: {stats['violation_count']}")
```

### 4. PIPL分类器 (PIPLClassifier)

#### 初始化

```python
from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier

config = {
    'pipl_classification': {
        'threshold': 0.8,
        'categories': ['personal_info', 'sensitive_info', 'biometric_data', 'general']
    }
}

classifier = PIPLClassifier(config)
```

#### 主要方法

##### `classify(text: str) -> Dict[str, Any]`

对文本进行PIPL分类。

**参数**:
- `text` (str): 待分类的文本

**返回**:
- `Dict[str, Any]`: 分类结果

**返回格式**:
```python
{
    'category': 'personal_info',
    'confidence': 0.95,
    'risk_level': 'high',
    'requires_protection': True
}
```

**示例**:
```python
text = "用户身份证号：110101199001011234"
result = classifier.classify(text)
print(f"分类: {result['category']}")
print(f"置信度: {result['confidence']}")
print(f"风险级别: {result['risk_level']}")
```

### 5. 风险评估器 (RiskEvaluator)

#### 初始化

```python
from test_algorithms.privacy_detection.risk_evaluator import RiskEvaluator

config = {
    'risk_weights': {
        'structured_pii': 0.8,
        'named_entities': 0.6,
        'semantic_context': 0.4,
        'entity_density': 0.5,
        'cross_border_risk': 0.7
    }
}

evaluator = RiskEvaluator(config)
```

#### 主要方法

##### `evaluate_risk(data: Dict[str, Any], context: str) -> Dict[str, Any]`

评估数据风险。

**参数**:
- `data` (Dict[str, Any]): 数据字典
- `context` (str): 上下文信息

**返回**:
- `Dict[str, Any]`: 风险评估结果

**返回格式**:
```python
{
    'risk_level': 'high',
    'risk_score': 0.8,
    'factors': ['Sensitive personal information'],
    'recommendations': ['Apply strong encryption', 'Implement access controls'],
    'timestamp': '2025-10-23T02:00:34.189967'
}
```

**示例**:
```python
data = {
    'type': 'phone',
    'value': '13812345678'
}
context = 'cross_border medical data processing'

result = evaluator.evaluate_risk(data, context)
print(f"风险级别: {result['risk_level']}")
print(f"风险分数: {result['risk_score']}")
print(f"风险因素: {result['factors']}")
```

## 配置管理API

### 配置加载器 (ConfigLoader)

#### 初始化

```python
from test_algorithms.common.config_loader import ConfigLoader

# 基本初始化
config_loader = ConfigLoader()

# 带配置文件初始化
config_loader = ConfigLoader(
    config_path='config.yaml',
    auto_reload=True,
    required_sections=['privacy', 'compliance']
)
```

#### 主要方法

##### `get(key: str, default: Any = None) -> Any`

获取配置值。

**参数**:
- `key` (str): 配置键（支持点号分隔，如 'privacy.budget_limit'）
- `default` (Any): 默认值

**返回**:
- `Any`: 配置值

**示例**:
```python
# 获取隐私预算限制
budget = config_loader.get('privacy.budget_limit', 10.0)

# 获取API密钥
api_key = config_loader.get('api_keys.edge_api_key')

# 获取日志级别
log_level = config_loader.get('logging.log_level', 'INFO')
```

##### `set(key: str, value: Any)`

设置配置值。

**参数**:
- `key` (str): 配置键
- `value` (Any): 配置值

**示例**:
```python
# 设置隐私预算
config_loader.set('privacy.budget_limit', 20.0)

# 设置API密钥
config_loader.set('api_keys.edge_api_key', 'new_key_123')

# 设置日志级别
config_loader.set('logging.log_level', 'DEBUG')
```

##### `get_all() -> Dict[str, Any]`

获取所有配置。

**返回**:
- `Dict[str, Any]`: 完整配置字典

**示例**:
```python
all_config = config_loader.get_all()
print(f"配置节: {list(all_config.keys())}")
```

##### `save_config(file_path: Optional[str] = None)`

保存配置到文件。

**参数**:
- `file_path` (Optional[str]): 保存路径，默认使用初始化时的路径

**示例**:
```python
# 保存到默认路径
config_loader.save_config()

# 保存到指定路径
config_loader.save_config('backup_config.yaml')
```

##### `export_env_file(file_path: str = '.env')`

导出环境变量文件。

**参数**:
- `file_path` (str): 导出文件路径

**示例**:
```python
# 导出到默认.env文件
config_loader.export_env_file()

# 导出到指定文件
config_loader.export_env_file('production.env')
```

### 全局配置函数

```python
from test_algorithms.common.config_loader import get_config, set_config

# 获取全局配置
budget = get_config('privacy.budget_limit', 10.0)

# 设置全局配置
set_config('privacy.budget_limit', 15.0)
```

## 错误处理API

### 异常类

```python
from test_algorithms.common.exceptions import (
    PIPLException, PrivacyBudgetExceededException, 
    ComplianceViolationException, ModelLoadException,
    ConfigurationException, DataProcessingException
)

# 基础异常
try:
    # 某些操作
    pass
except PIPLException as e:
    print(f"PIPL异常: {e.message}")
    print(f"错误代码: {e.error_code}")
    print(f"详细信息: {e.details}")

# 隐私预算超限异常
try:
    # 使用隐私预算
    pass
except PrivacyBudgetExceededException as e:
    print(f"隐私预算超限: {e.message}")
    print(f"当前预算: {e.details['current_budget']}")
    print(f"请求预算: {e.details['requested_budget']}")

# 合规性违规异常
try:
    # 合规性检查
    pass
except ComplianceViolationException as e:
    print(f"合规性违规: {e.message}")
    print(f"违规类型: {e.details['violation_type']}")
    print(f"严重程度: {e.details['severity']}")
```

### 错误处理装饰器

```python
from test_algorithms.common.error_handling import (
    handle_errors, retry_on_failure, timeout, rate_limit
)

# 错误处理装饰器
@handle_errors(max_retries=3, retry_delay=1.0)
def risky_operation():
    # 可能失败的操作
    pass

# 重试装饰器
@retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
def network_operation():
    # 网络操作
    pass

# 超时装饰器
@timeout(seconds=30.0)
def long_operation():
    # 长时间操作
    pass

# 速率限制装饰器
@rate_limit(calls_per_second=2.0)
def api_call():
    # API调用
    pass
```

### 资源管理

```python
from test_algorithms.common.error_handling import resource_manager

# 资源管理上下文
def cleanup_resource(resource):
    print(f"清理资源: {resource}")

with resource_manager("database_connection", cleanup_resource):
    # 使用资源
    print("使用数据库连接...")
```

## 工作流程API

### 完整隐私保护工作流程

```python
def privacy_protection_workflow(text: str, features: np.ndarray) -> Dict[str, Any]:
    """
    完整的隐私保护工作流程
    
    Args:
        text: 输入文本
        features: 特征向量
        
    Returns:
        处理结果
    """
    from test_algorithms.privacy_detection.pii_detector import PIIDetector
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
    from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier
    
    # 初始化模块
    config = {
        'detection_methods': {
            'regex_patterns': ['phone', 'email', 'id_card'],
            'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD']
        },
        'differential_privacy': {
            'general': {'epsilon': 1.2, 'delta': 0.00001, 'clipping_norm': 1.0}
        },
        'compliance_monitoring': {
            'audit_level': 'detailed',
            'cross_border_policy': 'strict'
        }
    }
    
    detector = PIIDetector(config)
    dp = DifferentialPrivacy(config)
    monitor = ComplianceMonitor(config)
    classifier = PIPLClassifier(config)
    
    # 步骤1: PII检测
    pii_result = detector.detect(text)
    
    # 步骤2: PIPL分类
    classification = classifier.classify(text)
    
    # 步骤3: 差分隐私保护
    dp_params = {
        'epsilon': 1.0,
        'delta': 0.00001,
        'clipping_norm': 1.0
    }
    dp_result = dp.add_noise(features, dp_params)
    
    # 步骤4: 合规性检查
    compliance_data = {
        'type': classification.get('category', 'general'),
        'content': text,
        'risk_level': classification.get('risk_level', 'low'),
        'cross_border': False
    }
    compliance = monitor.check_compliance(compliance_data)
    
    # 步骤5: 记录操作
    monitor.log_operation({
        'operation_id': 'workflow_001',
        'operation_type': 'privacy_protection',
        'user_id': 'user_001',
        'data_type': classification.get('category', 'general'),
        'details': {
            'pii_count': len(pii_result),
            'privacy_budget_used': dp_result.get('epsilon_used', 0),
            'compliance_status': compliance.get('status', 'unknown')
        }
    })
    
    return {
        'pii_detected': len(pii_result),
        'pii_entities': pii_result,
        'classification': classification,
        'privacy_protection': dp_result,
        'compliance': compliance,
        'workflow_status': 'completed',
        'timestamp': datetime.now().isoformat()
    }
```

### 批量处理工作流程

```python
def batch_privacy_protection_workflow(texts: List[str], features_list: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    批量隐私保护工作流程
    
    Args:
        texts: 文本列表
        features_list: 特征向量列表
        
    Returns:
        处理结果列表
    """
    results = []
    
    for i, (text, features) in enumerate(zip(texts, features_list)):
        try:
            result = privacy_protection_workflow(text, features)
            result['batch_index'] = i
            results.append(result)
        except Exception as e:
            logger.error(f"处理第{i}个文本时出错: {e}")
            results.append({
                'batch_index': i,
                'error': str(e),
                'workflow_status': 'failed'
            })
    
    return results
```

## 部署和运维API

### 系统监控

```python
from test_algorithms.common.error_handling import ResourceMonitor

# 创建资源监控器
monitor = ResourceMonitor({
    'memory_threshold': 80.0,
    'cpu_threshold': 80.0,
    'disk_threshold': 90.0
})

# 检查系统资源
resources = monitor.check_system_resources()
print(f"内存使用率: {resources['memory_percent']:.1f}%")
print(f"CPU使用率: {resources['cpu_percent']:.1f}%")
print(f"磁盘使用率: {resources['disk_percent']:.1f}%")

# 检查是否需要限流
if monitor.should_throttle():
    print("系统资源紧张，建议限流")
```

### 配置热更新

```python
# 启用配置热更新
config_loader = ConfigLoader(
    config_path='config.yaml',
    auto_reload=True
)

# 配置文件修改后会自动重新加载
# 可以通过以下方式手动重新加载
config_loader.reload_config('config.yaml')
```

## 示例代码

### 示例1: 基本PII检测

```python
from test_algorithms.privacy_detection.pii_detector import PIIDetector

# 初始化检测器
config = {
    'detection_methods': {
        'regex_patterns': ['phone', 'email', 'id_card'],
        'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD']
    }
}
detector = PIIDetector(config)

# 检测文本
text = """
用户信息：
姓名：张三
电话：13812345678
邮箱：zhangsan@example.com
身份证：110101199001011234
地址：北京市朝阳区
"""

entities = detector.detect(text)

print("检测结果:")
for entity in entities:
    print(f"- {entity['type']}: {entity['text']}")
    print(f"  风险级别: {entity['risk_level']}")
    print(f"  需要保护: {entity['requires_protection']}")
    print()
```

### 示例2: 差分隐私保护

```python
import numpy as np
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy

# 初始化差分隐私模块
config = {
    'differential_privacy': {
        'general': {
            'epsilon': 1.2,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
    },
    'budget_management': {
        'session_limit': 10.0,
        'rate_limit': 5
    }
}
dp = DifferentialPrivacy(config)

# 准备数据
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# 设置隐私参数
dp_params = {
    'epsilon': 1.0,
    'delta': 0.00001,
    'clipping_norm': 1.0
}

# 添加噪声
result = dp.add_noise(data, dp_params)

print(f"原始数据: {data}")
print(f"噪声数据: {result['noisy_data']}")
print(f"噪声规模: {result['noise_scale']:.4f}")
print(f"使用的Epsilon: {result['epsilon_used']}")
print(f"剩余隐私预算: {result['privacy_budget_remaining']}")
```

### 示例3: 合规性监控

```python
from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor

# 初始化合规性监控器
config = {
    'compliance_monitoring': {
        'audit_level': 'detailed',
        'cross_border_policy': 'strict',
        'pipl_version': '2021'
    }
}
monitor = ComplianceMonitor(config)

# 检查合规性
data = {
    'type': 'personal_info',
    'content': '用户身份证号：110101199001011234',
    'cross_border': False,
    'risk_level': 'high'
}

compliance_result = monitor.check_compliance(data)
print(f"合规状态: {compliance_result['status']}")
print(f"风险级别: {compliance_result['risk_level']}")
print(f"违规项: {compliance_result['violations']}")
print(f"建议: {compliance_result['recommendations']}")

# 记录操作
monitor.log_operation({
    'operation_id': 'op_001',
    'operation_type': 'data_processing',
    'user_id': 'user_001',
    'data_type': 'personal_info',
    'details': {'processing_method': 'differential_privacy'}
})

# 获取审计报告
report = monitor.get_audit_report()
print(f"审计报告: {report['total_entries']} 条记录")
```

### 示例4: 完整工作流程

```python
import numpy as np
from datetime import datetime

def complete_privacy_workflow():
    """完整的隐私保护工作流程示例"""
    
    # 初始化所有模块
    config = {
        'detection_methods': {
            'regex_patterns': ['phone', 'email', 'id_card'],
            'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD']
        },
        'differential_privacy': {
            'general': {'epsilon': 1.2, 'delta': 0.00001, 'clipping_norm': 1.0}
        },
        'compliance_monitoring': {
            'audit_level': 'detailed',
            'cross_border_policy': 'strict'
        },
        'pipl_classification': {
            'threshold': 0.8,
            'categories': ['personal_info', 'sensitive_info', 'general']
        }
    }
    
    # 导入模块
    from test_algorithms.privacy_detection.pii_detector import PIIDetector
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
    from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier
    
    # 初始化
    detector = PIIDetector(config)
    dp = DifferentialPrivacy(config)
    monitor = ComplianceMonitor(config)
    classifier = PIPLClassifier(config)
    
    # 测试数据
    text = "用户姓名：张三，电话：13812345678，邮箱：zhangsan@example.com"
    features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    print("=== 完整隐私保护工作流程 ===")
    print(f"输入文本: {text}")
    print(f"特征向量: {features}")
    print()
    
    # 步骤1: PII检测
    print("步骤1: PII检测")
    pii_result = detector.detect(text)
    print(f"检测到 {len(pii_result)} 个PII实体")
    for entity in pii_result:
        print(f"  - {entity['type']}: {entity['text']} (风险: {entity['risk_level']})")
    print()
    
    # 步骤2: PIPL分类
    print("步骤2: PIPL分类")
    classification = classifier.classify(text)
    print(f"分类: {classification['category']}")
    print(f"置信度: {classification['confidence']:.2f}")
    print(f"风险级别: {classification['risk_level']}")
    print()
    
    # 步骤3: 差分隐私保护
    print("步骤3: 差分隐私保护")
    dp_params = {
        'epsilon': 1.0,
        'delta': 0.00001,
        'clipping_norm': 1.0
    }
    dp_result = dp.add_noise(features, dp_params)
    print(f"原始数据: {features}")
    print(f"噪声数据: {dp_result['noisy_data']}")
    print(f"使用的Epsilon: {dp_result['epsilon_used']}")
    print(f"剩余隐私预算: {dp_result['privacy_budget_remaining']}")
    print()
    
    # 步骤4: 合规性检查
    print("步骤4: 合规性检查")
    compliance_data = {
        'type': classification.get('category', 'general'),
        'content': text,
        'risk_level': classification.get('risk_level', 'low'),
        'cross_border': False
    }
    compliance = monitor.check_compliance(compliance_data)
    print(f"合规状态: {compliance['status']}")
    print(f"风险级别: {compliance['risk_level']}")
    if compliance['recommendations']:
        print(f"建议: {compliance['recommendations']}")
    print()
    
    # 步骤5: 记录操作
    print("步骤5: 记录操作")
    monitor.log_operation({
        'operation_id': 'workflow_001',
        'operation_type': 'privacy_protection',
        'user_id': 'user_001',
        'data_type': classification.get('category', 'general'),
        'details': {
            'pii_count': len(pii_result),
            'privacy_budget_used': dp_result.get('epsilon_used', 0),
            'compliance_status': compliance.get('status', 'unknown')
        }
    })
    print("操作已记录到审计日志")
    print()
    
    # 步骤6: 生成报告
    print("步骤6: 生成报告")
    audit_report = monitor.get_audit_report()
    compliance_stats = monitor.get_compliance_statistics()
    
    print(f"审计报告: {audit_report['total_entries']} 条记录")
    print(f"合规统计: {compliance_stats['compliance_rate']:.2%} 合规率")
    print()
    
    print("=== 工作流程完成 ===")
    
    return {
        'pii_detected': len(pii_result),
        'classification': classification,
        'privacy_protection': dp_result,
        'compliance': compliance,
        'audit_report': audit_report,
        'compliance_stats': compliance_stats,
        'workflow_status': 'completed',
        'timestamp': datetime.now().isoformat()
    }

# 运行完整工作流程
if __name__ == "__main__":
    result = complete_privacy_workflow()
    print(f"最终结果: {result['workflow_status']}")
```

## 故障排除

### 常见问题

#### 1. 模块导入错误

**问题**: `ModuleNotFoundError: No module named 'test_algorithms'`

**解决方案**:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

#### 2. 配置加载失败

**问题**: 配置文件格式错误或路径不存在

**解决方案**:
```python
# 检查配置文件是否存在
import os
if not os.path.exists('config.yaml'):
    print("配置文件不存在，使用默认配置")

# 验证配置文件格式
import yaml
try:
    with open('config.yaml', 'r') as f:
        yaml.safe_load(f)
    print("配置文件格式正确")
except yaml.YAMLError as e:
    print(f"配置文件格式错误: {e}")
```

#### 3. 隐私预算超限

**问题**: `PrivacyBudgetExceededException`

**解决方案**:
```python
# 检查当前隐私预算
dp = DifferentialPrivacy(config)
params = dp.get_privacy_parameters('general')
print(f"当前预算: {params['epsilon']}")

# 重置隐私预算（如果支持）
# dp.reset_privacy_budget()

# 使用更小的epsilon值
dp_params = {
    'epsilon': 0.5,  # 减小epsilon值
    'delta': 0.00001,
    'clipping_norm': 1.0
}
```

#### 4. 合规性检查失败

**问题**: 合规性检查返回违规状态

**解决方案**:
```python
# 检查合规性结果
compliance = monitor.check_compliance(data)
if compliance['status'] != 'compliant':
    print(f"合规性问题: {compliance['violations']}")
    print(f"建议: {compliance['recommendations']}")
    
    # 根据建议调整数据或配置
    if 'Apply encryption' in compliance['recommendations']:
        # 应用加密
        pass
```

#### 5. 性能问题

**问题**: 处理速度慢或内存使用高

**解决方案**:
```python
# 检查系统资源
from test_algorithms.common.error_handling import ResourceMonitor

monitor = ResourceMonitor({
    'memory_threshold': 80.0,
    'cpu_threshold': 80.0
})

resources = monitor.check_system_resources()
if resources['status'] == 'warning':
    print("系统资源紧张，建议:")
    print("1. 减小批处理大小")
    print("2. 启用量化")
    print("3. 使用更高效的模型")
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 或者只启用特定模块的日志
logger = logging.getLogger('test_algorithms.privacy_detection.pii_detector')
logger.setLevel(logging.DEBUG)
```

#### 2. 使用配置验证

```python
from test_algorithms.common.config_loader import ConfigLoader

# 验证配置
config_loader = ConfigLoader('config.yaml')
try:
    # 尝试获取关键配置
    budget = config_loader.get('privacy.budget_limit')
    epsilon = config_loader.get('privacy.default_epsilon')
    print("配置验证通过")
except Exception as e:
    print(f"配置验证失败: {e}")
```

#### 3. 测试单个模块

```python
# 测试PII检测器
from test_algorithms.privacy_detection.pii_detector import PIIDetector

config = {'detection_methods': {'regex_patterns': ['phone']}}
detector = PIIDetector(config)

# 简单测试
test_text = "我的电话是13812345678"
result = detector.detect(test_text)
print(f"检测结果: {result}")
```

### 性能优化建议

#### 1. 批处理优化

```python
# 使用批处理而不是单个处理
def batch_process(texts, features_list):
    # 批量处理逻辑
    pass
```

#### 2. 缓存配置

```python
# 缓存配置加载器
global_config_loader = None

def get_config_loader():
    global global_config_loader
    if global_config_loader is None:
        global_config_loader = ConfigLoader('config.yaml')
    return global_config_loader
```

#### 3. 资源管理

```python
# 使用资源管理上下文
from test_algorithms.common.error_handling import resource_manager

def cleanup_model(model):
    # 清理模型资源
    pass

with resource_manager("model", cleanup_model):
    # 使用模型
    pass
```

---

## 📞 技术支持

如果您在使用过程中遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查日志文件中的错误信息
3. 验证配置文件的正确性
4. 联系技术支持团队

---

**版本**: 1.0.0  
**最后更新**: 2025-10-23  
**维护者**: PIPL框架开发团队
