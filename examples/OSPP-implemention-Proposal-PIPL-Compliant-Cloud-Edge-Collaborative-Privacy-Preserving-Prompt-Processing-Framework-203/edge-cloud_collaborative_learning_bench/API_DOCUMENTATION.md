# PIPL隐私保护LLM框架 API文档

## 1. 核心模块API

### 1.1 PII检测器 (PIIDetector)

```python
from test_algorithms.privacy_detection.pii_detector import PIIDetector

# 初始化
config = {
    'detection_methods': {
        'regex_patterns': ['phone', 'email', 'id_card'],
        'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD']
    }
}
detector = PIIDetector(config)

# 检测PII
text = "用户姓名：张三，电话：13812345678"
entities = detector.detect(text)

# 返回结果格式
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
```

### 1.2 差分隐私模块 (DifferentialPrivacy)

```python
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy

# 初始化
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

# 添加噪声
import numpy as np
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
dp_params = {
    'epsilon': 1.0,
    'delta': 0.00001,
    'clipping_norm': 1.0
}
result = dp.add_noise(data, dp_params)

# 返回结果格式
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

### 1.3 PIPL分类器 (PIPLClassifier)

```python
from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier

# 初始化
config = {
    'pipl_classification': {
        'threshold': 0.8,
        'categories': ['personal_info', 'sensitive_info', 'biometric_data']
    }
}
classifier = PIPLClassifier(config)

# 分类文本
text = "用户身份证号：110101199001011234"
classification = classifier.classify(text)

# 返回结果格式
{
    'category': 'personal_info',
    'confidence': 0.95,
    'risk_level': 'high',
    'requires_protection': True
}
```

### 1.4 合规性监控器 (ComplianceMonitor)

```python
from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor

# 初始化
config = {
    'compliance_monitoring': {
        'audit_level': 'detailed',
        'cross_border_policy': 'strict'
    }
}
monitor = ComplianceMonitor(config)

# 检查合规性
data = {
    'type': 'personal_info',
    'content': '用户信息',
    'cross_border': False
}
compliance = monitor.check_compliance(data)

# 返回结果格式
{
    'status': 'compliant',
    'violations': [],
    'recommendations': [],
    'risk_level': 'low'
}
```

## 2. 工作流程API

### 2.1 完整隐私保护工作流程

```python
def privacy_protection_workflow(text: str, features: np.ndarray) -> Dict[str, Any]:
    """完整的隐私保护工作流程"""
    
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
        'risk_level': classification.get('risk_level', 'low')
    }
    compliance = monitor.check_compliance(compliance_data)
    
    return {
        'pii_detected': len(pii_result),
        'classification': classification,
        'privacy_protection': dp_result,
        'compliance': compliance,
        'workflow_status': 'completed'
    }
```

## 3. 配置API

### 3.1 配置加载

```python
from config_loader import ConfigLoader

# 加载配置
config_loader = ConfigLoader('config.yaml')
config = config_loader.get_all()

# 获取特定配置
epsilon = config_loader.get('privacy.default_epsilon', 1.2)
api_key = config_loader.get('api_keys.edge_api_key')
```

### 3.2 环境变量配置

```python
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取配置
edge_api_key = os.getenv('EDGE_API_KEY')
privacy_budget = float(os.getenv('PRIVACY_BUDGET_LIMIT', '10.0'))
```

## 4. 错误处理API

### 4.1 异常处理

```python
from exceptions import PIPLException, PrivacyBudgetExceededException

try:
    result = dp.add_noise(data, dp_params)
except PrivacyBudgetExceededException as e:
    logger.error(f"Privacy budget exceeded: {e}")
    # 处理预算超限
except PIPLException as e:
    logger.error(f"PIPL framework error: {e}")
    # 处理框架错误
```

### 4.2 重试机制

```python
from error_handling import retry

@retry(max_attempts=3, delay=1.0)
def api_call():
    # API调用代码
    pass
```

## 5. 监控和日志API

### 5.1 日志配置

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipl_framework.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### 5.2 审计日志

```python
# 记录操作
monitor.log_operation({
    'operation_id': 'op_001',
    'operation_type': 'data_processing',
    'user_id': 'user_001',
    'data_type': 'personal_info'
})

# 获取审计日志
audit_log = monitor.get_audit_log()
```

## 6. 性能监控API

### 6.1 性能指标

```python
import time
import psutil

def monitor_performance(func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        performance_metrics = {
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'function_name': func.__name__
        }
        
        logger.info(f"Performance metrics: {performance_metrics}")
        return result
    
    return wrapper
```

## 7. 使用示例

### 7.1 基本使用

```python
# 初始化所有模块
config = ConfigLoader('config.yaml').get_all()
detector = PIIDetector(config['privacy_detection'])
dp = DifferentialPrivacy(config['privacy_encryption'])
classifier = PIPLClassifier(config['compliance'])
monitor = ComplianceMonitor(config['compliance'])

# 处理文本
text = "用户信息：张三，电话13812345678，邮箱zhangsan@example.com"
features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# 执行隐私保护工作流程
result = privacy_protection_workflow(text, features)
print(f"处理结果: {result}")
```

### 7.2 批量处理

```python
def batch_process(texts: List[str], features_list: List[np.ndarray]) -> List[Dict[str, Any]]:
    """批量处理"""
    results = []
    
    for text, features in zip(texts, features_list):
        try:
            result = privacy_protection_workflow(text, features)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process text: {e}")
            results.append({'error': str(e)})
    
    return results
```
