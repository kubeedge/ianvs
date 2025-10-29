# PIPL隐私保护LLM框架 - 快速入门指南

## 🚀 5分钟快速开始

### 步骤1: 环境准备

```bash
# 1. 确保Python版本 >= 3.8
python --version

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置环境变量（可选）
export EDGE_API_KEY="your_edge_api_key"
export CLOUD_API_KEY="your_cloud_api_key"
export PRIVACY_BUDGET_LIMIT="10.0"
```

### 步骤2: 基本使用

```python
#!/usr/bin/env python3
"""
PIPL隐私保护LLM框架 - 快速入门示例
"""

import sys
import os
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from test_algorithms.privacy_detection.pii_detector import PIIDetector
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier

def quick_start_example():
    """快速入门示例"""
    
    print("🚀 PIPL隐私保护LLM框架 - 快速入门")
    print("=" * 50)
    
    # 1. 初始化配置
    config = {
        'detection_methods': {
            'regex_patterns': ['phone', 'email', 'id_card'],
            'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD']
        },
        'differential_privacy': {
            'general': {
                'epsilon': 1.2,
                'delta': 0.00001,
                'clipping_norm': 1.0
            }
        },
        'compliance_monitoring': {
            'audit_level': 'detailed',
            'cross_border_policy': 'strict',
            'pipl_version': '2021'
        },
        'pipl_classification': {
            'threshold': 0.8,
            'categories': ['personal_info', 'sensitive_info', 'general']
        }
    }
    
    # 2. 初始化模块
    print("📦 初始化模块...")
    detector = PIIDetector(config)
    dp = DifferentialPrivacy(config)
    monitor = ComplianceMonitor(config)
    classifier = PIPLClassifier(config)
    print("✅ 模块初始化完成")
    
    # 3. 测试数据
    test_text = "用户姓名：张三，电话：13812345678，邮箱：zhangsan@example.com，身份证：110101199001011234"
    test_features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    print(f"\n📝 测试文本: {test_text}")
    print(f"📊 特征向量: {test_features}")
    
    # 4. PII检测
    print("\n🔍 步骤1: PII检测")
    pii_result = detector.detect(test_text)
    print(f"检测到 {len(pii_result)} 个PII实体:")
    for entity in pii_result:
        print(f"  - {entity['type']}: {entity['text']} (风险: {entity['risk_level']})")
    
    # 5. PIPL分类
    print("\n📋 步骤2: PIPL分类")
    classification = classifier.classify(test_text)
    print(f"分类: {classification['category']}")
    print(f"置信度: {classification['confidence']:.2f}")
    print(f"风险级别: {classification['risk_level']}")
    
    # 6. 差分隐私保护
    print("\n🔒 步骤3: 差分隐私保护")
    dp_params = {
        'epsilon': 1.0,
        'delta': 0.00001,
        'clipping_norm': 1.0
    }
    dp_result = dp.add_noise(test_features, dp_params)
    print(f"原始数据: {test_features}")
    print(f"噪声数据: {dp_result['noisy_data']}")
    print(f"使用的Epsilon: {dp_result['epsilon_used']}")
    print(f"剩余隐私预算: {dp_result['privacy_budget_remaining']}")
    
    # 7. 合规性检查
    print("\n⚖️ 步骤4: 合规性检查")
    compliance_data = {
        'type': classification.get('category', 'general'),
        'content': test_text,
        'risk_level': classification.get('risk_level', 'low'),
        'cross_border': False
    }
    compliance = monitor.check_compliance(compliance_data)
    print(f"合规状态: {compliance['status']}")
    print(f"风险级别: {compliance['risk_level']}")
    if compliance['recommendations']:
        print(f"建议: {compliance['recommendations']}")
    
    # 8. 记录操作
    print("\n📝 步骤5: 记录操作")
    monitor.log_operation({
        'operation_id': 'quick_start_001',
        'operation_type': 'privacy_protection',
        'user_id': 'demo_user',
        'data_type': classification.get('category', 'general'),
        'details': {
            'pii_count': len(pii_result),
            'privacy_budget_used': dp_result.get('epsilon_used', 0),
            'compliance_status': compliance.get('status', 'unknown')
        }
    })
    print("✅ 操作已记录到审计日志")
    
    # 9. 生成报告
    print("\n📊 步骤6: 生成报告")
    audit_report = monitor.get_audit_report()
    compliance_stats = monitor.get_compliance_statistics()
    
    print(f"审计报告: {audit_report['total_entries']} 条记录")
    print(f"合规统计: {compliance_stats['compliance_rate']:.2%} 合规率")
    
    # 10. 总结
    print("\n🎉 快速入门完成!")
    print("=" * 50)
    print("✅ PII检测: 成功")
    print("✅ PIPL分类: 成功")
    print("✅ 差分隐私: 成功")
    print("✅ 合规检查: 成功")
    print("✅ 审计记录: 成功")
    print("✅ 报告生成: 成功")
    
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

if __name__ == "__main__":
    try:
        result = quick_start_example()
        print(f"\n🏆 最终结果: {result['workflow_status']}")
        print(f"⏰ 完成时间: {result['timestamp']}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
```

### 步骤3: 运行示例

```bash
# 运行快速入门示例
python quick_start_example.py
```

## 📚 下一步学习

### 1. 深入学习核心模块

- [PII检测器详细文档](API_DOCUMENTATION.md#pii检测器-piidetector)
- [差分隐私模块详细文档](API_DOCUMENTATION.md#差分隐私模块-differentialprivacy)
- [合规性监控器详细文档](API_DOCUMENTATION.md#合规性监控器-compliancemonitor)

### 2. 配置管理

```python
from test_algorithms.common.config_loader import ConfigLoader

# 创建配置加载器
config_loader = ConfigLoader('config.yaml')

# 获取配置
budget = config_loader.get('privacy.budget_limit', 10.0)
print(f"隐私预算: {budget}")

# 设置配置
config_loader.set('privacy.budget_limit', 20.0)
```

### 3. 错误处理

```python
from test_algorithms.common.error_handling import handle_errors, retry_on_failure

@handle_errors(max_retries=3, retry_delay=1.0)
@retry_on_failure(max_attempts=3, delay=1.0)
def safe_operation():
    # 可能失败的操作
    pass
```

### 4. 批量处理

```python
def batch_process(texts, features_list):
    """批量处理示例"""
    results = []
    
    for text, features in zip(texts, features_list):
        try:
            # 处理单个文本
            result = privacy_protection_workflow(text, features)
            results.append(result)
        except Exception as e:
            print(f"处理失败: {e}")
            results.append({'error': str(e)})
    
    return results
```

## 🔧 常见问题解决

### 问题1: 模块导入错误

```python
# 解决方案：添加项目路径
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

### 问题2: 配置加载失败

```python
# 解决方案：检查配置文件
import os
if not os.path.exists('config.yaml'):
    print("使用默认配置")
    config_loader = ConfigLoader()
else:
    config_loader = ConfigLoader('config.yaml')
```

### 问题3: 隐私预算超限

```python
# 解决方案：检查预算使用情况
dp = DifferentialPrivacy(config)
params = dp.get_privacy_parameters('general')
print(f"当前预算: {params['epsilon']}")

# 使用更小的epsilon值
dp_params = {
    'epsilon': 0.5,  # 减小epsilon值
    'delta': 0.00001,
    'clipping_norm': 1.0
}
```

## 📖 更多资源

- [完整API文档](COMPREHENSIVE_API_DOCUMENTATION.md)
- [配置管理指南](CONFIGURATION_GUIDE.md)
- [错误处理指南](ERROR_HANDLING_GUIDE.md)
- [部署指南](DEPLOYMENT_GUIDE.md)
- [测试指南](TESTING_GUIDE.md)

## 🆘 获取帮助

如果您需要帮助，请：

1. 查看本文档的常见问题部分
2. 阅读完整的API文档
3. 检查日志文件中的错误信息
4. 联系技术支持团队

---

**祝您使用愉快！** 🎉
