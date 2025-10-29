# 🧪 综合测试指南

## 📋 概述

本指南包含所有测试相关的文档和说明。


## TESTING_GUIDE.md

# 测试指南

## 1. 单元测试

### 1.1 测试框架设置

```python
# test_framework.py
import unittest
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestFramework:
    """测试框架基类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_config = {
            'privacy_detection': {
                'detection_methods': {
                    'regex_patterns': ['phone', 'email'],
                    'entity_types': ['PERSON', 'PHONE', 'EMAIL']
                }
            },
            'privacy_encryption': {
                'differential_privacy': {
                    'general': {
                        'epsilon': 1.2,
                        'delta': 0.00001,
                        'clipping_norm': 1.0
                    }
                }
            }
        }
    
    def create_test_data(self):
        """创建测试数据"""
        return {
            'text_with_pii': "用户姓名：张三，电话：13812345678，邮箱：zhangsan@example.com",
            'text_without_pii': "这是一段普通的文本，不包含敏感信息",
            'feature_vector': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'batch_features': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        }
```

### 1.2 PII检测器单元测试

```python
# test_pii_detector_unit.py
import unittest
from test_algorithms.privacy_detection.pii_detector import PIIDetector

class TestPIIDetector(unittest.TestCase):
    """PII检测器单元测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'detection_methods': {
                'regex_patterns': ['phone', 'email', 'id_card'],
                'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD']
            }
        }
        self.detector = PIIDetector(self.config)
    
    def test_phone_detection(self):
        """测试电话号码检测"""
        text = "我的电话是13812345678"
        result = self.detector.detect(text)
        
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]['type'], 'PHONE')
        self.assertEqual(result[0]['text'], '13812345678')
        self.assertTrue(result[0]['requires_protection'])
    
    def test_email_detection(self):
        """测试邮箱检测"""
        text = "联系邮箱：zhangsan@example.com"
        result = self.detector.detect(text)
        
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]['type'], 'EMAIL')
        self.assertEqual(result[0]['text'], 'zhangsan@example.com')
    
    def test_id_card_detection(self):
        """测试身份证检测"""
        text = "身份证号：110101199001011234"
        result = self.detector.detect(text)
        
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]['type'], 'ID_CARD')
        self.assertEqual(result[0]['text'], '110101199001011234')
    
    def test_no_pii_detection(self):
        """测试无PII文本"""
        text = "这是一段普通的文本"
        result = self.detector.detect(text)
        
        self.assertEqual(len(result), 0)
    
    def test_risk_level_assessment(self):
        """测试风险级别评估"""
        text = "电话：13812345678"
        result = self.detector.detect(text)
        
        self.assertIn(result[0]['risk_level'], ['high', 'critical'])
    
    def test_context_extraction(self):
        """测试上下文提取"""
        text = "请拨打13812345678联系我"
        result = self.detector.detect(text)
        
        self.assertIn('[PHONE]', result[0]['context'])
        self.assertIn('13812345678', result[0]['context'])
```

### 1.3 差分隐私模块单元测试

```python
# test_differential_privacy_unit.py
import unittest
import numpy as np
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy

class TestDifferentialPrivacy(unittest.TestCase):
    """差分隐私模块单元测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
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
        self.dp = DifferentialPrivacy(self.config)
        self.test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_noise_addition(self):
        """测试噪声添加"""
        dp_params = {
            'epsilon': 1.0,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        
        result = self.dp.add_noise(self.test_data, dp_params)
        
        self.assertIn('noisy_data', result)
        self.assertIn('noise_scale', result)
        self.assertIn('privacy_budget_remaining', result)
        self.assertEqual(result['noisy_data'].shape, self.test_data.shape)
    
    def test_privacy_budget_consumption(self):
        """测试隐私预算消耗"""
        initial_budget = self.dp.get_privacy_parameters('general')['epsilon']
        
        dp_params = {
            'epsilon': 1.0,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        
        result = self.dp.add_noise(self.test_data, dp_params)
        remaining_budget = result['privacy_budget_remaining']
        
        self.assertLess(remaining_budget, initial_budget)
        self.assertEqual(remaining_budget, initial_budget - dp_params['epsilon'])
    
    def test_different_epsilon_values(self):
        """测试不同epsilon值"""
        epsilon_values = [0.5, 1.0, 2.0]
        noise_scales = []
        
        for epsilon in epsilon_values:
            dp_params = {
                'epsilon': epsilon,
                'delta': 0.00001,
                'clipping_norm': 1.0
            }
            result = self.dp.add_noise(self.test_data, dp_params)
            noise_scales.append(result['noise_scale'])
        
        # 验证epsilon越大，噪声越小
        for i in range(len(noise_scales) - 1):
            self.assertGreaterEqual(noise_scales[i], noise_scales[i + 1])
    
    def test_batch_processing(self):
        """测试批量处理"""
        batch_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        dp_params = {
            'epsilon': 1.0,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        
        result = self.dp.add_noise(batch_data, dp_params)
        
        self.assertEqual(result['noisy_data'].shape, batch_data.shape)
    
    def test_privacy_parameters_validation(self):
        """测试隐私参数验证"""
        # 测试无效的epsilon值
        with self.assertRaises(ValueError):
            invalid_params = {
                'epsilon': -1.0,
                'delta': 0.00001,
                'clipping_norm': 1.0
            }
            self.dp.add_noise(self.test_data, invalid_params)
        
        # 测试无效的delta值
        with self.assertRaises(ValueError):
            invalid_params = {
                'epsilon': 1.0,
                'delta': -0.00001,
                'clipping_norm': 1.0
            }
            self.dp.add_noise(self.test_data, invalid_params)
```

### 1.4 PIPL分类器单元测试

```python
# test_pipl_classifier_unit.py
import unittest
from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier

class TestPIPLClassifier(unittest.TestCase):
    """PIPL分类器单元测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'pipl_classification': {
                'threshold': 0.8,
                'categories': ['personal_info', 'sensitive_info', 'general']
            }
        }
        self.classifier = PIPLClassifier(self.config)
    
    def test_personal_info_classification(self):
        """测试个人信息分类"""
        text = "用户姓名：张三，身份证号：110101199001011234"
        result = self.classifier.classify(text)
        
        self.assertIn('category', result)
        self.assertIn('confidence', result)
    
    def test_sensitive_info_classification(self):
        """测试敏感信息分类"""
        text = "银行卡号：6222021234567890"
        result = self.classifier.classify(text)
        
        self.assertIn('category', result)
        self.assertIn('confidence', result)
    
    def test_general_text_classification(self):
        """测试普通文本分类"""
        text = "这是一段普通的文本内容"
        result = self.classifier.classify(text)
        
        self.assertIn('category', result)
        self.assertEqual(result['category'], 'general')
    
    def test_classification_confidence(self):
        """测试分类置信度"""
        text = "用户信息：张三"
        result = self.classifier.classify(text)
        
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
```

## 2. 集成测试

### 2.1 端到端集成测试

```python
# test_integration.py
import unittest
import numpy as np
from test_algorithms.privacy_detection.pii_detector import PIIDetector
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'privacy_detection': {
                'detection_methods': {
                    'regex_patterns': ['phone', 'email'],
                    'entity_types': ['PERSON', 'PHONE', 'EMAIL']
                }
            },
            'privacy_encryption': {
                'differential_privacy': {
                    'general': {
                        'epsilon': 1.2,
                        'delta': 0.00001,
                        'clipping_norm': 1.0
                    }
                }
            },
            'compliance': {
                'pipl_classification': {
                    'threshold': 0.8,
                    'categories': ['personal_info', 'sensitive_info', 'general']
                }
            }
        }
        
        self.detector = PIIDetector(self.config['privacy_detection'])
        self.dp = DifferentialPrivacy(self.config['privacy_encryption'])
        self.classifier = PIPLClassifier(self.config['compliance'])
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        text = "用户信息：张三，电话13812345678，邮箱zhangsan@example.com"
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # 步骤1: PII检测
        pii_result = self.detector.detect(text)
        self.assertGreater(len(pii_result), 0)
        
        # 步骤2: PIPL分类
        classification = self.classifier.classify(text)
        self.assertIn('category', classification)
        
        # 步骤3: 差分隐私保护
        dp_params = {
            'epsilon': 1.0,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        dp_result = self.dp.add_noise(features, dp_params)
        self.assertIn('noisy_data', dp_result)
        
        # 验证工作流程完整性
        self.assertTrue(len(pii_result) > 0)
        self.assertIsNotNone(classification)
        self.assertIsNotNone(dp_result)
    
    def test_privacy_budget_management(self):
        """测试隐私预算管理"""
        features = np.array([1.0, 2.0, 3.0])
        dp_params = {
            'epsilon': 1.0,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        
        # 多次使用隐私预算
        for i in range(3):
            result = self.dp.add_noise(features, dp_params)
            self.assertGreater(result['privacy_budget_remaining'], 0)
        
        # 验证预算正确消耗
        final_result = self.dp.add_noise(features, dp_params)
        self.assertLess(final_result['privacy_budget_remaining'], 10.0)
```

## 3. 性能测试

### 3.1 性能基准测试

```python
# test_performance.py
import unittest
import time
import numpy as np
from test_algorithms.privacy_detection.pii_detector import PIIDetector
from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy

class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'privacy_detection': {
                'detection_methods': {
                    'regex_patterns': ['phone', 'email'],
                    'entity_types': ['PERSON', 'PHONE', 'EMAIL']
                }
            },
            'privacy_encryption': {
                'differential_privacy': {
                    'general': {
                        'epsilon': 1.2,
                        'delta': 0.00001,
                        'clipping_norm': 1.0
                    }
                }
            }
        }
        
        self.detector = PIIDetector(self.config['privacy_detection'])
        self.dp = DifferentialPrivacy(self.config['privacy_encryption'])
    
    def test_pii_detection_performance(self):
        """测试PII检测性能"""
        test_texts = [
            "用户电话：13812345678",
            "邮箱：zhangsan@example.com",
            "普通文本内容"
        ] * 100  # 重复100次
        
        start_time = time.time()
        
        for text in test_texts:
            result = self.detector.detect(text)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证性能要求（每秒至少处理10个文本）
        texts_per_second = len(test_texts) / total_time
        self.assertGreater(texts_per_second, 10)
    
    def test_differential_privacy_performance(self):
        """测试差分隐私性能"""
        test_features = [np.random.rand(100) for _ in range(50)]
        
        dp_params = {
            'epsilon': 1.0,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        
        start_time = time.time()
        
        for features in test_features:
            result = self.dp.add_noise(features, dp_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证性能要求（每秒至少处理20个特征向量）
        features_per_second = len(test_features) / total_time
        self.assertGreater(features_per_second, 20)
```

## 4. 测试运行

### 4.1 运行所有测试

```bash
# 运行单元测试
python -m pytest test_pii_detector_unit.py -v
python -m pytest test_differential_privacy_unit.py -v
python -m pytest test_pipl_classifier_unit.py -v

# 运行集成测试
python -m pytest test_integration.py -v

# 运行性能测试
python -m pytest test_performance.py -v

# 运行所有测试
python -m pytest . -v --tb=short
```

### 4.2 测试覆盖率

```bash
# 安装覆盖率工具
pip install pytest-cov

# 运行测试并生成覆盖率报告
python -m pytest . --cov=test_algorithms --cov-report=html --cov-report=term
```

### 4.3 持续集成

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        python -m pytest . --cov=test_algorithms --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```


---

## FUNCTIONAL_TESTING_GUIDE.md

# PIPL隐私保护LLM框架 - 功能测试指南

## 🧪 测试概述

本指南提供PIPL隐私保护LLM框架的完整功能测试方案，包括快速测试和全面测试两种模式。

### 测试目标

- 验证PIPL隐私保护功能是否正常工作
- 确保PII检测、差分隐私、合规监控等核心功能正常
- 验证端到端工作流程的完整性
- 评估系统性能和稳定性
- 确保错误处理机制有效

## 📋 测试文件

### 1. 快速功能测试
- **文件**: `quick_functional_test.py`
- **用途**: 快速验证核心功能是否正常
- **耗时**: 约2-3分钟
- **适用场景**: 日常验证、快速检查

### 2. 完整功能测试
- **文件**: `comprehensive_functional_test.py`
- **用途**: 全面的功能测试和性能评估
- **耗时**: 约10-15分钟
- **适用场景**: 正式测试、性能评估

## 🚀 快速开始

### 步骤1: 准备环境

确保您已经在Colab中完成了以下步骤：

1. **加载Qwen2.5-7B模型** (通过Unsloth)
2. **运行PIPL集成代码**:
   ```python
   exec(open('colab_pipl_integration.py').read())
   ```

### 步骤2: 运行快速测试

```python
# 运行快速功能测试
exec(open('quick_functional_test.py').read())
```

### 步骤3: 查看测试结果

测试完成后，您将看到：
- 各模块测试结果
- 总体成功率
- 性能指标
- 测试报告文件

## 📊 测试内容

### 1. PII检测功能测试

**测试目标**: 验证PII检测器能否正确识别敏感信息

**测试案例**:
- 包含多种PII信息的文本
- 身份证号码检测
- 无PII信息的文本
- 中文姓名检测

**预期结果**: 能够准确识别电话、邮箱、身份证、姓名等敏感信息

### 2. 差分隐私功能测试

**测试目标**: 验证差分隐私模块能否正确添加噪声

**测试案例**:
- 基础差分隐私测试
- 高隐私保护测试
- 低隐私保护测试

**预期结果**: 能够根据epsilon参数添加适当的噪声

### 3. 合规性监控测试

**测试目标**: 验证合规性监控功能是否正常

**测试案例**:
- 低风险合规测试
- 高风险合规测试
- 跨境传输测试
- 操作记录测试

**预期结果**: 能够正确评估合规状态并记录操作

### 4. 端到端工作流程测试

**测试目标**: 验证完整的隐私保护工作流程

**测试案例**:
- 普通文本处理
- 包含PII的文本处理
- 包含姓名的文本处理
- 无敏感信息文本处理

**预期结果**: 能够完成从输入到输出的完整处理流程

### 5. 性能基准测试

**测试目标**: 评估系统性能

**测试指标**:
- 平均响应时间
- 响应时间标准差
- 最快/最慢响应时间

**预期结果**: 响应时间在合理范围内

### 6. 错误处理测试

**测试目标**: 验证错误处理机制

**测试案例**:
- 空输入测试
- 空字符串测试
- 超长文本测试

**预期结果**: 能够正确处理各种异常情况

### 7. 批量处理测试

**测试目标**: 验证批量处理功能

**测试案例**:
- 多个文本的批量处理
- 不同风险级别的文本混合

**预期结果**: 能够正确处理批量文本

## 📈 测试结果解读

### 成功标准

- **总体成功率**: ≥ 80%
- **PII检测**: ≥ 90%
- **差分隐私**: ≥ 80%
- **合规监控**: ≥ 90%
- **端到端流程**: ≥ 80%
- **错误处理**: ≥ 80%

### 性能指标

- **平均响应时间**: < 5秒
- **响应时间标准差**: < 2秒
- **内存使用**: 合理范围内

### 测试报告

测试完成后会生成以下报告文件：

1. **quick_test_report.json** - 快速测试报告
2. **comprehensive_test_report.json** - 完整测试报告

报告包含：
- 测试时间
- 各模块测试结果
- 性能指标
- 错误日志
- 总体统计

## 🔧 故障排除

### 常见问题

#### 1. 模型未加载
```
❌ 请先运行PIPL集成代码创建privacy_qwen对象
```
**解决方案**: 先运行 `exec(open('colab_pipl_integration.py').read())`

#### 2. 内存不足
```
❌ CUDA out of memory
```
**解决方案**: 
- 清理GPU缓存: `torch.cuda.empty_cache()`
- 减少批处理大小
- 使用更小的模型

#### 3. 网络连接问题
```
❌ 网络连接失败
```
**解决方案**: 
- 检查网络连接
- 重试测试
- 使用本地模型

#### 4. 依赖包缺失
```
❌ ModuleNotFoundError
```
**解决方案**: 
- 安装缺失的包
- 检查Python环境

### 调试技巧

1. **查看详细错误信息**:
   ```python
   import traceback
   traceback.print_exc()
   ```

2. **检查模型状态**:
   ```python
   print(f"模型设备: {next(model.parameters()).device}")
   print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
   ```

3. **监控内存使用**:
   ```python
   import psutil
   print(f"内存使用: {psutil.virtual_memory().percent}%")
   ```

## 📚 高级测试

### 自定义测试用例

```python
# 添加自定义测试用例
def custom_test_case(privacy_qwen):
    """自定义测试用例"""
    test_text = "您的自定义测试文本"
    result = privacy_qwen.generate_with_privacy_protection(test_text)
    print(f"测试结果: {result}")
    return result
```

### 性能压力测试

```python
# 性能压力测试
def stress_test(privacy_qwen, iterations=100):
    """性能压力测试"""
    for i in range(iterations):
        result = privacy_qwen.generate_with_privacy_protection(f"测试文本 {i}")
        if i % 10 == 0:
            print(f"已完成 {i} 次测试")
```

### 长期稳定性测试

```python
# 长期稳定性测试
def stability_test(privacy_qwen, duration_hours=1):
    """长期稳定性测试"""
    import time
    start_time = time.time()
    end_time = start_time + duration_hours * 3600
    
    while time.time() < end_time:
        result = privacy_qwen.generate_with_privacy_protection("稳定性测试")
        time.sleep(60)  # 每分钟测试一次
```

## 🎯 测试最佳实践

### 1. 测试环境准备
- 确保Colab环境稳定
- 检查GPU内存使用情况
- 验证网络连接

### 2. 测试执行顺序
1. 先运行快速测试验证基本功能
2. 再运行完整测试进行深入验证
3. 根据需要进行自定义测试

### 3. 结果分析
- 关注成功率指标
- 分析性能瓶颈
- 检查错误日志

### 4. 问题修复
- 根据测试结果调整配置
- 优化性能参数
- 修复发现的问题

## 📞 技术支持

如果您在测试过程中遇到问题，请：

1. 查看测试报告中的错误信息
2. 检查系统日志
3. 参考故障排除指南
4. 联系技术支持团队

---

**测试完成时间**: 2025-10-23  
**测试版本**: PIPL Framework v1.0.0  
**测试环境**: Google Colab + Unsloth + Qwen2.5-7B


---
