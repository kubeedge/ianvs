# 🚀 Ianvs StoryManager 测评结果导出指南

## 📋 概述

本指南详细介绍了如何使用Ianvs的StoryManager模块导出测评结果，包括排名、可视化、报告生成等功能。

## 🎯 StoryManager功能

### ✅ 核心功能
- **排名管理**: 自动生成算法排名和对比
- **可视化**: 生成热力图、对比图等可视化图表
- **报告导出**: 生成详细的测评报告
- **数据分析**: 性能、隐私、合规性分析
- **推荐建议**: 基于结果生成优化建议

### 🚀 支持的功能
- **多维度排名**: 按准确率、隐私分数、合规率等维度排名
- **可视化图表**: 热力图、柱状图、对比图等
- **综合报告**: 包含所有测评结果的详细报告
- **性能分析**: CPU、内存、GPU使用率分析
- **隐私分析**: PII检测率、隐私保护率分析
- **合规分析**: PIPL合规性分析

## 🛠️ 使用方法

### 方法1: 使用导出器类（推荐）

```python
from ianvs_storymanager_exporter import IanvsStoryManagerExporter

# 初始化导出器
exporter = IanvsStoryManagerExporter("/content/ianvs_pipl_framework")

# 准备数据
datasets = {...}  # 数据集信息
models = {...}    # 模型信息
workflow_results = [...]  # 工作流结果
monitoring_results = {...}  # 监控结果

# 导出所有结果
export_summary = exporter.export_all(
    datasets, models, workflow_results, monitoring_results
)
```

### 方法2: 使用Jupyter Notebook

```python
# 在Colab_Ianvs_PIPL_Framework.ipynb中运行阶段8
# 自动使用StoryManager导出测评结果
```

### 方法3: 直接运行Python脚本

```bash
python ianvs_storymanager_exporter.py
```

## 📊 导出内容

### 排名文件
- **all_rank.csv**: 所有测试用例的完整排名
- **selected_rank.csv**: 筛选后的排名结果

### 可视化文件
- **comprehensive_analysis.png**: 综合分析图表
- **performance_heatmap.png**: 性能指标热力图
- **privacy_heatmap.png**: 隐私保护指标热力图

### 报告文件
- **comprehensive_evaluation_report.json**: 综合测评报告
- **visualization_report.json**: 可视化报告
- **rankings_report.json**: 排名报告

## 🔧 配置选项

### 排名配置
```python
rank_config = {
    "sort_by": ["accuracy", "privacy_score", "compliance_rate"],
    "visualization": {
        "mode": "selected_only",
        "method": "print_table"
    },
    "selected_dataitem": {
        "paradigms": ["all"],
        "modules": ["all"],
        "hyperparameters": ["all"],
        "metrics": ["all"]
    },
    "save_mode": "selected_and_all_and_picture"
}
```

### 可视化配置
```python
visualization_config = {
    "performance_metrics": ["accuracy", "privacy_score", "compliance_rate"],
    "privacy_metrics": ["pii_detection_rate", "privacy_protection_rate"],
    "chart_types": ["heatmap", "bar_chart", "line_chart"],
    "output_format": "png"
}
```

## 📈 数据分析

### 性能指标分析
- **准确率**: 模型预测准确率
- **隐私分数**: 隐私保护效果评分
- **合规率**: PIPL合规性评分
- **吞吐量**: 系统处理能力
- **延迟**: 响应时间

### 隐私保护分析
- **PII检测率**: 个人身份信息检测准确率
- **隐私保护率**: 隐私保护措施覆盖率
- **隐私预算使用**: 差分隐私预算消耗情况
- **合规违规数**: 违反隐私法规的次数

### 合规性分析
- **PIPL合规率**: 个人信息保护法合规率
- **跨境违规数**: 跨境数据传输违规次数
- **数据本地化合规**: 数据本地化存储合规性
- **总违规数**: 所有合规性违规总数

## 🎯 输出示例

### 排名输出
```
算法名称          准确率    隐私分数    合规率    吞吐量    延迟
Qwen2.5-7B-Edge  0.95      0.92       0.98     120.5     0.15
Qwen2.5-7B-Cloud 0.93      0.89       0.96     115.2     0.18
```

### 可视化输出
- 性能指标热力图
- 隐私保护指标对比图
- 算法性能柱状图
- 隐私预算使用情况图

### 报告输出
```json
{
  "framework_info": {
    "name": "Ianvs PIPL隐私保护云边协同提示处理框架",
    "version": "1.0.0",
    "compliance": "PIPL-Compliant"
  },
  "test_summary": {
    "total_test_cases": 6,
    "successful_tests": 6,
    "failed_tests": 0
  },
  "performance_analysis": {
    "average_accuracy": 0.94,
    "average_privacy_score": 0.91,
    "average_compliance_rate": 0.97
  }
}
```

## 🚨 故障排除

### 常见问题

1. **导入错误**
   ```python
   # 解决方案：检查Ianvs核心模块路径
   import sys
   sys.path.append('/path/to/ianvs/core')
   ```

2. **文件路径错误**
   ```python
   # 解决方案：检查输出目录权限
   import os
   os.makedirs(output_dir, exist_ok=True)
   ```

3. **数据格式错误**
   ```python
   # 解决方案：验证数据格式
   def validate_data(data):
       required_fields = ['algorithm', 'metrics', 'performance']
       for field in required_fields:
           if field not in data:
               raise ValueError(f"Missing required field: {field}")
   ```

4. **可视化失败**
   ```python
   # 解决方案：检查matplotlib后端
   import matplotlib
   matplotlib.use('Agg')  # 使用非交互式后端
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用调试模式
exporter = IanvsStoryManagerExporter(debug=True)
```

## 📚 最佳实践

### 数据准备
1. **标准化格式**: 使用统一的数据格式
2. **完整性检查**: 确保所有必需字段都存在
3. **数据验证**: 验证数据的有效性和一致性
4. **版本控制**: 为数据集添加版本信息

### 性能优化
1. **批量处理**: 批量处理大量数据
2. **内存管理**: 合理使用内存资源
3. **并行处理**: 使用多进程处理
4. **缓存机制**: 缓存中间结果

### 结果展示
1. **清晰的可视化**: 使用清晰的图表和颜色
2. **详细的报告**: 包含所有重要信息
3. **交互式界面**: 提供交互式的结果展示
4. **导出格式**: 支持多种导出格式

## 🎯 高级功能

### 自定义排名
```python
# 自定义排名规则
def custom_ranking(test_results):
    def score(result):
        accuracy = result.get('metrics', {}).get('accuracy', 0)
        privacy = result.get('metrics', {}).get('privacy_score', 0)
        compliance = result.get('metrics', {}).get('compliance_rate', 0)
        return accuracy * 0.4 + privacy * 0.3 + compliance * 0.3
    
    return sorted(test_results, key=score, reverse=True)
```

### 自定义可视化
```python
# 自定义可视化函数
def custom_visualization(data, output_path):
    plt.figure(figsize=(12, 8))
    # 自定义绘图逻辑
    plt.savefig(output_path)
    plt.close()
```

### 自定义报告
```python
# 自定义报告生成
def custom_report(test_results):
    report = {
        "summary": generate_summary(test_results),
        "analysis": perform_analysis(test_results),
        "recommendations": generate_recommendations(test_results)
    }
    return report
```

## 🏆 总结

本指南提供了完整的Ianvs StoryManager测评结果导出解决方案，包括：

- ✅ **排名管理**: 多维度算法排名
- ✅ **可视化**: 丰富的图表和可视化
- ✅ **报告生成**: 详细的测评报告
- ✅ **数据分析**: 性能、隐私、合规性分析
- ✅ **推荐建议**: 基于结果的优化建议
- ✅ **故障排除**: 完善的错误处理
- ✅ **最佳实践**: 性能优化和结果展示

**🎉 现在您可以轻松使用Ianvs StoryManager导出专业的测评结果了！**
