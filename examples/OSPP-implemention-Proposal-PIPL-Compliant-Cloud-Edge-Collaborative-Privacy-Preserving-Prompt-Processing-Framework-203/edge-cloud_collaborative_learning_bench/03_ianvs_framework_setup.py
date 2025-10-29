#!/usr/bin/env python3
"""
阶段3: Ianvs框架设置

配置和初始化Ianvs框架，设置算法注册、测试环境、基准测试配置
"""

import os
import sys
import yaml
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_ianvs_environment():
    """设置Ianvs环境"""
    print("🏗️ 设置Ianvs环境...")
    
    # 设置Ianvs环境变量
    os.environ['IANVS_HOME'] = '/content/ianvs_pipl_framework'
    os.environ['PYTHONPATH'] = '/content/ianvs_pipl_framework:/content/ianvs'
    
    # 创建Ianvs工作目录
    ianvs_dirs = [
        'algorithms',
        'testenvs', 
        'datasets',
        'results',
        'logs'
    ]
    
    for directory in ianvs_dirs:
        os.makedirs(f'/content/ianvs_pipl_framework/{directory}', exist_ok=True)
        print(f"创建目录: {directory}")
    
    print("✅ Ianvs环境设置完成")
    return True

def create_algorithm_config():
    """创建算法配置"""
    print("\n🔧 创建算法配置...")
    
    algorithm_config = {
        "algorithm": {
            "name": "privacy_preserving_llm",
            "type": "jointinference",
            "modules": {
                "edge_model": {
                    "name": "Qwen2.5-7B-Edge",
                    "type": "edge_model",
                    "quantization": "4bit",
                    "optimization": "unsloth"
                },
                "cloud_model": {
                    "name": "Qwen2.5-7B-Cloud", 
                    "type": "cloud_model",
                    "quantization": "8bit",
                    "optimization": "unsloth"
                },
                "privacy_detection": {
                    "name": "PIIDetector",
                    "type": "privacy_detection",
                    "config": {
                        "detection_methods": ["regex", "ner", "spacy"],
                        "risk_levels": ["high", "medium", "low"]
                    }
                },
                "privacy_encryption": {
                    "name": "DifferentialPrivacy",
                    "type": "privacy_encryption",
                    "config": {
                        "epsilon": 1.2,
                        "delta": 0.00001,
                        "clipping_norm": 1.0
                    }
                },
                "compliance_monitor": {
                    "name": "ComplianceMonitor",
                    "type": "compliance_monitor",
                    "config": {
                        "pipl_compliance": True,
                        "cross_border_check": True
                    }
                }
            },
            "hyperparameters": {
                "privacy_budget": 1.2,
                "epsilon": 1.2,
                "delta": 0.00001,
                "clipping_norm": 1.0,
                "batch_size": 8,
                "max_length": 512
            }
        }
    }
    
    # 保存算法配置
    algorithm_file = '/content/ianvs_pipl_framework/algorithms/algorithm.yaml'
    with open(algorithm_file, 'w', encoding='utf-8') as f:
        yaml.dump(algorithm_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"算法配置文件: {algorithm_file}")
    return algorithm_file

def create_testenv_config():
    """创建测试环境配置"""
    print("\n🧪 创建测试环境配置...")
    
    testenv_config = {
        "testenv": {
            "name": "pipl_privacy_protection_testenv",
            "dataset": {
                "name": "ChnSentiCorp-Lite",
                "format": "jsonl",
                "path": "/content/ianvs_pipl_framework/data/processed",
                "splits": {
                    "train": "chnsenticorp_lite_train.jsonl",
                    "val": "chnsenticorp_lite_val.jsonl", 
                    "test": "chnsenticorp_lite_test.jsonl"
                }
            },
            "metrics": {
                "performance_metrics": [
                    "accuracy",
                    "throughput", 
                    "latency",
                    "cpu_usage",
                    "memory_usage",
                    "gpu_usage"
                ],
                "privacy_metrics": [
                    "pii_detection_rate",
                    "privacy_protection_rate",
                    "privacy_budget_usage",
                    "compliance_violations"
                ],
                "compliance_metrics": [
                    "pipl_compliance_rate",
                    "cross_border_violations",
                    "total_violations"
                ]
            },
            "evaluation": {
                "test_cases": 6,
                "batch_size": 8,
                "max_length": 512,
                "device": "cuda"
            }
        }
    }
    
    # 保存测试环境配置
    testenv_file = '/content/ianvs_pipl_framework/testenvs/testenv.yaml'
    with open(testenv_file, 'w', encoding='utf-8') as f:
        yaml.dump(testenv_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"测试环境配置文件: {testenv_file}")
    return testenv_file

def create_benchmarking_config():
    """创建基准测试配置"""
    print("\n📊 创建基准测试配置...")
    
    benchmarking_config = {
        "benchmarking": {
            "name": "pipl_privacy_protection_benchmark",
            "version": "1.0.0",
            "description": "PIPL隐私保护云边协同提示处理框架基准测试",
            "algorithm": {
                "name": "privacy_preserving_llm",
                "type": "jointinference",
                "config_file": "/content/ianvs_pipl_framework/algorithms/algorithm.yaml"
            },
            "testenv": {
                "name": "pipl_privacy_protection_testenv",
                "config_file": "/content/ianvs_pipl_framework/testenvs/testenv.yaml"
            },
            "output": {
                "result_dir": "/content/ianvs_pipl_framework/results",
                "log_dir": "/content/ianvs_pipl_framework/logs"
            },
            "storymanager": {
                "rank": {
                    "sort_by": ["accuracy", "privacy_score", "compliance_rate"],
                    "visualization": {
                        "mode": "selected_only",
                        "method": "print_table"
                    },
                    "save_mode": "selected_and_all_and_picture"
                }
            }
        }
    }
    
    # 保存基准测试配置
    benchmarking_file = '/content/ianvs_pipl_framework/benchmarkingjob.yaml'
    with open(benchmarking_file, 'w', encoding='utf-8') as f:
        yaml.dump(benchmarking_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"基准测试配置文件: {benchmarking_file}")
    return benchmarking_file

def register_algorithms():
    """注册算法模块"""
    print("\n📝 注册算法模块...")
    
    # 创建算法注册脚本
    registration_script = '''
import sys
import os
sys.path.append('/content/ianvs_pipl_framework')

# 导入Ianvs核心模块
try:
    from core.cmd.obj.benchmarkingjob import BenchmarkingJob
    from core.storymanager.rank.rank import Rank
    from core.storymanager.visualization.visualization import print_table
    print("✅ Ianvs核心模块导入成功")
except ImportError as e:
    print(f"⚠️ Ianvs核心模块导入失败: {e}")
    print("将使用模拟的Ianvs功能")

# 注册隐私保护算法
def register_privacy_algorithms():
    """注册隐私保护算法"""
    algorithms = {
        "PIIDetector": {
            "class": "test_algorithms.privacy_detection.pii_detector.PIIDetector",
            "type": "privacy_detection"
        },
        "DifferentialPrivacy": {
            "class": "test_algorithms.privacy_encryption.differential_privacy.DifferentialPrivacy", 
            "type": "privacy_encryption"
        },
        "ComplianceMonitor": {
            "class": "test_algorithms.privacy_encryption.compliance_monitor.ComplianceMonitor",
            "type": "compliance_monitor"
        },
        "PrivacyPreservingLLM": {
            "class": "test_algorithms.privacy_preserving_llm.privacy_preserving_llm.PrivacyPreservingLLM",
            "type": "privacy_preserving_llm"
        }
    }
    
    print("注册的算法:")
    for name, info in algorithms.items():
        print(f"  {name}: {info['class']}")
    
    return algorithms

if __name__ == "__main__":
    algorithms = register_privacy_algorithms()
    print(f"✅ 成功注册 {len(algorithms)} 个算法")
'''
    
    # 保存注册脚本
    registration_file = '/content/ianvs_pipl_framework/scripts/register_algorithms.py'
    os.makedirs('/content/ianvs_pipl_framework/scripts', exist_ok=True)
    
    with open(registration_file, 'w', encoding='utf-8') as f:
        f.write(registration_script)
    
    print(f"算法注册脚本: {registration_file}")
    return registration_file

def create_dataset_config():
    """创建数据集配置"""
    print("\n📁 创建数据集配置...")
    
    dataset_config = {
        "dataset": {
            "name": "ChnSentiCorp-Lite",
            "description": "中文情感分析数据集（轻量版）",
            "format": "jsonl",
            "path": "/content/ianvs_pipl_framework/data/processed",
            "splits": {
                "train": {
                    "file": "chnsenticorp_lite_train.jsonl",
                    "samples": 1000,
                    "description": "训练集"
                },
                "val": {
                    "file": "chnsenticorp_lite_val.jsonl", 
                    "samples": 200,
                    "description": "验证集"
                },
                "test": {
                    "file": "chnsenticorp_lite_test.jsonl",
                    "samples": 200,
                    "description": "测试集"
                }
            },
            "schema": {
                "sample_id": "string",
                "text": "string", 
                "label": "string",
                "privacy_level": "string",
                "pii_entities": "list",
                "pipl_cross_border": "boolean",
                "synthetic_pii": "object",
                "privacy_budget_cost": "float",
                "metadata": "object"
            },
            "statistics": {
                "total_samples": 1400,
                "train_samples": 1000,
                "val_samples": 200,
                "test_samples": 200,
                "avg_text_length": 50,
                "label_distribution": {
                    "positive": 0.5,
                    "negative": 0.5
                }
            }
        }
    }
    
    # 保存数据集配置
    dataset_file = '/content/ianvs_pipl_framework/datasets/dataset_config.yaml'
    with open(dataset_file, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"数据集配置文件: {dataset_file}")
    return dataset_file

def create_metrics_config():
    """创建指标配置"""
    print("\n📈 创建指标配置...")
    
    metrics_config = {
        "metrics": {
            "performance_metrics": {
                "accuracy": {
                    "name": "准确率",
                    "description": "模型预测准确率",
                    "target": ">0.85",
                    "weight": 0.3
                },
                "throughput": {
                    "name": "吞吐量",
                    "description": "系统处理能力（样本/秒）",
                    "target": ">100",
                    "weight": 0.2
                },
                "latency": {
                    "name": "延迟",
                    "description": "响应时间（秒）",
                    "target": "<0.5",
                    "weight": 0.2
                },
                "cpu_usage": {
                    "name": "CPU使用率",
                    "description": "CPU使用率（%）",
                    "target": "<80",
                    "weight": 0.1
                },
                "memory_usage": {
                    "name": "内存使用率",
                    "description": "内存使用率（%）",
                    "target": "<80",
                    "weight": 0.1
                },
                "gpu_usage": {
                    "name": "GPU使用率",
                    "description": "GPU使用率（%）",
                    "target": "<90",
                    "weight": 0.1
                }
            },
            "privacy_metrics": {
                "pii_detection_rate": {
                    "name": "PII检测率",
                    "description": "个人身份信息检测准确率",
                    "target": ">0.90",
                    "weight": 0.3
                },
                "privacy_protection_rate": {
                    "name": "隐私保护率",
                    "description": "隐私保护措施覆盖率",
                    "target": ">0.85",
                    "weight": 0.3
                },
                "privacy_budget_usage": {
                    "name": "隐私预算使用率",
                    "description": "差分隐私预算使用情况",
                    "target": "<0.9",
                    "weight": 0.2
                },
                "compliance_violations": {
                    "name": "合规违规数",
                    "description": "违反隐私法规的次数",
                    "target": "=0",
                    "weight": 0.2
                }
            },
            "compliance_metrics": {
                "pipl_compliance_rate": {
                    "name": "PIPL合规率",
                    "description": "个人信息保护法合规率",
                    "target": "=1.0",
                    "weight": 0.4
                },
                "cross_border_violations": {
                    "name": "跨境违规数",
                    "description": "跨境数据传输违规次数",
                    "target": "=0",
                    "weight": 0.3
                },
                "total_violations": {
                    "name": "总违规数",
                    "description": "所有合规性违规总数",
                    "target": "=0",
                    "weight": 0.3
                }
            }
        }
    }
    
    # 保存指标配置
    metrics_file = '/content/ianvs_pipl_framework/metrics_config.yaml'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        yaml.dump(metrics_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"指标配置文件: {metrics_file}")
    return metrics_file

def test_ianvs_setup():
    """测试Ianvs设置"""
    print("\n🧪 测试Ianvs设置...")
    
    try:
        # 测试配置文件
        config_files = [
            '/content/ianvs_pipl_framework/algorithms/algorithm.yaml',
            '/content/ianvs_pipl_framework/testenvs/testenv.yaml',
            '/content/ianvs_pipl_framework/benchmarkingjob.yaml',
            '/content/ianvs_pipl_framework/datasets/dataset_config.yaml',
            '/content/ianvs_pipl_framework/metrics_config.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✅ {config_file}")
            else:
                print(f"❌ {config_file}")
                return False
        
        # 测试目录结构
        required_dirs = [
            '/content/ianvs_pipl_framework/algorithms',
            '/content/ianvs_pipl_framework/testenvs',
            '/content/ianvs_pipl_framework/datasets',
            '/content/ianvs_pipl_framework/results',
            '/content/ianvs_pipl_framework/logs'
        ]
        
        for directory in required_dirs:
            if os.path.exists(directory):
                print(f"✅ 目录存在: {directory}")
            else:
                print(f"❌ 目录不存在: {directory}")
                return False
        
        print("✅ Ianvs设置测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Ianvs设置测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 阶段3: Ianvs框架设置")
    print("=" * 50)
    
    try:
        # 1. 设置Ianvs环境
        if not setup_ianvs_environment():
            return False
        
        # 2. 创建算法配置
        algorithm_file = create_algorithm_config()
        
        # 3. 创建测试环境配置
        testenv_file = create_testenv_config()
        
        # 4. 创建基准测试配置
        benchmarking_file = create_benchmarking_config()
        
        # 5. 注册算法模块
        registration_file = register_algorithms()
        
        # 6. 创建数据集配置
        dataset_file = create_dataset_config()
        
        # 7. 创建指标配置
        metrics_file = create_metrics_config()
        
        # 8. 测试Ianvs设置
        if not test_ianvs_setup():
            return False
        
        # 9. 保存设置报告
        setup_report = {
            "timestamp": datetime.now().isoformat(),
            "algorithm_config": algorithm_file,
            "testenv_config": testenv_file,
            "benchmarking_config": benchmarking_file,
            "registration_script": registration_file,
            "dataset_config": dataset_file,
            "metrics_config": metrics_file,
            "status": "success"
        }
        
        report_file = '/content/ianvs_pipl_framework/logs/ianvs_setup_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(setup_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Ianvs框架设置完成！")
        print(f"算法配置: {algorithm_file}")
        print(f"测试环境配置: {testenv_file}")
        print(f"基准测试配置: {benchmarking_file}")
        print(f"数据集配置: {dataset_file}")
        print(f"指标配置: {metrics_file}")
        print(f"设置报告: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ianvs框架设置失败: {e}")
        logger.error(f"Ianvs框架设置失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 阶段3完成，可以继续执行阶段4")
    else:
        print("\n❌ 阶段3失败，请检查错误信息")
