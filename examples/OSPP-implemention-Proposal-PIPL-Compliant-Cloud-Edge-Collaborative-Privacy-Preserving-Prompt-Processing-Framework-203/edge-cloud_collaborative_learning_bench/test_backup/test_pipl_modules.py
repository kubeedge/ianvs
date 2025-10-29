#!/usr/bin/env python3
"""
测试PIPL隐私保护LLM模块的基本功能
"""

import sys
import os
import json
import logging
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.append('.')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pii_detector():
    """测试PII检测器"""
    try:
        from test_algorithms.privacy_detection.pii_detector import PIIDetector
        
        # 创建测试配置
        config = {
            'detection_methods': {
                'regex_patterns': ['phone', 'email', 'name'],
                'entity_types': ['PERSON', 'ORG', 'LOC', 'PHONE', 'EMAIL']
            }
        }
        
        # 初始化检测器
        detector = PIIDetector(config)
        logger.info("✅ PII检测器初始化成功")
        
        # 测试文本
        test_text = "Zhang San phone is 13812345678, email is zhangsan@example.com"
        
        # 执行检测
        result = detector.detect(test_text)
        logger.info(f"✅ PII检测结果: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PII检测器测试失败: {e}")
        return False

def test_privacy_encryption():
    """测试隐私加密模块"""
    try:
        from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
        
        # 创建测试配置
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
        
        # 初始化差分隐私模块
        dp = DifferentialPrivacy(config)
        logger.info("✅ 差分隐私模块初始化成功")
        
        # 测试数据
        import numpy as np
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 创建DP参数
        dp_params = {
            'epsilon': 1.2,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        
        # 添加噪声
        noisy_data = dp.add_noise(test_data, dp_params)
        logger.info(f"✅ 差分隐私噪声添加成功，原始数据: {test_data}, 噪声数据: {noisy_data}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 隐私加密模块测试失败: {e}")
        return False

def test_privacy_preserving_llm():
    """测试隐私保护LLM主模块"""
    try:
        from test_algorithms.privacy_preserving_llm.privacy_preserving_llm import PrivacyPreservingLLM
        
        # 创建测试配置
        config = {
            'edge_model': {
                'name': 'test-model',
                'path': './models/test'
            },
            'cloud_model': {
                'name': 'gpt-4o-mini',
                'api_base': 'https://api.openai.com/v1'
            },
            'privacy_detection': {
                'detection_methods': {
                    'regex_patterns': ['phone', 'email'],
                    'entity_types': ['PERSON']
                }
            },
            'privacy_encryption': {
                'differential_privacy': {
                    'epsilon': 1.2,
                    'delta': 0.00001
                }
            }
        }
        
        # 初始化隐私保护LLM
        llm = PrivacyPreservingLLM()
        logger.info("✅ 隐私保护LLM模块初始化成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 隐私保护LLM模块测试失败: {e}")
        return False

def test_data_loading():
    """测试数据加载功能"""
    try:
        # 检查数据文件是否存在
        data_files = [
            './data/chnsenticorp_lite/train.jsonl',
            './data/chnsenticorp_lite/test.jsonl',
            './data/chnsenticorp_lite/val.jsonl'
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    logger.info(f"✅ 数据文件 {file_path} 加载成功，包含 {len(lines)} 条记录")
            else:
                logger.warning(f"⚠️ 数据文件 {file_path} 不存在")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据加载测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始测试PIPL隐私保护LLM模块...")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("数据加载", test_data_loading()))
    test_results.append(("PII检测器", test_pii_detector()))
    test_results.append(("隐私加密", test_privacy_encryption()))
    test_results.append(("隐私保护LLM", test_privacy_preserving_llm()))
    
    # 输出测试结果
    logger.info("\n📊 测试结果汇总:")
    logger.info("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！PIPL模块部署成功！")
        return True
    else:
        logger.warning(f"⚠️ 有 {total - passed} 项测试失败，请检查相关配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
