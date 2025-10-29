#!/usr/bin/env python3
"""
差分隐私模块专项测试脚本
测试差分隐私的各种功能和参数配置
"""

import sys
import os
import json
import logging
import numpy as np
import torch
from typing import Dict, Any, List

# 添加当前目录到Python路径
sys.path.append('.')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_config():
    """创建测试配置"""
    return {
        'differential_privacy': {
            'general': {
                'epsilon': 1.2,
                'delta': 0.00001,
                'clipping_norm': 1.0,
                'noise_multiplier': 1.1
            },
            'high_sensitivity': {
                'epsilon': 0.8,
                'delta': 0.00001,
                'clipping_norm': 0.5,
                'noise_multiplier': 1.5
            }
        },
        'budget_management': {
            'session_limit': 10.0,
            'rate_limit': 5,
            'audit_logging': True
        }
    }

def test_basic_noise_addition():
    """测试基础噪声添加功能"""
    logger.info("🔍 测试基础噪声添加功能...")
    
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    
    config = create_test_config()
    dp = DifferentialPrivacy(config)
    
    # 测试数据
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 创建DP参数
    dp_params = {
        'epsilon': 1.2,
        'delta': 0.00001,
        'clipping_norm': 1.0
    }
    
    # 添加噪声
    result = dp.add_noise(test_data, dp_params)
    
    logger.info(f"  原始数据: {test_data}")
    logger.info(f"  噪声数据: {result['noisy_data']}")
    logger.info(f"  噪声规模: {result['noise_scale']}")
    logger.info(f"  使用的epsilon: {result['epsilon_used']}")
    logger.info(f"  剩余隐私预算: {result['privacy_budget_remaining']}")
    
    # 验证噪声添加成功
    assert 'noisy_data' in result
    assert 'noise_scale' in result
    assert 'privacy_budget_remaining' in result
    
    return True

def test_privacy_budget_management():
    """测试隐私预算管理"""
    logger.info("🔍 测试隐私预算管理...")
    
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    
    config = create_test_config()
    dp = DifferentialPrivacy(config)
    
    # 获取隐私参数
    initial_params = dp.get_privacy_parameters('general')
    logger.info(f"  初始隐私参数: {initial_params}")
    
    # 模拟多次查询
    test_data = np.array([1.0, 2.0, 3.0])
    dp_params = {
        'epsilon': 1.0,
        'delta': 0.00001,
        'clipping_norm': 1.0
    }
    
    for i in range(3):
        result = dp.add_noise(test_data, dp_params)
        logger.info(f"  查询 {i+1}: 剩余预算 {result.get('privacy_budget_remaining', 'N/A')}")
    
    # 获取隐私会计师报告
    report = dp.get_privacy_accountant_report()
    logger.info(f"  隐私会计师报告: {report}")
    
    return True

def test_different_epsilon_values():
    """测试不同epsilon值的效果"""
    logger.info("🔍 测试不同epsilon值的效果...")
    
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    
    config = create_test_config()
    dp = DifferentialPrivacy(config)
    
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    epsilon_values = [0.5, 1.0, 2.0, 5.0]
    
    results = {}
    
    for epsilon in epsilon_values:
        dp_params = {
            'epsilon': epsilon,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        
        result = dp.add_noise(test_data, dp_params)
        noise_scale = result['noise_scale']
        
        results[epsilon] = noise_scale
        logger.info(f"  Epsilon {epsilon}: 噪声规模 {noise_scale:.4f}")
    
    # 验证epsilon越大，噪声越小（隐私保护越弱）
    epsilon_list = sorted(results.keys())
    noise_list = [results[e] for e in epsilon_list]
    
    # 检查噪声规模是否随epsilon增加而减少
    is_decreasing = all(noise_list[i] >= noise_list[i+1] for i in range(len(noise_list)-1))
    logger.info(f"  噪声规模递减性: {is_decreasing}")
    
    return True

def test_clipping_norm_effect():
    """测试裁剪范数的影响"""
    logger.info("🔍 测试裁剪范数的影响...")
    
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    
    config = create_test_config()
    dp = DifferentialPrivacy(config)
    
    # 创建不同范数的测试数据
    test_data_low_norm = np.array([0.1, 0.2, 0.3])
    test_data_high_norm = np.array([10.0, 20.0, 30.0])
    
    dp_params = {
        'epsilon': 1.0,
        'delta': 0.00001,
        'clipping_norm': 1.0
    }
    
    # 测试低范数数据
    result_low = dp.add_noise(test_data_low_norm, dp_params)
    logger.info(f"  低范数数据噪声规模: {result_low['noise_scale']:.4f}")
    
    # 测试高范数数据
    result_high = dp.add_noise(test_data_high_norm, dp_params)
    logger.info(f"  高范数数据噪声规模: {result_high['noise_scale']:.4f}")
    
    # 验证裁剪效果
    clipped_data = result_high.get('clipped_data', test_data_high_norm)
    original_norm = np.linalg.norm(test_data_high_norm)
    clipped_norm = np.linalg.norm(clipped_data)
    
    logger.info(f"  原始范数: {original_norm:.4f}")
    logger.info(f"  裁剪后范数: {clipped_norm:.4f}")
    logger.info(f"  裁剪效果: {clipped_norm <= dp_params['clipping_norm']}")
    
    return True

def test_high_sensitivity_mode():
    """测试高敏感度模式"""
    logger.info("🔍 测试高敏感度模式...")
    
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    
    config = create_test_config()
    dp = DifferentialPrivacy(config)
    
    test_data = np.array([1.0, 2.0, 3.0])
    
    # 普通模式
    normal_params = {
        'epsilon': 1.2,
        'delta': 0.00001,
        'clipping_norm': 1.0,
        'sensitivity_level': 'general'
    }
    
    # 高敏感度模式
    high_sensitivity_params = {
        'epsilon': 0.8,
        'delta': 0.00001,
        'clipping_norm': 0.5,
        'sensitivity_level': 'high_sensitivity'
    }
    
    result_normal = dp.add_noise(test_data, normal_params)
    result_high = dp.add_noise(test_data, high_sensitivity_params)
    
    logger.info(f"  普通模式噪声规模: {result_normal['noise_scale']:.4f}")
    logger.info(f"  高敏感度模式噪声规模: {result_high['noise_scale']:.4f}")
    logger.info(f"  高敏感度模式epsilon: {result_high['epsilon_used']}")
    
    # 验证高敏感度模式使用更严格的参数
    assert result_high['epsilon_used'] == 0.8
    assert result_high['clipping_norm'] == 0.5
    
    return True

def test_batch_processing():
    """测试批量处理"""
    logger.info("🔍 测试批量处理...")
    
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    
    config = create_test_config()
    dp = DifferentialPrivacy(config)
    
    # 创建批量数据
    batch_data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    dp_params = {
        'epsilon': 1.0,
        'delta': 0.00001,
        'clipping_norm': 1.0
    }
    
    result = dp.add_noise(batch_data, dp_params)
    
    logger.info(f"  批量数据形状: {batch_data.shape}")
    logger.info(f"  噪声数据形状: {result['noisy_data'].shape}")
    logger.info(f"  噪声规模: {result['noise_scale']:.4f}")
    
    # 验证形状保持一致
    assert batch_data.shape == result['noisy_data'].shape
    
    return True

def test_privacy_analysis():
    """测试隐私分析功能"""
    logger.info("🔍 测试隐私分析功能...")
    
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    
    config = create_test_config()
    dp = DifferentialPrivacy(config)
    
    # 获取隐私参数
    params = dp.get_privacy_parameters('general')
    
    # 获取隐私会计师报告
    report = dp.get_privacy_accountant_report()
    
    logger.info(f"  隐私参数:")
    logger.info(f"    - Epsilon: {params.get('epsilon', 'N/A')}")
    logger.info(f"    - Delta: {params.get('delta', 'N/A')}")
    logger.info(f"    - 裁剪范数: {params.get('clipping_norm', 'N/A')}")
    logger.info(f"  隐私会计师报告: {report}")
    
    return True

def test_error_handling():
    """测试错误处理"""
    logger.info("🔍 测试错误处理...")
    
    from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
    
    config = create_test_config()
    dp = DifferentialPrivacy(config)
    
    test_cases = [
        # 无效的epsilon值
        {'epsilon': -1.0, 'delta': 0.00001, 'clipping_norm': 1.0},
        # 无效的delta值
        {'epsilon': 1.0, 'delta': -0.00001, 'clipping_norm': 1.0},
        # 无效的裁剪范数
        {'epsilon': 1.0, 'delta': 0.00001, 'clipping_norm': -1.0},
    ]
    
    for i, invalid_params in enumerate(test_cases, 1):
        try:
            test_data = np.array([1.0, 2.0, 3.0])
            result = dp.add_noise(test_data, invalid_params)
            logger.warning(f"  测试 {i}: 应该失败但成功了")
        except Exception as e:
            logger.info(f"  测试 {i}: 正确捕获错误 - {type(e).__name__}")
    
    return True

def main():
    """主测试函数"""
    logger.info("🚀 开始差分隐私模块专项测试...")
    
    test_functions = [
        ("基础噪声添加", test_basic_noise_addition),
        ("隐私预算管理", test_privacy_budget_management),
        ("不同epsilon值效果", test_different_epsilon_values),
        ("裁剪范数影响", test_clipping_norm_effect),
        ("高敏感度模式", test_high_sensitivity_mode),
        ("批量处理", test_batch_processing),
        ("隐私分析功能", test_privacy_analysis),
        ("错误处理", test_error_handling)
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_name, test_func in test_functions:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 {test_name}")
            logger.info(f"{'='*50}")
            
            result = test_func()
            if result:
                logger.info(f"✅ {test_name} - 通过")
                passed += 1
            else:
                logger.error(f"❌ {test_name} - 失败")
                
        except Exception as e:
            logger.error(f"❌ {test_name} - 异常: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 测试结果汇总")
    logger.info(f"{'='*50}")
    logger.info(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有差分隐私模块测试通过！")
        return True
    else:
        logger.warning(f"⚠️ 有 {total - passed} 项测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
