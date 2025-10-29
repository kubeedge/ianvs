#!/usr/bin/env python3
"""
配置管理模块测试脚本

测试PIPL框架的配置管理功能，包括：
- 环境变量加载
- 配置文件解析
- 配置验证
- 配置合并
- 配置热更新
"""

import sys
import os
import tempfile
import logging
import time
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入配置管理模块
from test_algorithms.common.config_loader import (
    ConfigLoader, get_global_config_loader, set_global_config_loader,
    get_config, set_config
)


def test_environment_loading():
    """测试环境变量加载"""
    logger.info("🧪 测试环境变量加载")
    logger.info("=" * 50)
    
    # 设置测试环境变量
    test_env_vars = {
        'EDGE_API_KEY': 'test_edge_key_123',
        'CLOUD_API_KEY': 'test_cloud_key_456',
        'PRIVACY_BUDGET_LIMIT': '15.0',
        'DEFAULT_EPSILON': '2.0',
        'PIPL_VERSION': '2021',
        'LOG_LEVEL': 'DEBUG'
    }
    
    # 保存原始环境变量
    original_env = {}
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        
        # 验证环境变量加载
        assert config_loader.get('api_keys.edge_api_key') == 'test_edge_key_123'
        assert config_loader.get('api_keys.cloud_api_key') == 'test_cloud_key_456'
        assert config_loader.get('privacy.budget_limit') == 15.0
        assert config_loader.get('privacy.default_epsilon') == 2.0
        assert config_loader.get('compliance.pipl_version') == '2021'
        assert config_loader.get('logging.log_level') == 'DEBUG'
        
        logger.info("✅ 环境变量加载测试通过")
        logger.info(f"   Edge API Key: {config_loader.get('api_keys.edge_api_key')}")
        logger.info(f"   Privacy Budget: {config_loader.get('privacy.budget_limit')}")
        logger.info(f"   Default Epsilon: {config_loader.get('privacy.default_epsilon')}")
        
    finally:
        # 恢复原始环境变量
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
    
    return True


def test_config_file_loading():
    """测试配置文件加载"""
    logger.info("\n🧪 测试配置文件加载")
    logger.info("=" * 50)
    
    # 创建临时配置文件
    test_config = {
        'privacy': {
            'budget_limit': 20.0,
            'default_epsilon': 1.5,
            'default_delta': 0.0001
        },
        'compliance': {
            'pipl_version': '2021',
            'compliance_mode': 'strict',
            'cross_border_policy': 'encrypted'
        },
        'models': {
            'edge_model_name': 'custom/edge-model',
            'cloud_model_name': 'custom/cloud-model'
        },
        'performance': {
            'max_batch_size': 64,
            'max_sequence_length': 1024,
            'enable_quantization': True
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(test_config, f)
        temp_file = f.name
    
    try:
        # 测试YAML配置文件加载
        config_loader = ConfigLoader(config_path=temp_file)
        
        # 验证配置加载
        assert config_loader.get('privacy.budget_limit') == 20.0
        assert config_loader.get('privacy.default_epsilon') == 1.5
        assert config_loader.get('compliance.compliance_mode') == 'strict'
        assert config_loader.get('models.edge_model_name') == 'custom/edge-model'
        assert config_loader.get('performance.max_batch_size') == 64
        
        logger.info("✅ YAML配置文件加载测试通过")
        logger.info(f"   Privacy Budget: {config_loader.get('privacy.budget_limit')}")
        logger.info(f"   Compliance Mode: {config_loader.get('compliance.compliance_mode')}")
        logger.info(f"   Max Batch Size: {config_loader.get('performance.max_batch_size')}")
        
        # 测试JSON配置文件加载
        json_config = {
            'privacy': {
                'budget_limit': 25.0,
                'default_epsilon': 2.5
            },
            'logging': {
                'log_level': 'WARNING'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(json_config, f)
            json_file = f.name
        
        try:
            json_loader = ConfigLoader(config_path=json_file)
            assert json_loader.get('privacy.budget_limit') == 25.0
            assert json_loader.get('privacy.default_epsilon') == 2.5
            assert json_loader.get('logging.log_level') == 'WARNING'
            
            logger.info("✅ JSON配置文件加载测试通过")
            
        finally:
            os.unlink(json_file)
    
    finally:
        os.unlink(temp_file)
    
    return True


def test_config_validation():
    """测试配置验证"""
    logger.info("\n🧪 测试配置验证")
    logger.info("=" * 50)
    
    # 测试有效配置
    valid_config = {
        'privacy': {
            'budget_limit': 10.0,
            'default_epsilon': 1.2,
            'default_delta': 0.00001
        },
        'performance': {
            'max_batch_size': 32,
            'max_sequence_length': 512
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(valid_config, f)
        valid_file = f.name
    
    try:
        config_loader = ConfigLoader(config_path=valid_file, required_sections=['privacy', 'performance'])
        logger.info("✅ 有效配置验证通过")
        
    finally:
        os.unlink(valid_file)
    
    # 测试无效配置
    invalid_config = {
        'privacy': {
            'budget_limit': 10.0,
            'default_epsilon': -1.0,  # 无效值
            'default_delta': 1.5      # 无效值
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(invalid_config, f)
        invalid_file = f.name
    
    try:
        try:
            config_loader = ConfigLoader(config_path=invalid_file)
            logger.error("❌ 无效配置验证应该失败")
            return False
        except Exception as e:
            logger.info(f"✅ 无效配置验证: {e}")
    
    finally:
        os.unlink(invalid_file)
    
    return True


def test_config_merging():
    """测试配置合并"""
    logger.info("\n🧪 测试配置合并")
    logger.info("=" * 50)
    
    # 创建基础配置
    base_config = {
        'privacy': {
            'budget_limit': 10.0,
            'default_epsilon': 1.2
        },
        'compliance': {
            'pipl_version': '2021'
        }
    }
    
    # 创建覆盖配置
    override_config = {
        'privacy': {
            'budget_limit': 20.0,  # 覆盖
            'default_delta': 0.0001  # 新增
        },
        'models': {  # 新增节
            'edge_model_name': 'test-model'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(override_config, f)
        override_file = f.name
    
    try:
        config_loader = ConfigLoader(config_path=override_file)
        
        # 验证配置合并
        assert config_loader.get('privacy.budget_limit') == 20.0  # 被覆盖
        assert config_loader.get('privacy.default_epsilon') == 1.2  # 保持原值
        assert config_loader.get('privacy.default_delta') == 0.0001  # 新增
        assert config_loader.get('compliance.pipl_version') == '2021'  # 保持原值
        assert config_loader.get('models.edge_model_name') == 'test-model'  # 新增
        
        logger.info("✅ 配置合并测试通过")
        logger.info(f"   Privacy Budget (覆盖): {config_loader.get('privacy.budget_limit')}")
        logger.info(f"   Privacy Epsilon (保持): {config_loader.get('privacy.default_epsilon')}")
        logger.info(f"   Privacy Delta (新增): {config_loader.get('privacy.default_delta')}")
        logger.info(f"   Edge Model (新增): {config_loader.get('models.edge_model_name')}")
    
    finally:
        os.unlink(override_file)
    
    return True


def test_config_get_set():
    """测试配置获取和设置"""
    logger.info("\n🧪 测试配置获取和设置")
    logger.info("=" * 50)
    
    config_loader = ConfigLoader()
    
    # 测试获取配置
    edge_api_key = config_loader.get('api_keys.edge_api_key')
    privacy_budget = config_loader.get('privacy.budget_limit')
    log_level = config_loader.get('logging.log_level')
    
    logger.info(f"   Edge API Key: {edge_api_key}")
    logger.info(f"   Privacy Budget: {privacy_budget}")
    logger.info(f"   Log Level: {log_level}")
    
    # 测试设置配置
    config_loader.set('test.new_key', 'test_value')
    config_loader.set('privacy.budget_limit', 25.0)
    config_loader.set('logging.log_level', 'DEBUG')
    
    # 验证设置
    assert config_loader.get('test.new_key') == 'test_value'
    assert config_loader.get('privacy.budget_limit') == 25.0
    assert config_loader.get('logging.log_level') == 'DEBUG'
    
    logger.info("✅ 配置获取和设置测试通过")
    logger.info(f"   New Key: {config_loader.get('test.new_key')}")
    logger.info(f"   Updated Budget: {config_loader.get('privacy.budget_limit')}")
    logger.info(f"   Updated Log Level: {config_loader.get('logging.log_level')}")
    
    return True


def test_config_save_export():
    """测试配置保存和导出"""
    logger.info("\n🧪 测试配置保存和导出")
    logger.info("=" * 50)
    
    config_loader = ConfigLoader()
    
    # 修改一些配置
    config_loader.set('privacy.budget_limit', 30.0)
    config_loader.set('compliance.compliance_mode', 'relaxed')
    config_loader.set('performance.max_batch_size', 128)
    
    # 测试保存配置
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        save_file = f.name
    
    try:
        config_loader.save_config(save_file)
        
        # 验证保存的配置
        import yaml
        with open(save_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['privacy']['budget_limit'] == 30.0
        assert saved_config['compliance']['compliance_mode'] == 'relaxed'
        assert saved_config['performance']['max_batch_size'] == 128
        
        logger.info("✅ 配置保存测试通过")
        logger.info(f"   保存文件: {save_file}")
    
    finally:
        os.unlink(save_file)
    
    # 测试导出环境变量文件
    with tempfile.NamedTemporaryFile(suffix='.env', delete=False) as f:
        env_file = f.name
    
    try:
        config_loader.export_env_file(env_file)
        
        # 验证导出的环境变量文件
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        assert 'PRIVACY_BUDGET_LIMIT=30.0' in env_content
        assert 'COMPLIANCE_MODE=relaxed' in env_content
        assert 'MAX_BATCH_SIZE=128' in env_content
        
        logger.info("✅ 环境变量导出测试通过")
        logger.info(f"   导出文件: {env_file}")
    
    finally:
        os.unlink(env_file)
    
    return True


def test_global_config_loader():
    """测试全局配置加载器"""
    logger.info("\n🧪 测试全局配置加载器")
    logger.info("=" * 50)
    
    # 获取全局配置加载器
    global_loader = get_global_config_loader()
    
    # 测试全局配置函数
    original_budget = get_config('privacy.budget_limit')
    set_config('privacy.budget_limit', 50.0)
    
    assert get_config('privacy.budget_limit') == 50.0
    
    # 恢复原始值
    set_config('privacy.budget_limit', original_budget)
    
    logger.info("✅ 全局配置加载器测试通过")
    logger.info(f"   原始预算: {original_budget}")
    logger.info(f"   设置预算: 50.0")
    logger.info(f"   恢复预算: {get_config('privacy.budget_limit')}")
    
    return True


def test_config_summary():
    """测试配置摘要"""
    logger.info("\n🧪 测试配置摘要")
    logger.info("=" * 50)
    
    config_loader = ConfigLoader()
    summary = config_loader.get_config_summary()
    
    # 验证摘要内容
    assert 'config_path' in summary
    assert 'auto_reload' in summary
    assert 'required_sections' in summary
    assert 'sections' in summary
    assert 'last_updated' in summary
    
    logger.info("✅ 配置摘要测试通过")
    logger.info(f"   配置路径: {summary['config_path']}")
    logger.info(f"   自动重载: {summary['auto_reload']}")
    logger.info(f"   配置节: {summary['sections']}")
    logger.info(f"   最后更新: {summary['last_updated']}")
    
    return True


def main():
    """主测试函数"""
    logger.info("🚀 开始配置管理模块专项测试...")
    logger.info("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("环境变量加载", test_environment_loading),
        ("配置文件加载", test_config_file_loading),
        ("配置验证", test_config_validation),
        ("配置合并", test_config_merging),
        ("配置获取和设置", test_config_get_set),
        ("配置保存和导出", test_config_save_export),
        ("全局配置加载器", test_global_config_loader),
        ("配置摘要", test_config_summary)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name}测试失败: {e}")
            test_results.append((test_name, False))
    
    # 输出测试结果
    logger.info("\n📊 测试结果汇总")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有配置管理模块测试通过！")
        return True
    else:
        logger.error(f"⚠️ 有 {total - passed} 项测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
