#!/usr/bin/env python3
"""系统模块综合测试"""

import os
import sys
import json
import time
import logging

from datetime import datetime



# === test_config_management.py ===

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





# === test_error_handling.py ===

def test_custom_exceptions():
    """测试自定义异常类"""
    logger.info("🧪 测试自定义异常类")
    logger.info("=" * 50)
    
    try:
        # 测试隐私预算超限异常
        raise PrivacyBudgetExceededException(5.0, 10.0, "session_001")
    except PrivacyBudgetExceededException as e:
        logger.info(f"✅ 隐私预算超限异常: {e.message}")
        logger.info(f"   错误代码: {e.error_code}")
        logger.info(f"   详细信息: {e.details}")
    
    try:
        # 测试合规性违规异常
        raise ComplianceViolationException(
            "cross_border_transmission",
            "Data transmitted without proper encryption",
            "high",
            "PIPL"
        )
    except ComplianceViolationException as e:
        logger.info(f"✅ 合规性违规异常: {e.message}")
        logger.info(f"   违规类型: {e.details['violation_type']}")
        logger.info(f"   严重程度: {e.details['severity']}")
    
    try:
        # 测试模型加载异常
        raise ModelLoadException("meta-llama/Llama-3-8B-Instruct", "Model not found", "llm")
    except ModelLoadException as e:
        logger.info(f"✅ 模型加载异常: {e.message}")
        logger.info(f"   模型名称: {e.details['model_name']}")
        logger.info(f"   模型类型: {e.details['model_type']}")
    
    try:
        # 测试配置异常
        raise ConfigurationException("epsilon", "positive_float", -1.0, "privacy_encryption")
    except ConfigurationException as e:
        logger.info(f"✅ 配置异常: {e.message}")
        logger.info(f"   配置键: {e.details['config_key']}")
        logger.info(f"   期望类型: {e.details['expected_type']}")
    
    logger.info("✅ 自定义异常类测试通过")
    return True


def test_error_handling_decorators():
    """测试错误处理装饰器"""
    logger.info("\n🧪 测试错误处理装饰器")
    logger.info("=" * 50)
    
    # 测试handle_exception装饰器
    @handle_exception
    def test_function_success():
        return "操作成功"
    
    @handle_exception
    def test_function_failure():
        raise ValueError("测试错误")
    
    # 测试成功情况
    result = test_function_success()
    logger.info(f"✅ 成功函数测试: {result}")
    
    # 测试失败情况
    try:
        test_function_failure()
    except PIPLException as e:
        logger.info(f"✅ 失败函数测试: {e.message}")
        logger.info(f"   错误代码: {e.error_code}")
    
    # 测试safe_execute函数
    def risky_function():
        raise RuntimeError("危险操作失败")
    
    result = safe_execute(risky_function)
    logger.info(f"✅ 安全执行测试: {result}")
    
    logger.info("✅ 错误处理装饰器测试通过")
    return True


def test_retry_mechanism():
    """测试重试机制"""
    logger.info("\n🧪 测试重试机制")
    logger.info("=" * 50)
    
    # 测试重试装饰器
    call_count = 0
    
    @retry_on_failure(max_attempts=3, delay=0.1, backoff=2.0)
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise NetworkException("api.example.com", "request", "Connection timeout")
        return f"成功在第{call_count}次尝试"
    
    try:
        result = flaky_function()
        logger.info(f"✅ 重试机制测试: {result}")
        logger.info(f"   总尝试次数: {call_count}")
    except Exception as e:
        logger.error(f"❌ 重试机制测试失败: {e}")
        return False
    
    # 测试错误处理器
    error_handler = ErrorHandler({
        'max_errors_per_hour': 5,
        'circuit_breaker_threshold': 3,
        'circuit_breaker_timeout': 1
    })
    
    @handle_errors(error_handler=error_handler, operation_name="test_operation", max_retries=2)
    def failing_operation():
        raise NetworkException("test.com", "request", "Always fails")
    
    try:
        failing_operation()
    except PIPLException as e:
        logger.info(f"✅ 错误处理器测试: {e.message}")
    
    logger.info("✅ 重试机制测试通过")
    return True


def test_validation_functions():
    """测试验证函数"""
    logger.info("\n🧪 测试验证函数")
    logger.info("=" * 50)
    
    # 测试必需字段验证
    try:
        data = {'name': 'test', 'age': 25}
        validate_required_fields(data, ['name', 'age', 'email'])
    except ValidationException as e:
        logger.info(f"✅ 必需字段验证: {e.message}")
        logger.info(f"   缺失字段: {e.details['validation_rule']}")
    
    # 测试字段类型验证
    try:
        validate_field_type("not_a_number", int, "age")
    except ValidationException as e:
        logger.info(f"✅ 字段类型验证: {e.message}")
        logger.info(f"   期望类型: {e.details.get('expected_type', 'unknown')}")
        logger.info(f"   实际类型: {e.details.get('actual_type', 'unknown')}")
    
    # 测试范围验证
    try:
        validate_range(150, 0, 100, "percentage")
    except ValidationException as e:
        logger.info(f"✅ 范围验证: {e.message}")
        logger.info(f"   验证规则: {e.details['validation_rule']}")
    
    logger.info("✅ 验证函数测试通过")
    return True


def test_resource_monitoring():
    """测试资源监控"""
    logger.info("\n🧪 测试资源监控")
    logger.info("=" * 50)
    
    # 测试资源监控器
    monitor = ResourceMonitor({
        'memory_threshold': 80.0,
        'cpu_threshold': 80.0,
        'disk_threshold': 90.0
    })
    
    resources = monitor.check_system_resources()
    logger.info(f"✅ 系统资源状态: {resources['status']}")
    logger.info(f"   内存使用率: {resources.get('memory_percent', 0):.1f}%")
    logger.info(f"   CPU使用率: {resources.get('cpu_percent', 0):.1f}%")
    logger.info(f"   磁盘使用率: {resources.get('disk_percent', 0):.1f}%")
    
    if resources.get('warnings'):
        logger.info(f"   警告: {resources['warnings']}")
    
    # 测试限流检查
    should_throttle = monitor.should_throttle()
    logger.info(f"   是否需要限流: {should_throttle}")
    
    logger.info("✅ 资源监控测试通过")
    return True


def test_config_validation():
    """测试配置验证"""
    logger.info("\n🧪 测试配置验证")
    logger.info("=" * 50)
    
    # 测试有效配置
    valid_config = {
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
            'pipl_version': '2021'
        }
    }
    
    try:
        validate_config(valid_config, ['privacy_encryption', 'compliance'])
        logger.info("✅ 有效配置验证通过")
    except Exception as e:
        logger.error(f"❌ 有效配置验证失败: {e}")
        return False
    
    # 测试无效配置
    invalid_config = {
        'privacy_encryption': {
            'differential_privacy': {
                'general': {
                    'epsilon': -1.0,  # 无效值
                    'delta': 1.5,     # 无效值
                    'clipping_norm': 1.0
                }
            }
        }
    }
    
    try:
        validate_config(invalid_config, ['privacy_encryption'])
    except ConfigurationException as e:
        logger.info(f"✅ 无效配置验证: {e.message}")
        logger.info(f"   配置键: {e.details.get('config_key', 'unknown')}")
    
    logger.info("✅ 配置验证测试通过")
    return True


def test_monitoring_decorators():
    """测试监控装饰器"""
    logger.info("\n🧪 测试监控装饰器")
    logger.info("=" * 50)
    
    # 测试操作监控装饰器
    @monitor_operation("test_operation")
    def monitored_function():
        time.sleep(0.1)  # 模拟操作
        return "操作完成"
    
    result = monitored_function()
    logger.info(f"✅ 监控装饰器测试: {result}")
    
    # 测试速率限制装饰器
    @rate_limit(calls_per_second=2.0)
    def rate_limited_function():
        return f"调用时间: {time.time()}"
    
    start_time = time.time()
    for i in range(3):
        result = rate_limited_function()
        logger.info(f"   调用 {i+1}: {result}")
    
    total_time = time.time() - start_time
    logger.info(f"   总耗时: {total_time:.2f}s (应该 >= 1.0s)")
    
    logger.info("✅ 监控装饰器测试通过")
    return True


def test_resource_management():
    """测试资源管理"""
    logger.info("\n🧪 测试资源管理")
    logger.info("=" * 50)
    
    # 测试资源管理器上下文
    def cleanup_resource(resource):
        logger.info(f"清理资源: {resource}")
    
    with resource_manager("test_resource", cleanup_resource):
        logger.info("   使用资源中...")
        time.sleep(0.1)
    
    logger.info("✅ 资源管理测试通过")
    return True


def test_global_error_handler():
    """测试全局错误处理器"""
    logger.info("\n🧪 测试全局错误处理器")
    logger.info("=" * 50)
    
    # 获取全局错误处理器
    global_handler = get_global_error_handler()
    
    # 测试熔断器
    @handle_errors(error_handler=global_handler, operation_name="global_test", max_retries=1)
    def global_test_function():
        raise NetworkException("global.test.com", "request", "Test error")
    
    try:
        global_test_function()
    except PIPLException as e:
        logger.info(f"✅ 全局错误处理器测试: {e.message}")
    
    logger.info("✅ 全局错误处理器测试通过")
    return True



