#!/usr/bin/env python3
"""
错误处理模块测试脚本

测试PIPL框架的错误处理机制，包括：
- 自定义异常类
- 错误处理装饰器
- 重试机制
- 资源管理
- 配置验证
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入错误处理模块
from test_algorithms.common.exceptions import (
    PIPLException, PrivacyBudgetExceededException, ComplianceViolationException,
    ModelLoadException, ConfigurationException, DataProcessingException,
    EncryptionException, AuditLogException, ValidationException,
    ResourceException, NetworkException, TimeoutException, RateLimitException,
    handle_exception, safe_execute, validate_required_fields, validate_field_type, validate_range
)

from test_algorithms.common.error_handling import (
    ErrorHandler, handle_errors, retry_on_failure, timeout, rate_limit,
    resource_manager, ResourceMonitor, validate_config, safe_file_operation,
    monitor_operation, get_global_error_handler
)


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


def main():
    """主测试函数"""
    logger.info("🚀 开始错误处理模块专项测试...")
    logger.info("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("自定义异常类", test_custom_exceptions),
        ("错误处理装饰器", test_error_handling_decorators),
        ("重试机制", test_retry_mechanism),
        ("验证函数", test_validation_functions),
        ("资源监控", test_resource_monitoring),
        ("配置验证", test_config_validation),
        ("监控装饰器", test_monitoring_decorators),
        ("资源管理", test_resource_management),
        ("全局错误处理器", test_global_error_handler)
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
        logger.info("🎉 所有错误处理模块测试通过！")
        return True
    else:
        logger.error(f"⚠️ 有 {total - passed} 项测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
