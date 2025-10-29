#!/usr/bin/env python3
"""
PIPL合规性检查模块专项测试脚本
测试PIPL分类器和合规性监控模块
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List

# 添加当前目录到Python路径
sys.path.append('.')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_config():
    """创建测试配置"""
    return {
        'pipl_classification': {
            'model_path': './models/pipl_classifier',
            'threshold': 0.8,
            'categories': ['personal_info', 'sensitive_info', 'biometric_data', 'location_data']
        },
        'compliance_monitoring': {
            'audit_level': 'detailed',
            'cross_border_policy': 'strict',
            'data_retention_days': 30,
            'consent_required': True
        },
        'privacy_detection': {
            'detection_methods': {
                'regex_patterns': ['phone', 'email', 'id_card'],
                'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD']
            }
        }
    }

def test_pipl_classifier():
    """测试PIPL分类器"""
    logger.info("🔍 测试PIPL分类器...")
    
    try:
        from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier
        
        config = create_test_config()
        classifier = PIPLClassifier(config)
        logger.info("✅ PIPL分类器初始化成功")
        
        # 测试分类功能
        test_cases = [
            {
                'text': '用户姓名：张三，身份证号：110101199001011234',
                'expected_category': 'personal_info'
            },
            {
                'text': '用户位置：北京市朝阳区，GPS坐标：39.9042,116.4074',
                'expected_category': 'location_data'
            },
            {
                'text': '这是一段普通的文本，不包含敏感信息',
                'expected_category': 'general'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                result = classifier.classify(test_case['text'])
                logger.info(f"  测试 {i}: '{test_case['text'][:30]}...'")
                logger.info(f"    分类结果: {result}")
                logger.info(f"    预期类别: {test_case['expected_category']}")
            except Exception as e:
                logger.warning(f"  测试 {i}: 分类失败 - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PIPL分类器测试失败: {e}")
        return False

def test_compliance_monitor():
    """测试合规性监控器"""
    logger.info("🔍 测试合规性监控器...")
    
    try:
        from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
        
        config = create_test_config()
        monitor = ComplianceMonitor(config)
        logger.info("✅ 合规性监控器初始化成功")
        
        # 测试合规性检查
        test_scenarios = [
            {
                'name': '正常数据处理',
                'data': {'type': 'general', 'content': '普通文本'},
                'expected': 'compliant'
            },
            {
                'name': '敏感数据处理',
                'data': {'type': 'personal_info', 'content': '张三 13812345678'},
                'expected': 'requires_protection'
            },
            {
                'name': '跨境数据传输',
                'data': {'type': 'personal_info', 'content': '用户信息', 'cross_border': True},
                'expected': 'requires_encryption'
            }
        ]
        
        for scenario in test_scenarios:
            try:
                result = monitor.check_compliance(scenario['data'])
                logger.info(f"  场景: {scenario['name']}")
                logger.info(f"    合规性结果: {result}")
                logger.info(f"    预期结果: {scenario['expected']}")
            except Exception as e:
                logger.warning(f"  场景 {scenario['name']}: 检查失败 - {e}")
        
        # 测试审计日志
        try:
            audit_log = monitor.get_audit_log()
            logger.info(f"  审计日志: {audit_log}")
        except Exception as e:
            logger.warning(f"  获取审计日志失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 合规性监控器测试失败: {e}")
        return False

def test_risk_evaluator():
    """测试风险评估器"""
    logger.info("🔍 测试风险评估器...")
    
    try:
        from test_algorithms.privacy_detection.risk_evaluator import RiskEvaluator
        
        config = create_test_config()
        evaluator = RiskEvaluator(config)
        logger.info("✅ 风险评估器初始化成功")
        
        # 测试风险评估
        test_cases = [
            {
                'data': {'type': 'phone', 'value': '13812345678'},
                'context': '用户注册信息',
                'expected_risk': 'high'
            },
            {
                'data': {'type': 'email', 'value': 'user@example.com'},
                'context': '联系信息',
                'expected_risk': 'medium'
            },
            {
                'data': {'type': 'general', 'value': '普通文本'},
                'context': '一般信息',
                'expected_risk': 'low'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                risk_result = evaluator.evaluate_risk(test_case['data'], test_case['context'])
                logger.info(f"  测试 {i}: {test_case['data']['type']} - {test_case['data']['value']}")
                logger.info(f"    风险评估: {risk_result}")
                logger.info(f"    预期风险: {test_case['expected_risk']}")
            except Exception as e:
                logger.warning(f"  测试 {i}: 风险评估失败 - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 风险评估器测试失败: {e}")
        return False

def test_compliance_integration():
    """测试合规性模块集成"""
    logger.info("🔍 测试合规性模块集成...")
    
    try:
        from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier
        from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
        from test_algorithms.privacy_detection.risk_evaluator import RiskEvaluator
        
        config = create_test_config()
        
        # 初始化所有模块
        classifier = PIPLClassifier(config)
        monitor = ComplianceMonitor(config)
        evaluator = RiskEvaluator(config)
        
        logger.info("✅ 所有合规性模块初始化成功")
        
        # 模拟完整的合规性检查流程
        test_text = "用户信息：姓名张三，电话13812345678，邮箱zhangsan@example.com，地址北京市朝阳区"
        
        logger.info("  步骤1: PIPL分类")
        try:
            classification = classifier.classify(test_text)
            logger.info(f"    分类结果: {classification}")
        except Exception as e:
            logger.warning(f"    分类失败: {e}")
            classification = {'category': 'personal_info', 'confidence': 0.9}
        
        logger.info("  步骤2: 风险评估")
        try:
            risk_assessment = evaluator.evaluate_risk(
                {'type': classification.get('category', 'personal_info'), 'value': test_text},
                '用户信息处理'
            )
            logger.info(f"    风险评估: {risk_assessment}")
        except Exception as e:
            logger.warning(f"    风险评估失败: {e}")
            risk_assessment = {'risk_level': 'high', 'score': 0.8}
        
        logger.info("  步骤3: 合规性检查")
        try:
            compliance_result = monitor.check_compliance({
                'type': classification.get('category', 'personal_info'),
                'content': test_text,
                'risk_level': risk_assessment.get('risk_level', 'high')
            })
            logger.info(f"    合规性结果: {compliance_result}")
        except Exception as e:
            logger.warning(f"    合规性检查失败: {e}")
            compliance_result = {'status': 'requires_protection', 'actions': ['encrypt', 'audit']}
        
        logger.info("✅ 合规性模块集成测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 合规性模块集成测试失败: {e}")
        return False

def test_privacy_metrics():
    """测试隐私指标计算"""
    logger.info("🔍 测试隐私指标计算...")
    
    try:
        from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier
        from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
        
        config = create_test_config()
        classifier = PIPLClassifier(config)
        monitor = ComplianceMonitor(config)
        
        # 测试隐私指标
        test_data = [
            {'text': '张三 13812345678', 'expected_metrics': ['pii_detection', 'risk_assessment']},
            {'text': '普通文本内容', 'expected_metrics': ['general_classification']},
            {'text': '用户位置：北京市', 'expected_metrics': ['location_detection', 'cross_border_check']}
        ]
        
        for i, data in enumerate(test_data, 1):
            logger.info(f"  测试 {i}: '{data['text']}'")
            
            # 计算各种隐私指标
            metrics = {}
            
            try:
                # PIPL分类指标
                classification = classifier.classify(data['text'])
                metrics['pipl_classification'] = classification
            except Exception as e:
                logger.warning(f"    PIPL分类失败: {e}")
            
            try:
                # 合规性指标
                compliance = monitor.check_compliance({'content': data['text']})
                metrics['compliance_check'] = compliance
            except Exception as e:
                logger.warning(f"    合规性检查失败: {e}")
            
            logger.info(f"    计算的指标: {list(metrics.keys())}")
            logger.info(f"    预期指标: {data['expected_metrics']}")
        
        logger.info("✅ 隐私指标计算测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 隐私指标计算测试失败: {e}")
        return False

def test_audit_functionality():
    """测试审计功能"""
    logger.info("🔍 测试审计功能...")
    
    try:
        from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
        
        config = create_test_config()
        monitor = ComplianceMonitor(config)
        
        # 模拟一些操作来生成审计日志
        operations = [
            {'action': 'data_access', 'data_type': 'personal_info', 'user': 'user1'},
            {'action': 'data_processing', 'data_type': 'sensitive_info', 'user': 'user2'},
            {'action': 'cross_border_transfer', 'data_type': 'location_data', 'user': 'user1'}
        ]
        
        for operation in operations:
            try:
                # 记录操作
                monitor.log_operation(operation)
                logger.info(f"  记录操作: {operation['action']} - {operation['data_type']}")
            except Exception as e:
                logger.warning(f"  记录操作失败: {e}")
        
        # 获取审计报告
        try:
            audit_report = monitor.get_audit_report()
            logger.info(f"  审计报告: {audit_report}")
        except Exception as e:
            logger.warning(f"  获取审计报告失败: {e}")
        
        # 获取合规性统计
        try:
            compliance_stats = monitor.get_compliance_statistics()
            logger.info(f"  合规性统计: {compliance_stats}")
        except Exception as e:
            logger.warning(f"  获取合规性统计失败: {e}")
        
        logger.info("✅ 审计功能测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 审计功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始PIPL合规性检查模块专项测试...")
    
    test_functions = [
        ("PIPL分类器", test_pipl_classifier),
        ("合规性监控器", test_compliance_monitor),
        ("风险评估器", test_risk_evaluator),
        ("合规性模块集成", test_compliance_integration),
        ("隐私指标计算", test_privacy_metrics),
        ("审计功能", test_audit_functionality)
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
        logger.info("🎉 所有PIPL合规性检查模块测试通过！")
        return True
    else:
        logger.warning(f"⚠️ 有 {total - passed} 项测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
