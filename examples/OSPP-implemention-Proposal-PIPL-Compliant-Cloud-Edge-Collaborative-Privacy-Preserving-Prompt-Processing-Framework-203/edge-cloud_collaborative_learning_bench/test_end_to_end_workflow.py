#!/usr/bin/env python3
"""
端到端工作流程测试脚本
测试完整的隐私保护LLM工作流程，不涉及模型下载
"""

import sys
import os
import json
import logging
import time
import numpy as np
from typing import Dict, Any, List

# 添加当前目录到Python路径
sys.path.append('.')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_workflow_config():
    """创建工作流程配置"""
    return {
        'privacy_detection': {
            'detection_methods': {
                'regex_patterns': ['phone', 'email', 'id_card', 'address', 'name'],
                'entity_types': ['PERSON', 'PHONE', 'EMAIL', 'ID_CARD', 'ADDRESS'],
                'ner_model': 'hfl/chinese-bert-wwm-ext'
            }
        },
        'privacy_encryption': {
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
        },
        'compliance': {
            'pipl_version': '2021',
            'audit_level': 'detailed',
            'cross_border_policy': 'strict'
        }
    }

def test_data_preprocessing_workflow():
    """测试数据预处理工作流程"""
    logger.info("🔍 测试数据预处理工作流程...")
    
    try:
        from test_algorithms.privacy_detection.pii_detector import PIIDetector
        from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier
        
        config = create_workflow_config()
        
        # 初始化模块
        pii_detector = PIIDetector(config['privacy_detection'])
        pipl_classifier = PIPLClassifier(config['compliance'])
        
        # 测试数据
        test_data = [
            "用户信息：姓名张三，电话13812345678，邮箱zhangsan@example.com",
            "地址：北京市朝阳区建国门外大街1号",
            "这是一段普通的文本，不包含敏感信息",
            "身份证号：110101199001011234，银行卡：6222021234567890"
        ]
        
        processed_results = []
        
        for i, text in enumerate(test_data, 1):
            logger.info(f"  处理数据 {i}: '{text[:30]}...'")
            
            # 步骤1: PII检测
            pii_result = pii_detector.detect(text)
            logger.info(f"    检测到 {len(pii_result)} 个敏感实体")
            
            # 步骤2: PIPL分类
            classification = pipl_classifier.classify(text)
            logger.info(f"    分类结果: {classification}")
            
            # 步骤3: 数据预处理
            processed_text = text
            protection_applied = []
            
            for entity in pii_result:
                if entity['requires_protection']:
                    # 应用保护措施
                    if entity['type'] in ['PHONE', 'ID_CARD']:
                        # 部分掩码
                        masked_value = entity['text'][:3] + '*' * (len(entity['text']) - 6) + entity['text'][-3:]
                        processed_text = processed_text.replace(entity['text'], masked_value)
                        protection_applied.append(f"部分掩码: {entity['type']}")
                    elif entity['type'] == 'EMAIL':
                        # 完全掩码
                        processed_text = processed_text.replace(entity['text'], '[EMAIL_MASKED]')
                        protection_applied.append(f"完全掩码: {entity['type']}")
            
            result = {
                'original_text': text,
                'processed_text': processed_text,
                'pii_entities': len(pii_result),
                'classification': classification,
                'protection_applied': protection_applied
            }
            processed_results.append(result)
            
            logger.info(f"    保护措施: {protection_applied}")
        
        logger.info(f"✅ 数据预处理工作流程完成，处理了 {len(processed_results)} 条数据")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据预处理工作流程失败: {e}")
        return False

def test_privacy_protection_workflow():
    """测试隐私保护工作流程"""
    logger.info("🔍 测试隐私保护工作流程...")
    
    try:
        from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
        from test_algorithms.privacy_detection.pii_detector import PIIDetector
        
        config = create_workflow_config()
        
        # 初始化模块
        dp = DifferentialPrivacy(config['privacy_encryption'])
        pii_detector = PIIDetector(config['privacy_detection'])
        
        # 测试数据
        test_text = "用户特征向量：[0.1, 0.2, 0.3, 0.4, 0.5]，用户信息：张三 13812345678"
        
        logger.info(f"  处理文本: '{test_text}'")
        
        # 步骤1: 检测敏感信息
        pii_result = pii_detector.detect(test_text)
        logger.info(f"    检测到 {len(pii_result)} 个敏感实体")
        
        # 步骤2: 提取特征向量（模拟）
        feature_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        logger.info(f"    特征向量: {feature_vector}")
        
        # 步骤3: 应用差分隐私
        dp_params = {
            'epsilon': 1.0,
            'delta': 0.00001,
            'clipping_norm': 1.0
        }
        
        dp_result = dp.add_noise(feature_vector, dp_params)
        logger.info(f"    差分隐私处理完成")
        logger.info(f"    噪声规模: {dp_result['noise_scale']:.4f}")
        logger.info(f"    剩余预算: {dp_result['privacy_budget_remaining']}")
        
        # 步骤4: 生成保护报告
        protection_report = {
            'original_features': feature_vector.tolist(),
            'protected_features': dp_result['noisy_data'].tolist(),
            'privacy_parameters': {
                'epsilon_used': dp_result['epsilon_used'],
                'delta_used': dp_result['delta_used'],
                'noise_scale': dp_result['noise_scale']
            },
            'pii_detected': len(pii_result),
            'protection_level': 'high' if len(pii_result) > 0 else 'low'
        }
        
        logger.info(f"    保护报告生成完成")
        logger.info(f"    保护级别: {protection_report['protection_level']}")
        
        logger.info("✅ 隐私保护工作流程完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 隐私保护工作流程失败: {e}")
        return False

def test_compliance_monitoring_workflow():
    """测试合规性监控工作流程"""
    logger.info("🔍 测试合规性监控工作流程...")
    
    try:
        from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
        from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier
        
        config = create_workflow_config()
        
        # 初始化模块
        monitor = ComplianceMonitor(config['compliance'])
        classifier = PIPLClassifier(config['compliance'])
        
        # 模拟一系列操作
        operations = [
            {
                'operation_id': 'op_001',
                'operation_type': 'data_access',
                'data_content': '用户姓名：张三',
                'timestamp': time.time(),
                'user_id': 'user_001'
            },
            {
                'operation_id': 'op_002',
                'operation_type': 'data_processing',
                'data_content': '电话：13812345678',
                'timestamp': time.time(),
                'user_id': 'user_002'
            },
            {
                'operation_id': 'op_003',
                'operation_type': 'cross_border_transfer',
                'data_content': '邮箱：zhangsan@example.com',
                'timestamp': time.time(),
                'user_id': 'user_001'
            }
        ]
        
        compliance_results = []
        
        for operation in operations:
            logger.info(f"  监控操作: {operation['operation_type']} - {operation['data_content']}")
            
            # 步骤1: 数据分类
            classification = classifier.classify(operation['data_content'])
            logger.info(f"    数据分类: {classification}")
            
            # 步骤2: 合规性检查（模拟）
            compliance_status = 'compliant'
            if operation['operation_type'] == 'cross_border_transfer':
                compliance_status = 'requires_encryption'
            elif 'phone' in operation['data_content'] or 'email' in operation['data_content']:
                compliance_status = 'requires_protection'
            
            logger.info(f"    合规性状态: {compliance_status}")
            
            # 步骤3: 记录操作（模拟）
            operation_record = {
                'operation_id': operation['operation_id'],
                'timestamp': operation['timestamp'],
                'classification': classification,
                'compliance_status': compliance_status,
                'actions_taken': ['logged', 'monitored']
            }
            compliance_results.append(operation_record)
        
        # 步骤4: 生成合规性报告
        compliance_report = {
            'total_operations': len(operations),
            'compliant_operations': len([r for r in compliance_results if r['compliance_status'] == 'compliant']),
            'requires_protection': len([r for r in compliance_results if r['compliance_status'] == 'requires_protection']),
            'requires_encryption': len([r for r in compliance_results if r['compliance_status'] == 'requires_encryption']),
            'compliance_rate': len([r for r in compliance_results if r['compliance_status'] == 'compliant']) / len(operations)
        }
        
        logger.info(f"  合规性报告:")
        logger.info(f"    总操作数: {compliance_report['total_operations']}")
        logger.info(f"    合规率: {compliance_report['compliance_rate']:.2%}")
        
        logger.info("✅ 合规性监控工作流程完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 合规性监控工作流程失败: {e}")
        return False

def test_integrated_workflow():
    """测试集成工作流程"""
    logger.info("🔍 测试集成工作流程...")
    
    try:
        from test_algorithms.privacy_detection.pii_detector import PIIDetector
        from test_algorithms.privacy_detection.pipl_classifier import PIPLClassifier
        from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
        from test_algorithms.privacy_encryption.compliance_monitor import ComplianceMonitor
        
        config = create_workflow_config()
        
        # 初始化所有模块
        pii_detector = PIIDetector(config['privacy_detection'])
        pipl_classifier = PIPLClassifier(config['compliance'])
        dp = DifferentialPrivacy(config['privacy_encryption'])
        monitor = ComplianceMonitor(config['compliance'])
        
        logger.info("✅ 所有模块初始化成功")
        
        # 模拟完整的端到端工作流程
        test_cases = [
            {
                'text': '用户信息：张三，电话13812345678，邮箱zhangsan@example.com',
                'features': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'operation_type': 'data_processing'
            },
            {
                'text': '地址：北京市朝阳区建国门外大街1号',
                'features': np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
                'operation_type': 'location_analysis'
            },
            {
                'text': '普通文本内容，无敏感信息',
                'features': np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
                'operation_type': 'general_analysis'
            }
        ]
        
        workflow_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"  处理案例 {i}: {test_case['operation_type']}")
            
            # 步骤1: 数据预处理和PII检测
            pii_result = pii_detector.detect(test_case['text'])
            classification = pipl_classifier.classify(test_case['text'])
            
            # 步骤2: 隐私保护处理
            dp_params = {
                'epsilon': 1.0,
                'delta': 0.00001,
                'clipping_norm': 1.0
            }
            dp_result = dp.add_noise(test_case['features'], dp_params)
            
            # 步骤3: 合规性检查
            compliance_status = 'compliant'
            if len(pii_result) > 0:
                compliance_status = 'requires_protection'
            
            # 步骤4: 生成处理结果
            result = {
                'case_id': i,
                'operation_type': test_case['operation_type'],
                'pii_entities_detected': len(pii_result),
                'data_classification': classification,
                'privacy_protection_applied': True,
                'compliance_status': compliance_status,
                'privacy_budget_used': dp_result['epsilon_used'],
                'processing_time': time.time()
            }
            
            workflow_results.append(result)
            
            logger.info(f"    PII实体: {len(pii_result)}")
            logger.info(f"    数据分类: {classification}")
            logger.info(f"    合规状态: {compliance_status}")
            logger.info(f"    隐私预算使用: {dp_result['epsilon_used']}")
        
        # 生成工作流程报告
        workflow_report = {
            'total_cases': len(test_cases),
            'successful_cases': len(workflow_results),
            'total_pii_detected': sum(r['pii_entities_detected'] for r in workflow_results),
            'total_privacy_budget_used': sum(r['privacy_budget_used'] for r in workflow_results),
            'compliance_rate': len([r for r in workflow_results if r['compliance_status'] == 'compliant']) / len(workflow_results),
            'average_processing_time': np.mean([r['processing_time'] for r in workflow_results])
        }
        
        logger.info(f"  工作流程报告:")
        logger.info(f"    总案例数: {workflow_report['total_cases']}")
        logger.info(f"    成功案例数: {workflow_report['successful_cases']}")
        logger.info(f"    总PII检测数: {workflow_report['total_pii_detected']}")
        logger.info(f"    总隐私预算使用: {workflow_report['total_privacy_budget_used']:.2f}")
        logger.info(f"    合规率: {workflow_report['compliance_rate']:.2%}")
        
        logger.info("✅ 集成工作流程完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 集成工作流程失败: {e}")
        return False

def test_performance_benchmark():
    """测试性能基准"""
    logger.info("🔍 测试性能基准...")
    
    try:
        from test_algorithms.privacy_detection.pii_detector import PIIDetector
        from test_algorithms.privacy_encryption.differential_privacy import DifferentialPrivacy
        
        config = create_workflow_config()
        
        # 初始化模块
        pii_detector = PIIDetector(config['privacy_detection'])
        dp = DifferentialPrivacy(config['privacy_encryption'])
        
        # 性能测试数据
        test_texts = [
            "用户信息：张三，电话13812345678，邮箱zhangsan@example.com",
            "地址：北京市朝阳区建国门外大街1号",
            "身份证号：110101199001011234",
            "普通文本内容，无敏感信息",
            "银行卡号：6222021234567890"
        ] * 10  # 重复10次进行性能测试
        
        test_features = [np.random.rand(100) for _ in range(50)]  # 50个100维特征向量
        
        # PII检测性能测试
        logger.info("  测试PII检测性能...")
        start_time = time.time()
        pii_results = []
        for text in test_texts:
            result = pii_detector.detect(text)
            pii_results.append(result)
        pii_time = time.time() - start_time
        
        # 差分隐私性能测试
        logger.info("  测试差分隐私性能...")
        start_time = time.time()
        dp_results = []
        for features in test_features:
            dp_params = {
                'epsilon': 1.0,
                'delta': 0.00001,
                'clipping_norm': 1.0
            }
            result = dp.add_noise(features, dp_params)
            dp_results.append(result)
        dp_time = time.time() - start_time
        
        # 计算性能指标
        performance_metrics = {
            'pii_detection': {
                'total_texts': len(test_texts),
                'total_time': pii_time,
                'avg_time_per_text': pii_time / len(test_texts),
                'texts_per_second': len(test_texts) / pii_time
            },
            'differential_privacy': {
                'total_features': len(test_features),
                'total_time': dp_time,
                'avg_time_per_feature': dp_time / len(test_features),
                'features_per_second': len(test_features) / dp_time
            },
            'overall': {
                'total_operations': len(test_texts) + len(test_features),
                'total_time': pii_time + dp_time,
                'avg_time_per_operation': (pii_time + dp_time) / (len(test_texts) + len(test_features))
            }
        }
        
        logger.info(f"  性能指标:")
        logger.info(f"    PII检测: {performance_metrics['pii_detection']['texts_per_second']:.2f} 文本/秒")
        logger.info(f"    差分隐私: {performance_metrics['differential_privacy']['features_per_second']:.2f} 特征/秒")
        logger.info(f"    总体性能: {performance_metrics['overall']['avg_time_per_operation']:.4f} 秒/操作")
        
        logger.info("✅ 性能基准测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 性能基准测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始端到端工作流程测试...")
    
    test_functions = [
        ("数据预处理工作流程", test_data_preprocessing_workflow),
        ("隐私保护工作流程", test_privacy_protection_workflow),
        ("合规性监控工作流程", test_compliance_monitoring_workflow),
        ("集成工作流程", test_integrated_workflow),
        ("性能基准测试", test_performance_benchmark)
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_name, test_func in test_functions:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 {test_name}")
            logger.info(f"{'='*60}")
            
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
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 端到端工作流程测试结果汇总")
    logger.info(f"{'='*60}")
    logger.info(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有端到端工作流程测试通过！")
        return True
    else:
        logger.warning(f"⚠️ 有 {total - passed} 项测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
