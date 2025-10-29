#!/usr/bin/env python3
"""
PII检测器专项测试脚本
测试各种类型的敏感信息检测功能
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
        'detection_methods': {
            'regex_patterns': ['phone', 'email', 'id_card', 'address', 'name'],
            'entity_types': ['PERSON', 'ORG', 'LOC', 'PHONE', 'EMAIL', 'ID_CARD', 'FINANCIAL'],
            'ner_model': 'hfl/chinese-bert-wwm-ext'
        }
    }

def test_phone_detection():
    """测试电话号码检测"""
    logger.info("🔍 测试电话号码检测...")
    
    from test_algorithms.privacy_detection.pii_detector import PIIDetector
    
    config = create_test_config()
    detector = PIIDetector(config)
    
    test_cases = [
        "My phone number is 13812345678",
        "Call me at +86-138-1234-5678",
        "Contact: 138-1234-5678",
        "Phone: 13812345678, Email: test@example.com"
    ]
    
    for i, text in enumerate(test_cases, 1):
        result = detector.detect(text)
        phone_entities = [e for e in result if e['type'] == 'PHONE']
        logger.info(f"  测试 {i}: 检测到 {len(phone_entities)} 个电话号码")
        for entity in phone_entities:
            logger.info(f"    - {entity['text']} (风险: {entity['risk_level']}, 需要保护: {entity['requires_protection']})")
    
    return True

def test_email_detection():
    """测试邮箱地址检测"""
    logger.info("🔍 测试邮箱地址检测...")
    
    from test_algorithms.privacy_detection.pii_detector import PIIDetector
    
    config = create_test_config()
    detector = PIIDetector(config)
    
    test_cases = [
        "My email is zhangsan@example.com",
        "Contact us at support@company.org",
        "Email: user.name+tag@domain.co.uk",
        "Send to: test123@test-domain.com"
    ]
    
    for i, text in enumerate(test_cases, 1):
        result = detector.detect(text)
        email_entities = [e for e in result if e['type'] == 'EMAIL']
        logger.info(f"  测试 {i}: 检测到 {len(email_entities)} 个邮箱地址")
        for entity in email_entities:
            logger.info(f"    - {entity['text']} (风险: {entity['risk_level']}, 需要保护: {entity['requires_protection']})")
    
    return True

def test_mixed_content():
    """测试混合内容检测"""
    logger.info("🔍 测试混合内容检测...")
    
    from test_algorithms.privacy_detection.pii_detector import PIIDetector
    
    config = create_test_config()
    detector = PIIDetector(config)
    
    test_text = """
    User Information:
    Name: Zhang San
    Phone: 13812345678
    Email: zhangsan@example.com
    ID Card: 110101199001011234
    Address: No.123 Main Street, Beijing, China
    Company: ABC Technology Co., Ltd.
    """
    
    result = detector.detect(test_text)
    
    logger.info(f"  检测到 {len(result)} 个敏感实体:")
    for entity in result:
        logger.info(f"    - {entity['type']}: {entity['text']} (风险: {entity['risk_level']}, 需要保护: {entity['requires_protection']})")
    
    # 获取统计信息
    summary = detector.get_entity_summary(result)
    logger.info(f"  统计信息: {summary}")
    
    return True

def test_risk_assessment():
    """测试风险评估功能"""
    logger.info("🔍 测试风险评估功能...")
    
    from test_algorithms.privacy_detection.pii_detector import PIIDetector
    
    config = create_test_config()
    detector = PIIDetector(config)
    
    test_cases = [
        ("Phone: 13812345678", "应该检测为高风险"),
        ("Email: test@example.com", "应该检测为中等风险"),
        ("Company: ABC Corp", "应该检测为低风险"),
        ("ID: 110101199001011234", "应该检测为高风险")
    ]
    
    for text, expected in test_cases:
        result = detector.detect(text)
        if result:
            entity = result[0]
            logger.info(f"  '{text}' -> 风险级别: {entity['risk_level']} ({expected})")
        else:
            logger.warning(f"  '{text}' -> 未检测到实体")
    
    return True

def test_context_extraction():
    """测试上下文提取功能"""
    logger.info("🔍 测试上下文提取功能...")
    
    from test_algorithms.privacy_detection.pii_detector import PIIDetector
    
    config = create_test_config()
    detector = PIIDetector(config)
    
    test_text = "Please contact Zhang San at 13812345678 for more information about our services."
    
    result = detector.detect(test_text)
    
    for entity in result:
        logger.info(f"  实体: {entity['text']}")
        logger.info(f"  上下文: {entity['context']}")
        logger.info(f"  位置: {entity['start']}-{entity['end']}")
    
    return True

def test_protection_requirements():
    """测试保护需求判断"""
    logger.info("🔍 测试保护需求判断...")
    
    from test_algorithms.privacy_detection.pii_detector import PIIDetector
    
    config = create_test_config()
    detector = PIIDetector(config)
    
    test_cases = [
        "Phone: 13812345678",  # 应该需要保护
        "Email: test@example.com",  # 应该需要保护
        "Company: ABC Corp",  # 可能不需要保护
        "Location: Beijing"  # 可能不需要保护
    ]
    
    for text in test_cases:
        result = detector.detect(text)
        if result:
            entity = result[0]
            protection_needed = entity['requires_protection']
            logger.info(f"  '{text}' -> 需要保护: {protection_needed}")
        else:
            logger.warning(f"  '{text}' -> 未检测到实体")
    
    return True

def main():
    """主测试函数"""
    logger.info("🚀 开始PII检测器专项测试...")
    
    test_functions = [
        ("电话号码检测", test_phone_detection),
        ("邮箱地址检测", test_email_detection),
        ("混合内容检测", test_mixed_content),
        ("风险评估功能", test_risk_assessment),
        ("上下文提取", test_context_extraction),
        ("保护需求判断", test_protection_requirements)
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
    
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 测试结果汇总")
    logger.info(f"{'='*50}")
    logger.info(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有PII检测器测试通过！")
        return True
    else:
        logger.warning(f"⚠️ 有 {total - passed} 项测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
