#!/usr/bin/env python3
"""
ChnSentiCorp-Lite数据集验证器
验证数据集质量和PIPL合规性
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, data_dir: str = "./data/chnsenticorp_lite"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
    
    def load_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """加载数据集"""
        print("加载数据集...")
        
        # 加载训练集
        with open(self.data_dir / "train.jsonl", "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
        
        # 加载验证集
        with open(self.data_dir / "val.jsonl", "r", encoding="utf-8") as f:
            val_data = [json.loads(line) for line in f]
        
        # 加载测试集
        with open(self.data_dir / "test.jsonl", "r", encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f]
        
        print(f"数据集加载完成: 训练集{len(train_data)}, 验证集{len(val_data)}, 测试集{len(test_data)}")
        return train_data, val_data, test_data
    
    def validate_schema(self, data: List[Dict]) -> Dict[str, Any]:
        """验证数据模式"""
        print("验证数据模式...")
        
        required_fields = [
            "sample_id", "text", "label", "privacy_level", 
            "pii_entities", "pipl_cross_border", "privacy_budget_cost", "metadata"
        ]
        
        schema_errors = []
        valid_samples = 0
        
        for i, sample in enumerate(data):
            # 检查必需字段
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                schema_errors.append(f"样本{i}: 缺少字段 {missing_fields}")
                continue
            
            # 检查字段类型
            if not isinstance(sample["text"], str):
                schema_errors.append(f"样本{i}: text字段类型错误")
                continue
            
            if sample["label"] not in ["positive", "negative"]:
                schema_errors.append(f"样本{i}: label字段值错误")
                continue
            
            if sample["privacy_level"] not in ["general", "high_sensitivity"]:
                schema_errors.append(f"样本{i}: privacy_level字段值错误")
                continue
            
            if not isinstance(sample["pii_entities"], list):
                schema_errors.append(f"样本{i}: pii_entities字段类型错误")
                continue
            
            if not isinstance(sample["pipl_cross_border"], bool):
                schema_errors.append(f"样本{i}: pipl_cross_border字段类型错误")
                continue
            
            if not isinstance(sample["privacy_budget_cost"], (int, float)):
                schema_errors.append(f"样本{i}: privacy_budget_cost字段类型错误")
                continue
            
            valid_samples += 1
        
        return {
            "total_samples": len(data),
            "valid_samples": valid_samples,
            "schema_errors": schema_errors,
            "validity_rate": valid_samples / len(data) if data else 0
        }
    
    def validate_pii_detection(self, data: List[Dict]) -> Dict[str, Any]:
        """验证PII检测准确性"""
        print("验证PII检测准确性...")
        
        pii_patterns = {
            "PERSON": r'[张王李赵刘陈杨黄周吴徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤][\u4e00-\u9fa5]{1,2}',
            "PHONE": r'1[3-9]\d{9}',
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "ID_CARD": r'\d{17}[\dXx]'
        }
        
        detection_results = {
            "total_samples": len(data),
            "pii_detected_samples": 0,
            "pii_accuracy": 0,
            "entity_accuracy": {}
        }
        
        for sample in data:
            text = sample["text"]
            detected_entities = sample["pii_entities"]
            
            # 手动检测PII实体
            manual_entities = []
            for entity_type, pattern in pii_patterns.items():
                if re.search(pattern, text):
                    manual_entities.append(entity_type)
            
            # 计算检测准确性
            if manual_entities or detected_entities:
                detection_results["pii_detected_samples"] += 1
                
                # 计算实体检测准确性
                for entity_type in pii_patterns.keys():
                    manual_detected = entity_type in manual_entities
                    auto_detected = entity_type in detected_entities
                    
                    if entity_type not in detection_results["entity_accuracy"]:
                        detection_results["entity_accuracy"][entity_type] = {"correct": 0, "total": 0}
                    
                    detection_results["entity_accuracy"][entity_type]["total"] += 1
                    if manual_detected == auto_detected:
                        detection_results["entity_accuracy"][entity_type]["correct"] += 1
        
        # 计算总体准确性
        if detection_results["pii_detected_samples"] > 0:
            total_correct = sum(acc["correct"] for acc in detection_results["entity_accuracy"].values())
            total_entities = sum(acc["total"] for acc in detection_results["entity_accuracy"].values())
            detection_results["pii_accuracy"] = total_correct / total_entities if total_entities > 0 else 0
        
        return detection_results
    
    def validate_privacy_compliance(self, data: List[Dict]) -> Dict[str, Any]:
        """验证隐私合规性"""
        print("🔍 验证隐私合规性...")
        
        compliance_results = {
            "total_samples": len(data),
            "cross_border_samples": 0,
            "high_sensitivity_samples": 0,
            "compliance_violations": [],
            "privacy_budget_stats": {}
        }
        
        for i, sample in enumerate(data):
            # 检查跨境传输合规性
            if sample["pipl_cross_border"]:
                compliance_results["cross_border_samples"] += 1
                
                # 高敏感度数据不应该允许跨境传输
                if sample["privacy_level"] == "high_sensitivity":
                    compliance_results["compliance_violations"].append(
                        f"样本{i}: 高敏感度数据不允许跨境传输"
                    )
            
            # 统计高敏感度样本
            if sample["privacy_level"] == "high_sensitivity":
                compliance_results["high_sensitivity_samples"] += 1
            
            # 检查隐私预算成本合理性
            budget_cost = sample["privacy_budget_cost"]
            if budget_cost < 0:
                compliance_results["compliance_violations"].append(
                    f"样本{i}: 隐私预算成本不能为负数"
                )
        
        # 计算隐私预算统计
        budget_costs = [sample["privacy_budget_cost"] for sample in data]
        compliance_results["privacy_budget_stats"] = {
            "mean": round(np.mean(budget_costs), 3),
            "std": round(np.std(budget_costs), 3),
            "min": round(np.min(budget_costs), 3),
            "max": round(np.max(budget_costs), 3)
        }
        
        return compliance_results
    
    def validate_data_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """验证数据质量"""
        print("🔍 验证数据质量...")
        
        quality_results = {
            "total_samples": len(data),
            "empty_text_samples": 0,
            "short_text_samples": 0,
            "long_text_samples": 0,
            "duplicate_samples": 0,
            "text_length_stats": {}
        }
        
        text_lengths = []
        text_set = set()
        
        for sample in data:
            text = sample["text"]
            text_length = len(text)
            text_lengths.append(text_length)
            
            # 检查空文本
            if not text.strip():
                quality_results["empty_text_samples"] += 1
            
            # 检查短文本
            if text_length < 5:
                quality_results["short_text_samples"] += 1
            
            # 检查长文本
            if text_length > 500:
                quality_results["long_text_samples"] += 1
            
            # 检查重复文本
            if text in text_set:
                quality_results["duplicate_samples"] += 1
            else:
                text_set.add(text)
        
        # 计算文本长度统计
        if text_lengths:
            quality_results["text_length_stats"] = {
                "mean": round(np.mean(text_lengths), 2),
                "std": round(np.std(text_lengths), 2),
                "min": min(text_lengths),
                "max": max(text_lengths)
            }
        
        return quality_results
    
    def validate_dataset_balance(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """验证数据集平衡性"""
        print("🔍 验证数据集平衡性...")
        
        balance_results = {
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "total_size": len(train_data) + len(val_data) + len(test_data),
            "split_ratios": {},
            "label_distribution": {},
            "privacy_level_distribution": {}
        }
        
        # 计算分割比例
        total_size = balance_results["total_size"]
        balance_results["split_ratios"] = {
            "train": round(len(train_data) / total_size, 3),
            "val": round(len(val_data) / total_size, 3),
            "test": round(len(test_data) / total_size, 3)
        }
        
        # 统计标签分布
        all_data = train_data + val_data + test_data
        for sample in all_data:
            label = sample["label"]
            balance_results["label_distribution"][label] = balance_results["label_distribution"].get(label, 0) + 1
        
        # 统计隐私级别分布
        for sample in all_data:
            level = sample["privacy_level"]
            balance_results["privacy_level_distribution"][level] = balance_results["privacy_level_distribution"].get(level, 0) + 1
        
        return balance_results
    
    def run_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        print("🧪 开始数据集验证...")
        print("="*60)
        
        # 加载数据集
        train_data, val_data, test_data = self.load_dataset()
        
        # 验证数据模式
        print("\n1. 验证数据模式...")
        train_schema = self.validate_schema(train_data)
        val_schema = self.validate_schema(val_data)
        test_schema = self.validate_schema(test_data)
        
        # 验证PII检测
        print("\n2. 验证PII检测...")
        train_pii = self.validate_pii_detection(train_data)
        val_pii = self.validate_pii_detection(val_data)
        test_pii = self.validate_pii_detection(test_data)
        
        # 验证隐私合规性
        print("\n3. 验证隐私合规性...")
        train_compliance = self.validate_privacy_compliance(train_data)
        val_compliance = self.validate_privacy_compliance(val_data)
        test_compliance = self.validate_privacy_compliance(test_data)
        
        # 验证数据质量
        print("\n4. 验证数据质量...")
        train_quality = self.validate_data_quality(train_data)
        val_quality = self.validate_data_quality(val_data)
        test_quality = self.validate_data_quality(test_data)
        
        # 验证数据集平衡性
        print("\n5. 验证数据集平衡性...")
        balance = self.validate_dataset_balance(train_data, val_data, test_data)
        
        # 汇总验证结果
        validation_results = {
            "schema_validation": {
                "train": train_schema,
                "val": val_schema,
                "test": test_schema
            },
            "pii_detection": {
                "train": train_pii,
                "val": val_pii,
                "test": test_pii
            },
            "privacy_compliance": {
                "train": train_compliance,
                "val": val_compliance,
                "test": test_compliance
            },
            "data_quality": {
                "train": train_quality,
                "val": val_quality,
                "test": test_quality
            },
            "dataset_balance": balance
        }
        
        return validation_results
    
    def print_validation_report(self, results: Dict[str, Any]):
        """打印验证报告"""
        print("\n" + "="*80)
        print("📊 数据集验证报告")
        print("="*80)
        
        # 数据模式验证
        print("\n1. 数据模式验证:")
        for split, schema in results["schema_validation"].items():
            print(f"  {split}: {schema['valid_samples']}/{schema['total_samples']} 有效样本")
            if schema['schema_errors']:
                print(f"    错误: {len(schema['schema_errors'])} 个")
        
        # PII检测验证
        print("\n2. PII检测验证:")
        for split, pii in results["pii_detection"].items():
            print(f"  {split}: 准确性 {pii['pii_accuracy']:.3f}")
            for entity, acc in pii['entity_accuracy'].items():
                accuracy = acc['correct'] / acc['total'] if acc['total'] > 0 else 0
                print(f"    {entity}: {accuracy:.3f}")
        
        # 隐私合规性验证
        print("\n3. 隐私合规性验证:")
        for split, compliance in results["privacy_compliance"].items():
            print(f"  {split}: 跨境传输 {compliance['cross_border_samples']} 个")
            print(f"    高敏感度: {compliance['high_sensitivity_samples']} 个")
            if compliance['compliance_violations']:
                print(f"    合规违规: {len(compliance['compliance_violations'])} 个")
        
        # 数据质量验证
        print("\n4. 数据质量验证:")
        for split, quality in results["data_quality"].items():
            print(f"  {split}: 空文本 {quality['empty_text_samples']} 个")
            print(f"    短文本: {quality['short_text_samples']} 个")
            print(f"    长文本: {quality['long_text_samples']} 个")
            print(f"    重复文本: {quality['duplicate_samples']} 个")
        
        # 数据集平衡性验证
        print("\n5. 数据集平衡性验证:")
        balance = results["dataset_balance"]
        print(f"  总样本数: {balance['total_size']}")
        print(f"  分割比例: 训练集{balance['split_ratios']['train']}, 验证集{balance['split_ratios']['val']}, 测试集{balance['split_ratios']['test']}")
        print(f"  标签分布: {balance['label_distribution']}")
        print(f"  隐私级别分布: {balance['privacy_level_distribution']}")
        
        print("="*80)

def main():
    """主函数"""
    print("ChnSentiCorp-Lite数据集验证器")
    print("验证数据集质量和PIPL合规性")
    print("="*60)
    
    # 创建验证器
    validator = DatasetValidator()
    
    # 运行验证
    results = validator.run_validation()
    
    # 打印验证报告
    validator.print_validation_report(results)
    
    # 保存验证结果
    with open("./data/chnsenticorp_lite/validation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n🎉 数据集验证完成！")
    print("验证报告已保存到: validation_report.json")

if __name__ == "__main__":
    main()
