#!/usr/bin/env python3
"""
ChnSentiCorp-Lite数据集验证器 (简化版)
验证数据集质量和PIPL合规性
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

class SimpleDatasetValidator:
    """简化数据集验证器"""
    
    def __init__(self, data_dir: str = "./data/chnsenticorp_lite"):
        self.data_dir = Path(data_dir)
    
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
    
    def validate_basic_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """验证基本数据质量"""
        print("验证基本数据质量...")
        
        results = {
            "total_samples": len(data),
            "valid_samples": 0,
            "empty_text": 0,
            "short_text": 0,
            "label_distribution": {},
            "privacy_level_distribution": {},
            "cross_border_distribution": {}
        }
        
        for sample in data:
            # 检查基本字段
            if not all(key in sample for key in ["text", "label", "privacy_level", "pipl_cross_border"]):
                continue
            
            results["valid_samples"] += 1
            
            # 检查文本质量
            text = sample["text"]
            if not text.strip():
                results["empty_text"] += 1
            elif len(text) < 5:
                results["short_text"] += 1
            
            # 统计分布
            label = sample["label"]
            results["label_distribution"][label] = results["label_distribution"].get(label, 0) + 1
            
            level = sample["privacy_level"]
            results["privacy_level_distribution"][level] = results["privacy_level_distribution"].get(level, 0) + 1
            
            cross_border = sample["pipl_cross_border"]
            results["cross_border_distribution"][str(cross_border)] = results["cross_border_distribution"].get(str(cross_border), 0) + 1
        
        return results
    
    def validate_privacy_compliance(self, data: List[Dict]) -> Dict[str, Any]:
        """验证隐私合规性"""
        print("验证隐私合规性...")
        
        results = {
            "total_samples": len(data),
            "compliance_violations": 0,
            "high_sensitivity_cross_border": 0,
            "privacy_budget_stats": {}
        }
        
        budget_costs = []
        
        for sample in data:
            # 检查高敏感度数据是否错误地允许跨境传输
            if (sample["privacy_level"] == "high_sensitivity" and 
                sample["pipl_cross_border"]):
                results["compliance_violations"] += 1
                results["high_sensitivity_cross_border"] += 1
            
            # 收集隐私预算成本
            if "privacy_budget_cost" in sample:
                budget_costs.append(sample["privacy_budget_cost"])
        
        # 计算隐私预算统计
        if budget_costs:
            results["privacy_budget_stats"] = {
                "mean": round(np.mean(budget_costs), 3),
                "std": round(np.std(budget_costs), 3),
                "min": round(np.min(budget_costs), 3),
                "max": round(np.max(budget_costs), 3)
            }
        
        return results
    
    def run_validation(self) -> Dict[str, Any]:
        """运行验证"""
        print("开始数据集验证...")
        print("="*60)
        
        # 加载数据集
        train_data, val_data, test_data = self.load_dataset()
        
        # 验证基本质量
        print("\n1. 验证基本数据质量...")
        train_quality = self.validate_basic_quality(train_data)
        val_quality = self.validate_basic_quality(val_data)
        test_quality = self.validate_basic_quality(test_data)
        
        # 验证隐私合规性
        print("\n2. 验证隐私合规性...")
        train_compliance = self.validate_privacy_compliance(train_data)
        val_compliance = self.validate_privacy_compliance(val_data)
        test_compliance = self.validate_privacy_compliance(test_data)
        
        # 汇总结果
        validation_results = {
            "basic_quality": {
                "train": train_quality,
                "val": val_quality,
                "test": test_quality
            },
            "privacy_compliance": {
                "train": train_compliance,
                "val": val_compliance,
                "test": test_compliance
            }
        }
        
        return validation_results
    
    def print_validation_report(self, results: Dict[str, Any]):
        """打印验证报告"""
        print("\n" + "="*80)
        print("数据集验证报告")
        print("="*80)
        
        # 基本质量验证
        print("\n1. 基本数据质量验证:")
        for split, quality in results["basic_quality"].items():
            print(f"  {split}:")
            print(f"    总样本数: {quality['total_samples']}")
            print(f"    有效样本数: {quality['valid_samples']}")
            print(f"    空文本: {quality['empty_text']}")
            print(f"    短文本: {quality['short_text']}")
            print(f"    标签分布: {quality['label_distribution']}")
            print(f"    隐私级别分布: {quality['privacy_level_distribution']}")
            print(f"    跨境传输分布: {quality['cross_border_distribution']}")
        
        # 隐私合规性验证
        print("\n2. 隐私合规性验证:")
        for split, compliance in results["privacy_compliance"].items():
            print(f"  {split}:")
            print(f"    总样本数: {compliance['total_samples']}")
            print(f"    合规违规: {compliance['compliance_violations']}")
            print(f"    高敏感度错误跨境: {compliance['high_sensitivity_cross_border']}")
            if compliance['privacy_budget_stats']:
                print(f"    隐私预算统计: {compliance['privacy_budget_stats']}")
        
        print("="*80)

def main():
    """主函数"""
    print("ChnSentiCorp-Lite数据集验证器 (简化版)")
    print("验证数据集质量和PIPL合规性")
    print("="*60)
    
    # 创建验证器
    validator = SimpleDatasetValidator()
    
    # 运行验证
    results = validator.run_validation()
    
    # 打印验证报告
    validator.print_validation_report(results)
    
    # 保存验证结果
    with open("./data/chnsenticorp_lite/simple_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n数据集验证完成！")
    print("验证报告已保存到: simple_validation_report.json")

if __name__ == "__main__":
    main()
