#!/usr/bin/env python3
"""
ChnSentiCorp-Lite数据集使用示例 (简化版)
展示如何使用PIPL合规的云边协同LLM基准数据集
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

class SimpleDatasetUsage:
    """简化数据集使用示例"""
    
    def __init__(self, data_dir: str = "./data/chnsenticorp_lite"):
        self.data_dir = Path(data_dir)
    
    def load_dataset(self) -> Dict[str, List[Dict]]:
        """加载数据集"""
        print("加载ChnSentiCorp-Lite数据集...")
        
        dataset = {}
        for split in ["train", "val", "test"]:
            file_path = self.data_dir / f"{split}.jsonl"
            with open(file_path, "r", encoding="utf-8") as f:
                dataset[split] = [json.loads(line) for line in f]
            print(f"  {split}: {len(dataset[split])} 个样本")
        
        return dataset
    
    def demonstrate_privacy_levels(self, dataset: Dict[str, List[Dict]]):
        """演示隐私级别处理"""
        print("\n隐私级别处理演示")
        print("="*50)
        
        # 统计隐私级别分布
        privacy_levels = {}
        for split, data in dataset.items():
            for sample in data:
                level = sample["privacy_level"]
                privacy_levels[level] = privacy_levels.get(level, 0) + 1
        
        print("隐私级别分布:")
        for level, count in privacy_levels.items():
            print(f"  {level}: {count} 个样本")
        
        # 展示不同隐私级别的样本
        print("\n通用敏感度样本示例:")
        general_samples = [s for s in dataset["train"] if s["privacy_level"] == "general"][:3]
        for i, sample in enumerate(general_samples):
            print(f"  样本{i+1}: {sample['text']}")
            print(f"    标签: {sample['label']}")
            print(f"    跨境传输: {sample['pipl_cross_border']}")
            print(f"    隐私预算成本: {sample['privacy_budget_cost']}")
        
        print("\n高敏感度样本示例:")
        high_sensitivity_samples = [s for s in dataset["train"] if s["privacy_level"] == "high_sensitivity"][:3]
        for i, sample in enumerate(high_sensitivity_samples):
            print(f"  样本{i+1}: {sample['text']}")
            print(f"    标签: {sample['label']}")
            print(f"    PII实体: {sample['pii_entities']}")
            print(f"    跨境传输: {sample['pipl_cross_border']}")
            print(f"    隐私预算成本: {sample['privacy_budget_cost']}")
            if sample["synthetic_pii"]:
                print(f"    合成PII: {sample['synthetic_pii']}")
    
    def demonstrate_pii_detection(self, dataset: Dict[str, List[Dict]]):
        """演示PII检测"""
        print("\nPII检测演示")
        print("="*50)
        
        # 统计PII实体分布
        pii_entities = {}
        for split, data in dataset.items():
            for sample in data:
                for entity in sample["pii_entities"]:
                    pii_entities[entity] = pii_entities.get(entity, 0) + 1
        
        print("PII实体分布:")
        for entity, count in pii_entities.items():
            print(f"  {entity}: {count} 个")
        
        # 展示包含PII的样本
        print("\n包含PII的样本:")
        pii_samples = []
        for split, data in dataset.items():
            for sample in data:
                if sample["pii_entities"]:
                    pii_samples.append(sample)
                    if len(pii_samples) >= 5:
                        break
            if len(pii_samples) >= 5:
                break
        
        for i, sample in enumerate(pii_samples):
            print(f"  样本{i+1}: {sample['text']}")
            print(f"    PII实体: {sample['pii_entities']}")
            print(f"    隐私级别: {sample['privacy_level']}")
            print(f"    跨境传输: {sample['pipl_cross_border']}")
    
    def demonstrate_cross_border_compliance(self, dataset: Dict[str, List[Dict]]):
        """演示跨境传输合规性"""
        print("\n跨境传输合规性演示")
        print("="*50)
        
        # 统计跨境传输分布
        cross_border_stats = {
            "total_samples": 0,
            "cross_border_allowed": 0,
            "cross_border_denied": 0,
            "high_sensitivity_cross_border": 0
        }
        
        for split, data in dataset.items():
            for sample in data:
                cross_border_stats["total_samples"] += 1
                
                if sample["pipl_cross_border"]:
                    cross_border_stats["cross_border_allowed"] += 1
                    
                    # 检查高敏感度数据是否错误地允许跨境传输
                    if sample["privacy_level"] == "high_sensitivity":
                        cross_border_stats["high_sensitivity_cross_border"] += 1
                else:
                    cross_border_stats["cross_border_denied"] += 1
        
        print("跨境传输统计:")
        print(f"  总样本数: {cross_border_stats['total_samples']}")
        print(f"  允许跨境: {cross_border_stats['cross_border_allowed']}")
        print(f"  禁止跨境: {cross_border_stats['cross_border_denied']}")
        print(f"  高敏感度错误跨境: {cross_border_stats['high_sensitivity_cross_border']}")
        
        # 展示合规性检查
        print("\n合规性检查示例:")
        compliance_samples = []
        for split, data in dataset.items():
            for sample in data:
                if sample["privacy_level"] == "high_sensitivity" and sample["pipl_cross_border"]:
                    compliance_samples.append(sample)
                    if len(compliance_samples) >= 3:
                        break
            if len(compliance_samples) >= 3:
                break
        
        if compliance_samples:
            print("  发现合规性问题:")
            for i, sample in enumerate(compliance_samples):
                print(f"    样本{i+1}: {sample['text']}")
                print(f"      隐私级别: {sample['privacy_level']}")
                print(f"      跨境传输: {sample['pipl_cross_border']}")
                print(f"      问题: 高敏感度数据不应允许跨境传输")
        else:
            print("  未发现合规性问题")
    
    def demonstrate_privacy_budget_management(self, dataset: Dict[str, List[Dict]]):
        """演示隐私预算管理"""
        print("\n隐私预算管理演示")
        print("="*50)
        
        # 统计隐私预算成本
        budget_costs = []
        for split, data in dataset.items():
            for sample in data:
                budget_costs.append(sample["privacy_budget_cost"])
        
        print("隐私预算成本统计:")
        print(f"  均值: {np.mean(budget_costs):.3f}")
        print(f"  标准差: {np.std(budget_costs):.3f}")
        print(f"  最小值: {np.min(budget_costs):.3f}")
        print(f"  最大值: {np.max(budget_costs):.3f}")
        
        # 按隐私级别分析预算成本
        print("\n按隐私级别分析预算成本:")
        for level in ["general", "high_sensitivity"]:
            level_costs = []
            for split, data in dataset.items():
                for sample in data:
                    if sample["privacy_level"] == level:
                        level_costs.append(sample["privacy_budget_cost"])
            
            if level_costs:
                print(f"  {level}:")
                print(f"    均值: {np.mean(level_costs):.3f}")
                print(f"    标准差: {np.std(level_costs):.3f}")
                print(f"    样本数: {len(level_costs)}")
        
        # 展示高预算成本样本
        print("\n高预算成本样本示例:")
        high_budget_samples = sorted(dataset["train"], key=lambda x: x["privacy_budget_cost"], reverse=True)[:3]
        for i, sample in enumerate(high_budget_samples):
            print(f"  样本{i+1}: {sample['text']}")
            print(f"    隐私级别: {sample['privacy_level']}")
            print(f"    PII实体: {sample['pii_entities']}")
            print(f"    预算成本: {sample['privacy_budget_cost']}")
    
    def demonstrate_dataset_usage(self, dataset: Dict[str, List[Dict]]):
        """演示数据集使用"""
        print("\n数据集使用演示")
        print("="*50)
        
        # 展示数据集基本信息
        print("数据集基本信息:")
        for split, data in dataset.items():
            print(f"  {split}: {len(data)} 个样本")
        
        # 展示标签分布
        print("\n标签分布:")
        label_dist = {}
        for split, data in dataset.items():
            for sample in data:
                label = sample["label"]
                label_dist[label] = label_dist.get(label, 0) + 1
        
        for label, count in label_dist.items():
            print(f"  {label}: {count} 个")
        
        # 展示数据源分布
        print("\n数据源分布:")
        source_dist = {}
        for split, data in dataset.items():
            for sample in data:
                source = sample["metadata"].get("source", "unknown")
                source_dist[source] = source_dist.get(source, 0) + 1
        
        for source, count in source_dist.items():
            print(f"  {source}: {count} 个")
        
        # 展示领域分布
        print("\n领域分布:")
        domain_dist = {}
        for split, data in dataset.items():
            for sample in data:
                domain = sample["metadata"].get("domain", "unknown")
                domain_dist[domain] = domain_dist.get(domain, 0) + 1
        
        for domain, count in domain_dist.items():
            print(f"  {domain}: {count} 个")
    
    def run_demonstration(self):
        """运行完整演示"""
        print("ChnSentiCorp-Lite数据集使用演示")
        print("展示PIPL合规的云边协同LLM基准数据集")
        print("="*60)
        
        # 加载数据集
        dataset = self.load_dataset()
        
        # 演示各种功能
        self.demonstrate_privacy_levels(dataset)
        self.demonstrate_pii_detection(dataset)
        self.demonstrate_cross_border_compliance(dataset)
        self.demonstrate_privacy_budget_management(dataset)
        self.demonstrate_dataset_usage(dataset)
        
        print("\n数据集使用演示完成！")
        print("数据集已准备好用于PIPL合规的云边协同LLM基准测试")

def main():
    """主函数"""
    # 创建使用示例
    example = SimpleDatasetUsage()
    
    # 运行演示
    example.run_demonstration()

if __name__ == "__main__":
    main()
