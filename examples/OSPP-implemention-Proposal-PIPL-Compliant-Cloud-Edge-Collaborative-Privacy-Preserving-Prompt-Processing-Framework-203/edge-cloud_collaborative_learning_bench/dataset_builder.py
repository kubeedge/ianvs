#!/usr/bin/env python3
"""
ChnSentiCorp-Lite数据集构建器
根据PR方案构建PIPL合规的云边协同LLM基准数据集
"""

import json
import random
import re
import time
import uuid
from typing import Dict, List, Any, Tuple
import numpy as np
from pathlib import Path

class ChnSentiCorpLiteBuilder:
    """ChnSentiCorp-Lite数据集构建器"""
    
    def __init__(self):
        self.dataset_size = 3000
        self.train_size = 2000
        self.val_size = 500
        self.test_size = 500
        
        # 基础情感文本模板
        self.base_templates = {
            "positive": [
                "这个产品真的很不错，我很满意。",
                "质量很好，价格也合理，推荐购买。",
                "服务态度很好，处理问题很及时。",
                "这个软件功能很强大，推荐使用。",
                "性价比很高，值得购买。",
                "整体来说比较满意，会继续使用。",
                "这家餐厅的服务真的很不错",
                "产品质量超出预期，非常满意。",
                "客服响应很快，问题解决得很及时。",
                "这个品牌值得信赖，推荐给大家。"
            ],
            "negative": [
                "这个产品真的很糟糕，完全不推荐。",
                "质量很差，价格还贵，不值得购买。",
                "服务态度很差，处理问题很慢。",
                "这个软件有很多bug，不推荐使用。",
                "性价比很低，不值得购买。",
                "整体来说很不满意，不会再使用。",
                "这家餐厅的服务真的很差",
                "产品质量很差，非常失望。",
                "客服响应很慢，问题解决得很不及时。",
                "这个品牌不值得信赖，不推荐给大家。"
            ]
        }
        
        # 合成PII模板
        self.pii_templates = {
            "complaint_with_contact": [
                "请联系{person_name}，电话是{phone}处理订单问题",
                "用户{person_name}，电话{phone}，对这个产品很满意",
                "张三觉得这个服务很糟糕，完全不推荐",
                "李四的体验很差，客服态度不好",
                "王五对这次购买很失望"
            ],
            "order_with_personal_info": [
                "订单号123456，收件人{person_name}，电话{phone}",
                "用户{person_name}，身份证{id_card}，电话{phone}",
                "客户{person_name}，邮箱{email}，电话{phone}",
                "联系人{person_name}，地址{address}，电话{phone}"
            ],
            "service_with_contact": [
                "请联系{person_name}，电话{phone}，邮箱{email}",
                "用户{person_name}，电话{phone}，反映产品质量问题",
                "客户{person_name}，电话{phone}，投诉服务态度"
            ]
        }
        
        # 中文姓名库
        self.names = [
            "张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
            "郑十一", "王十二", "冯十三", "陈十四", "褚十五", "卫十六",
            "蒋十七", "沈十八", "韩十九", "杨二十", "朱二十一", "秦二十二"
        ]
        
        # 电话号码模板
        self.phone_templates = [
            "138{phone_suffix}", "139{phone_suffix}", "150{phone_suffix}",
            "151{phone_suffix}", "152{phone_suffix}", "188{phone_suffix}"
        ]
        
        # 邮箱模板
        self.email_templates = [
            "{name}@example.com", "{name}@company.com", "{name}@gmail.com"
        ]
        
        # 身份证模板
        self.id_card_templates = [
            "11010119900101{id_suffix}", "32010219900101{id_suffix}",
            "44010319900101{id_suffix}", "51010419900101{id_suffix}"
        ]
        
        # 地址模板
        self.address_templates = [
            "北京市朝阳区建国路88号",
            "上海市浦东新区陆家嘴金融贸易区",
            "广州市天河区珠江新城",
            "深圳市南山区科技园"
        ]
    
    def generate_phone_number(self) -> str:
        """生成电话号码"""
        template = random.choice(self.phone_templates)
        suffix = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        return template.format(phone_suffix=suffix)
    
    def generate_email(self, name: str) -> str:
        """生成邮箱地址"""
        template = random.choice(self.email_templates)
        return template.format(name=name.lower())
    
    def generate_id_card(self) -> str:
        """生成身份证号码"""
        template = random.choice(self.id_card_templates)
        suffix = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        return template.format(id_suffix=suffix)
    
    def generate_address(self) -> str:
        """生成地址"""
        return random.choice(self.address_templates)
    
    def generate_synthetic_pii(self, template_type: str) -> Dict[str, str]:
        """生成合成PII数据"""
        name = random.choice(self.names)
        
        pii_data = {
            "person_name": name,
            "phone": self.generate_phone_number(),
            "email": self.generate_email(name),
            "id_card": self.generate_id_card(),
            "address": self.generate_address()
        }
        
        return pii_data
    
    def detect_pii_entities(self, text: str) -> List[str]:
        """检测PII实体类型"""
        entities = []
        
        # 检测姓名
        if any(name in text for name in self.names):
            entities.append("PERSON")
        
        # 检测电话
        if re.search(r'1[3-9]\d{9}', text):
            entities.append("PHONE")
        
        # 检测邮箱
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            entities.append("EMAIL")
        
        # 检测身份证
        if re.search(r'\d{17}[\dXx]', text):
            entities.append("ID_CARD")
        
        return entities
    
    def calculate_privacy_budget_cost(self, privacy_level: str, pii_entities: List[str]) -> float:
        """计算隐私预算成本"""
        base_cost = 0.0
        
        if privacy_level == "general":
            base_cost = 0.0
        elif privacy_level == "high_sensitivity":
            base_cost = 1.0
        
        # 根据PII实体类型增加成本
        entity_costs = {
            "PERSON": 0.2,
            "PHONE": 0.3,
            "EMAIL": 0.2,
            "ID_CARD": 0.5
        }
        
        for entity in pii_entities:
            base_cost += entity_costs.get(entity, 0.1)
        
        return round(base_cost, 2)
    
    def generate_sample(self, sample_id: str, is_high_sensitivity: bool = False) -> Dict[str, Any]:
        """生成单个样本"""
        # 选择情感标签
        sentiment = random.choice(["positive", "negative"])
        
        if is_high_sensitivity:
            # 生成高敏感度样本
            template_type = random.choice(list(self.pii_templates.keys()))
            template = random.choice(self.pii_templates[template_type])
            
            # 生成合成PII
            pii_data = self.generate_synthetic_pii(template_type)
            
            # 替换模板中的占位符
            text = template.format(**pii_data)
            
            # 检测PII实体
            pii_entities = self.detect_pii_entities(text)
            
            # 计算隐私预算成本
            privacy_budget_cost = self.calculate_privacy_budget_cost("high_sensitivity", pii_entities)
            
            sample = {
                "sample_id": sample_id,
                "text": text,
                "label": sentiment,
                "privacy_level": "high_sensitivity",
                "pii_entities": pii_entities,
                "pipl_cross_border": False,  # 高敏感度不允许跨境
                "synthetic_pii": pii_data,
                "privacy_budget_cost": privacy_budget_cost,
                "metadata": {
                    "source": "synthetic_generation",
                    "domain": "customer_service",
                    "length": len(text),
                    "mia_test_subset": True
                }
            }
        else:
            # 生成通用敏感度样本
            text = random.choice(self.base_templates[sentiment])
            
            # 检测PII实体
            pii_entities = self.detect_pii_entities(text)
            
            # 计算隐私预算成本
            privacy_budget_cost = self.calculate_privacy_budget_cost("general", pii_entities)
            
            sample = {
                "sample_id": sample_id,
                "text": text,
                "label": sentiment,
                "privacy_level": "general",
                "pii_entities": pii_entities,
                "pipl_cross_border": True,  # 通用敏感度允许跨境
                "synthetic_pii": None,
                "privacy_budget_cost": privacy_budget_cost,
                "metadata": {
                    "source": "ChnSentiCorp",
                    "domain": "restaurant_review",
                    "length": len(text),
                    "mia_test_subset": False
                }
            }
        
        return sample
    
    def generate_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """生成完整数据集"""
        print("开始生成ChnSentiCorp-Lite数据集...")
        
        # 生成训练集
        print("生成训练集...")
        train_data = []
        for i in range(self.train_size):
            sample_id = f"chnsc_{i+1:06d}"
            is_high_sensitivity = random.random() < 0.3  # 30%高敏感度
            sample = self.generate_sample(sample_id, is_high_sensitivity)
            train_data.append(sample)
        
        # 生成验证集
        print("生成验证集...")
        val_data = []
        for i in range(self.val_size):
            sample_id = f"chnsc_{i+2001:06d}"
            is_high_sensitivity = random.random() < 0.3  # 30%高敏感度
            sample = self.generate_sample(sample_id, is_high_sensitivity)
            val_data.append(sample)
        
        # 生成测试集
        print("生成测试集...")
        test_data = []
        for i in range(self.test_size):
            sample_id = f"chnsc_{i+2501:06d}"
            is_high_sensitivity = random.random() < 0.3  # 30%高敏感度
            sample = self.generate_sample(sample_id, is_high_sensitivity)
            test_data.append(sample)
        
        print("数据集生成完成！")
        return train_data, val_data, test_data
    
    def save_dataset(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], output_dir: str = "./data/chnsenticorp_lite"):
        """保存数据集到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"保存数据集到 {output_path}...")
        
        # 保存训练集
        with open(output_path / "train.jsonl", "w", encoding="utf-8") as f:
            for sample in train_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        # 保存验证集
        with open(output_path / "val.jsonl", "w", encoding="utf-8") as f:
            for sample in val_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        # 保存测试集
        with open(output_path / "test.jsonl", "w", encoding="utf-8") as f:
            for sample in test_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print("数据集保存完成！")
    
    def generate_dataset_statistics(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """生成数据集统计信息"""
        all_data = train_data + val_data + test_data
        
        # 统计隐私级别分布
        privacy_levels = {}
        for sample in all_data:
            level = sample["privacy_level"]
            privacy_levels[level] = privacy_levels.get(level, 0) + 1
        
        # 统计PII实体分布
        pii_entities = {}
        for sample in all_data:
            for entity in sample["pii_entities"]:
                pii_entities[entity] = pii_entities.get(entity, 0) + 1
        
        # 统计标签分布
        labels = {}
        for sample in all_data:
            label = sample["label"]
            labels[label] = labels.get(label, 0) + 1
        
        # 统计跨境传输分布
        cross_border = {}
        for sample in all_data:
            cb = sample["pipl_cross_border"]
            cross_border[str(cb)] = cross_border.get(str(cb), 0) + 1
        
        # 统计隐私预算成本
        budget_costs = [sample["privacy_budget_cost"] for sample in all_data]
        
        statistics = {
            "total_samples": len(all_data),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "privacy_levels": privacy_levels,
            "pii_entities": pii_entities,
            "labels": labels,
            "cross_border": cross_border,
            "privacy_budget_stats": {
                "mean": round(np.mean(budget_costs), 3),
                "std": round(np.std(budget_costs), 3),
                "min": round(np.min(budget_costs), 3),
                "max": round(np.max(budget_costs), 3)
            }
        }
        
        return statistics
    
    def print_dataset_info(self, statistics: Dict[str, Any]):
        """打印数据集信息"""
        print("\n" + "="*60)
        print("ChnSentiCorp-Lite数据集统计信息")
        print("="*60)
        
        print(f"总样本数: {statistics['total_samples']}")
        print(f"训练集: {statistics['train_samples']}")
        print(f"验证集: {statistics['val_samples']}")
        print(f"测试集: {statistics['test_samples']}")
        
        print(f"\n隐私级别分布:")
        for level, count in statistics['privacy_levels'].items():
            print(f"  {level}: {count}")
        
        print(f"\nPII实体分布:")
        for entity, count in statistics['pii_entities'].items():
            print(f"  {entity}: {count}")
        
        print(f"\n标签分布:")
        for label, count in statistics['labels'].items():
            print(f"  {label}: {count}")
        
        print(f"\n跨境传输分布:")
        for cb, count in statistics['cross_border'].items():
            print(f"  {cb}: {count}")
        
        print(f"\n隐私预算成本统计:")
        budget_stats = statistics['privacy_budget_stats']
        print(f"  均值: {budget_stats['mean']}")
        print(f"  标准差: {budget_stats['std']}")
        print(f"  最小值: {budget_stats['min']}")
        print(f"  最大值: {budget_stats['max']}")
        
        print("="*60)

def main():
    """主函数"""
    print("ChnSentiCorp-Lite数据集构建器")
    print("根据PR方案构建PIPL合规的云边协同LLM基准数据集")
    print("="*60)
    
    # 创建数据集构建器
    builder = ChnSentiCorpLiteBuilder()
    
    # 生成数据集
    train_data, val_data, test_data = builder.generate_dataset()
    
    # 保存数据集
    builder.save_dataset(train_data, val_data, test_data)
    
    # 生成统计信息
    statistics = builder.generate_dataset_statistics(train_data, val_data, test_data)
    
    # 打印统计信息
    builder.print_dataset_info(statistics)
    
    # 保存统计信息
    with open("./data/chnsenticorp_lite/statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    print("\n数据集构建完成！")
    print("数据集文件:")
    print("  - train.jsonl: 训练集")
    print("  - val.jsonl: 验证集")
    print("  - test.jsonl: 测试集")
    print("  - statistics.json: 统计信息")

if __name__ == "__main__":
    main()
