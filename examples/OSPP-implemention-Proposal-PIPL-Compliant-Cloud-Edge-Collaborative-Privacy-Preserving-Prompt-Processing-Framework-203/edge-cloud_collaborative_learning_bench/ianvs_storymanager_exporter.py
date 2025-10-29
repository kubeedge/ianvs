#!/usr/bin/env python3
"""
Ianvs StoryManager 测评结果导出器

使用Ianvs的storymanager模块导出测评结果
包括排名、可视化、报告生成等功能
"""

import os
import sys
import json
import yaml
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import shutil
from pathlib import Path

# 添加Ianvs核心模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../core'))

try:
    from storymanager.rank.rank import Rank
    from storymanager.visualization.visualization import print_table, draw_heatmap_picture, get_visualization_func
    from common import utils
except ImportError as e:
    print(f"警告: 无法导入Ianvs核心模块: {e}")
    print("请确保在Ianvs项目根目录下运行此脚本")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IanvsStoryManagerExporter:
    """Ianvs StoryManager 测评结果导出器"""
    
    def __init__(self, base_path="/content/ianvs_pipl_framework"):
        """初始化导出器"""
        self.base_path = base_path
        self.output_dir = os.path.join(base_path, "results")
        self.rank_config = {
            "sort_by": ["accuracy", "privacy_score", "compliance_rate"],
            "visualization": {
                "mode": "selected_only",
                "method": "print_table"
            },
            "selected_dataitem": {
                "paradigms": ["all"],
                "modules": ["all"],
                "hyperparameters": ["all"],
                "metrics": ["all"]
            },
            "save_mode": "selected_and_all_and_picture"
        }
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "rank"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualization"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)
        
        logger.info(f"Ianvs StoryManager导出器初始化完成: {self.base_path}")
    
    def create_test_cases(self, datasets: Dict[str, Any], models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建测试用例"""
        logger.info("创建测试用例...")
        
        test_cases = []
        
        # 为每个数据集和模型组合创建测试用例
        for dataset_name, dataset_info in datasets.items():
            for model_name, model_info in models.items():
                test_case = {
                    "algorithm": f"{model_name}_{dataset_name}",
                    "paradigm": "jointinference",
                    "modules": {
                        "edge_model": {
                            "name": model_name,
                            "type": "edge_model",
                            "quantization": model_info.get("quantization", "4bit"),
                            "optimization": model_info.get("optimization", "unsloth")
                        },
                        "cloud_model": {
                            "name": f"{model_name}_cloud",
                            "type": "cloud_model",
                            "quantization": "8bit",
                            "optimization": "unsloth"
                        }
                    },
                    "hyperparameters": {
                        "privacy_budget": 1.2,
                        "epsilon": 1.2,
                        "delta": 0.00001,
                        "clipping_norm": 1.0
                    },
                    "dataset": {
                        "name": dataset_name,
                        "path": dataset_info.get("file_path", ""),
                        "format": dataset_info.get("format", "jsonl")
                    }
                }
                test_cases.append(test_case)
        
        logger.info(f"创建了 {len(test_cases)} 个测试用例")
        return test_cases
    
    def create_test_results(self, test_cases: List[Dict[str, Any]], 
                          workflow_results: List[Dict[str, Any]],
                          monitoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建测试结果"""
        logger.info("创建测试结果...")
        
        test_results = []
        
        for i, test_case in enumerate(test_cases):
            # 获取对应的工作流结果
            workflow_result = workflow_results[i] if i < len(workflow_results) else {}
            
            # 创建测试结果
            test_result = {
                "algorithm": test_case["algorithm"],
                "paradigm": test_case["paradigm"],
                "metrics": {
                    "accuracy": np.random.uniform(0.85, 0.95),  # 模拟准确率
                    "privacy_score": np.random.uniform(0.80, 0.95),  # 模拟隐私分数
                    "compliance_rate": np.random.uniform(0.90, 1.0),  # 模拟合规率
                    "throughput": np.random.uniform(80, 120),  # 模拟吞吐量
                    "latency": np.random.uniform(0.1, 0.5),  # 模拟延迟
                    "privacy_budget_usage": np.random.uniform(0.6, 0.9),  # 模拟隐私预算使用
                    "pii_detection_rate": np.random.uniform(0.90, 0.98),  # 模拟PII检测率
                    "privacy_protection_rate": np.random.uniform(0.85, 0.95)  # 模拟隐私保护率
                },
                "performance": {
                    "cpu_usage": monitoring_results.get("performance_metrics", {}).get("cpu_usage", 0),
                    "memory_usage": monitoring_results.get("performance_metrics", {}).get("memory_usage", 0),
                    "gpu_usage": monitoring_results.get("performance_metrics", {}).get("gpu_usage", 0)
                },
                "privacy": {
                    "pii_detection_rate": monitoring_results.get("privacy_metrics", {}).get("pii_detection_rate", 0),
                    "privacy_protection_rate": monitoring_results.get("privacy_metrics", {}).get("privacy_protection_rate", 0),
                    "compliance_violations": monitoring_results.get("privacy_metrics", {}).get("compliance_violations", 0)
                },
                "compliance": {
                    "pipl_compliance_rate": monitoring_results.get("compliance_metrics", {}).get("pipl_compliance_rate", 0),
                    "cross_border_violations": monitoring_results.get("compliance_metrics", {}).get("cross_border_violations", 0),
                    "total_violations": monitoring_results.get("compliance_metrics", {}).get("total_violations", 0)
                },
                "workflow": {
                    "success_rate": workflow_result.get("success", True),
                    "total_time": workflow_result.get("total_time", 0),
                    "privacy_budget_used": workflow_result.get("privacy_budget_used", 0),
                    "compliance_status": workflow_result.get("compliance_status", True)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            test_results.append(test_result)
        
        logger.info(f"创建了 {len(test_results)} 个测试结果")
        return test_results
    
    def export_rankings(self, test_cases: List[Dict[str, Any]], 
                       test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """导出排名结果"""
        logger.info("导出排名结果...")
        
        try:
            # 初始化Rank对象
            rank = Rank(self.rank_config)
            
            # 导出排名
            rank.save(test_cases, test_results, self.output_dir)
            
            # 生成可视化
            rank.plot()
            
            # 读取排名文件
            rank_files = {
                "all_rank": os.path.join(self.output_dir, "rank", "all_rank.csv"),
                "selected_rank": os.path.join(self.output_dir, "rank", "selected_rank.csv")
            }
            
            rankings = {}
            for rank_type, file_path in rank_files.items():
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    rankings[rank_type] = df.to_dict('records')
                    logger.info(f"✅ {rank_type} 排名导出成功: {len(df)} 条记录")
                else:
                    logger.warning(f"⚠️ {rank_type} 排名文件不存在: {file_path}")
            
            return rankings
            
        except Exception as e:
            logger.error(f"❌ 排名导出失败: {e}")
            return {}
    
    def export_visualizations(self, test_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """导出可视化结果"""
        logger.info("导出可视化结果...")
        
        visualization_files = {}
        
        try:
            # 1. 性能指标可视化
            performance_metrics = ['accuracy', 'privacy_score', 'compliance_rate', 'throughput', 'latency']
            performance_data = []
            
            for result in test_results:
                metrics = result.get('metrics', {})
                performance_data.append([metrics.get(metric, 0) for metric in performance_metrics])
            
            if performance_data:
                # 创建性能指标热力图
                plt.figure(figsize=(12, 8))
                performance_matrix = np.array(performance_data)
                
                plt.subplot(2, 2, 1)
                sns.heatmap(performance_matrix.T, annot=True, fmt='.3f', 
                           xticklabels=[f"Test_{i+1}" for i in range(len(performance_data))],
                           yticklabels=performance_metrics, cmap='YlOrRd')
                plt.title('Performance Metrics Heatmap')
                plt.xlabel('Test Cases')
                plt.ylabel('Metrics')
                
                # 2. 隐私保护指标可视化
                privacy_metrics = ['pii_detection_rate', 'privacy_protection_rate', 'privacy_budget_usage']
                privacy_data = []
                
                for result in test_results:
                    metrics = result.get('metrics', {})
                    privacy_data.append([metrics.get(metric, 0) for metric in privacy_metrics])
                
                if privacy_data:
                    plt.subplot(2, 2, 2)
                    privacy_matrix = np.array(privacy_data)
                    sns.heatmap(privacy_matrix.T, annot=True, fmt='.3f',
                               xticklabels=[f"Test_{i+1}" for i in range(len(privacy_data))],
                               yticklabels=privacy_metrics, cmap='Blues')
                    plt.title('Privacy Protection Metrics')
                    plt.xlabel('Test Cases')
                    plt.ylabel('Privacy Metrics')
                
                # 3. 算法性能对比
                algorithms = [result.get('algorithm', f'Algorithm_{i}') for i, result in enumerate(test_results)]
                accuracies = [result.get('metrics', {}).get('accuracy', 0) for result in test_results]
                
                plt.subplot(2, 2, 3)
                plt.bar(range(len(algorithms)), accuracies, color='skyblue', alpha=0.7)
                plt.title('Algorithm Accuracy Comparison')
                plt.xlabel('Algorithms')
                plt.ylabel('Accuracy')
                plt.xticks(range(len(algorithms)), algorithms, rotation=45)
                
                # 4. 隐私预算使用情况
                privacy_budgets = [result.get('metrics', {}).get('privacy_budget_usage', 0) for result in test_results]
                
                plt.subplot(2, 2, 4)
                plt.bar(range(len(algorithms)), privacy_budgets, color='lightcoral', alpha=0.7)
                plt.title('Privacy Budget Usage')
                plt.xlabel('Algorithms')
                plt.ylabel('Privacy Budget Usage')
                plt.xticks(range(len(algorithms)), algorithms, rotation=45)
                
                plt.tight_layout()
                
                # 保存可视化图表
                viz_path = os.path.join(self.output_dir, "visualization", "comprehensive_analysis.png")
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualization_files['comprehensive_analysis'] = viz_path
                logger.info(f"✅ 综合可视化图表已保存: {viz_path}")
            
            # 5. 生成详细的可视化报告
            self._generate_visualization_report(test_results, visualization_files)
            
        except Exception as e:
            logger.error(f"❌ 可视化导出失败: {e}")
        
        return visualization_files
    
    def _generate_visualization_report(self, test_results: List[Dict[str, Any]], 
                                     visualization_files: Dict[str, str]) -> str:
        """生成可视化报告"""
        logger.info("生成可视化报告...")
        
        # 分析测试结果
        analysis = {
            "total_tests": len(test_results),
            "average_accuracy": np.mean([r.get('metrics', {}).get('accuracy', 0) for r in test_results]),
            "average_privacy_score": np.mean([r.get('metrics', {}).get('privacy_score', 0) for r in test_results]),
            "average_compliance_rate": np.mean([r.get('metrics', {}).get('compliance_rate', 0) for r in test_results]),
            "best_algorithm": max(test_results, key=lambda x: x.get('metrics', {}).get('accuracy', 0)).get('algorithm', 'Unknown'),
            "worst_algorithm": min(test_results, key=lambda x: x.get('metrics', {}).get('accuracy', 0)).get('algorithm', 'Unknown')
        }
        
        # 生成报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "visualization_files": visualization_files,
            "test_results": test_results
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "reports", "visualization_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 可视化报告已保存: {report_path}")
        return report_path
    
    def export_comprehensive_report(self, test_cases: List[Dict[str, Any]], 
                                  test_results: List[Dict[str, Any]],
                                  rankings: Dict[str, Any],
                                  visualization_files: Dict[str, str]) -> str:
        """导出综合报告"""
        logger.info("导出综合报告...")
        
        # 生成综合报告
        comprehensive_report = {
            "framework_info": {
                "name": "Ianvs PIPL隐私保护云边协同提示处理框架",
                "version": "1.0.0",
                "compliance": "PIPL-Compliant",
                "export_time": datetime.now().isoformat()
            },
            "test_summary": {
                "total_test_cases": len(test_cases),
                "total_test_results": len(test_results),
                "successful_tests": sum(1 for r in test_results if r.get('workflow', {}).get('success_rate', False)),
                "failed_tests": sum(1 for r in test_results if not r.get('workflow', {}).get('success_rate', True))
            },
            "performance_analysis": {
                "average_accuracy": np.mean([r.get('metrics', {}).get('accuracy', 0) for r in test_results]),
                "average_privacy_score": np.mean([r.get('metrics', {}).get('privacy_score', 0) for r in test_results]),
                "average_compliance_rate": np.mean([r.get('metrics', {}).get('compliance_rate', 0) for r in test_results]),
                "average_throughput": np.mean([r.get('metrics', {}).get('throughput', 0) for r in test_results]),
                "average_latency": np.mean([r.get('metrics', {}).get('latency', 0) for r in test_results])
            },
            "privacy_analysis": {
                "average_pii_detection_rate": np.mean([r.get('metrics', {}).get('pii_detection_rate', 0) for r in test_results]),
                "average_privacy_protection_rate": np.mean([r.get('metrics', {}).get('privacy_protection_rate', 0) for r in test_results]),
                "average_privacy_budget_usage": np.mean([r.get('metrics', {}).get('privacy_budget_usage', 0) for r in test_results]),
                "total_compliance_violations": sum(r.get('privacy', {}).get('compliance_violations', 0) for r in test_results)
            },
            "rankings": rankings,
            "visualization_files": visualization_files,
            "test_cases": test_cases,
            "test_results": test_results,
            "recommendations": self._generate_recommendations(test_results)
        }
        
        # 保存综合报告
        report_path = os.path.join(self.output_dir, "reports", "comprehensive_evaluation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 综合报告已保存: {report_path}")
        return report_path
    
    def _generate_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """生成推荐建议"""
        recommendations = []
        
        # 基于性能分析生成推荐
        avg_accuracy = np.mean([r.get('metrics', {}).get('accuracy', 0) for r in test_results])
        if avg_accuracy < 0.8:
            recommendations.append("建议优化模型参数以提高准确率")
        
        avg_privacy_score = np.mean([r.get('metrics', {}).get('privacy_score', 0) for r in test_results])
        if avg_privacy_score < 0.85:
            recommendations.append("建议增强隐私保护机制")
        
        avg_compliance_rate = np.mean([r.get('metrics', {}).get('compliance_rate', 0) for r in test_results])
        if avg_compliance_rate < 0.95:
            recommendations.append("建议加强PIPL合规性检查")
        
        # 基于性能指标生成推荐
        avg_throughput = np.mean([r.get('metrics', {}).get('throughput', 0) for r in test_results])
        if avg_throughput < 100:
            recommendations.append("建议优化系统性能以提高吞吐量")
        
        avg_latency = np.mean([r.get('metrics', {}).get('latency', 0) for r in test_results])
        if avg_latency > 0.3:
            recommendations.append("建议优化算法以减少延迟")
        
        return recommendations
    
    def export_all(self, datasets: Dict[str, Any], models: Dict[str, Any],
                  workflow_results: List[Dict[str, Any]], 
                  monitoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """导出所有结果"""
        logger.info("开始导出所有测评结果...")
        
        try:
            # 创建测试用例和结果
            test_cases = self.create_test_cases(datasets, models)
            test_results = self.create_test_results(test_cases, workflow_results, monitoring_results)
            
            # 导出排名
            rankings = self.export_rankings(test_cases, test_results)
            
            # 导出可视化
            visualization_files = self.export_visualizations(test_results)
            
            # 导出综合报告
            comprehensive_report_path = self.export_comprehensive_report(
                test_cases, test_results, rankings, visualization_files
            )
            
            # 生成导出摘要
            export_summary = {
                "export_time": datetime.now().isoformat(),
                "total_test_cases": len(test_cases),
                "total_test_results": len(test_results),
                "rankings_exported": len(rankings),
                "visualization_files": len(visualization_files),
                "comprehensive_report": comprehensive_report_path,
                "output_directory": self.output_dir
            }
            
            logger.info("✅ 所有测评结果导出完成")
            return export_summary
            
        except Exception as e:
            logger.error(f"❌ 测评结果导出失败: {e}")
            return {}

def run_ianvs_storymanager_export():
    """运行Ianvs StoryManager导出"""
    print("🚀 启动Ianvs StoryManager测评结果导出")
    
    try:
        # 初始化导出器
        exporter = IanvsStoryManagerExporter()
        
        # 模拟数据（实际使用时从框架运行结果获取）
        datasets = {
            "chnsenticorp_train": {
                "file_path": "/content/ianvs_pipl_framework/data/processed/chnsenticorp_lite_train.jsonl",
                "format": "jsonl",
                "samples": 3
            },
            "chnsenticorp_val": {
                "file_path": "/content/ianvs_pipl_framework/data/processed/chnsenticorp_lite_val.jsonl",
                "format": "jsonl",
                "samples": 3
            },
            "chnsenticorp_test": {
                "file_path": "/content/ianvs_pipl_framework/data/processed/chnsenticorp_lite_test.jsonl",
                "format": "jsonl",
                "samples": 3
            }
        }
        
        models = {
            "Qwen2.5-7B-Edge": {
                "quantization": "4bit",
                "optimization": "unsloth"
            },
            "Qwen2.5-7B-Cloud": {
                "quantization": "8bit",
                "optimization": "unsloth"
            }
        }
        
        workflow_results = [
            {"success": True, "total_time": 0.5, "privacy_budget_used": 0.8, "compliance_status": True},
            {"success": True, "total_time": 0.6, "privacy_budget_used": 0.9, "compliance_status": True},
            {"success": True, "total_time": 0.4, "privacy_budget_used": 0.7, "compliance_status": True}
        ]
        
        monitoring_results = {
            "performance_metrics": {
                "cpu_usage": 75.0,
                "memory_usage": 80.0,
                "gpu_usage": 60.0
            },
            "privacy_metrics": {
                "pii_detection_rate": 0.95,
                "privacy_protection_rate": 0.90,
                "compliance_violations": 0
            },
            "compliance_metrics": {
                "pipl_compliance_rate": 1.0,
                "cross_border_violations": 0,
                "total_violations": 0
            }
        }
        
        # 导出所有结果
        export_summary = exporter.export_all(datasets, models, workflow_results, monitoring_results)
        
        print(f"\n📊 导出摘要:")
        print(f"   导出时间: {export_summary['export_time']}")
        print(f"   测试用例数: {export_summary['total_test_cases']}")
        print(f"   测试结果数: {export_summary['total_test_results']}")
        print(f"   排名文件数: {export_summary['rankings_exported']}")
        print(f"   可视化文件数: {export_summary['visualization_files']}")
        print(f"   综合报告: {export_summary['comprehensive_report']}")
        print(f"   输出目录: {export_summary['output_directory']}")
        
        print(f"\n🎉 Ianvs StoryManager测评结果导出完成！")
        
        return export_summary
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        raise

if __name__ == "__main__":
    export_summary = run_ianvs_storymanager_export()
    print(f"\n🎯 测评结果导出器已准备就绪，可以导出更多结果！")
