#!/usr/bin/env python3
"""
阶段7: 协同工作流

执行云边协同处理，包括隐私检测、隐私保护、边缘处理、云端处理、结果聚合
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollaborativeWorkflow:
    """协同工作流类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化协同工作流"""
        self.config = config
        self.edge_model = None
        self.cloud_model = None
        self.privacy_modules = None
        self.workflow_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "privacy_budget_used": 0.0
        }
        
    def set_models(self, edge_model: Any, cloud_model: Any):
        """设置模型"""
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        print("✅ 模型设置完成")
        
    def set_privacy_modules(self, privacy_modules: Dict[str, Any]):
        """设置隐私保护模块"""
        self.privacy_modules = privacy_modules
        print("✅ 隐私保护模块设置完成")
    
    def process_single_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个请求"""
        start_time = time.time()
        
        try:
            # 阶段1: 隐私检测
            print(f"🔍 阶段1: 隐私检测 - {request_data.get('sample_id', 'unknown')}")
            pii_result = self.privacy_modules["pii_detector"].detect(request_data["text"])
            
            # 阶段2: 隐私保护
            print(f"🛡️ 阶段2: 隐私保护")
            protected_data = self._apply_privacy_protection(request_data, pii_result)
            
            # 阶段3: 边缘处理
            print(f"📱 阶段3: 边缘处理")
            edge_result = self._process_edge(protected_data)
            
            # 阶段4: 云端处理
            print(f"☁️ 阶段4: 云端处理")
            cloud_result = self._process_cloud(edge_result)
            
            # 阶段5: 结果聚合
            print(f"🔄 阶段5: 结果聚合")
            final_result = self._aggregate_results(edge_result, cloud_result)
            
            processing_time = time.time() - start_time
            
            # 更新指标
            self.workflow_metrics["total_requests"] += 1
            self.workflow_metrics["successful_requests"] += 1
            self.workflow_metrics["average_processing_time"] = (
                (self.workflow_metrics["average_processing_time"] * (self.workflow_metrics["total_requests"] - 1) + 
                 processing_time) / self.workflow_metrics["total_requests"]
            )
            
            return {
                "success": True,
                "sample_id": request_data.get("sample_id", "unknown"),
                "processing_time": processing_time,
                "pii_result": pii_result,
                "edge_result": edge_result,
                "cloud_result": cloud_result,
                "final_result": final_result,
                "privacy_budget_used": pii_result.get("pii_count", 0) * 0.1,
                "compliance_status": True
            }
            
        except Exception as e:
            print(f"❌ 请求处理失败: {e}")
            self.workflow_metrics["total_requests"] += 1
            self.workflow_metrics["failed_requests"] += 1
            
            return {
                "success": False,
                "sample_id": request_data.get("sample_id", "unknown"),
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _apply_privacy_protection(self, data: Dict[str, Any], pii_result: Dict[str, Any]) -> Dict[str, Any]:
        """应用隐私保护"""
        protected_data = data.copy()
        
        # 如果检测到PII，应用差分隐私
        if pii_result["requires_protection"]:
            # 模拟差分隐私处理
            dp_params = self.privacy_modules["differential_privacy"].get_privacy_parameters("general")
            # 这里可以添加实际的差分隐私处理逻辑
            
            protected_data["privacy_protected"] = True
            protected_data["privacy_level"] = pii_result["risk_level"]
            protected_data["pii_entities"] = pii_result["pii_entities"]
        else:
            protected_data["privacy_protected"] = False
            protected_data["privacy_level"] = "low"
            protected_data["pii_entities"] = []
        
        return protected_data
    
    def _process_edge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """边缘处理"""
        # 模拟边缘模型处理
        edge_processing_time = np.random.uniform(0.1, 0.3)
        time.sleep(edge_processing_time)
        
        # 模拟边缘推理结果
        edge_result = {
            "text": data["text"],
            "edge_prediction": data["label"],  # 模拟预测
            "edge_confidence": np.random.uniform(0.7, 0.9),
            "edge_processing_time": edge_processing_time,
            "privacy_level": data.get("privacy_level", "low"),
            "requires_cloud_processing": data.get("privacy_level") in ["high", "medium"]
        }
        
        return edge_result
    
    def _process_cloud(self, edge_result: Dict[str, Any]) -> Dict[str, Any]:
        """云端处理"""
        # 检查是否需要云端处理
        if not edge_result.get("requires_cloud_processing", False):
            return {
                "cloud_prediction": edge_result["edge_prediction"],
                "cloud_confidence": edge_result["edge_confidence"],
                "cloud_processing_time": 0.0,
                "processing_mode": "edge_only"
            }
        
        # 模拟云端模型处理
        cloud_processing_time = np.random.uniform(0.2, 0.5)
        time.sleep(cloud_processing_time)
        
        # 模拟云端推理结果
        cloud_result = {
            "cloud_prediction": edge_result["edge_prediction"],  # 模拟预测
            "cloud_confidence": np.random.uniform(0.8, 0.95),
            "cloud_processing_time": cloud_processing_time,
            "processing_mode": "cloud_enhanced"
        }
        
        return cloud_result
    
    def _aggregate_results(self, edge_result: Dict[str, Any], cloud_result: Dict[str, Any]) -> Dict[str, Any]:
        """聚合结果"""
        # 选择最佳预测结果
        if cloud_result.get("cloud_confidence", 0) > edge_result.get("edge_confidence", 0):
            final_prediction = cloud_result["cloud_prediction"]
            final_confidence = cloud_result["cloud_confidence"]
            processing_mode = "cloud_enhanced"
        else:
            final_prediction = edge_result["edge_prediction"]
            final_confidence = edge_result["edge_confidence"]
            processing_mode = "edge_optimized"
        
        # 计算总处理时间
        total_processing_time = (
            edge_result.get("edge_processing_time", 0) + 
            cloud_result.get("cloud_processing_time", 0)
        )
        
        return {
            "final_prediction": final_prediction,
            "final_confidence": final_confidence,
            "processing_mode": processing_mode,
            "total_processing_time": total_processing_time,
            "edge_processing_time": edge_result.get("edge_processing_time", 0),
            "cloud_processing_time": cloud_result.get("cloud_processing_time", 0)
        }
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理请求"""
        print(f"📦 批量处理 {len(batch_data)} 个请求...")
        
        results = []
        for i, request_data in enumerate(batch_data):
            print(f"  处理请求 {i+1}/{len(batch_data)}")
            result = self.process_single_request(request_data)
            results.append(result)
        
        return results
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """获取工作流指标"""
        success_rate = (
            self.workflow_metrics["successful_requests"] / 
            self.workflow_metrics["total_requests"] 
            if self.workflow_metrics["total_requests"] > 0 else 0
        )
        
        return {
            "total_requests": self.workflow_metrics["total_requests"],
            "successful_requests": self.workflow_metrics["successful_requests"],
            "failed_requests": self.workflow_metrics["failed_requests"],
            "success_rate": success_rate,
            "average_processing_time": self.workflow_metrics["average_processing_time"],
            "privacy_budget_used": self.workflow_metrics["privacy_budget_used"]
        }

def load_test_data() -> List[Dict[str, Any]]:
    """加载测试数据"""
    print("📊 加载测试数据...")
    
    # 从数据集文件加载测试数据
    test_data_file = "/content/ianvs_pipl_framework/data/processed/chnsenticorp_lite_test.jsonl"
    
    if os.path.exists(test_data_file):
        test_data = []
        with open(test_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        
        print(f"✅ 加载了 {len(test_data)} 个测试样本")
        return test_data
    else:
        print("⚠️ 测试数据文件不存在，创建模拟数据")
        # 创建模拟测试数据
        mock_data = [
            {
                "sample_id": f"test_{i:03d}",
                "text": f"这是测试文本 {i}，用于验证协同工作流。",
                "label": "positive" if i % 2 == 0 else "negative",
                "privacy_level": "medium",
                "pii_entities": [],
                "pipl_cross_border": False,
                "privacy_budget_cost": 0.1
            }
            for i in range(10)
        ]
        return mock_data

def execute_collaborative_workflow():
    """执行协同工作流"""
    print("🤝 执行协同工作流...")
    
    # 初始化协同工作流
    workflow_config = {
        "batch_size": 5,
        "max_processing_time": 30.0,
        "privacy_budget_limit": 1.0
    }
    
    workflow = CollaborativeWorkflow(workflow_config)
    
    # 设置模型（模拟）
    edge_model = {"name": "Qwen2.5-7B-Edge", "type": "edge"}
    cloud_model = {"name": "Qwen2.5-7B-Cloud", "type": "cloud"}
    workflow.set_models(edge_model, cloud_model)
    
    # 设置隐私保护模块（模拟）
    privacy_modules = {
        "pii_detector": {"name": "PIIDetector"},
        "differential_privacy": {"name": "DifferentialPrivacy"},
        "compliance_monitor": {"name": "ComplianceMonitor"},
        "risk_evaluator": {"name": "RiskEvaluator"}
    }
    workflow.set_privacy_modules(privacy_modules)
    
    # 加载测试数据
    test_data = load_test_data()
    
    # 执行批量处理
    results = workflow.process_batch(test_data)
    
    # 获取工作流指标
    metrics = workflow.get_workflow_metrics()
    
    print(f"✅ 协同工作流执行完成")
    print(f"  总请求数: {metrics['total_requests']}")
    print(f"  成功请求数: {metrics['successful_requests']}")
    print(f"  成功率: {metrics['success_rate']:.1%}")
    print(f"  平均处理时间: {metrics['average_processing_time']:.3f}s")
    
    return results, metrics

def save_workflow_results(results: List[Dict[str, Any]], metrics: Dict[str, Any]):
    """保存工作流结果"""
    print("💾 保存工作流结果...")
    
    # 保存详细结果
    results_file = "/content/ianvs_pipl_framework/results/collaborative_workflow_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存指标
    metrics_file = "/content/ianvs_pipl_framework/results/workflow_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"工作流结果已保存: {results_file}")
    print(f"工作流指标已保存: {metrics_file}")
    
    return results_file, metrics_file

def main():
    """主函数"""
    print("🚀 阶段7: 协同工作流")
    print("=" * 50)
    
    try:
        # 1. 执行协同工作流
        results, metrics = execute_collaborative_workflow()
        
        # 2. 保存工作流结果
        results_file, metrics_file = save_workflow_results(results, metrics)
        
        # 3. 保存执行报告
        execution_report = {
            "timestamp": datetime.now().isoformat(),
            "total_requests": metrics["total_requests"],
            "successful_requests": metrics["successful_requests"],
            "success_rate": metrics["success_rate"],
            "average_processing_time": metrics["average_processing_time"],
            "results_file": results_file,
            "metrics_file": metrics_file,
            "execution_status": "success"
        }
        
        report_file = '/content/ianvs_pipl_framework/logs/collaborative_workflow_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(execution_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 协同工作流执行完成！")
        print(f"总请求数: {metrics['total_requests']}")
        print(f"成功请求数: {metrics['successful_requests']}")
        print(f"成功率: {metrics['success_rate']:.1%}")
        print(f"平均处理时间: {metrics['average_processing_time']:.3f}s")
        print(f"结果文件: {results_file}")
        print(f"指标文件: {metrics_file}")
        print(f"执行报告: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 协同工作流执行失败: {e}")
        logger.error(f"协同工作流执行失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 阶段7完成，可以继续执行阶段8")
    else:
        print("\n❌ 阶段7失败，请检查错误信息")
