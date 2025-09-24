# Copyright 2024 New Government Agent Project
# 新版政府代理测试环境实现

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
from sedna.common.class_factory import ClassType, ClassFactory

def _calculate_vlm_score(predictions: List[Dict[str, Any]], 
                        ground_truths: List[Dict[str, Any]]) -> float:
    """计算VLM评分（10分制）"""
    try:
        if not predictions:
            return 0.0
        
        total_score = 0.0
        valid_predictions = 0
        predictions = predictions["results"]

        for pred in predictions:
            evaluation = pred['evaluation']
            if evaluation and 'score' in evaluation:
                score = evaluation['score']
                total_score += score
                valid_predictions += 1
        
        if valid_predictions == 0:
            return 0.0
        
        avg_score = total_score / valid_predictions
        return min(10.0, max(0.0, avg_score))
        
    except Exception as e:
        logging.warning(f"计算VLM评分失败: {str(e)}")
        return 0.0

@ClassFactory.register(ClassType.GENERAL, alias="score")
def score(ground_truths, predictions, metric_name: str = 'score') -> float:
    """
    新版政府代理准确度评估函数
    
    实现Ianvs框架要求的评估接口
    支持多种评估指标的计算
    
    Args:
        ground_truths: 真实标签列表
        predictions: 预测结果列表
        metric_name: 指标名称
        
    Returns:
        float: 指标分数
    """
    try:

        # 支持的指标映射
        metrics = {
            'score': _calculate_vlm_score
        }
        
        if metric_name not in metrics:
            logging.warning(f"未知的指标名称: {metric_name}")
            return 0.0
        
        # 计算指标
        metric_func = metrics[metric_name]
        score = metric_func(predictions, ground_truths)
        
        logging.info(f"计算指标 {metric_name}: {score:.4f}")
        return str(score)
        
    except Exception as e:
        logging.error(f"计算指标 {metric_name} 失败: {str(e)}")
        return 0.0
