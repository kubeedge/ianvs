<<<<<<< HEAD
# Copyright 2024 New Government Agent Project
# New government agent test environment implementation

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
from sedna.common.class_factory import ClassType, ClassFactory

def _calculate_vlm_score(predictions: List[Dict[str, Any]], 
                        ground_truths: List[Dict[str, Any]]) -> float:
    """Calculate VLM score (10-point scale)"""
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
        logging.warning(f"Failed to calculate VLM score: {str(e)}")
        return 0.0

@ClassFactory.register(ClassType.GENERAL, alias="score")
def score(ground_truths, predictions, metric_name: str = 'score') -> float:
    """
    New government agent accuracy evaluation function
    
    Implements the evaluation interface required by the Ianvs framework
    Supports calculation of multiple evaluation metrics
    
    Args:
        ground_truths: Ground truth labels list
        predictions: Predictions results list
        metric_name: Metric name
        
    Returns:
        float: Metric score
    """
    try:

        # Supported metrics mapping
        metrics = {
            'score': _calculate_vlm_score
        }
        
        if metric_name not in metrics:
            logging.warning(f"Unknown metric name: {metric_name}")
            return 0.0
        
        # Calculate metric
        metric_func = metrics[metric_name]
        score = metric_func(predictions, ground_truths)
        
        logging.info(f"Calculated metric {metric_name}: {score:.4f}")
        return float(score)
        
    except Exception as e:
        logging.error(f"Failed to calculate metric {metric_name}: {str(e)}")
        return 0.0
=======
version https://git-lfs.github.com/spec/v1
oid sha256:e09fe588b92bd2ba8754a4d573e164240aa0d3ef532cc3747382cf256fbf10b5
size 2323
>>>>>>> 9676c3e (ya toh aar ya toh par)
