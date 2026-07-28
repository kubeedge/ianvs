import evaluate
import numpy as np
from sedna.common.class_factory import ClassType, ClassFactory
from transformers import AutoTokenizer,AutoModelForCausalLM
import logging
from rouge_score import rouge_scorer

@ClassFactory.register(ClassType.GENERAL, alias="rouge1")
def rouge1(y_true, y_pred, **kwargs):
    scorer=rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    y_prednew=[str(item) for item in y_pred]
    scores=[scorer.score(ref, pred)['rouge1'].fmeasure for ref, pred in zip(y_true, y_prednew)]
    if not scores:
        return 0.0
    return (sum(scores) / len(scores)) * 10

@ClassFactory.register(ClassType.GENERAL, alias="rouge2")
def rouge2(y_true, y_pred, **kwargs):
    scorer=rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    y_prednew=[str(item) for item in y_pred]
    scores=[scorer.score(ref, pred)['rouge2'].fmeasure for ref, pred in zip(y_true, y_prednew)]
    if not scores:
        return 0.0
    return (sum(scores) / len(scores)) * 10

@ClassFactory.register(ClassType.GENERAL, alias="rougeL")
def rougeL(y_true, y_pred, **kwargs):
    scorer=rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    y_prednew=[str(item) for item in y_pred]
    scores=[scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(y_true, y_prednew)]
    if not scores:
        return 0.0
    return (sum(scores) / len(scores)) * 10

def calculate_mean(lst):
    logging.info(lst)
    if not isinstance(lst,list):
        return lst
    if not lst:
        return None
    return sum(lst) / len(lst)
