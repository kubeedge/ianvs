import evaluate
import numpy as np
from sedna.common.class_factory import ClassType, ClassFactory
from transformers import AutoTokenizer,AutoModelForCausalLM
import logging

@ClassFactory.register(ClassType.GENERAL, alias="rouge1")
def rouge1(y_true, y_pred, **kwargs):
    rouge=evaluate.load('./examples/LLM-Agent-Benchmark/evaluate/metrics/rouge')
    y_prednew=[]
    for i in range(len(y_pred)):
        y_prednew.append(y_pred[i]["generated_text"])
    rou_score = rouge.compute(predictions = y_prednew, references=y_true, use_aggregator=True)
    rouge1 = rou_score['rouge1'] * 10
    return rouge1

@ClassFactory.register(ClassType.GENERAL, alias="rouge2")
def rouge2(y_true, y_pred, **kwargs):
    rouge=evaluate.load('./examples/LLM-Agent-Benchmark/evaluate/metrics/rouge')
    y_prednew=[]
    for i in range(len(y_pred)):
        y_prednew.append(y_pred[i]["generated_text"])
    rou_score = rouge.compute(predictions = y_prednew, references=y_true, use_aggregator=True)
    rouge2 = rou_score['rouge2'] * 10
    return rouge2

@ClassFactory.register(ClassType.GENERAL, alias="rougeL")
def rougeL(y_true, y_pred, **kwargs):
    rouge=evaluate.load('./examples/LLM-Agent-Benchmark/evaluate/metrics/rouge')
    y_prednew=[]
    for i in range(len(y_pred)):
        y_prednew.append(y_pred[i]["generated_text"])
    rou_score = rouge.compute(predictions = y_prednew, references=y_true, use_aggregator=True)
    rougeL = rou_score['rougeL'] * 10
    return rougeL

def calculate_mean(lst):
    logging.info(lst)
    if not isinstance(lst,list):
        return lst
    if not lst:
        return None
    return sum(lst) / len(lst)