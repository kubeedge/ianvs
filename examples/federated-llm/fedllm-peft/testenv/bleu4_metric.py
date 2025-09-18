"""Metric callback that computes BLEU-4 only."""
from evaluate import load as load_metric
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ["bleu4_metric"]

@ClassFactory.register(ClassType.GENERAL, alias="bleu4_metric")
def bleu4_metric(y_pred, y_true, **kwargs):
    preds = list(y_pred.values()) if isinstance(y_pred, dict) else list(y_pred)
    refs  = list(y_true.values()) if isinstance(y_true, dict) else list(y_true)
    bleu = load_metric("bleu")
    score = bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
    return round(float(score), 4)