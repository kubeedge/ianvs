"""Metric callback that computes ROUGE-1 only."""
from evaluate import load as load_metric
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ["rouge1_metric"]

@ClassFactory.register(ClassType.GENERAL, alias="rouge1_metric")
def rouge1_metric(y_pred, y_true, **kwargs):
    # normalise inputs
    preds = list(y_pred.values()) if isinstance(y_pred, dict) else list(y_pred)
    refs  = list(y_true.values()) if isinstance(y_true, dict) else list(y_true)
    rouge = load_metric("rouge")
    score = rouge.compute(predictions=preds, references=refs, use_stemmer=True)["rouge1"]
    return round(float(score), 4)
