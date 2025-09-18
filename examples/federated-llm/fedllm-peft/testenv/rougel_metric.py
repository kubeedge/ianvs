"""Metric callback that computes ROUGE-L only."""
from evaluate import load as load_metric
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ["rougel_metric"]

@ClassFactory.register(ClassType.GENERAL, alias="rougel_metric")
def rougel_metric(y_pred, y_true, **kwargs):
    preds = list(y_pred.values()) if isinstance(y_pred, dict) else list(y_pred)
    refs  = list(y_true.values()) if isinstance(y_true, dict) else list(y_true)
    rouge = load_metric("rouge")
    score = rouge.compute(predictions=preds, references=refs, use_stemmer=True)["rougeL"]
    return round(float(score), 4)