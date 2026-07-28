import json

from sedna.common.class_factory import ClassType, ClassFactory
__all__ = ["accuracy"]

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
@ClassFactory.register(ClassType.GENERAL, alias="accuracy")
def accuracy(y_true, y_pred, **kwargs):
    "this y_pred is a jsonl file , recommend task, episode andsuccess"
    """
        result["Task"] = task_description
        result["episode"] = task_episodes
        result["Success"] = True/False
    """
    pred_data = read_jsonl(y_pred)
    total_count = len(pred_data)
    correct_count = sum(1 for item in pred_data if item.get("Success") is True)
    acc = correct_count / total_count if total_count > 0 else 0.0
    print("loading test")
    return acc

