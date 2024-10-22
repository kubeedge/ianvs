from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ('accuracy')

@ClassFactory.register(ClassType.GENERAL, alias="accuracy")
def accuracy(y_true, y_pred, **kwargs):
    y_pred = y_pred.get("pred")
    total = len(y_pred)
    y_true_ = [int(y_true[i].split('/')[-1]) for (_, i) in y_pred]
    y_pred_ = [int(i) for (i, _) in y_pred]
    correct_predictions = sum(yt == yp for yt, yp in zip(y_true_, y_pred_))
    accuracy = (correct_predictions / total) * 100  
    print("Accuracy: {:.2f}%".format(accuracy))
    return accuracy
