from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="Time to First Token")
def time_to_first_token(y_true, y_pred):
    
    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    average_ttft = sum([pred.result.time_to_first_token for pred in infer_res]) / len(infer_res)
    
    return round(average_ttft, 3)