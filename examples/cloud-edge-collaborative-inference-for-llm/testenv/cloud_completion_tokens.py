from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="Cloud Completion Tokens")
def cloud_completion_tokens(y_true, y_pred):
    
    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    cloud_completion_tokens = sum([pred.cloud_result.completion_tokens for pred in infer_res])
    
    return cloud_completion_tokens