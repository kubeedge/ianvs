from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="Cloud Prompt Tokens")
def cloud_prompt_tokens(y_true, y_pred):
    
    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    cloud_prompt_tokens = sum([pred.cloud_result.prompt_tokens for pred in infer_res])
    
    return cloud_prompt_tokens