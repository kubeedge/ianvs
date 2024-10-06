from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="Edge Prompt Tokens")
def edge_prompt_tokens(y_true, y_pred):
    
    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    edge_prompt_tokens = sum([pred.edge_result.prompt_tokens for pred in infer_res])
    
    return edge_prompt_tokens