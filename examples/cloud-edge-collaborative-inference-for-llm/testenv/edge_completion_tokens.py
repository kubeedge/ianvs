from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="Edge Completion Tokens")
def edge_completion_tokens(y_true, y_pred):
    
    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    edge_completion_tokens = sum([pred.edge_result.completion_tokens for pred in infer_res])
    
    return edge_completion_tokens