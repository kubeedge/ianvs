from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="edge-rate")
def edge_rate(y_true, y_pred):
    
    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    y_pred = [pred.is_hard_example for pred in infer_res]

    return 1 - sum(y_pred) / len(y_pred)
        
