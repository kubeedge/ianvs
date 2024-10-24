from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="Throughput")
def throughput(y_true, y_pred):
    
    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    average_itl = sum([pred.result.internal_token_latency for pred in infer_res]) / len(infer_res)
    
    average_throughput = 1 / average_itl
    
    return round(average_throughput,2)