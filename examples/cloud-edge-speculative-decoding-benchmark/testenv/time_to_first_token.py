from sedna.common.class_factory import ClassFactory, ClassType

from result_parser import parse_effective_joint_inference_results


@ClassFactory.register(ClassType.GENERAL, alias="Time to First Token")
def time_to_first_token(_, y_pred):
    infer_res = parse_effective_joint_inference_results(y_pred)
    if not infer_res:
        return ""
    average_ttft = sum(item.result.time_to_first_token for item in infer_res) / len(infer_res)
    return round(average_ttft, 3)
