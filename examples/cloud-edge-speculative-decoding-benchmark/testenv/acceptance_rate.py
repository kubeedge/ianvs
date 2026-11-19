from sedna.common.class_factory import ClassFactory, ClassType

from result_parser import parse_effective_joint_inference_results


@ClassFactory.register(ClassType.GENERAL, alias="Acceptance Rate")
def acceptance_rate(_, y_pred):
    infer_res = parse_effective_joint_inference_results(y_pred)
    if not infer_res:
        return ""
    simulations = [item.result.simulation for item in infer_res]
    # Weight by draft tokens so short requests do not count as much as long ones.
    total_draft_tokens = sum(sim.total_draft_tokens for sim in simulations)
    if total_draft_tokens > 0:
        accepted_draft_tokens = sum(sim.accepted_draft_tokens for sim in simulations)
        return round(accepted_draft_tokens / total_draft_tokens, 3)
    # Older result files may carry only the per-request rate.
    acceptance_values = [
        sim.acceptance_rate for sim in simulations if sim.acceptance_rate is not None
    ]
    if not acceptance_values:
        return ""
    average_acceptance = sum(acceptance_values) / len(acceptance_values)
    return round(average_acceptance, 3)
