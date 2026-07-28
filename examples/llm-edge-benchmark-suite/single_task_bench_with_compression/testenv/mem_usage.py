from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["mem_usage"]

@ClassFactory.register(ClassType.GENERAL, alias="mem_usage")
def mem_usage(y_true, y_pred):
    results_list = y_pred.get('results', [])
    total_mem_usage = 0.0
    num_requests = len(results_list)
    for result in results_list:
        total_mem_usage += result['mem_usage']
    average_mem_usage = total_mem_usage / num_requests
    return average_mem_usage