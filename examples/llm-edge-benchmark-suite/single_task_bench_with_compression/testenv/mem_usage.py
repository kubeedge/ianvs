from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["mem_usage"]

@ClassFactory.register(ClassType.GENERAL, alias="mem_usage")
def mem_usage(y_true, y_pred):
    results_list = y_pred.get('results', [])
    total_mem_usage = 0.0
    count = 0
    for result in results_list:
        if isinstance(result, dict) and 'mem_usage' in result:
            mem_usage_bytes = result['mem_usage']
            mem_usage_mb = mem_usage_bytes / (1024 * 1024)  # byte -> MB
            total_mem_usage += mem_usage_mb
            count += 1
    average_mem_usage = total_mem_usage / count if count > 0 else 0.0
    print(f"Average Memory Usage: {average_mem_usage} MB")
    return average_mem_usage