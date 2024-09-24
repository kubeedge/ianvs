from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["prefill_latency"]

@ClassFactory.register(ClassType.GENERAL, alias="prefill_latency")
def prefill_latency(y_true, y_pred):
    results_list = y_pred.get('results', [])
    
    total_prefill_time = 0.0
    count = 0
    for result in results_list:
        if isinstance(result, dict) and 'prefill_latency' in result:
            total_prefill_time += result['prefill_latency']
            count += 1
    average_prefill_latency = total_prefill_time / count if count > 0 else 0.0
    print(f"Average Prefill Latency: {average_prefill_latency} ms")
    return average_prefill_latency