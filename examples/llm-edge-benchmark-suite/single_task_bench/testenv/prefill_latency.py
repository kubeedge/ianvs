<<<<<<< HEAD
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["prefill_latency"]

@ClassFactory.register(ClassType.GENERAL, alias="prefill_latency")
def prefill_latency(y_true, y_pred):
    results_list = y_pred.get('results', [])
    num_requests = len(results_list)
    total_prefill_latency = 0.0
    for result in results_list:
        total_prefill_latency += result['prefill_latency']
    avg_prefill_latency = total_prefill_latency / num_requests
    return avg_prefill_latency
=======
version https://git-lfs.github.com/spec/v1
oid sha256:e562f8d1e2f9816d9fc2309462a604ee47556faa457e6e554629f84c02d1c839
size 497
>>>>>>> 9676c3e (ya toh aar ya toh par)
