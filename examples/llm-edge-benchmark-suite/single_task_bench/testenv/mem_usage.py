<<<<<<< HEAD
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
=======
version https://git-lfs.github.com/spec/v1
oid sha256:3b71003d175c1cdde4fdc4bebd566fde400a0a22f553e321e8996a73260080b3
size 451
>>>>>>> 9676c3e (ya toh aar ya toh par)
