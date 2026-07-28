import sys
import os

from sedna.common.class_factory import ClassType, ClassFactory

import matplotlib.pyplot as plt

__all__ = ('peak_memory')

@ClassFactory.register(ClassType.GENERAL, alias="peak_memory")
def peak_power(y_true, y_pred, **kwargs):
    mem_usage_per_device = y_pred.get("mem_usage_per_device")
    plt.figure()
    peak_mem = -sys.maxsize
    for device, mem_list in mem_usage_per_device.items():
        plt.bar(device, max(mem_list), label=f'{device}')
        peak_mem = max(peak_mem, max(mem_list))
    plt.axhline(y=peak_mem, color='red', linewidth=2, label='Peak Memory')

    plt.xticks(rotation=45)
    plt.ylabel('Memory')
    plt.xlabel('Device')
    plt.legend() 

    dir = './multiedge_inference_bench/workspace/classification_job/images/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    from datetime import datetime
    now = datetime.now().strftime("%H_%M_%S")
    plt.savefig(dir + 'peak_mem_per_device' + now + '.png')
    
    return peak_mem
