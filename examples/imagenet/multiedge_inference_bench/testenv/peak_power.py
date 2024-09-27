import sys
import os

from sedna.common.class_factory import ClassType, ClassFactory

import matplotlib.pyplot as plt

__all__ = ('peak_power')

@ClassFactory.register(ClassType.GENERAL, alias="peak_power")
def peak_power(y_true, y_pred, **kwargs):
    power_usage_per_device = y_pred.get("power_usage_per_device")
    plt.figure()
    peak_power = -sys.maxsize
    for device, power_list in power_usage_per_device.items():
        plt.plot(power_list, label=device)
        peak_power = max(peak_power, max(power_list))
    plt.axhline(y=peak_power, color='red', linewidth=2, label='Peak Power')

    plt.xticks(rotation=45)
    plt.ylabel('Power')
    plt.xlabel('Device')
    plt.legend() 

    dir = './multiedge_inference_bench/workspace/classification_job/images/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    from datetime import datetime
    now = datetime.now().strftime("%H_%M_%S")
    plt.savefig(dir + 'power_usage_per_device' + now + '.png')
    
    return peak_power
