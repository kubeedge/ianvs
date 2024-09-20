import sys
import os

from sedna.common.class_factory import ClassType, ClassFactory

import matplotlib.pyplot as plt

__all__ = ('fps')

@ClassFactory.register(ClassType.GENERAL, alias="fps")
def fps(y_true, y_pred, **kwargs):
    total = len(y_pred.get("pred"))
    inference_time_per_device = y_pred.get("inference_time_per_device")
    plt.figure()
    min_fps = sys.maxsize
    for device, time in inference_time_per_device.items():
        fps = total / time
        plt.bar(device, fps, label=f'{device}')
        min_fps = min(fps, min_fps)
    plt.axhline(y=min_fps, color='red', linewidth=2, label='Min FPS')

    plt.xticks(rotation=45)
    plt.ylabel('FPS')
    plt.xlabel('Device')
    plt.legend() 

    dir = './multiedge_inference_bench/workspace/classification_job/images/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    from datetime import datetime
    now = datetime.now().strftime("%H_%M_%S")
    plt.savefig(dir + 'FPS_per_device' + now + '.png')

    return min_fps
