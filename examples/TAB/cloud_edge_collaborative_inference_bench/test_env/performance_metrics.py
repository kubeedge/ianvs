import time

class PerformanceAssessment:
    def __init__(self):
        self.baseline_metrics = {}

    def measure_latency(self, fn, *args, **kwargs):
        start = time.time()
        _ = fn(*args, **kwargs)
        end = time.time()
        return end - start


