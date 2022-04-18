from core.testenvmanager.dataset import Dataset


class TestEnv:
    def __init__(self):
        self.dataset = Dataset()
        self.model_eval = {
            "model_metric": {
                "name": "",
                "parameters": {},
            },
            "threshold": 0.9,
            "operator": ">"
        }
        self.metrics = []
        self.incremental_rounds = 2

    def prepare(self, workerspace):
        """ prepare env"""
        try:
            self.dataset.process_dataset(workerspace)
        except Exception as err:
            raise Exception(f"prepare dataset failed, error: {err}.")

    def check_fields(self):
        self.dataset.check_fields()
        if not self.metrics:
            raise ValueError(f"not found testenv metrics({self.metrics}).")
