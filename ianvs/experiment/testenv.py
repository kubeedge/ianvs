class TestEnv:
    def __init__(self):
        self.dataset_url = ""
        self.train_ratio = 0.8
        self.label = []
        self.model_eval = {}
        self.metrics = []
        self.rank = {"switch": False}
        self.visualization_models = "off"
        self.output_url = "./test/"
