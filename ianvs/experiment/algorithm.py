class Algorithm:
    def __init__(self):
        self.paradigm = Paradigm()
        self.basemodel = BaseModel()


class BaseModel:
    def __init__(self):
        self.name = ""
        self.hyperparameters = {}
        self.hyperparameter_file = ""
        self.multi_parameters = []


class Paradigm:
    def __init__(self):
        self.kind = ""
        self.incremental_rounds = 2
        self.funcs = []
