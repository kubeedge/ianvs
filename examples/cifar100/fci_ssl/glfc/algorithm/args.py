class ClientArgs:
    def __init__(self,  **kwargs):
        self.epochs = 1
        self.batch_size = 32
        self.task_size = 10
        self.learning_rate = 0.001
        
        self.memory_size = 2000
        
        

class ServerArgs:
    def __init__(self,  **kwargs):
        self.epochs = 1
        self.batch_size = 32
        self.task_size = 10
        self.memory_size = 2000