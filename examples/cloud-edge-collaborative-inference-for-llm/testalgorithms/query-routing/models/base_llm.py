class BaseLLM:
    def __init__(self, **kwargs) -> None:
        BaseLLM.__init__(self, **kwargs)
        self.quantization = kwargs.get("quantization", "full")

    def load(self):
        raise NotImplementedError

    def inference(self, datas):
        answer_list = []
        for line in datas:
            response = self._infer(line)
            answer_list.append(response)
        return answer_list
    
    def _infer(self, data):
        raise NotImplementedError