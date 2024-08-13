class BaseLLM:
    def __init__(self, **kwargs) -> None:
        self.quantization = kwargs.get("quantization", "full")

    def load(self):
        raise NotImplementedError

    def inference(self, data):
        if isinstance(data, list):
            return [self._infer(line) for line in data]
        elif isinstance(data, str):
            return  self._infer(data)
        else:
            raise ValueError(f"DataType {type(data)} is not supported, it must be `list` or `str`")
        
    def get_message_chain(self, prompt, system = None):
        system = "You are a helpful assistant, please help sovle user's question. "

        if system:   
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]

        return messages

    
    def _infer(self, data):
        raise NotImplementedError