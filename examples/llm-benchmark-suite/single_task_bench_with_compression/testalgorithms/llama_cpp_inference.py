from sedna.common.class_factory import ClassFactory, ClassType

@ClassFactory.register(ClassType.STP, alias="LlamaCppInference")
class LlamaCppInference:
    def __init__(self, batch_size=1, max_tokens=32, stop=None, echo=False, **kwargs):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.stop = stop
        self.echo = echo

    def __call__(self, model, data):
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            responses = model.predict(
                batch, 
                max_tokens=self.max_tokens, 
                stop=self.stop,
                echo=self.echo
            )
            results.extend(responses)
        
        for r in results:
            print(r['choices'][0]['text'], "\n", "-" * 80, "\n")
        
        return results