import os 
from openai import OpenAI

from models.base_llm import BaseLLM
from sedna.core.joint_inference.joint_inference import BigModelService

class APIBasedLLM(BaseLLM):
    def __init__(self, model_name, **kwargs) -> None:

        api_key=os.environ.get("OPENAI_API_KEY")
        base_url=os.environ.get("OPENAI_BASE_URL")

        self.model = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _infer(self, prompt, system=None):
        messages = self.get_message_chain(prompt, system)
        
        self.chat_completion = self.client.chat.completions.create(
            messages = messages,
            model=self.model,
        )

        response = self.chat_completion.choices[0].message.content

        return response
    
if __name__ == '__main__':
    llm = APIBasedLLM(model_name="gpt-4o-mini")
    data = ["你好吗？介绍一下自己"]
    res = llm.inference(data)
    print(res)