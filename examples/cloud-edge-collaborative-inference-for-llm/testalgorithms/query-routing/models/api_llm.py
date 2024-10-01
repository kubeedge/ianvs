import os 
import time
from openai import OpenAI

from models.base_llm import BaseLLM

class APIBasedLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        BaseLLM.__init__(self, **kwargs)

        api_key=os.environ.get("OPENAI_API_KEY")
        base_url=os.environ.get("OPENAI_BASE_URL")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def load(self, model):
        self.model = model
    
    def _infer(self, messages):
        # messages = self.get_message_chain(question, system_prompt)

        time_to_first_token = 0.0
        internal_token_latency = []
        st = time.perf_counter()
        most_recent_timestamp = st
        generated_text = ""

        stream = self.client.chat.completions.create(
            messages = messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.repetition_penalty,
            stream=True,
            stream_options={"include_usage":True}
        )

        for chunk in stream:
            timestamp = time.perf_counter()
            if time_to_first_token == 0.0:
                time_to_first_token = time.perf_counter() - st
            else:
                internal_token_latency.append(timestamp - most_recent_timestamp)
            most_recent_timestamp = timestamp
            if chunk.choices:
                generated_text += chunk.choices[0].delta.content or ""
            if chunk.usage:
                usage = chunk.usage

        text = generated_text
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        internal_token_latency = sum(internal_token_latency) / len(internal_token_latency)
        throughput = 1 / internal_token_latency

        response = self._format_response(
            text, 
            prompt_tokens, 
            completion_tokens,
            time_to_first_token,
            internal_token_latency,
            throughput
        )

        return response

if __name__ == '__main__':
    llm = APIBasedLLM(model="gpt-4o-mini")
    data = ["你好吗？介绍一下自己"]
    res = llm.inference(data)
    print(res)