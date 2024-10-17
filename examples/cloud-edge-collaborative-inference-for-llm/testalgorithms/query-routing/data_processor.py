import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType

PROMPTS = {
    "system_prompt": {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    "ice_template": [
        {
            "role": "user",
            "content": "There is a single choice question about {level_4_dim}. Answer the question by replying A, B, C or D.\n{query}\nAnswer: "
        },
        {
            "role": "assistant",
            "content": "{response}\n"
        }
    ],
    "prompt_template": {
        "role": "user",
        "content": "There is a single choice question about {level_4_dim}. Answer the question by replying A, B, C or D.\n{query}\nAnswer: "
    }
}

@ClassFactory.register(ClassType.GENERAL, alias="MultiShotGenertor")
class Multi_Shot_Generator:
    def __init__(self, **kwargs):
        self.shot_nums = kwargs.get("shot_nums", 0)
    def load_prompts(self):
        self.system_prompt = PROMPTS.get("system_prompt", None)
        self.ice_template = PROMPTS.get('ice_template', None)
        self.prompt_template = PROMPTS.get('prompt_template', None)

    def multi_shot_generation(self, dataset, shot_nums = 0):
        data = [{"query":query, "response":response, "level_4_dim":level_4_dim}
            for query, response,level_4_dim in  zip(dataset.x, dataset.y, dataset.level_4)]

        format_chat = lambda chat, item: {key: value.format(**item) for key, value in chat.items()}
        
        data_array = np.array(data)
        data_index = np.arange(len(data))

        x = []

        for i, item in enumerate(data):
            messages = []
            if self.system_prompt:
                messages.append(self.system_prompt)
            if self.ice_template:
                shots = np.random.choice(data_array[data_index != i], size=shot_nums, replace=False)
                for shot in shots:
                    formatted_chat = [format_chat(chat, shot) for chat in self.ice_template]
                    messages.extend(formatted_chat)
            final_chat = format_chat(self.prompt_template, item)
            messages.append(final_chat)
            
            x.append({"messages":messages,"gold": item["response"]})

        dataset.x = x

        return dataset

    def __call__(self, dataset):            
        self.load_prompts()
        return self.multi_shot_generation(dataset, self.shot_nums)