from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate

MODEL_PATH = "Qwen/Qwen1.5-1.8B-Chat"
MODEL_ABBR = "qwen1.5-1.8b-chat-hf"

with read_base():
    from core.op_extra.datasets.cmmlu.cmmlu_gen import cmmlu_datasets

datasets = [*cmmlu_datasets]

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=MODEL_ABBR,
        path=MODEL_PATH,
        max_out_len=1024,
        batch_size=2,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
