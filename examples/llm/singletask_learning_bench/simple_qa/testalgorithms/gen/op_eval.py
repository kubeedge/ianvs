from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate
# import sys
# sys.path.append('/home/icyfeather/project/ianvs')

with read_base():
    from core.op_extra.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
    
datasets = [*cmmlu_datasets]

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen1.5-1.8b-chat-hf',
        path='/home/icyfeather/models/Qwen1.5-1.8B-Chat',
        max_out_len=1024,
        batch_size=2,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
