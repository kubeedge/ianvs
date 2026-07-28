"""Common device configuration."""
from typing import Tuple, Union
import torch

# The torch.device to use for computation
DEVICE = None

def forward_pre_hook_to_device(_module, inputs) \
    -> Union[Tuple[torch.tensor], Tuple[Tuple[torch.Tensor]]]:
    """Move tensors to the compute device (e.g., GPU), if needed."""
    assert isinstance(inputs, tuple)
    assert len(inputs) == 1
    if isinstance(inputs[0], torch.Tensor):
        inputs = (inputs,)
    tensors_dev = tuple(t.to(device=DEVICE) for t in inputs[0])
    return tensors_dev if len(tensors_dev) == 1 else (tensors_dev,)

def forward_hook_to_cpu(_module, _inputs, outputs) -> Union[torch.tensor, Tuple[torch.Tensor]]:
    """Move tensors to the CPU, if needed."""
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    assert isinstance(outputs, tuple)
    tensors_cpu = tuple(t.cpu() for t in outputs)
    return tensors_cpu[0] if len(tensors_cpu) == 1 else tensors_cpu
