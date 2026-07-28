"""Transformers module."""
from typing import Tuple, Type, Union
from torch import Tensor

TransformerShardData: Type = Union[Tensor, Tuple[Tensor, Tensor]]
"""A transformer shard input/output type."""
