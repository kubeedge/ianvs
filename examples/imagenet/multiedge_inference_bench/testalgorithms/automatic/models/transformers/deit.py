"""DeiT Transformers."""
from collections.abc import Mapping
import logging
import math
from typing import Optional, Union
import numpy as np
import torch
from torch import nn
from transformers import DeiTConfig
from transformers.models.deit.modeling_deit import DeiTEmbeddings
from transformers.models.vit.modeling_vit import (
    ViTIntermediate, ViTOutput, ViTSelfAttention, ViTSelfOutput
)
from .. import ModuleShard, ModuleShardConfig
from . import TransformerShardData


logger = logging.getLogger(__name__)

_HUB_MODEL_NAMES = {
    'facebook/deit-base-distilled-patch16-224': 'deit_base_distilled_patch16_224',
    'facebook/deit-small-distilled-patch16-224': 'deit_small_distilled_patch16_224',
    'facebook/deit-tiny-distilled-patch16-224': 'deit_tiny_distilled_patch16_224',
}


class DeiTLayerShard(ModuleShard):
    """Module shard based on `DeiTLayer` (copied from `.vit.ViTLayerShard`)."""

    def __init__(self, config: DeiTConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.layernorm_before = None
        self.self_attention = None
        self.self_output = None
        self.layernorm_after = None
        self.intermediate = None
        self.output = None
        self._build_shard()

    def _build_shard(self):
        if self.has_layer(0):
            self.layernorm_before = nn.LayerNorm(self.config.hidden_size,
                                                 eps=self.config.layer_norm_eps)
            self.self_attention = ViTSelfAttention(self.config)
        if self.has_layer(1):
            self.self_output = ViTSelfOutput(self.config)
        if self.has_layer(2):
            self.layernorm_after = nn.LayerNorm(self.config.hidden_size,
                                                eps=self.config.layer_norm_eps)
            self.intermediate = ViTIntermediate(self.config)
        if self.has_layer(3):
            self.output = ViTOutput(self.config)

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute layer shard."""
        if self.has_layer(0):
            data_norm = self.layernorm_before(data)
            data = (self.self_attention(data_norm)[0], data)
        if self.has_layer(1):
            skip = data[1]
            data = self.self_output(data[0], skip)
            data += skip
        if self.has_layer(2):
            data_norm = self.layernorm_after(data)
            data = (self.intermediate(data_norm), data)
        if self.has_layer(3):
            data = self.output(data[0], data[1])
        return data


class DeiTModelShard(ModuleShard):
    """Module shard based on `DeiTModel`."""

    def __init__(self, config: DeiTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)
        self.embeddings = None
        # DeiTModel uses an encoder here, but we'll just add the layers here instead.
        # Since we just do inference, a DeiTEncoderShard class wouldn't provide real benefit.
        self.layers = nn.ModuleList()
        self.layernorm = None

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        if self.shard_config.is_first:
            logger.debug(">>>> Load embeddings layer for the first shard")
            self.embeddings = DeiTEmbeddings(self.config)
            self._load_weights_first(weights)

        layer_curr = self.shard_config.layer_start
        while layer_curr <= self.shard_config.layer_end:
            layer_id = math.ceil(layer_curr / 4) - 1
            sublayer_start = (layer_curr - 1) % 4
            if layer_id == math.ceil(self.shard_config.layer_end / 4) - 1:
                sublayer_end = (self.shard_config.layer_end - 1) % 4
            else:
                sublayer_end = 3
            logger.debug(">>>> Load layer %d, sublayers %d-%d",
                         layer_id, sublayer_start, sublayer_end)
            layer_config = ModuleShardConfig(layer_start=sublayer_start, layer_end=sublayer_end)
            layer = DeiTLayerShard(self.config, layer_config)
            self._load_weights_layer(weights, layer_id, layer)
            self.layers.append(layer)
            layer_curr += sublayer_end - sublayer_start + 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_first(self, weights):
        self.embeddings.cls_token.copy_(torch.from_numpy(weights["cls_token"]))
        self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["pos_embed"])))
        self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(weights["patch_embed.proj.weight"]))
        self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["patch_embed.proj.bias"]))

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.layernorm.weight.copy_(torch.from_numpy(weights["norm.weight"]))
        self.layernorm.bias.copy_(torch.from_numpy(weights["norm.bias"]))

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer_id, layer):
        root = f"blocks.{layer_id}."
        embed_dim = self.config.hidden_size
        if layer.has_layer(0):
            layer.layernorm_before.weight.copy_(torch.from_numpy(weights[root + "norm1.weight"]))
            layer.layernorm_before.bias.copy_(torch.from_numpy(weights[root + "norm1.bias"]))
            qkv_weight = weights[root + "attn.qkv.weight"]
            layer.self_attention.query.weight.copy_(torch.from_numpy(qkv_weight[0:embed_dim,:]))
            layer.self_attention.key.weight.copy_(torch.from_numpy(qkv_weight[embed_dim:embed_dim*2,:]))
            layer.self_attention.value.weight.copy_(torch.from_numpy(qkv_weight[embed_dim*2:embed_dim*3,:]))
            qkv_bias = weights[root + "attn.qkv.bias"]
            layer.self_attention.query.bias.copy_(torch.from_numpy(qkv_bias[0:embed_dim,]))
            layer.self_attention.key.bias.copy_(torch.from_numpy(qkv_bias[embed_dim:embed_dim*2]))
            layer.self_attention.value.bias.copy_(torch.from_numpy(qkv_bias[embed_dim*2:embed_dim*3]))
        if layer.has_layer(1):
            layer.self_output.dense.weight.copy_(torch.from_numpy(weights[root + "attn.proj.weight"]))
            layer.self_output.dense.bias.copy_(torch.from_numpy(weights[root + "attn.proj.bias"]))
        if layer.has_layer(2):
            layer.layernorm_after.weight.copy_(torch.from_numpy(weights[root + "norm2.weight"]))
            layer.layernorm_after.bias.copy_(torch.from_numpy(weights[root + "norm2.bias"]))
            layer.intermediate.dense.weight.copy_(torch.from_numpy(weights[root + "mlp.fc1.weight"]))
            layer.intermediate.dense.bias.copy_(torch.from_numpy(weights[root + "mlp.fc1.bias"]))
        if layer.has_layer(3):
            layer.output.dense.weight.copy_(torch.from_numpy(weights[root + "mlp.fc2.weight"]))
            layer.output.dense.bias.copy_(torch.from_numpy(weights[root + "mlp.fc2.bias"]))

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        if self.shard_config.is_first:
            data = self.embeddings(data)
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.layernorm(data)
        return data

    # NOTE: repo has a dependency on the timm package, which isn't an automatic torch dependency
    @staticmethod
    def save_weights(model_name: str, model_file: str, hub_repo: str='facebookresearch/deit:main',
                     hub_model_name: Optional[str]=None) -> None:
        """Save the model weights file."""
        if hub_model_name is None:
            if model_name in _HUB_MODEL_NAMES:
                hub_model_name = _HUB_MODEL_NAMES[model_name]
                logger.debug("Mapping model name to torch hub equivalent: %s: %s", model_name,
                             hub_model_name)
            else:
                hub_model_name = model_name
        model = torch.hub.load(hub_repo, hub_model_name, pretrained=True)
        state_dict = model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        np.savez(model_file, **weights)


class DeiTShardForImageClassification(ModuleShard):
    """Module shard based on `DeiTForImageClassification`."""

    def __init__(self, config: DeiTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)
        self.deit = None
        self.classifier = None

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        ## all shards use the inner DeiT model
        self.deit = DeiTModelShard(self.config, self.shard_config, weights)

        if self.shard_config.is_last:
            logger.debug(">>>> Load classifier for the last shard")
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.classifier.weight.copy_(torch.from_numpy(weights["head.weight"]))
        self.classifier.bias.copy_(torch.from_numpy(weights["head.bias"]))

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        data = self.deit(data)
        if self.shard_config.is_last:
            data = self.classifier(data[:, 0, :])
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str, hub_repo: str='facebookresearch/deit:main',
                     hub_model_name: Optional[str]=None) -> None:
        """Save the model weights file."""
        DeiTModelShard.save_weights(model_name, model_file, hub_repo=hub_repo,
                                    hub_model_name=hub_model_name)
