"""BERT transformers."""
from collections.abc import Mapping
import logging
import math
from typing import Union
import numpy as np
import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification, BertModel
from transformers.models.bert.modeling_bert import (
    BertEmbeddings, BertIntermediate, BertOutput, BertPooler, BertSelfAttention, BertSelfOutput
)
from .. import ModuleShard, ModuleShardConfig
from . import TransformerShardData


logger = logging.getLogger(__name__)


class BertLayerShard(ModuleShard):
    """Module shard based on `BertLayer`."""

    def __init__(self, config: BertConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.self_attention = None
        self.self_output = None
        self.intermediate = None
        self.output = None
        self._build_shard()

    def _build_shard(self):
        if self.has_layer(0):
            self.self_attention = BertSelfAttention(self.config)
        if self.has_layer(1):
            self.self_output = BertSelfOutput(self.config)
        if self.has_layer(2):
            self.intermediate = BertIntermediate(self.config)
        if self.has_layer(3):
            self.output = BertOutput(self.config)

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute layer shard."""
        if self.has_layer(0):
            data = (self.self_attention(data)[0], data)
        if self.has_layer(1):
            data = self.self_output(data[0], data[1])
        if self.has_layer(2):
            data = (self.intermediate(data), data)
        if self.has_layer(3):
            data = self.output(data[0], data[1])
        return data


class BertModelShard(ModuleShard):
    """Module shard based on `BertModel`."""

    def __init__(self, config: BertConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)
        self.embeddings = None
        # BertModel uses an encoder here, but we'll just add the layers here instead.
        # Since we just do inference, a BertEncoderShard class wouldn't provide real benefit.
        self.layers = nn.ModuleList()
        self.pooler = None

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
            self.embeddings = BertEmbeddings(self.config)
            self.embeddings.eval()
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
            layer = BertLayerShard(self.config, layer_config)
            self._load_weights_layer(weights, layer_id, layer)
            self.layers.append(layer)
            layer_curr += sublayer_end - sublayer_start + 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load pooler for the last shard")
            self.pooler = BertPooler(self.config)
            self.pooler.eval()
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_first(self, weights):
        self.embeddings.position_ids.copy_(torch.from_numpy((weights["embeddings.position_ids"])))
        self.embeddings.word_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.word_embeddings.weight']))
        self.embeddings.position_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.position_embeddings.weight']))
        self.embeddings.token_type_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.token_type_embeddings.weight']))
        self.embeddings.LayerNorm.weight.copy_(torch.from_numpy(weights['embeddings.LayerNorm.weight']))
        self.embeddings.LayerNorm.bias.copy_(torch.from_numpy(weights['embeddings.LayerNorm.bias']))

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.pooler.dense.weight.copy_(torch.from_numpy(weights["pooler.dense.weight"]))
        self.pooler.dense.bias.copy_(torch.from_numpy(weights['pooler.dense.bias']))

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer_id, layer):
        root = f"encoder.layer.{layer_id}."
        if layer.has_layer(0):
            layer.self_attention.query.weight.copy_(torch.from_numpy(weights[root + "attention.self.query.weight"]))
            layer.self_attention.key.weight.copy_(torch.from_numpy(weights[root + "attention.self.key.weight"]))
            layer.self_attention.value.weight.copy_(torch.from_numpy(weights[root + "attention.self.value.weight"]))
            layer.self_attention.query.bias.copy_(torch.from_numpy(weights[root + "attention.self.query.bias"]))
            layer.self_attention.key.bias.copy_(torch.from_numpy(weights[root + "attention.self.key.bias"]))
            layer.self_attention.value.bias.copy_(torch.from_numpy(weights[root + "attention.self.value.bias"]))
        if layer.has_layer(1):
            layer.self_output.dense.weight.copy_(torch.from_numpy(weights[root + "attention.output.dense.weight"]))
            layer.self_output.LayerNorm.weight.copy_(torch.from_numpy(weights[root + "attention.output.LayerNorm.weight"]))
            layer.self_output.dense.bias.copy_(torch.from_numpy(weights[root + "attention.output.dense.bias"]))
            layer.self_output.LayerNorm.bias.copy_(torch.from_numpy(weights[root + "attention.output.LayerNorm.bias"]))
        if layer.has_layer(2):
            layer.intermediate.dense.weight.copy_(torch.from_numpy(weights[root + "intermediate.dense.weight"]))
            layer.intermediate.dense.bias.copy_(torch.from_numpy(weights[root + "intermediate.dense.bias"]))
        if layer.has_layer(3):
            layer.output.dense.weight.copy_(torch.from_numpy(weights[root + "output.dense.weight"]))
            layer.output.dense.bias.copy_(torch.from_numpy(weights[root + "output.dense.bias"]))
            layer.output.LayerNorm.weight.copy_(torch.from_numpy(weights[root + "output.LayerNorm.weight"]))
            layer.output.LayerNorm.bias.copy_(torch.from_numpy(weights[root + "output.LayerNorm.bias"]))

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        if self.shard_config.is_first:
            data = self.embeddings(data)
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.pooler(data)
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str) -> None:
        """Save the model weights file."""
        model = BertModel.from_pretrained(model_name)
        state_dict = model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        np.savez(model_file, **weights)


class BertShardForSequenceClassification(ModuleShard):
    """Module shard based on `BertForSequenceClassification`."""

    def __init__(self, config: BertConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)
        self.bert = None
        self.classifier = None

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        ## all shards use the inner BERT model
        self.bert = BertModelShard(self.config, self.shard_config,
                                   self._extract_weights_bert(weights))

        if self.shard_config.is_last:
            logger.debug(">>>> Load classifier for the last shard")
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
            self._load_weights_last(weights)

    def _extract_weights_bert(self, weights):
        bert_weights = {}
        for key, val in weights.items():
            if key.startswith('bert.'):
                bert_weights[key[len('bert.'):]] = val
        return bert_weights

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.classifier.weight.copy_(torch.from_numpy(weights['classifier.weight']))
        self.classifier.bias.copy_(torch.from_numpy(weights['classifier.bias']))

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        data = self.bert(data)
        if self.shard_config.is_last:
            data = self.classifier(data)
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str) -> None:
        """Save the model weights file."""
        model = BertForSequenceClassification.from_pretrained(model_name)
        state_dict = model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        np.savez(model_file, **weights)
