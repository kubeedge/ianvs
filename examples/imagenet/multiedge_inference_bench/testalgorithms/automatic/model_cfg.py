"""Model configurations and default parameters."""
import logging
from typing import Any, Callable, List, Optional, Tuple
from transformers import AutoConfig
from models import ModuleShard, ModuleShardConfig
from models.transformers import bert, deit, vit
import devices

_logger = logging.getLogger(__name__)

_MODEL_CONFIGS = {}

def _model_cfg_add(name, layers, weights_file, shard_module):
    _MODEL_CONFIGS[name] = {
        'name': name,
        'layers': layers,
        'weights_file': weights_file,
        'shard_module': shard_module,
    }

# Transformer blocks can be split 4 ways, e.g., where ViT-Base has 12 layers, we specify 12*4=48
_model_cfg_add('google/vit-base-patch16-224', 48, './initial_model/ViT-B_16-224.npz',
               vit.ViTShardForImageClassification)
_model_cfg_add('google/vit-large-patch16-224', 96, 'ViT-L_16-224.npz',
               vit.ViTShardForImageClassification)
# NOTE: This ViT-Huge model doesn't include classification, so the config must be extended
_model_cfg_add('google/vit-huge-patch14-224-in21k', 128, 'ViT-H_14.npz',
               vit.ViTShardForImageClassification)
# NOTE: BertModelShard alone doesn't do classification
_model_cfg_add('bert-base-uncased', 48, 'BERT-B.npz',
               bert.BertModelShard)
_model_cfg_add('bert-large-uncased', 96, 'BERT-L.npz',
               bert.BertModelShard)
_model_cfg_add('textattack/bert-base-uncased-CoLA', 48, 'BERT-B-CoLA.npz',
               bert.BertShardForSequenceClassification)
_model_cfg_add('facebook/deit-base-distilled-patch16-224', 48, 'DeiT_B_distilled.npz',
               deit.DeiTShardForImageClassification)
_model_cfg_add('facebook/deit-small-distilled-patch16-224', 48, 'DeiT_S_distilled.npz',
               deit.DeiTShardForImageClassification)
_model_cfg_add('facebook/deit-tiny-distilled-patch16-224', 48, 'DeiT_T_distilled.npz',
               deit.DeiTShardForImageClassification)

def get_model_names() -> List[str]:
    """Get a list of available model names."""
    return list(_MODEL_CONFIGS.keys())

def get_model_dict(model_name: str) -> dict:
    """Get a model's key/value properties - modify at your own risk."""
    return _MODEL_CONFIGS[model_name]

def get_model_layers(model_name: str) -> int:
    """Get a model's layer count."""
    return _MODEL_CONFIGS[model_name]['layers']

def get_model_config(model_name: str) -> Any:
    """Get a model's config."""
    # We'll need more complexity if/when we add support for models not from `transformers`
    config = AutoConfig.from_pretrained(model_name)
    # Config overrides
    if model_name == 'google/vit-huge-patch14-224-in21k':
        # ViT-Huge doesn't include classification, so we have to set this ourselves
        # NOTE: not setting 'id2label' or 'label2id'
        config.num_labels = 21843
    return config

def get_model_default_weights_file(model_name: str) -> str:
    """Get a model's default weights file name."""
    return _MODEL_CONFIGS[model_name]['weights_file']

def save_model_weights_file(model_name: str, model_file: Optional[str]=None) -> None:
    """Save a model's weights file."""
    if model_file is None:
        model_file = get_model_default_weights_file(model_name)
    # This works b/c all shard implementations have the same save_weights interface
    module = _MODEL_CONFIGS[model_name]['shard_module']
    module.save_weights(model_name, model_file)

def module_shard_factory(model_name: str, model_file: Optional[str], layer_start: int,
                         layer_end: int, stage: int) -> ModuleShard:
    """Get a shard instance on the globally-configured `devices.DEVICE`."""
    # This works b/c all shard implementations have the same constructor interface
    if model_file is None:
        model_file = get_model_default_weights_file(model_name)
    config = get_model_config(model_name)
    is_first = layer_start == 1
    is_last = layer_end == get_model_layers(model_name)
    shard_config = ModuleShardConfig(layer_start=layer_start, layer_end=layer_end,
                                     is_first=is_first, is_last=is_last)
    module = _MODEL_CONFIGS[model_name]['shard_module']
    shard = module(config, shard_config, model_file)
    _logger.info("======= %s Stage %d =======", module.__name__, stage)
    shard.to(device=devices.DEVICE)
    return shard