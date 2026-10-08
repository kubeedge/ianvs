import logging
import random
from typing import Callable, Optional, Sequence
import os
import shutil

from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoImageProcessor as ViTFeatureExtractor
from torchvision.datasets import ImageNet, ImageFolder


def load_dataset_imagenet(feature_extractor: Callable, root: str, split: str='train') -> Dataset:
    """Get the ImageNet dataset.

    Falls back to ImageFolder if the ImageNet devkit archive is not present,
    which allows running without ILSVRC2012_devkit_t12.tar.gz (e.g. when
    the dataset has already been extracted and organized into split/class/ folders).
    """
    def transform(img):
        pixels = feature_extractor(images=img.convert('RGB'), return_tensors='pt')['pixel_values']
        return pixels[0]
    # If root ends with the split name, step up one level
    # (ianvs may double-nest paths when resolving relative index files)
    if os.path.basename(root) == split:
        root = os.path.dirname(root)
    # Normalize root: remove duplicate trailing path segment if present
    # (ianvs double-nests paths when resolving relative index file entries)
    root = os.path.normpath(root)
    head, tail = os.path.split(root)
    if tail and os.path.basename(head) == tail:
        root = head
    try:
        return ImageNet(root, split=split, transform=transform)
    except RuntimeError:
        logging.warning(
            "ImageNet devkit archive not found, falling back to ImageFolder. "
            "Label indices may not match official ImageNet class ordering."
        )
        split_dir = os.path.join(root, split)
        dataset = ImageFolder(split_dir, transform=transform)
        dataset.imgs = [(path, label) for path, label in dataset.imgs]
        return dataset


def load_dataset_subset(dataset: Dataset, indices: Optional[Sequence[int]]=None,
                        max_size: Optional[int]=None, shuffle: bool=False) -> Dataset:
    """Get a Dataset subset."""
    if indices is None:
        indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    if max_size is not None:
        indices = indices[:max_size]
    image_paths = []
    for index in indices:
        image_paths.append(dataset.imgs[index][0])
    return Subset(dataset, indices), image_paths, indices


def load_dataset(dataset_cfg: dict, model_name: str, batch_size: int) -> Dataset:
    """Load inputs based on model."""
    def _get_feature_extractor():
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        return feature_extractor
    dataset_name = dataset_cfg['name']
    dataset_root = dataset_cfg['root']
    dataset_split = dataset_cfg['split']
    indices = dataset_cfg['indices']
    dataset_shuffle = dataset_cfg['shuffle']
    if dataset_name == 'ImageNet':
        if dataset_root is None:
            dataset_root = 'ImageNet'
            logging.info("Dataset root not set, assuming: %s", dataset_root)
        feature_extractor = _get_feature_extractor()
        dataset = load_dataset_imagenet(feature_extractor, dataset_root, split=dataset_split)
        dataset, paths, ids = load_dataset_subset(dataset, indices=indices, max_size=batch_size,
                                                  shuffle=dataset_shuffle)
    return dataset, paths, ids


if __name__ == '__main__':
    dataset_cfg = {
        'name': "ImageNet",
        'root': './dataset',
        'split': 'val',
        'indices': None,
        'shuffle': False,
    }
    model_name = "google/vit-base-patch16-224"
    ## Total images to be inferenced.
    data_size = 1000
    dataset, paths, _ = load_dataset(dataset_cfg, model_name, data_size)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    with open('./dataset/train.txt', 'w') as f:
        for i, (image, label) in enumerate(data_loader):
            f.write(f"{paths[i]} {label.item()}\n")
    shutil.copy('./dataset/train.txt', './dataset/test.txt')
