"""Manage YAML files."""
import os
import yaml


def _yaml_load_map(file):
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as yfile:
            yml = yaml.safe_load(yfile)
    else:
        yml = {}
    return yml


def yaml_models_load(file) -> dict:
    """Load a YAML models file."""
    # models files are a map of model names to yaml_model values.
    return _yaml_load_map(file)


def yaml_device_types_load(file) -> dict:
    """Load a YAML device types file."""
    # device types files are a map of device type names to yaml_device_type values.
    return _yaml_load_map(file)


def yaml_devices_load(file) -> dict:
    """Load a YAML devices file."""
    # devices files are a map of device type names to lists of hosts.
    return _yaml_load_map(file)


def yaml_device_neighbors_load(file) -> dict:
    """Load a YAML device neighbors file."""
    # device neighbors files are a map of neighbor hostnames to yaml_device_neighbors_type values.
    return _yaml_load_map(file)


def yaml_device_neighbors_world_load(file) -> dict:
    """Load a YAML device neighbors world file."""
    # device neighbors world files are a map of hostnames to a map of neighbor hostnames to
    # yaml_device_neighbors_type values.
    return _yaml_load_map(file)


def yaml_save(yml, file):
    """Save a YAML file."""
    with open(file, 'w', encoding='utf-8') as yfile:
        yaml.safe_dump(yml, yfile, default_flow_style=None, encoding='utf-8')
