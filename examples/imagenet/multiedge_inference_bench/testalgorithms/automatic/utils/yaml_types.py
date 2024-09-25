"""YAML types."""
from typing import List, Optional, Union


def _assert_list_type(lst, dtype):
    assert isinstance(lst, list)
    for var in lst:
        assert isinstance(var, dtype)


def yaml_model(num_layers: int, parameters_in: int, parameters_out: List[int],
               mem_MB: Union[List[int], List[float]]) -> dict:
    """Create a YAML model."""
    assert isinstance(num_layers, int)
    assert isinstance(parameters_in, int)
    _assert_list_type(parameters_out, int)
    _assert_list_type(mem_MB, (int, float))
    return {
        'layers': num_layers,
        'parameters_in': parameters_in,
        'parameters_out': parameters_out,
        'mem_MB': mem_MB,
    }


def yaml_model_profile(dtype: str, batch_size: int, time_s: Union[List[int], List[float]]) -> dict:
    """Create a YAML model profile."""
    assert isinstance(dtype, str)
    assert isinstance(batch_size, int)
    _assert_list_type(time_s, (int, float))
    return {
        'dtype': dtype,
        'batch_size': batch_size,
        'time_s': time_s,
    }


def _assert_model_profile(model_prof):
    assert isinstance(model_prof, dict)
    for model_prof_prop in model_prof:
        # only 'time_s' is supported
        assert model_prof_prop == 'time_s'
        _assert_list_type(model_prof['time_s'], (int, float))


def _assert_model_profiles(model_profiles):
    assert isinstance(model_profiles, dict)
    for model in model_profiles:
        assert isinstance(model, str)
        _assert_model_profile(model_profiles[model])


def yaml_device_type(mem_MB: Union[int, float], bw_Mbps: Union[int, float],
                     model_profiles: Optional[dict]) -> dict:
    """Create a YAML device type."""
    assert isinstance(mem_MB, (int, float))
    assert isinstance(bw_Mbps, (int, float))
    if model_profiles is None:
        model_profiles = {}
    _assert_model_profiles(model_profiles)
    return {
        'mem_MB': mem_MB,
        'bw_Mbps': bw_Mbps,
        'model_profiles': model_profiles,
    }

def yaml_device_neighbors_type(bw_Mbps: Union[int, float]) -> dict:
    """Create a YAML device neighbors type."""
    assert isinstance(bw_Mbps, (int, float))
    return {
        'bw_Mbps': bw_Mbps,
        # Currently only one field, but could be extended, e.g., to include latency_{ms,us}.
    }

def yaml_device_neighbors(neighbors: List[str], bws_Mbps: Union[List[int], List[float]]) -> dict:
    """Create a YAML device neighbors."""
    _assert_list_type(neighbors, str)
    _assert_list_type(bws_Mbps, (int, float))
    return {
        neighbor: yaml_device_neighbors_type(bw_Mbps)
            for neighbor, bw_Mbps in zip(neighbors, bws_Mbps)
    }
