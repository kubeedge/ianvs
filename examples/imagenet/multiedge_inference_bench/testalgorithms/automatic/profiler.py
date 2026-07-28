"""Module shard profiler."""
import argparse
import gc
import os
import time
import numpy as np
import psutil
import torch
import torch.multiprocessing as mp
import yaml
from transformers import BertTokenizer
import devices
import model_cfg


def get_shapes(tensors):
    """Get the tensor shapes, excluding the outer dimension (microbatch size)."""
    if isinstance(tensors, tuple):
        shape = []
        for tensor in tensors:
            shape.append(tuple(tensor.shape[1:]))
    else:
        shape = [tuple(tensors.shape[1:])]
    return shape


def create_module_shard(module_cfg, stage_cfg):
    """Create a module shard."""
    model_name = module_cfg['name']
    model_file = module_cfg['file']
    stage = stage_cfg['stage']
    layer_start = stage_cfg['layer_start']
    layer_end = stage_cfg['layer_end']
    return model_cfg.module_shard_factory(model_name, model_file, layer_start, layer_end, stage)


def profile_module_shard(module_cfg, stage_cfg, stage_inputs, warmup, iterations):
    """Profile a module shard."""
    process = psutil.Process(os.getpid())

    # Measure memory (create shard) on the CPU.
    # This avoids capturing additional memory overhead when using other devices, like GPUs.
    # It's OK if the model fits in DRAM but not on the "device" - we'll just fail later.
    # We consider memory requirements to be a property of the model, not the device/platform.
    assert devices.DEVICE is None
    # Capturing memory behavior in Python is extremely difficult and results are subject to many
    # factors beyond our ability to control or reliably detect/infer.
    # This works best when run once per process execution with only minimal work done beforehand.
    gc.collect()
    stage_start_mem = process.memory_info().rss / 1000000
    module = create_module_shard(module_cfg, stage_cfg)
    gc.collect()
    stage_end_mem = process.memory_info().rss / 1000000

    # Now move the module to the specified device
    device = module_cfg['device']
    if device is not None:
        devices.DEVICE = torch.device(device)
    if devices.DEVICE is not None and devices.DEVICE.type == 'cuda':
        torch.cuda.init()
    module.to(device=device)
    module.register_forward_pre_hook(devices.forward_pre_hook_to_device)
    module.register_forward_hook(devices.forward_hook_to_cpu)

    # Measure data input
    shape_in = get_shapes(stage_inputs)

    # Optional warmup
    if warmup:
        module(stage_inputs)

    # Measure timing (execute shard) - includes data movement overhead (performed in hooks)
    stage_times = []
    for _ in range(iterations):
        stage_start_time = time.time()
        stage_outputs = module(stage_inputs)
        stage_end_time = time.time()
        stage_times.append(stage_end_time - stage_start_time)
    stage_time_avg = sum(stage_times) / len(stage_times)

    # Measure data output
    shape_out = get_shapes(stage_outputs)

    results = {
        'shape_in': shape_in,
        'shape_out': shape_out,
        'memory': stage_end_mem - stage_start_mem,
        'time': stage_time_avg,
    }
    return (stage_outputs, results)


def profile_module_shard_mp_queue(queue, evt_done, args):
    """Multiprocessing target function for `profile_module_shard` which adds output to queue."""
    queue.put(profile_module_shard(*args))
    evt_done.wait()


def profile_module_shard_mp(args):
    """Run `profile_module_shard` with multiprocessing (for more accurate memory results)."""
    # First, a non-optional module warmup in case PyTorch needs to fetch/cache models on first use
    print("Performing module warmup...")
    proc = mp.Process(target=create_module_shard, args=(args[0], args[1]))
    proc.start()
    proc.join()

    # Now, the actual profiling
    print("Performing module profiling...")
    queue = mp.Queue()
    # The child process sometimes exits before we read the queue items, even though it should have
    # flushed all data to the underlying pipe before that, so use an event to keep it alive.
    evt_done = mp.Event()
    proc = mp.Process(target=profile_module_shard_mp_queue, args=(queue, evt_done, args))
    proc.start()
    tensors, prof_dict = queue.get()
    evt_done.set()
    proc.join()
    return (tensors, prof_dict)


def profile_layers(module_cfg, tensors, layer_start, layer_end, warmup, iterations):
    """Profile a shard with layer_start through layer_end."""
    shard = {
        'stage': 0,
        'layer_start': layer_start,
        'layer_end': layer_end,
    }
    _, prof_dict = profile_module_shard_mp(args=(module_cfg, shard, tensors, warmup, iterations))
    prof_dict['layer'] = 0
    return [prof_dict]


def profile_layers_individually(module_cfg, tensors, layer_start, layer_end, warmup, iterations):
    """Profile module shards for each layer individually."""
    results = []
    for layer in range(layer_start, layer_end + 1):
        shard = {
            'stage': layer,
            'layer_start': layer,
            'layer_end': layer,
        }
        tensors, prof_dict = profile_module_shard_mp(args=(module_cfg, shard, tensors, warmup, iterations))
        prof_dict['layer'] = layer
        results.append(prof_dict)
    return results


def profile_layers_cumulatively(module_cfg, tensors, layer_start, layer_end, warmup, iterations):
    """Profile module shards with increasing numbers of layers."""
    results = []
    for layer in range(1, layer_end + 1):
        shard = {
            'stage': layer,
            'layer_start': layer_start,
            'layer_end': layer,
        }
        _, prof_dict = profile_module_shard_mp(args=(module_cfg, shard, tensors, warmup, iterations))
        prof_dict['layer'] = layer
        results.append(prof_dict)
    return results


def validate_profile_results(profile_results, args, inputs, model_layers, layer_end):
    """Validate that we can work with existing profiling results"""
    assert profile_results['model_name'] == args.model_name, "model name mismatch with existing results"
    dtype = inputs[0].dtype if isinstance(inputs, tuple) else inputs.dtype
    assert profile_results['dtype'] == str(dtype), "dtype mismatch with existing results"
    assert profile_results['batch_size'] == args.batch_size, "batch size mismatch with existing results"
    assert profile_results['layers'] == model_layers, "layer count mismatch with existing results"
    # check for overlap with existing results data
    for _layer in range(args.layer_start, layer_end + 1):
        for _pd in profile_results['profile_data']:
            assert _layer != _pd['layer'], "layer to be profiled already in existing results"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Module Shard Profiler",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--results-yml", default="./examples/imagenet/multiedge_inference_bench/testalgorithms/automatic/profiler_results.yml", type=str,
                        help="output YAML file")
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="compute device type to use, with optional ordinal, "
                             "e.g.: 'cpu', 'cuda', 'cuda:1'")
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str,
                        help="the model file, if not in working directory")
    parser.add_argument("-l", "--layer-start", default=1, type=int, help="start layer")
    parser.add_argument("-L", "--layer-end", type=int, help="end layer; default: last layer in the model")
    parser.add_argument("-s", "--shape-input", type=str, action='append',
                        help="comma-delimited shape input, e.g., '3,224,224' (required for start_layer != 1)")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("-w", "--warmup", action="store_true", default=True,
                        help="perform a warmup iteration "
                             "(strongly recommended, esp. with device='cuda' or iterations>1)")
    parser.add_argument("--no-warmup", action="store_false", dest="warmup",
                        help="don't perform a warmup iteration")
    parser.add_argument("-i", "--iterations", default=1, type=int,
                        help="iterations to average runtime for")
    args = parser.parse_args()

    if args.shape_input is not None:
        shapes = []
        for shp in args.shape_input:
            shapes.append(tuple(int(d) for d in shp.split(',')))
        if len(shapes) > 1:
            # tuple of tensors
            inputs = tuple(torch.randn(args.batch_size, *shp) for shp in shapes)
        else:
            # single tensor
            inputs = torch.randn(args.batch_size, *shapes[0])
    elif args.model_name in ['bert-base-uncased', 'bert-large-uncased']:
        with np.load("bert_input.npz") as bert_inputs:
            inputs_sentence = list(bert_inputs['input'][0: args.batch_size])
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        inputs = tokenizer(inputs_sentence, padding=True, truncation=True, return_tensors="pt")['input_ids']
    else:
        inputs = torch.randn(args.batch_size, 3, 224, 224)

    model_layers = model_cfg.get_model_layers(args.model_name)
    layer_end = args.layer_end
    if layer_end is None:
        layer_end = model_layers

    # get or create profile_results
    if os.path.exists(args.results_yml):
        print("Using existing results file")
        with open(args.results_yml, 'r', encoding='utf-8') as yfile:
            profile_results = yaml.safe_load(yfile)
        validate_profile_results(profile_results, args, inputs, model_layers, layer_end)
    else:
        profile_results = {
            'model_name': args.model_name,
            'dtype': str(inputs.dtype),
            'batch_size': args.batch_size,
            'layers': model_layers,
            'profile_data': [],
        }

    module_cfg = {
        'device': args.device,
        'name': args.model_name,
        'file': args.model_file,
    }
    if args.model_file is None:
        module_cfg['file'] = model_cfg.get_model_default_weights_file(args.model_name)
    # a single shard measurement can be a useful reference
    # results = profile_layers(module_cfg, inputs, args.layer_start, layer_end, args.warmup, args.iterations)
    # cumulative won't work if the whole model doesn't fit on the device
    # results = profile_layers_cumulatively(module_cfg, inputs, args.layer_start, layer_end, args.warmup, args.iterations)
    results = profile_layers_individually(module_cfg, inputs, args.layer_start, layer_end, args.warmup, args.iterations)

    # just a dump of the configuration and profiling results
    profile_results['profile_data'].extend(results)
    profile_results['profile_data'].sort(key=lambda pd: pd['layer'])
    with open(args.results_yml, 'w', encoding='utf-8') as yfile:
        yaml.safe_dump(profile_results, yfile, default_flow_style=None, encoding='utf-8')


if __name__=="__main__":
    main()
