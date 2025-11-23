<<<<<<< HEAD
# using_finetune.py
import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import Optional, Mapping

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from vla_component.vlaScripts.finetune import FinetuneConfig, finetune

__all__ = ["build_cfg", "run_finetune"]

def build_cfg(
    vla_path: str = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/models--openvla--openvla-7b",
    data_root_dir: str = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/data/openvla/modified_libero_rlds",
    dataset_name: str = "libero_10_no_noops",
    run_root_dir: str = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/finetune_model",
    adapter_tmp_dir: str = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/adapter",
    batch_size: int = 8,
    max_steps: int = 400,
    save_steps: int = 400,
    learning_rate: float = 5e-4,
    grad_accumulation_steps: int = 1,
    image_aug: bool = True,
    shuffle_buffer_size: int = 100_000,
    save_latest_checkpoint_only: bool = True,
    use_lora: bool = True,
    lora_rank: int = 32,
    lora_dropout: float = 0.0,
    use_quantization: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    run_id_note: Optional[str] = None,
) -> FinetuneConfig:
    return FinetuneConfig(
        vla_path=vla_path,
        data_root_dir=Path(data_root_dir),
        dataset_name=dataset_name,
        run_root_dir=Path(run_root_dir),
        adapter_tmp_dir=Path(adapter_tmp_dir),
        batch_size=batch_size,
        max_steps=max_steps,
        save_steps=save_steps,
        learning_rate=learning_rate,
        grad_accumulation_steps=grad_accumulation_steps,
        image_aug=image_aug,
        shuffle_buffer_size=shuffle_buffer_size,
        save_latest_checkpoint_only=save_latest_checkpoint_only,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        use_quantization=use_quantization,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_id_note=run_id_note,
    )

def _base_finetune_call(cfg: FinetuneConfig):
    fn = getattr(finetune, "__wrapped__", finetune)
    return fn(cfg)

def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def _choose_backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"

def _set_env_for_rank(master_addr: str, master_port: int, rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)  
    os.environ["WORLD_SIZE"] = str(world_size)

def _init_pg_if_needed(backend: str, rank: int, world_size: int, timeout_min: int = 30):
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=timeout_min),
        )

def _worker(rank: int, world_size: int, cfg: FinetuneConfig, backend: str, master_addr: str, master_port: int):
    _set_env_for_rank(master_addr, master_port, rank, world_size)
    _init_pg_if_needed(backend, rank, world_size)
    _base_finetune_call(cfg)

def run_finetune(
    nproc_per_node: int = 1,
    cfg: Optional[FinetuneConfig] = None,
    *,
    backend: Optional[str] = None,
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
    init_single_process_pg: bool = False,
    **cfg_overrides,
) -> None:
    """
        Unified Launcher: Supports both single and multi-GPU training; accepts either a config object or keyword overrides.
        Typical usage:
            from using_finetune import run_finetune, build_cfg
            # Method 1: Build config first then pass
            cfg = build_cfg(dataset_name="libero_10_no_noops", batch_size=8)
            run_finetune(nproc_per_node=4, cfg=cfg)

            # Method 2: Override hyperparameters directly at call site (internally calls build_cfg(**overrides))
            run_finetune(nproc_per_node=1, dataset_name="libero_10_no_noops", batch_size=8)

        Parameters:
        - nproc_per_node: Number of processes (= number of GPUs used)
        - cfg: Training configuration; if None, uses cfg_overrides to call build_cfg(...)
        - backend: "nccl"/"gloo"; automatically selected by default
        - master_addr/master_port: Rendezvous parameters; must be consistent across multi-process
        - env: Additional environment variables to inject (e.g., {"WANDB_MODE": "offline"})
        - init_single_process_pg: Whether to initialize process group for single process (default False.
                If your finetune() internally already calls _ensure_singleproc_ddp(), keep False to avoid duplicate init)
        - **cfg_overrides: Direct overrides for build_cfg parameters with same names
    """

    if env:
        os.environ.update({k: str(v) for k, v in env.items()})
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    if cfg is None:
        cfg = build_cfg(**cfg_overrides)
    assert isinstance(cfg, FinetuneConfig), "cfg must be an instance of FinetuneConfig"

    backend = backend or _choose_backend()
    master_port = master_port or _free_port()

    if nproc_per_node > 1:
        # Multi-GPU: Spawn multiple processes
        exp_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.use_quantization:
            exp_id += "+q-4bit"
        if cfg.run_id_note is not None:
            exp_id += f"--{cfg.run_id_note}"
        if cfg.image_aug:
            exp_id += "--image_aug"

        # Start =>> Build Directories
        run_dir = cfg.run_root_dir / exp_id
        spawn(
            _worker,
            nprocs=nproc_per_node,
            args=(nproc_per_node, cfg, backend, master_addr, master_port),
        )
        return run_dir
    else:
        # single gpu
        if init_single_process_pg:
            _set_env_for_rank(master_addr, master_port, 0, 1)
            _init_pg_if_needed(backend, 0, 1)
        checkpoint_path = _base_finetune_call(cfg)
        return checkpoint_path
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified OpenVLA finetune launcher")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo", "mpi"], default=None)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--init-single-process-pg", action="store_true")
    args, unknown = parser.parse_known_args()
    
    run_finetune(
        nproc_per_node=args.nproc_per_node,
        backend=args.backend,
        master_addr=args.master_addr,
        master_port=args.master_port,
        init_single_process_pg=args.init_single_process_pg,
    )
=======
version https://git-lfs.github.com/spec/v1
oid sha256:6dfcb4b48f8047a91170799bf6ae08a6e0be8d1bc8506df4f9ee0badf0802025
size 7493
>>>>>>> 9676c3e (ya toh aar ya toh par)
