# using_finetune.py
import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import Optional, Mapping

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

# 确保 PYTHONPATH 能找到 vlaScripts（你的模块命名以实际为准）
from vla_component.vlaScripts.finetune import FinetuneConfig, finetune

__all__ = ["build_cfg", "run_finetune"]

# ----------------------------
# Config 构造器（可被外部覆盖）
# ----------------------------
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

# ----------------------------
# 内部工具
# ----------------------------
def _base_finetune_call(cfg: FinetuneConfig):
    # 若 finetune 被 @draccus.wrap 装饰，__wrapped__ 指向真实函数
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
    os.environ["LOCAL_RANK"] = str(rank)  # 让 PartialState/local_process_index 正确
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

# ----------------------------
# 统一启动接口（可被外部 import）
# ----------------------------
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
    统一启动器：单/多卡皆可；既可传 cfg，也可用关键字覆盖构造配置。
    典型用法：
        from using_finetune import run_finetune, build_cfg
        # 方式一：先构 cfg 再传入
        cfg = build_cfg(dataset_name="libero_10_no_noops", batch_size=8)
        run_finetune(nproc_per_node=4, cfg=cfg)

        # 方式二：直接在调用处覆盖超参（内部会 build_cfg(**overrides)）
        run_finetune(nproc_per_node=1, dataset_name="libero_10_no_noops", batch_size=8)

    参数：
      - nproc_per_node: 进程数（=使用的 GPU 数）
      - cfg: 训练配置；若为空则用 cfg_overrides 调 build_cfg(...)
      - backend: "nccl"/"gloo"；默认自动选择
      - master_addr/master_port: rendezvous；多进程必须一致
      - env: 额外注入的环境变量（如 {"WANDB_MODE": "offline"}）
      - init_single_process_pg: 单进程时是否在此处初始化 PG（默认 False。
            若你的 finetune() 内部已做 _ensure_singleproc_ddp()，保持 False 避免重复 init）
      - **cfg_overrides: 直接覆盖 build_cfg 的同名参数
    """
    # 环境变量（可选）
    if env:
        os.environ.update({k: str(v) for k, v in env.items()})
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    # 构建/校验配置
    if cfg is None:
        cfg = build_cfg(**cfg_overrides)
    assert isinstance(cfg, FinetuneConfig), "cfg 必须是 FinetuneConfig 实例"

    backend = backend or _choose_backend()
    master_port = master_port or _free_port()

    if nproc_per_node > 1:
        # 多卡：spawn 多进程
        spawn(
            _worker,
            nprocs=nproc_per_node,
            args=(nproc_per_node, cfg, backend, master_addr, master_port),
        )
    else:
        # 单卡：默认不在这里 init PG，避免与 finetune 内部的单进程 init 重复
        if init_single_process_pg:
            _set_env_for_rank(master_addr, master_port, 0, 1)
            _init_pg_if_needed(backend, 0, 1)
        _base_finetune_call(cfg)

# ----------------------------
# 轻量 CLI（只负责并发/后端参数）
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified OpenVLA finetune launcher")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo", "mpi"], default=None)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--init-single-process-pg", action="store_true",
                        help="若 finetune() 未做单进程 PG 初始化，可开启此项由启动器来初始化。")
    args, unknown = parser.parse_known_args()

    # 这里不解析训练超参，保持由代码端传参构 cfg；需要 CLI 传参时，也可改为读取 JSON/YAML
    run_finetune(
        nproc_per_node=args.nproc_per_node,
        backend=args.backend,
        master_addr=args.master_addr,
        master_port=args.master_port,
        init_single_process_pg=args.init_single_process_pg,
    )
