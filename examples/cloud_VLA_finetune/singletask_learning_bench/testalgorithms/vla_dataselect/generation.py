from typing import Optional
from vla_component.experiments.robot.libero.run_libero_eval import GenerateConfig, eval_libero

def build_generation_cfg(
    model_family: str = "openvla",  # 默认值为 "openvla"
    pretrained_checkpoint: str = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/models--openvla--openvla-7b-finetuned-libero-10",
    task_suite_name: str = "libero_10",
    center_crop: bool = True,
    run_id_note: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "YOUR_WANDB_PROJECT",
    wandb_entity: str = "YOUR_WANDB_ENTITY",
    seed: int = 7,  # 默认随机种子
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    num_steps_wait: int = 10,
    num_trials_per_task: int = 2,
    local_log_dir: str = "./experiments/logs",
) -> GenerateConfig:
     # 构造并返回配置对象
    return GenerateConfig(
        model_family=model_family,
        pretrained_checkpoint=pretrained_checkpoint,
        task_suite_name=task_suite_name,
        center_crop=center_crop,
        run_id_note=run_id_note,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        seed=seed,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        num_steps_wait=num_steps_wait,
        num_trials_per_task=num_trials_per_task,
        local_log_dir=local_log_dir,
    )

def run_inference(cfg: GenerateConfig):
    """
    This function runs the inference based on the provided configuration (cfg).
    """
    # Access the original (unwrapped) eval_libero function via the `__wrapped__` attribute
    original_eval_libero = eval_libero.__wrapped__  # Access the original function
    original_eval_libero(cfg)  # Call the unwrapped version of eval_libero with cfg

def main():
    # 使用 build_cfg 构造配置对象
    cfg = build_generation_cfg()
    
    # 调用推理函数
    run_inference(cfg)

if __name__ == "__main__":
    main()