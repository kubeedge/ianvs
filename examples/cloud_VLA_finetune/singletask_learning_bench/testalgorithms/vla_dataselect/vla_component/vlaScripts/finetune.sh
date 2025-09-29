torchrun --standalone --nnodes 1 --nproc-per-node 1 /inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/openvla/vla-scripts/finetune.py \
  --vla_path "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/models--openvla--openvla-7b" \
  --data_root_dir /inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/data/openvla/modified_libero_rlds \
  --dataset_name libero_10_no_noops \
  --run_root_dir /inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/fitune_model \
  --adapter_tmp_dir /inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/adapter \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 400 \
  --max_steps 400 > /inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/openvla/log/test_log.log 2>&1

# --wandb_project <PROJECT> \
# --wandb_entity <ENTITY> \