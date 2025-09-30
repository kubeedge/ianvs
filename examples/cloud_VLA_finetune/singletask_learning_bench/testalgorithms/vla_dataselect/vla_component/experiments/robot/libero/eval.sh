# Launch LIBERO-10 (LIBERO-Long) evals
python /inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/openvla/experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/models--openvla--openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True