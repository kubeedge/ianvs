model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="ResNet", depth=18, num_stages=4, out_indices=(3,), style="pytorch"
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="MultiLabelLinearClsHead",
        num_classes=20,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0, use_soft=True),
    ),
)
dataset_type = "BDD_Performance"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(256, -1)),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(256, -1)),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type="BDD_Performance",
        data_prefix="",
        ann_file="/home/liyunzhe/Mobile-Inference/algorithm/labels/0129_real_world_multi_label_remo_xyxy_bdd_train.txt",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", size=(256, -1)),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="ToTensor", keys=["gt_label"]),
            dict(type="Collect", keys=["img", "gt_label"]),
        ],
    ),
    val=dict(
        type="BDD_Performance",
        data_prefix="",
        ann_file="/home/liyunzhe/Mobile-Inference/algorithm/labels/0129_real_world_multi_label_remo_xyxy_bdd_val.txt",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", size=(256, -1)),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
    test=dict(
        type="BDD_Performance",
        data_prefix="",
        ann_file="/home/liyunzhe/Mobile-Inference/algorithm/labels/0129_real_world_multi_label_remo_xyxy_bdd_val.txt",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", size=(256, -1)),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
)
evaluation = dict(interval=1, metric="mAP")
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy="step", step=[30, 60, 90])
runner = dict(type="EpochBasedRunner", max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type="TextLoggerHook")])
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
work_dir = "work_dirs/220208-bdd-best"
gpu_ids = range(0, 1)
