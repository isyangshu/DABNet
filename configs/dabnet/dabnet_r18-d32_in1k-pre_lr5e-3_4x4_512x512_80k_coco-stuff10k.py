_base_ = [
    '../_base_/models/dabnet_r18-d32-CoCo.py',
    '../_base_/datasets/coco-stuff10k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))),)
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.005)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)