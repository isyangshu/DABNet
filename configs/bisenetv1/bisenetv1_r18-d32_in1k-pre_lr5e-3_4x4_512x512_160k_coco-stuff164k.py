_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/coco-stuff164k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))),
    decode_head=dict(num_classes=171),
    auxiliary_head=[
        dict(num_classes=171),
        dict(num_classes=171),
    ])
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.005)
