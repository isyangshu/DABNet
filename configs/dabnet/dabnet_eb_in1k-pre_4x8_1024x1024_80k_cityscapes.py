_base_ = [
    '../_base_/models/dabnet_eb1-d32.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k-Adam.py'
]

optimizer = dict(type='Adam', lr=0.001, betas=(0.5, 0.999))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
)
