dataset_type = 'SODAADataset'
data_root = '/home/cv09f26/group9/CFINet/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'SODA-A/divData/train/Annotations/',
        img_prefix=data_root + 'SODA-A/divData/train/Images/',
        pipeline=train_pipeline,
        ori_ann_file=data_root + 'SODA-A/Annotations/train/',
        filter_empty_gt=True,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'SODA-A/divData/val/Annotations/',
        img_prefix=data_root + 'SODA-A/divData/val/Images/',
        pipeline=test_pipeline,
        ori_ann_file=data_root + 'SODA-A/Annotations/val/',
        filter_empty_gt=False,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'SODA-A/divData/test/Annotations/',
        img_prefix=data_root + 'SODA-A/divData/test/Images/',
        pipeline=test_pipeline,
        ori_ann_file=data_root + 'SODA-A/Annotations/test/',
        filter_empty_gt=False,
    ),
)