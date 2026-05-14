_base_ = ['/home/cv09f26/group9/mmyolo/mmyolo/configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py']

# SODA-D classes
class_name = (
    'people', 'rider', 'bicycle', 'motor', 'vehicle',
    'traffic-sign', 'traffic-light', 'traffic-camera', 'warning-cone')
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# Data paths (COCO-style split annotations/images)
data_root = '/home/cv09f26/group9/CFINet/data/divData/'
train_ann_file = 'Annotations/train.json'
val_ann_file = 'Annotations/val.json'
test_ann_file = 'Annotations/test.json'
train_data_prefix = 'Images/train/'
val_data_prefix = 'Images/val/'
test_data_prefix = 'Images/test/'

# Training schedule
max_epochs = 1
close_mosaic_epochs = 1
save_epoch_intervals = 1

# Explicitly define key pipeline hyper-params used below.
# YOLO feature maps require input dims divisible by 32.
img_scale = (1216, 1216)
affine_scale = 0.5
max_aspect_ratio = 100

# Throughput knobs (adjust if OOM)
train_batch_size_per_gpu = 8
train_num_workers = 4
val_batch_size_per_gpu = 1
val_num_workers = 2

model = dict(
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]

# Keep YOLOv8 augmentation flow but remove mmdet.Albu to avoid
# albumentations key mismatch issues in this environment.
last_transform = [
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)),
    *last_transform
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        metainfo=metainfo,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        metainfo=metainfo))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=test_data_prefix),
        metainfo=metainfo))

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = dict(ann_file=data_root + test_ann_file)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=2))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

