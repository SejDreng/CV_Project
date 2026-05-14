_base_ = ['./yolox_s_70e.py']
find_unused_parameters = True

# YOLOX-Nano is the smallest YOLOX variant available in this repo.
model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.25, use_depthwise=True),
    neck=dict(
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=True),
    bbox_head=dict(in_channels=64, feat_channels=64, use_depthwise=True))

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        dataset=dict(
            ann_file='/home/cv09f26/group9/CFINet/data/divData/Annotations/train.json',
            img_prefix='/home/cv09f26/group9/CFINet/data/divData/Images/train/',
            ori_ann_file='/home/cv09f26/group9/CFINet/data/SODA-D/rawData/Annotations/train.json')),
    val=dict(
        ann_file='/home/cv09f26/group9/CFINet/data/divData/Annotations/val.json',
        img_prefix='/home/cv09f26/group9/CFINet/data/divData/Images/val/',
        ori_ann_file='/home/cv09f26/group9/CFINet/data/SODA-D/rawData/Annotations/val.json'),
    test=dict(
        ann_file='/home/cv09f26/group9/CFINet/data/divData/Annotations/test.json',
        img_prefix='/home/cv09f26/group9/CFINet/data/divData/Images/test/',
        ori_ann_file='/home/cv09f26/group9/CFINet/data/SODA-D/rawData/Annotations/test.json'))

max_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
