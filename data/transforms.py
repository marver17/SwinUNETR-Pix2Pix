import numpy as np
import torch
from .datautils import get_window_viewing_value
import os
from monai import transforms
from data.solid_head_transform import GenerateSolidHeadMaskd,ApplyMaskAsBackgroundd


def get_transforms(opt):
    transform_list = [
        transforms.LoadImaged(keys=["CT", "MR",], image_only=False, reader="ITKReader"),
        transforms.EnsureChannelFirstd(keys=["CT", "MR"]),
        #transforms.EnsureTyped(keys=["CT", "MR"], device=torch.device("cuda:0"), track_meta=False),
    ]


    # Set Position - MRCT preprocessing plus ### this transformation is used for mr2ct 

    if opt.transform.fix_pose.enable : 
        transform_list.extend([
                GenerateSolidHeadMaskd(keys= ["CT", "MR"],
                                            new_key="head_mask"),
                ApplyMaskAsBackgroundd(keys=["CT", "MR"],
                                         mask_key="head_mask",
                                         background_values={"CT": -1024, "MR": 0}),
                #transforms.EnsureTyped(keys=["CT", "MR"], device=torch.device("cpu"), track_meta=False),
]    
        )
    if opt.transform.uniformize_dimension.enable : 
        transform_list.extend([ transforms.Spacingd(keys=["CT","MR"], pixdim=(1, 1, 1), mode=("bilinear", "bilinear")),
                                transforms.Orientationd(keys=["CT","MR"], axcodes="RAS"),
                                transforms.Resized(
                                    keys=["CT","MR"],
                                    spatial_size=opt.transform.resize.size,
                                    mode=opt.transform.resize.mode)] )




        # Resize transformation
    if opt.transform.resize.enable:
        transform_list.append(
            transforms.Resized(
                keys=["CT","MR"],
                spatial_size=opt.transform.resize.size,
                mode=opt.transform.resize.mode
            )
        )

    # Normalize transformation
    if opt.transform.normalize.enable:
        transform_list.append(
            transforms.NormalizeIntensityd(
                keys=opt.transform.normalize.keys,
                subtrahend=opt.transform.normalize.subtrahend,
                divisor=opt.transform.normalize.divisor
            )
        )



    # Scale intensity transformation
    if opt.transform.scaleintensity.enable:
        if opt.transform.scaleintensity.windowing.enable:
            window_width, window_level = get_window_viewing_value(opt.transform.scaleintensity.windowing.windows)
            window_min = window_level - window_width // 2
            window_max = window_level + window_width // 2
            
            transform_list.extend([
                transforms.ScaleIntensityRanged(
                    keys="CT",
                    a_min=window_min,
                    a_max=window_max,
                    b_min=opt.transform.scaleintensity.min,
                    b_max=opt.transform.scaleintensity.max,
                    clip=True
                ),
                transforms.ScaleIntensityd(
                    keys="MR",
                    minv=opt.transform.scaleintensity.min,
                    maxv=opt.transform.scaleintensity.max)])

        else:
            transform_list.append(
                transforms.ScaleIntensityd(
                    keys=["CT", "MR"],
                    minv=opt.transform.scaleintensity.min,
                    maxv=opt.transform.scaleintensity.max
                )
            )

    # Random crop transformation
    if opt.transform.random_crop.enable:
        transform_list.append(
            transforms.RandSpatialCropd(
                keys=["CT","MR"],
                roi_size=opt.transform.random_crop.size
            )
        )
    # Random flip transformation
    if opt.transform.random_flip.enable:
        transform_list.append(
            transforms.RandFlipd(
                keys=["CT","MR"],
                prob=opt.transform.random_flip.probability,
                spatial_axis=opt.transform.random_flip.axes
            )
        )

    # Random rotate transformation
    if opt.transform.random_rotate.enable:
        transform_list.append(
            transforms.RandRotated(
                keys=["CT","MR"],
                prob=opt.transform.random_rotate.probability,
                range_x=opt.transform.random_rotate.angle_range
            )
        )

    # Random intensity transformation
    if opt.transform.random_intensity.enable:
        transform_list.append(
            transforms.RandScaleIntensityd(
                keys=["CT","MR"],
                prob=opt.transform.random_intensity.probability,
                factors=opt.transform.random_intensity.factor_range
            )
        )

    return transforms.Compose(transform_list)



    