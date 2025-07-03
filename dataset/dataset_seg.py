import pickle
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    RandBiasFieldd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
)

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import os
import pandas as pd


class BaseVolumeDataset(Dataset):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__()
        self.img_dict = pd.read_csv(image_paths)
        if phase == 'train':
            if args.data_num > 0:
                # crop the dataset
                self.img_dict = self.img_dict.iloc[: args.data_num]
        print(f"Loading {phase} dataset with {len(self.img_dict)} samples")
        self.root = args.root
        self._set_dataset_stat()
        self.transforms = transforms  # self.get_transforms()

    def _set_dataset_stat(self):
        self.spacing = (0.5, 0.5, 1.0)
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.target_class = 1

    def __len__(self):
        return len(self.img_dict)

    def read(self, path):
        vol = nib.load(os.path.join(self.root, path))
        vol = vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        vol = torch.from_numpy(vol)
        return vol

    def __getitem__(self, idx):
        return NotImplemented


class UCLSet(BaseVolumeDataset):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])
        img = torch.stack([t2w, dwi, adc], 0)
        seg = self.read(path["lesion"]).unsqueeze(0)
        seg = seg > 0
        # print(img.shape)
        # seg = (seg == self.target_class).float()
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img, "label": seg})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img, seg = trans_dict["image"], trans_dict["label"]
        return img, seg, torch.tensor(idx, dtype=torch.long)

# TODO: need to update; unfinished
"""
class UCL2DSet(BaseVolumeDataset):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)
    
    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])

        seg = self.read(path["lesion"]).unsqueeze(0)
        seg = seg > 0

        seg_mask = seg.squeeze(0).numpy()
        non_zero_slices = np.where(seg_mask.any(axis=1,2))[0]
        if len(non_zero_slices) > 0:
            sampled_slices = np.random.choice(non_zero_slices, min(N, len(non_zero_slices)), replace=False)
            filtered_seg = np.zeros_like(seg_mask)
            filtered_seg[sampled_slices] = seg_mask[sampled_slices]
        else:
            filtered_seg = seg_mask
        
        img = torch.stack([t2w, dwi, adc], 0)
        seg = torch.tensor(filtered_seg, dtype=torch.float32).unsqueeze(0)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img, "label": seg})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img, seg = trans_dict["image"], trans_dict["label"]
        return img, seg, torch.tensor(idx, dtype=torch.long)
"""

class AnatomySet(BaseVolumeDataset):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)
    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        # img = t2w.unsqueeze(0)
        zero = torch.zeros_like(t2w)
        # modified to align img to 3 channel
        img = torch.stack([t2w, zero, zero], 0)
        seg = self.read(path["mask"]).unsqueeze(0)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img, "label": seg})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img, seg = trans_dict["image"], trans_dict["label"]
        return img, seg, torch.tensor(idx, dtype=torch.long)


class BpAnatomySet(BaseVolumeDataset):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        zero = torch.zeros_like(t2w)
        img = torch.stack([t2w, zero, zero], 0)
        seg = self.read(path["mask"]).unsqueeze(0)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img, "label": seg})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img, seg = trans_dict["image"], trans_dict["label"]
        return img, seg, torch.tensor(idx, dtype=torch.long)        

class PromisHist(BaseVolumeDataset):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])
        img = torch.stack([t2w, dwi, adc], 0)

        zone_mask = self.read(path["gland"]).unsqueeze(0)

        zone_level = list(map(int, path["zone_label"].split()))
        zone_level = torch.tensor(zone_level)
        
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img, "label": zone_mask})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img, zone_mask = trans_dict["image"], trans_dict["label"]

        return img, zone_mask, zone_level

class PromisZone(BaseVolumeDataset):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])
        img = torch.stack([t2w, dwi, adc], 0)

        zone_mask = self.read(path["zome_mask"]).unsqueeze(0)

        zone_level = list(map(int, path["zone_label"].split()))
        zone_level = torch.tensor(zone_level)
        
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img, "label": zone_mask})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img, zone_mask = trans_dict["image"], trans_dict["label"]

        return img, zone_mask, zone_level

def get_transforms(args):
    train_transforms = [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandRotated(
            keys=["image", "label"],
            prob=0.3,
            range_x=30 / 180 * np.pi,
            keep_size=False,
            mode=["bilinear", "nearest"],
        ),
        RandZoomd(
            keys=["image", "label"],
            prob=0.3,
            min_zoom=[1, 0.9, 0.9],
            max_zoom=[1, 1.1, 1.1],
            mode=["trilinear", "nearest"],
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=[round(i * 1.2) for i in args.crop_spatial_size],
        ),
        # RandCropByPosNegLabeld(
        #     keys=["image", "label"],
        #     spatial_size=[round(i * 1.2) for i in args.crop_spatial_size],
        #     label_key="label",
        #     pos=2,
        #     neg=1,
        #     num_samples=1,
        # ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=args.crop_spatial_size,
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # BinarizeLabeld(keys=["label"])
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.8),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.8),
        RandBiasFieldd(keys="image", prob=0.2),
        RandGaussianSmoothd(keys="image", prob=1.0)
    ]

    train_transforms = Compose(train_transforms)
    val_transforms = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CenterSpatialCropd(
                keys=["image", "label"], roi_size=args.crop_spatial_size
            ),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=[i for i in args.crop_spatial_size],
            ),
            # BinarizeLabeld(keys=["label"])
        ]
    )
    test_transforms = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CenterSpatialCropd(
                keys=["image", "label"], roi_size=args.crop_spatial_size
            ),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=[i for i in args.crop_spatial_size],
            ),
            # BinarizeLabeld(keys=["label"])
        ]
    )
    return train_transforms, val_transforms, test_transforms


def build_UCL_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.demo:
        test_set = UCLSet(args, "demo/data/UCL/test.csv", 'test', test_transforms)
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=14,
            drop_last=False,
        )
        args.in_channels = 3
        args.out_channels = 1
        args.num_classes = 1
        return test_loader
    else:
        if args.data20:
            train_set = UCLSet(args, "spilt/UCL/train_16.csv", 'train', train_transforms)
        else:
            train_set = UCLSet(args, "spilt/UCL/train.csv", 'train', train_transforms)
        val_set = UCLSet(args, "spilt/UCL/val.csv", 'val', val_transforms)
        test_set = UCLSet(args, "spilt/UCL/test.csv", 'test', test_transforms)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=14,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=14,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=14,
            drop_last=False,
        )
        args.in_channels = 3
        args.out_channels = 1
        args.num_classes = 1
        return train_loader, val_loader, test_loader


def build_Promis_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.data20:
        train_set = UCLSet(args, "spilt/promis567/train_20.csv", 'train', train_transforms)
    else:
        train_set = UCLSet(args, "spilt/promis567/train.csv", 'train', train_transforms)
    val_set = UCLSet(args, "spilt/promis567/val.csv", 'val', val_transforms)
    test_set = UCLSet(args, "spilt/promis567/test.csv", 'test', test_transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.out_channels = 1
    args.num_classes = 1
    return train_loader, val_loader, test_loader


def build_Anatomy_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.data20:
        train_set = AnatomySet(args, "spilt/anatomy/train_20.csv", 'train', train_transforms)
    else:
        train_set = AnatomySet(args, "spilt/anatomy/train.csv", 'train', train_transforms)
    val_set = AnatomySet(args, "spilt/anatomy/val.csv", 'val', val_transforms)
    test_set = AnatomySet(
        args,
        "spilt/anatomy/test.csv",
        'test',
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    if args.prompt:
        # TODO: need to update; currently not in use
        args.in_channels = 3
    else:
        args.in_channels = 3
    args.out_channels = 9
    args.num_classes = 8
    return train_loader, val_loader, test_loader


def build_BpAnatomy_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.data20:
        train_set = BpAnatomySet(args, "spilt/anatomy/train_20.csv", 'train', train_transforms)
    else:
        train_set = BpAnatomySet(args, "spilt/anatomy/train.csv", 'train', train_transforms)
    val_set = BpAnatomySet(args, "spilt/anatomy/val.csv", 'val', val_transforms)
    test_set = BpAnatomySet(
        args,
        "spilt/anatomy/test.csv",
        'test',
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )
    args.in_channels = 3
    args.out_channels = 9
    args.num_classes = 8
    return train_loader, val_loader, test_loader


def build_PromisHist_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.data20:
        train_set = PromisHist(args, "spilt/promis567_hist/train_20.csv", 'train', train_transforms)
    else:
        train_set = PromisHist(args, "spilt/promis567_hist/train.csv", 'train', train_transforms)
    val_set = PromisHist(args, "spilt/promis567_hist/val.csv", 'val', val_transforms)
    test_set = PromisHist(args, "spilt/promis567_hist/test.csv", 'test', test_transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.out_channels = 1
    args.num_classes = 1
    return train_loader, val_loader, test_loader

def build_PromisZone_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    train_set = PromisZone(args, "spilt/promis_zone/train.csv", 'train', train_transforms)
    val_set = PromisZone(args, "spilt/promis_zone/val.csv", 'val', val_transforms)
    test_set = PromisZone(args, "spilt/promis_zone/test.csv", 'test', test_transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.out_channels = 1
    args.num_classes = 1
    return train_loader, val_loader, test_loader


def build_PromisPirads3_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.data20:
        train_set = UCLSet(args, "spilt/promis_pirads3/train_15.csv", 'train', train_transforms)
    else:
        train_set = UCLSet(args, "spilt/promis_pirads3/train.csv", 'train', train_transforms)
    val_set = UCLSet(args, "spilt/promis_pirads3/val.csv", 'val', val_transforms)
    test_set = UCLSet(args, "spilt/promis_pirads3/test.csv", 'test', test_transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.out_channels = 1
    args.num_classes = 1
    return train_loader, val_loader, test_loader
