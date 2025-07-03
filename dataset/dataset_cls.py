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
    RandBiasFieldd,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
)

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import os
import pandas as pd
from ast import literal_eval


class RiskSet(Dataset):
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
        self.set_sampler()

    def set_sampler(self):
        class_counts = self.img_dict["pirads"].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        values = self.img_dict["pirads"].values.astype(int) - 2
        self.sampler_weight = class_weights[values]

    def cal_weight(self):
        class_counts = self.img_dict["pirads"].value_counts().sort_index().values
        return class_counts

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
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["highb"])
        adc = self.read(path["adc"])
        img = torch.stack([t2w, dwi, adc], 0)
        label = torch.tensor(int(path["pirads"]) - 2, dtype=torch.long)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img = trans_dict["image"]
        return img, label, torch.tensor(idx, dtype=torch.long)


class ScreeningSet(RiskSet):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase = phase, transforms=transforms)

    def set_sampler(self):
        class_counts = self.img_dict["result"].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        self.sampler_weight = class_weights[self.img_dict["result"].values]

    def cal_weight(self):
        class_counts = self.img_dict["result"].value_counts().sort_index().values
        return class_counts

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])
        img = torch.stack([t2w, dwi, adc], 0)
        label = torch.tensor(int(path["result"]), dtype=torch.long)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img = trans_dict["image"]
        return img, label, torch.tensor(idx, dtype=torch.long)


class PromisSet(RiskSet):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)

    def set_sampler(self):
        class_counts = self.img_dict["patient_level"].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        self.sampler_weight = class_weights[self.img_dict["patient_level"].values.astype(int)]

    def cal_weight(self):
        class_counts = self.img_dict["patient_level"].value_counts().sort_index().values
        return class_counts

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])
        img = torch.stack([t2w, dwi, adc], 0)
        zone_level = literal_eval(path["zone_level"])
        zone_level = torch.tensor(zone_level, dtype=torch.float32)
        #patient_level  = torch.tensor(int(path["patient_level"]), dtype=torch.float32)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img = trans_dict["image"]
        return img, zone_level, torch.tensor(idx, dtype=torch.long)

class Promis3HistSet(RiskSet):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)

    def set_sampler(self):
        class_counts = self.img_dict["def"].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        self.sampler_weight = class_weights[self.img_dict["def"].values.astype(int)]

    def cal_weight(self):
        class_counts = self.img_dict["def"].value_counts().sort_index().values
        return class_counts

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])
        img = torch.stack([t2w, dwi, adc], 0)
        label = torch.tensor(int(path["def"]), dtype=torch.long)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img = trans_dict["image"]
        return img, label, torch.tensor(idx, dtype=torch.long)

class Promis3GGSet(RiskSet):
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)

    def set_sampler(self):
        class_counts = self.img_dict["gleason"].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        self.sampler_weight = class_weights[self.img_dict["gleason"].values.astype(int)]

    def cal_weight(self):
        class_counts = self.img_dict["gleason"].value_counts().sort_index().values
        return class_counts

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])
        img = torch.stack([t2w, dwi, adc], 0)
        label = torch.tensor(int(path["gleason"]), dtype=torch.long)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img = trans_dict["image"]
        return img, label, torch.tensor(idx, dtype=torch.long)


def get_transforms(args):
    train_transforms = [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CenterSpatialCropd(keys="image", roi_size=(80, 300, 300)),
        RandRotated(
            keys="image",
            prob=0.3,
            range_x=10 / 180 * np.pi,
            range_y=10 / 180 * np.pi,
            range_z=10 / 180 * np.pi,
            keep_size=False,
            mode="bilinear",
        ),
        RandZoomd(
            keys="image",
            prob=0.3,
            min_zoom=[0.9, 0.9, 0.9],
            max_zoom=[1.1, 1.1, 1.1],
            mode="trilinear",
        ),
        SpatialPadd(
            keys="image",
            spatial_size=[round(i * 1.2) for i in args.crop_spatial_size],
        ),
        RandSpatialCropd(
            keys="image",
            roi_size=args.crop_spatial_size,
            random_size=False,
        ),
        RandFlipd(keys="image", prob=0.5, spatial_axis=2),
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
            CenterSpatialCropd(keys="image", roi_size=args.crop_spatial_size),
            SpatialPadd(keys="image", spatial_size=[i for i in args.crop_spatial_size]),
            # BinarizeLabeld(keys=["label"])
        ]
    )
    test_transforms = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CenterSpatialCropd(keys="image", roi_size=args.crop_spatial_size),
            SpatialPadd(keys="image", spatial_size=[i for i in args.crop_spatial_size]),
            # BinarizeLabeld(keys=["label"])
        ]
    )
    return train_transforms, val_transforms, test_transforms


def build_Risk_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    
    if args.demo:
        train_set = RiskSet(args, "demo/data/risk/train.csv", 'train', train_transforms)
        val_set = RiskSet(args, "demo/data/risk/val.csv", 'val', val_transforms)
        test_set = RiskSet(args, "demo/data/risk/test.csv", 'test', test_transforms)
    else:
        if args.data20:
            train_set = RiskSet(args, "spilt/risk/train_16.csv", 'train', train_transforms)
        else:
            train_set = RiskSet(args, "spilt/risk/train.csv", 'train', train_transforms)
        val_set = RiskSet(args, "spilt/risk/val.csv", 'val', val_transforms)
        test_set = RiskSet(args, "spilt/risk/test.csv", 'test', test_transforms)

    sampler = WeightedRandomSampler(
        weights=train_set.sampler_weight, num_samples=len(train_set), replacement=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
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
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.num_classes = 4
    return train_loader, val_loader, test_loader


def build_Screening_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.kfold is None:
        if args.data20:
            train_set = ScreeningSet(
                args, "spilt/screening/train_20.csv", 'train', train_transforms
            )
        else:
            train_set = ScreeningSet(
                args, "spilt/screening/train.csv", 'train', train_transforms
            )
        val_set = ScreeningSet(args, "spilt/screening/val.csv", 'val', val_transforms)
        test_set = ScreeningSet(args, "spilt/screening/test.csv", 'test', test_transforms)
        args.cls_account = train_set.cal_weight() / len(train_set)
    else:
        train_set = ScreeningSet(
            args, f"spilt/screening/train_{args.kfold}.csv", train_transforms
        )
        args.cls_account = train_set.cal_weight() / len(train_set)
        train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])
        val_set.transforms = val_transforms
        test_set = ScreeningSet(
            args, f"spilt/screening/test_{args.kfold}.csv", test_transforms
        )

    # sampler_weight = [train_set.dataset.sampler_weight[i] for i in train_set.indices]
    sampler = WeightedRandomSampler(
        weights=train_set.sampler_weight, num_samples=len(train_set), replacement=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
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
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.num_classes = 2
    return train_loader, val_loader, test_loader


# 4.0    453
# 3.0    206
# 5.0    195
# 2.0    174


def build_Promis_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.data20:
        train_set = PromisSet(args, "spilt/promis567_hist/train_20.csv", 'train', train_transforms)
    else:
        train_set = PromisSet(args, "spilt/promis567_hist/train.csv", 'train', train_transforms)
    val_set = PromisSet(args, "spilt/promis567_hist/val.csv", 'val', val_transforms)
    test_set = PromisSet(args, "spilt/promis567_hist/test.csv", 'test', test_transforms)

    # sampler = WeightedRandomSampler(
    #     weights=train_set.sampler_weight, num_samples=len(train_set), replacement=True
    # )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
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
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.num_classes = 20
    return train_loader, val_loader, test_loader

def build_Promis3_hist_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    train_set = Promis3HistSet(args, "spilt/promis_pirads3_hist/train.csv", 'train', train_transforms)
    val_set = Promis3HistSet(args, "spilt/promis_pirads3_hist/val.csv", 'val', val_transforms)
    test_set = Promis3HistSet(args, "spilt/promis_pirads3_hist/test.csv", 'test', test_transforms)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
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
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.num_classes = 3
    return train_loader, val_loader, test_loader

def build_Promis3_gg_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    train_set = Promis3GGSet(args, "spilt/promis_pirads3_gg/train.csv", 'train', train_transforms)
    val_set = Promis3GGSet(args, "spilt/promis_pirads3_gg/val.csv", 'val', val_transforms)
    test_set = Promis3GGSet(args, "spilt/promis_pirads3_gg/test.csv", 'test', test_transforms)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
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
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    args.num_classes = 5
    return train_loader, val_loader, test_loader
