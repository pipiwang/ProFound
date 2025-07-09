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
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Spacingd,
    Orientationd,
)

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import os
import pandas as pd
import util.misc as misc


class BinarizeLabeld(MapTransform):
    def __init__(
        self,
        keys,
        threshold: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d


class SSLDataset(Dataset):
    def __init__(self, args, image_paths, transforms=None):
        super().__init__()
        self.img_dict = pd.read_csv(image_paths)
        self.root = args.root
        self._set_dataset_stat()
        self.transforms = transforms  # self.get_transforms()

    def _set_dataset_stat(self):
        self.spacing = (1.0, 1.0, 1.0)
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
        dwi = self.read(path["dwi"])
        adc = self.read(path["adc"])
        # z_min = []
        # z_max = []
        # for img in [t2w, adc, dwi]:
        #     z,x,y = img.nonzero(as_tuple=True)
        #     z_min.append(z.min())
        #     z_max.append(z.max())

        # z_min = max(z_min)
        # z_max = min(z_max)

        # adc = adc[z_min:z_max]
        # t2w = t2w[z_min:z_max]
        # dwi = dwi[z_min:z_max]

        img = torch.stack([t2w, dwi, adc], 0)
        # seg = self.read(path['lesion']).unsqueeze(0)
        # print(img.shape)
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img = trans_dict["image"]
        return img, torch.tensor(idx, dtype=torch.long)


def get_ssl_train_transforms(args):
    train_transforms = [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandRotated(
        #     keys="image",
        #     prob=0.3,
        #     range_x=30 / 180 * np.pi,
        #     keep_size=False,
        # ),
        RandZoomd(
            keys="image",
            prob=1.0,
            min_zoom=[0.9, 0.9, 0.9],
            max_zoom=[1.1, 1.1, 1.1],
            mode="trilinear",
        ),
        RandSpatialCropd(
            keys="image",
            roi_size=args.crop_spatial_size,
            random_size=False,
        ),
        SpatialPadd(
            keys="image",
            spatial_size=args.crop_spatial_size,
            method="symmetric",
            mode="constant",
        ),
        RandFlipd(keys="image", prob=0.5, spatial_axis=2),
    ]
    train_transforms = Compose(train_transforms)
    return train_transforms


def get_visualize_transforms(args):
    test_transforms = [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CenterSpatialCropd(
            keys=[
                "image",
            ],
            roi_size=args.crop_spatial_size,
        ),
        SpatialPadd(
            keys="image",
            spatial_size=args.crop_spatial_size,
            method="symmetric",
            mode="constant",
        ),
    ]
    test_transforms = Compose(test_transforms)
    return test_transforms


def build_ssl_loader(args):
    train_set = SSLDataset(args, args.dataset_csv, get_ssl_train_transforms(args))
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    # sampler_train=None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader


def build_visualize_loader(args):
    set = SSLDataset(args, args.dataset_csv, get_visualize_transforms(args))
    loader = DataLoader(
        set,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
    )
    return loader


# def get_train_transforms(args):
#     train_transforms = [
#         RandRotated(
#             keys=["image", "label"],
#             prob=0.3,
#             range_x=30 / 180 * np.pi,
#             keep_size=False,
#                 ),
#         RandZoomd(
#             keys=["image", "label"],
#             prob=0.3,
#             min_zoom=[1, 0.9, 0.9],
#             max_zoom=[1, 1.1, 1.1],
#             mode=["trilinear", "trilinear"],
#         ),

#         SpatialPadd(
#         keys=["image", "label"],
#         spatial_size=[round(i * 1.2) for i in args.crop_spatial_size],
#         ),

#         RandSpatialCropd(
#             keys=["image", "label"],
#             roi_size=args.crop_spatial_size,
#             random_size=False,
#         ),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#         BinarizeLabeld(keys=["label"])
#     ]
#     train_transforms = Compose(train_transforms)
#     return train_transforms

# def build_loader(args):
#     train_set = BaseVolumeDataset(args, 'split/train.csv', get_train_transforms(args))
#     val_set = BaseVolumeDataset(args, 'split/valid.csv', get_test_transforms(args))
#     test_set = BaseVolumeDataset(args, 'split/test.csv', get_test_transforms(args))
#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
#     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
#     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
#     args.num_exemplar = len(train_set)
#     return train_loader, val_loader, test_loader
