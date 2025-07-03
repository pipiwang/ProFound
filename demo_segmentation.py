# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import torch
import torch.backends.cudnn as cudnn
from dataset.dataset_seg import (
    build_UCL_loader,
    build_Anatomy_loader,
    build_BpAnatomy_loader,
    build_Promis_loader,
    build_PromisPirads3_loader
)
import monai
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
import SimpleITK as sitk
from models.convnextv2 import convnextv2_tiny, remap_checkpoint_keys, load_state_dict
from models.convnext_unter import ConvnextUNETR
from models.upernet_module import UperNet


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)



def get_args_parser():
    parser = argparse.ArgumentParser("segmentation", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--root", default="/SAN/medic/foundation/downstream_data", type=str
    )
    parser.add_argument("--crop_spatial_size", default=(64, 256, 256), type=tuple_type)

    # Model parameters
    parser.add_argument("--model", help="model name")
    parser.add_argument(
        "--input_size", default=(64, 256, 256), type=tuple_type, help="images input size"
    )
    parser.add_argument(
        "--train",
        default="scratch",
        choices=["fintune", "freeze", "scratch"],
        help="train method",
    )
    parser.add_argument("--pretrain", default=None, type=str)
    parser.add_argument("--tolerance", default=5, type=int)
    parser.add_argument("--spacing", default=(1.0, 0.5, 0.5), type=tuple)
    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="weight decay (default: 1e-5)"
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--output_dir",
        default="./outputseg",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--file_name", default="")
    parser.add_argument("--ckpt_dir", default="./outputseg")
    parser.add_argument(
        "--log_dir", default="./outputseg", help="path where to tensorboard log"
    )
    parser.add_argument("--dataset", default="UCL", help="dataset name")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--data20", action="store_true", help="Use 20 training data")
    parser.set_defaults(data20=False)

    parser.add_argument("--data_num", default=0, type=int, help="number of train data")

    parser.add_argument("--save_fig", action="store_true")
    parser.set_defaults(save_fig=False)

    parser.add_argument(
        "--prompt", action="store_true", help="Use visual prompt tuning"
    )
    parser.set_defaults(prompt=False)

    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--demo", type=bool, default=True, help="Run in demo mode")
    return parser


def main(args):

    device = "cuda"
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if args.dataset == "UCL":
        _, _, data_loader_test = build_UCL_loader(args)
        args.sliding_window = False
    
    else:
        raise NotImplementedError(f"unknown schedule sampler: {args.dataset}")

    if args.model == "profound_conv":
        convnext = convnextv2_tiny(in_chans=3)
        model = UperNet(
            encoder=convnext,
            in_channels=[96, 192, 384, 768],
            out_channels=args.out_channels,
        )
        model = model.to(device)
    
    elif args.model == "profound_conv_unetr3d":
        convnext = convnextv2_tiny(in_chans=3)
        
        model = ConvnextUNETR(
            in_channels=3, out_channels=1, convnext=convnext, feature_size=32
        )
        model = model.to(device)

    else:
        raise NotImplementedError(f"unknown model: {args.model}")
    
    
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    # define the model
    filepath_best = os.path.join(args.ckpt_dir, "best.pth.tar")
    model.load_state_dict(torch.load(filepath_best)["model"])
    dice_list = []
    model.eval()
    with torch.no_grad():
        for idx, (img, gt, pid) in enumerate(data_loader_test):
            img, gt = img.to(args.device), gt.to(args.device)
            if args.sliding_window:
                pred = sliding_window_inference(
                    img, args.crop_spatial_size, 4, model, overlap=0.5
                )
            else:
                pred = model(img)

            if args.num_classes == 1:
                pred = torch.sigmoid(pred) > 0.5
                pred = pred.int()
            else:
                pred = torch.softmax(pred, dim=1)
                pred = torch.argmax(pred, dim=1, keepdim=True)

            dice = compute_dice(pred, gt)  # compute_dice(pred, gt, False,num_classes=9)
            print(pid, dice.item())
            if not torch.isnan(dice):
                dice_list.append(dice)
            # dice = int(dice.mean()*10000)
            img = img.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()
            if args.save_fig:
                if idx < 20:
                    # print(img.shape,pred.shape, gt.shape )
                    sitk.WriteImage(
                        sitk.GetImageFromArray(img[0]),
                        os.path.join(args.output_dir, f"{idx}_t2w.nii.gz"),
                    )
                    sitk.WriteImage(
                        sitk.GetImageFromArray(img[1]),
                        os.path.join(args.output_dir, f"{idx}_dwi.nii.gz"),
                    )
                    sitk.WriteImage(
                        sitk.GetImageFromArray(pred),
                        os.path.join(args.output_dir, f"{idx}_pred.nii.gz"),
                    )
                    sitk.WriteImage(
                        sitk.GetImageFromArray(gt),
                        os.path.join(args.output_dir, f"{idx}_gt.nii.gz"),
                    )
        dice_list = torch.stack(dice_list, 0)
        np.save(
            os.path.join(args.output_dir, f"{args.file_name}.npy"),
            dice_list.cpu().numpy(),
        )
        print("dice mean: ", dice_list.mean().item())


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
