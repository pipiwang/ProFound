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
from models.classifier import Classifier
from models.convnextv2 import convnextv2_tiny, remap_checkpoint_keys, load_state_dict
from dataset.dataset_cls import build_Risk_loader, build_Screening_loader, build_Promis_loader, build_Promis3_hist_loader
from engine.classification import test_risk

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
        "--root", default="./", type=str
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
        default="./outputcls",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--file_name", default="")
    parser.add_argument("--ckpt_dir", default="./outputcls")
    parser.add_argument(
        "--log_dir", default="./outputcls", help="path where to tensorboard log"
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
    parser.set_defaults(data20=False)

    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--kfold", type=int, default=None)
    parser.add_argument("--demo", type=bool, default=True, help="Run in demo mode")
    return parser


def main(args):

    device = "cuda"
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    

    if args.dataset == "risk":
        data_loader_test = build_Risk_loader(args)
    # elif args.dataset == "screening":
    #     data_loader_train, data_loader_val, data_loader_test = build_Screening_loader(
    #         args
    #     )
    # elif args.dataset == "promis":
    #     data_loader_train, data_loader_val, data_loader_test = build_Promis_loader(args)
    # elif args.dataset == "promis3hist":
    #     data_loader_train, data_loader_val, data_loader_test = build_Promis3_hist_loader(args)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {args.dataset}")
    print(f"Loaded dataset: {args.dataset}, test set size: {len(data_loader_test.dataset)}")
    
    if args.model == "profound_conv":
        convnext = convnextv2_tiny(in_chans=3)
        model = Classifier(convnext, args.num_classes)
    else:
        raise NotImplementedError(f"unknown model: {args.model}")

    args.output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    model.load_state_dict(torch.load(args.ckpt_dir, map_location='cpu')["model"])
    print(f"Loaded model from {args.ckpt_dir}")
    model.to(device)
    logits, gts = [], []
    model.eval()
    with torch.no_grad():
        for idx, (img, gt, pid) in enumerate(data_loader_test):
            img, gt = img.to(args.device), gt.to(args.device)
            logit = model(img)
            logits.append(logit)
            gts.append(gt)

        # if args.dataset == "risk":
        #     test_risk(logits, gts)
        logits = torch.cat(logits, 0).squeeze().cpu().numpy()
        gts = torch.cat(gts, 0).squeeze().cpu().numpy()
        print(f"test results: logits {logits}, gts {gts}")
        np.savez(os.path.join(args.output_dir, f"{args.file_name}.npz"), logits = logits, gts=gts)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
