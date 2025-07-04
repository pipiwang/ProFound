# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import os
import util.misc as misc
import util.lr_sched as lr_sched
from monai.losses import DiceCELoss, DiceLoss
import numpy as np
from monai.metrics import DiceHelper
import surface_distance
from surface_distance import metrics
from util.meter import DiceMeter, HausdorffMeter, SurfaceDistanceMeter

# from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference

# from monai.metrics import DiceMetric
# from monai.transforms import Activations
import pdb


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    if args.out_channels == 1:
        loss_cal = DiceCELoss(sigmoid=True)
    else:
        loss_cal = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

    
    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    last_norm = 0.0
    for data_iter_step, (img, gt, dataidx) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        img, gt = img.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        # print(img.shape, img.mean(), img.std())
        # with torch.cuda.amp.autocast():
        logit = model(img)       
        if isinstance(logit, list):
            loss = loss_cal(logit[0], gt) + 0.4*loss_cal(logit[1], gt)
        else:
            loss = loss_cal(logit, gt)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(
                "nan",
                torch.isnan(logit).any(),
                torch.isnan(img).any(),
                dataidx,
                last_norm,
            )
            print(
                "inf",
                torch.isinf(logit).any(),
                torch.isinf(img).any(),
                dataidx,
                last_norm,
            )
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),  1.0)
        optimizer.step()

        # last_norm = loss_scaler(loss, optimizer, parameters=model.parameters())
        # optimizer.zero_grad()
        # torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validation(model, data_loader_val, device, epoch, args):
    model.eval()
    if args.out_channels == 1:
        dice_loss = DiceLoss(sigmoid=True)
    else:
        dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)

    with torch.no_grad():
        loss_summary = []
        for idx, (img, gt, _) in enumerate(data_loader_val):
            img, gt = img.to(device), gt.to(device)
            mask = model(img)
            loss = dice_loss(mask, gt)
            loss_summary.append(loss.detach().cpu().numpy())
            print(
                "epoch: {}/{}, iter: {}/{}".format(
                    epoch, args.epochs, idx, len(data_loader_val)
                )
                + " loss:"
                + str(loss_summary[-1].flatten()[0])
            )
        avg_loss = np.mean(loss_summary)
        print("Averaged stats:", str(avg_loss))
    return avg_loss


def test(model, test_loader, args, sliding_window=False):
    model.eval()
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    model.load_state_dict(torch.load(filepath_best)["model"], weights_only=False)
    dice_meter = DiceMeter(args)
    hausdorff_meter = HausdorffMeter(args)
    sd_meter = SurfaceDistanceMeter(args)
    log_stats = {}
    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)
            if sliding_window:
                pred = sliding_window_inference(
                    img, args.crop_spatial_size, 4, model, overlap=0.5
                )
            else:
                pred = model(img)
            if args.num_classes == 1:
                pred = torch.sigmoid(pred) > 0.5
            else:
                pred = torch.softmax(pred, dim=1)
                pred = torch.argmax(pred, dim=1, keepdim=True)
            dice_meter.update(pred, gt)
            hausdorff_meter.update(pred, gt)
            sd_meter.update(pred, gt)

    print("- Test metrics Dice: ")
    dice_class_avg, dice_avg = dice_meter.get_average()
    print("Class wise: ", dice_class_avg)
    print("Avg.: ", dice_avg)

    print("- Test metrics Hausdorff95: ")
    hsd_class_avg, hsd_avg = hausdorff_meter.get_average()
    print("Class wise: ", hsd_class_avg)
    print("Avg.: ", hsd_avg)

    print("- Test metrics SurfaceDistance: ")
    sd_class_avg, sd_avg = sd_meter.get_average()
    print("Class wise: ", sd_class_avg)
    print("Avg.: ", sd_avg)
    log_stats = {
        "dice_class_avg": dice_class_avg.tolist() if isinstance(dice_class_avg, np.ndarray) else dice_class_avg,
        "dice_avg": dice_avg.tolist() if isinstance(dice_avg, np.ndarray) else dice_avg,
        "hsd_class_avg": hsd_class_avg.tolist() if isinstance(hsd_class_avg, np.ndarray) else hsd_class_avg,
        "hsd_avg": hsd_avg.tolist() if isinstance(hsd_avg, np.ndarray) else hsd_avg,
        "sd_class_avg": sd_class_avg.tolist() if isinstance(sd_class_avg, np.ndarray) else sd_class_avg,
        "sd_avg": sd_avg.tolist() if isinstance(sd_avg, np.ndarray) else sd_avg,
    }
    return log_stats
