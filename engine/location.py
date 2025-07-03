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
from torchmetrics.classification import (
    BinarySpecificityAtSensitivity,
    BinarySensitivityAtSpecificity,
)
# from monai.metrics import DiceMetric
# from monai.transforms import Activations
import pdb
from sklearn.metrics import (
    roc_auc_score,
    top_k_accuracy_score,
    f1_score,
    confusion_matrix,
)


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

    loss_cal = torch.nn.BCEWithLogitsLoss()
    
    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    last_norm = 0.0
    for data_iter_step, (img, zone_mask, gt) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        

        # we use a per iteration (instead of per epoch) lr scheduler
        img, zone_mask, gt = img.to(device, non_blocking=True), zone_mask.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        gt = gt.float()
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        logit = model(img, zone_mask)
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
                last_norm,
            )
            print(
                "inf",
                torch.isinf(logit).any(),
                torch.isinf(img).any(),
                last_norm,
            )
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),  1.0)
        optimizer.step()

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
    loss_cal = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        loss_summary = []
        
        for idx, (img, zone_mask, gt) in enumerate(data_loader_val):
            img, zone_mask, gt = img.to(device, non_blocking=True), zone_mask.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            gt = gt.float()
            logit = model(img, zone_mask)
            loss = loss_cal(logit, gt)
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
    model.load_state_dict(torch.load(filepath_best)["model"])
   
    log_stats = {}
    with torch.no_grad():
        prob, gts = [], []
       
        for idx, (img, zone_mask, gt) in enumerate(test_loader):
            img, zone_mask, gt = img.to(args.device, non_blocking=True), zone_mask.to(args.device, non_blocking=True), gt.to(args.device, non_blocking=True)
        
            logit = model(img, zone_mask)
            prob.append(logit)
            gts.append(gt)

            
    prob = torch.cat(prob, 0)
    prob = torch.sigmoid(prob).cpu()
    gts = torch.cat(gts, 0).cpu()



    print("- Zone level: ")
    zone_prob = prob.reshape(-1, prob.shape[-1])
    zone_gt = gts.reshape(-1, prob.shape[-1])
    zone_auc = roc_auc_score(zone_prob, zone_gt) * 100
    
    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(zone_prob, zone_gt)
        sig_specificity = sig_specificity * 100
        
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(zone_prob, zone_gt)
        sig_sensitivity = sig_sensitivity* 100

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity:.2f} \t {sig_sensitivity:.2f} ")
        log_stats[f"specificity_at_{i}"]=f"{sig_specificity:.2f}"
        log_stats[f"sensitivity_at_{i}"]=f"{sig_sensitivity:.2f}"


    print("- Patient level: ")
    p_prob = prob.max(1).values
    p_gt = gts.max(1).values

    p_auc = roc_auc_score(p_prob, p_gt) * 100

    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(p_prob, p_gt)
        sig_specificity = sig_specificity * 100
        
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(p_prob, p_gt)
        sig_sensitivity = sig_sensitivity* 100

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity:.2f} \t {sig_sensitivity:.2f} ")
        log_stats[f"specificity_at_{i}"]=f"{sig_specificity:.2f}"
        log_stats[f"sensitivity_at_{i}"]=f"{sig_sensitivity:.2f}"

    return log_stats
