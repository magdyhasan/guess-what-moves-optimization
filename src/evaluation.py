import logging

from tqdm import tqdm
logging.disable(logging.CRITICAL)

import os
import torch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from detectron2.checkpoint import DetectionCheckpointer

import config
import utils as ut
from mask_former_trainer import setup, Trainer
from collections import defaultdict
from torchvision.transforms.functional import pil_to_tensor


def iou(masks, gt, thres=0.5):
    masks = (masks > thres).float()
    intersect = torch.tensordot(masks, gt, dims=([-2, -1], [0, 1]))
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    return intersect / union.clip(min=1e-12)

def eval_unsupmf(cfg, val_loader):
    ious_davis_eval = defaultdict(list)
    ious = defaultdict(list)
    for idx, sample in enumerate(tqdm(val_loader)):
        t = 1
        sample = [e for s in sample for e in s]
        category = [s['category'] for s in sample]
        
        out_root = 'stv2_fts/'
        os.makedirs(out_root + category[0][0], exist_ok=True)
        gt_seg = torch.stack([x['sem_seg_ori'] for x in sample]).cpu()
        
        preds_path = '/storage/user/mahmo/temp/ssl-vos/data/SegTrackv2/opt_masks_bce_bce_e0.01_c1.0_u0_l1_ni5_nf1_olbfgs_farflow_cityscapes_percentile_90_s768_eig_90_b0.3_w0.1_s96/'
        pred = (pil_to_tensor(Image.open(preds_path + category[0][0] + '/' + category[0][1])) / 255.0).type(torch.FloatTensor)
        for i in range(1):
            iou_max = iou(pred, gt_seg[0].type(torch.FloatTensor), thres=0.5)
            ious[category[i][0]].append(iou_max)
            # ious[category[i][0]].append(iou_max)
            frame_id = category[i][1]
            ious_davis_eval[category[i][0]].append((frame_id.strip().replace('.png', ''), iou_max))

    frameious = sum(ious.values(), [])
    frame_mean_iou = torch.cat(frameious).sum().item() * 100 / len(frameious)
    if 'DAVIS' in cfg.GWM.DATASET.split('+')[0]:
        seq_scores = dict()
        for c in ious_davis_eval:
            seq_scores[c] = np.nanmean([v.item() for n, v in ious_davis_eval[c] if int(n) > 1])

        frame_mean_iou = np.nanmean(list(seq_scores.values())) * 100

    return frame_mean_iou


def main():
    dataset = "STv2"
    ckpt_path = "../checkpoints/STv2"
    experiment = Path('../outputs/') / ckpt_path
    args = SimpleNamespace(config_file='configs/maskformer/maskformer_R50_bs16_160k_dino.yaml', opts=["GWM.DATASET", dataset], wandb_sweep_mode=False, resume_path=str(experiment / 'checkpoints/checkpoint_best.pth'), eval_only=True)
    cfg = setup(args)
    # model, cfg = load_model_cfg('../checkpoints/FBMS', "FBMS")
    _, val_loader = config.loaders(cfg)
    print(eval_unsupmf(cfg, val_loader))

if __name__ == "__main__":
    main()