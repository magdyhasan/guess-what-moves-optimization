import math
import os
from pathlib import Path

import detectron2.data.transforms as DT
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from detectron2.data import detection_utils as d2_utils
from detectron2.structures import Instances, BitMasks
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import cv2

from utils.data import read_flow

def rescale_flow(flow, scale_factor=(60, 60), renormalize=True):
    u, v = cv2.resize(flow, (scale_factor[1], scale_factor[0])).transpose(2, 0, 1)
    if renormalize:
        u = u*(flow.shape[1] / scale_factor)
        v = v*(flow.shape[0] / scale_factor)
    return u, v


def FlowSSLVIDDetectron(data_dir, resolution, pair_list, val_seq, size_divisibility=None,
                small_val=0, flow_clip=1., norm=True, eval_size=True, frames_num=2):

    samples = []
    r_samples = []
    samples_fid = {}
    for v in val_seq:
        seq_dir = Path(data_dir[0]) / v
        frames_paths = sorted(seq_dir.glob('*.flo'))
        samples_fid[str(seq_dir)] = {fp: i for i, fp in enumerate(frames_paths)}
        samples.extend(frames_paths)

        r_seq_dir = Path(data_dir[0].parent / 'Flows_gap-1') / v
        r_frames_paths = sorted(r_seq_dir.glob('*.flo'))
        r_samples.extend(r_frames_paths)
    samples = [os.path.join(x.parent.name, x.name) for x in samples]
    r_samples = [os.path.join(x.parent.name, x.name) for x in r_samples]
    gaps = ['gap{}'.format(i) for i in pair_list]
    neg_gaps = ['gap{}'.format(-i) for i in pair_list]
    size_divisibility = size_divisibility
    ignore_label = -1
    transforms = DT.AugmentationList([
        DT.Resize(resolution, interp=Image.BICUBIC),
    ])
    flow_clip=flow_clip
    norm_flow=norm
    force1080p_transforms=None
    frames_num = frames_num


    dataset_dicts = []
    for f_idx in np.arange(len(samples)):
        dataset_dict = {}
        flow_dir = Path(data_dir[0]) / samples[f_idx]
        fid = samples_fid[str(flow_dir.parent)][flow_dir]
        
        flow_dir = Path('/storage/user/mahmo/temp/ssl-vos/data/SegTrackv2/flow_1_arflow_cityscapes_percentile_90') / samples[f_idx].replace('.flo', '.npy')
        r_flow_dir = Path('/storage/user/mahmo/temp/ssl-vos/data/SegTrackv2/flow_reverse_1_arflow_cityscapes_percentile_90') / r_samples[f_idx].replace('.flo', '.npy')
        

        flo = np.load(flow_dir)
        flo = np.stack(rescale_flow(flo, scale_factor=resolution, renormalize=False)).transpose(1, 2, 0)
        flo = torch.tensor(flo, dtype=torch.float)
        
        r_flo = np.load(r_flow_dir)
        r_flo = np.stack(rescale_flow(r_flo, scale_factor=resolution, renormalize=False)).transpose(1, 2, 0)
        r_flo = torch.tensor(r_flo, dtype=torch.float)

        flow_dir = Path('/storage/user/mahmo/temp/ssl-vos/data/SegTrackv2/Flows_gap1') / samples[f_idx]
        r_flow_dir = Path('/storage/user/mahmo/temp/ssl-vos/data/SegTrackv2/Flows_gap1') / r_samples[f_idx]
        
        flo = read_flow(str(flow_dir), resolution, False).transpose(1, 2, 0)
        r_flo = read_flow(str(r_flow_dir), resolution, False).transpose(1, 2, 0)
        
        dataset_dict["gap"] = 'gap1'

        suffix = '.jpg'
        rgb_dir = (data_dir[1] / samples[f_idx]).with_suffix(suffix)
        gt_dir = (data_dir[2] / samples[f_idx]).with_suffix('.png')

        rgb = d2_utils.read_image(str(rgb_dir)).astype(np.float32)
        original_rgb = torch.as_tensor(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)).clip(0., 255.))).float()
        input = DT.AugInput(rgb)

        # Apply the augmentation:
        preprocessing_transforms = transforms(input)  # type: DT.Transform
        rgb = input.image
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = rgb.clip(0., 255.)
        d2_utils.check_image_size(dataset_dict, flo)

        if gt_dir.exists():
            sem_seg_gt_ori = d2_utils.read_image(gt_dir)
            sem_seg_gt = preprocessing_transforms.apply_segmentation(sem_seg_gt_ori)
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
                sem_seg_gt_ori = sem_seg_gt_ori[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
                sem_seg_gt_ori = (sem_seg_gt_ori > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((resolution[0], resolution[1]))
            sem_seg_gt_ori = np.zeros((original_rgb.shape[-2], original_rgb.shape[-1]))

        gwm_dir = (Path(str(data_dir[2]).replace('Annotations', 'gwm')) / samples[f_idx]).with_suffix(
            '.png')
        gwm_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        rgb = torch.as_tensor(np.ascontiguousarray(rgb)).float()
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            sem_seg_gt_ori = torch.as_tensor(sem_seg_gt_ori.astype("long"))
        

        if size_divisibility > 0:
            image_size = (flo.shape[-2], flo.shape[-1])
            padding_size = [
                0,
                int(size_divisibility * math.ceil(image_size[1] // size_divisibility)) - image_size[1],
                0,
                int(size_divisibility * math.ceil(image_size[0] // size_divisibility)) - image_size[0],
            ]
            # flo = F.pad(flo, padding_size, value=0).contiguous()
            # r_flo = F.pad(r_flo, padding_size, value=0).contiguous()
            rgb = F.pad(rgb, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=ignore_label).contiguous()

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo
        dataset_dict["r_flow"] = r_flo
        dataset_dict["rgb"] = rgb


        dataset_dict["original_rgb"] = F.interpolate(original_rgb[None], mode='bicubic', size=sem_seg_gt_ori.shape[-2:], align_corners=False).clip(0.,255.)[0]

        dataset_dict["category"] = str(gt_dir).split('/')[-2:]
        dataset_dict['frame_id'] = fid

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
            dataset_dict["sem_seg_ori"] = sem_seg_gt_ori.long()

        dataset_dicts.append(dataset_dict)
    return dataset_dicts
