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

from utils.data import read_flow


class FlowPostDetectron(Dataset):
    def __init__(self, data_dir, resolution, pair_list, val_seq, size_divisibility=None,
                 small_val=0, flow_clip=1., norm=True, eval_size=True):
        self.val_seq = val_seq
        self.data_dir = data_dir
        self.pair_list = pair_list
        self.resolution = resolution

        self.eval_size = eval_size

        self.samples = []
        self.r_samples = []
        self.samples_fid = {}
        for v in self.val_seq:
            seq_dir = Path(self.data_dir[0]) / v
            frames_paths = sorted(seq_dir.glob('*.flo'))
            self.samples_fid[str(seq_dir)] = {fp: i for i, fp in enumerate(frames_paths)}
            self.samples.extend(frames_paths)

            r_seq_dir = Path(self.data_dir[0].parent / 'Flows_gap-1') / v
            r_frames_paths = sorted(r_seq_dir.glob('*.flo'))
            self.r_samples.extend(r_frames_paths)
        self.samples = [os.path.join(x.parent.name, x.name) for x in self.samples]
        self.r_samples = [os.path.join(x.parent.name, x.name) for x in self.r_samples]
        self.gaps = ['gap{}'.format(i) for i in pair_list]
        self.neg_gaps = ['gap{}'.format(-i) for i in pair_list]
        self.size_divisibility = size_divisibility
        self.ignore_label = -1
        self.transforms = DT.AugmentationList([
            DT.Resize(self.resolution, interp=Image.BICUBIC),
        ])
        self.flow_clip=flow_clip
        self.norm_flow=norm
        self.force1080p_transforms=None


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset_dicts = []
        dataset_dict = {}
        flow_dir = Path(self.data_dir[0]) / self.samples[idx]
        r_flow_dir = Path(self.data_dir[0].parent / 'Flows_gap-1') / self.r_samples[idx]
        fid = self.samples_fid[str(flow_dir.parent)][flow_dir]
        flo = einops.rearrange(read_flow(str(flow_dir), self.resolution), 'c h w -> h w c')
        r_flo = einops.rearrange(read_flow(str(r_flow_dir), self.resolution), 'c h w -> h w c')
        dataset_dict["gap"] = 'gap1'

        suffix = '.jpg'
        rgb_dir = (self.data_dir[1] / self.samples[idx]).with_suffix(suffix)
        gt_dir = (self.data_dir[2] / self.samples[idx]).with_suffix('.png')

        rgb2_dir = (self.data_dir[1] / self.r_samples[idx]).with_suffix(suffix)
        gt2_dir = (self.data_dir[2] / self.r_samples[idx]).with_suffix('.png')

        rgb = d2_utils.read_image(str(rgb_dir)).astype(np.float32)
        rgb2 = d2_utils.read_image(str(rgb2_dir)).astype(np.float32)
        original_rgb = torch.as_tensor(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)).clip(0., 255.))).float()
        original_rgb2 = torch.as_tensor(np.ascontiguousarray(np.transpose(rgb2, (2, 0, 1)).clip(0., 255.))).float()
        input = DT.AugInput(rgb)
        input2 = DT.AugInput(rgb2)

        # Apply the augmentation:
        preprocessing_transforms = self.transforms(input)  # type: DT.Transform
        rgb = input.image
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = rgb.clip(0., 255.)
        d2_utils.check_image_size(dataset_dict, flo)

        preprocessing_transforms2 = self.transforms(input2)  # type: DT.Transform
        rgb2 = input2.image
        rgb2 = np.transpose(rgb2, (2, 0, 1))
        rgb2 = rgb2.clip(0., 255.)
        d2_utils.check_image_size(dataset_dict, r_flo)

        if gt_dir.exists():
            sem_seg_gt_ori = d2_utils.read_image(gt_dir)
            sem_seg_gt = preprocessing_transforms.apply_segmentation(sem_seg_gt_ori)
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
                sem_seg_gt_ori = sem_seg_gt_ori[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
                sem_seg_gt_ori = (sem_seg_gt_ori > 128).astype(int)

            sem_seg_gt_ori2 = d2_utils.read_image(gt2_dir)
            sem_seg_gt2 = preprocessing_transforms2.apply_segmentation(sem_seg_gt_ori2)
            if sem_seg_gt2.ndim == 3:
                sem_seg_gt2 = sem_seg_gt2[:, :, 0]
                sem_seg_gt_ori2 = sem_seg_gt_ori2[:, :, 0]
            if sem_seg_gt2.max() == 255:
                sem_seg_gt2 = (sem_seg_gt2 > 128).astype(int)
                sem_seg_gt_ori2 = (sem_seg_gt_ori2 > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((self.resolution[0], self.resolution[1]))
            sem_seg_gt_ori = np.zeros((original_rgb.shape[-2], original_rgb.shape[-1]))

        gwm_dir = (Path(str(self.data_dir[2]).replace('Annotations', 'gwm')) / self.samples[idx]).with_suffix(
            '.png')
        gwm_seg_gt = None

        if sem_seg_gt is None or sem_seg_gt2 is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Pad image and segmentation label here!
        flo = torch.as_tensor(np.ascontiguousarray(flo.transpose(2, 0, 1)))
        if self.norm_flow:
            flo = flo/(flo ** 2).sum(0).max().sqrt()
        flo = flo.clip(-self.flow_clip, self.flow_clip)

        r_flo = torch.as_tensor(np.ascontiguousarray(r_flo.transpose(2, 0, 1)))
        if self.norm_flow:
            r_flo = r_flo/(r_flo ** 2).sum(0).max().sqrt()
        r_flo = r_flo.clip(-self.flow_clip, self.flow_clip)

        rgb = torch.as_tensor(np.ascontiguousarray(rgb)).float()
        rgb2 = torch.as_tensor(np.ascontiguousarray(rgb2)).float()
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            sem_seg_gt_ori = torch.as_tensor(sem_seg_gt_ori.astype("long"))
        
        if sem_seg_gt2 is not None:
            sem_seg_gt2 = torch.as_tensor(sem_seg_gt2.astype("long"))
            sem_seg_gt_ori2 = torch.as_tensor(sem_seg_gt_ori2.astype("long"))
        

        if self.size_divisibility > 0:
            image_size = (flo.shape[-2], flo.shape[-1])
            padding_size = [
                0,
                int(self.size_divisibility * math.ceil(image_size[1] // self.size_divisibility)) - image_size[1],
                0,
                int(self.size_divisibility * math.ceil(image_size[0] // self.size_divisibility)) - image_size[0],
            ]
            flo = F.pad(flo, padding_size, value=0).contiguous()
            r_flo = F.pad(r_flo, padding_size, value=0).contiguous()
            rgb = F.pad(rgb, padding_size, value=128).contiguous()
            rgb2 = F.pad(rgb2, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            if sem_seg_gt2 is not None:
                sem_seg_gt2 = F.pad(sem_seg_gt2, padding_size, value=self.ignore_label).contiguous()

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo
        dataset_dict["r_flow"] = r_flo
        dataset_dict["rgb"] = rgb
        dataset_dict["rgb2"] = rgb2


        dataset_dict["original_rgb"] = F.interpolate(original_rgb[None], mode='bicubic', size=sem_seg_gt_ori.shape[-2:], align_corners=False).clip(0.,255.)[0]
        dataset_dict["original_rgb2"] = F.interpolate(original_rgb2[None], mode='bicubic', size=sem_seg_gt_ori2.shape[-2:], align_corners=False).clip(0.,255.)[0]

        dataset_dict["category"] = str(gt_dir).split('/')[-2:]
        dataset_dict['frame_id'] = fid

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
            dataset_dict["sem_seg_ori"] = sem_seg_gt_ori.long()

        if sem_seg_gt2 is not None:
            dataset_dict["sem_seg2"] = sem_seg_gt2.long()
            dataset_dict["sem_seg_ori2"] = sem_seg_gt_ori2.long()
        dataset_dicts.append(dataset_dict)
        return dataset_dicts
