
import torch
import torch.nn.functional as F
from torch import nn
import utils

logger = utils.log.getLogger(__name__)

def iou(masks, gt, thres=0.5):
    masks = (masks > thres).float()
    intersect = torch.tensordot(masks, gt, dims=([-2, -1], [0, 1]))
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    return intersect / union.clip(min=1e-12)

def check_num_fg_corners(mask):
    # check number of corners belonging to the foreground
    top_l, top_r, bottom_l, bottom_r = mask[0][0], mask[0][-1], mask[-1][0], mask[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def select_mask(mask, gt_seg):
    # mask = F.interpolate(mask[None], size=(gt_seg.shape[-2], gt_seg.shape[-1]))
    # # mask = F.interpolate(mask, size=(gt_seg.shape[-2], gt_seg.shape[-1]))
    # mask_iou = iou(mask, gt_seg)
    # return int(mask_iou.max(dim=1)[1])
    binary_mask = mask.argmax(0)
    fr_idx = int(torch.sum(binary_mask == 0) > torch.sum(binary_mask == 1))
    fr_mask = binary_mask if fr_idx == 1 else 1 - binary_mask
    if check_num_fg_corners(fr_mask) >= 3:
        fr_idx = 1 - fr_idx
    return fr_idx



def warp_w_opflow(mask, flow):
    # mask.shape torch.Size([96, 96])
    # flow.shape torch.Size([96, 96, 2])

    theta = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(mask.device)
    theta = theta.view(-1, 2, 3)
    # h, w = mask.shape[1], mask.shape[2]
    h, w = mask.shape[0], mask.shape[1]
    flow[:, :, 0] = flow[:, :, 0] / (w-1)
    flow[:, :, 1] = flow[:, :, 1] / (h-1)
    flow = flow.unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, h, w))
    grid = grid + 2*flow

    # mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    # Align corner True or False shows similar results
    mask = F.grid_sample(mask, grid, mode='bilinear')
    return mask.squeeze()

def length_sq(x):
    # return torch.square(x)
    return torch.sum(torch.square(x), 1, keepdim=True)
    return tf.reduce_sum(tf.square(x), 3, keepdims=True)

def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    height, width, channels = x.shape
    normalization = torch.tensor(height * width * channels, dtype=torch.float32)

    error = torch.pow(torch.square(x * beta) + torch.square(torch.tensor(epsilon)), alpha)

    if mask is not None:
        error = torch.multiply(mask, error)

    if truncate is not None:
        error = torch.minimum(error, truncate)

    return torch.sum(error) / normalization


class UnFlowConsistencyLoss:
    def __init__(self, cfg, model):
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.device=model.device
        self.cfg = cfg

    def __call__(self, sample, masks_softmaxed, device):
        return self.loss2(sample, masks_softmaxed, device)

    def loss(self, sample, masks_softmaxed, device):
        loss = 0
        for batch, vid_masks_softmaxed in zip(sample, masks_softmaxed):
            for frame_id in range(len(batch) - 1):
                mask1 = vid_masks_softmaxed[frame_id][0]
                mask2 = vid_masks_softmaxed[frame_id + 1][0]

                gt_seg1 = batch[frame_id]['sem_seg_ori'].to(device).float()
                gt_seg2 = batch[frame_id + 1]['sem_seg_ori'].to(device).float()

                flow = batch[frame_id]['flow'].to(device)
                r_flow = batch[frame_id]['r_flow'].to(device)

                idx1 = select_mask(mask1, gt_seg1)
                idx2 = select_mask(mask2, gt_seg2)

                # Build FB masks
                mask_fw = warp_w_opflow(mask1[idx1], flow)
                mask_fw_bw = warp_w_opflow(mask1[idx1] + mask_fw, r_flow)
                mask_fw_all = mask_fw + mask_fw_bw

                mask_bw = warp_w_opflow(mask2[idx2], r_flow)
                mask_bw_fw = warp_w_opflow(mask2[idx2] + mask_bw, flow)
                mask_bw_all = mask_bw + mask_bw_fw

                # Build occlusion masks
                mag_sq_fw = length_sq(mask_fw) + length_sq(mask_fw_bw) 
                mag_sq_bw = length_sq(mask_bw) + length_sq(mask_bw_fw)
                occ_thresh_fw =  0.01 * mag_sq_fw + 0.5
                occ_thresh_bw =  0.01 * mag_sq_bw + 0.5
                occ_fw = (length_sq(mask_fw_all) > occ_thresh_fw).float()
                occ_bw = (length_sq(mask_bw_all) > occ_thresh_bw).float()

                # loss += self.loss_fn(mask_fw, mask2[idx2])
                # loss += self.loss_fn(mask_bw, mask1[idx1])
                loss += self.loss_fn(mask_fw * (1 - occ_bw), mask2[idx2] * (1 - occ_bw))
                loss += self.loss_fn(mask_bw * (1 - occ_fw), mask1[idx1] * (1 - occ_fw))
                # loss += charbonnier_loss(mask_fw_all, (1 - occ_fw))
                # loss += charbonnier_loss(mask_bw_all, (1 - occ_bw))
        return loss
    
    def loss2(self, sample, masks_softmaxed, device):
        loss = 0
        for batch, vid_masks_softmaxed in zip(sample, masks_softmaxed):
            for frame_id in range(len(batch) - 1):
                mask1 = vid_masks_softmaxed[frame_id][0]
                mask2 = vid_masks_softmaxed[frame_id + 1][0]

                gt_seg1 = batch[frame_id]['sem_seg_ori'].to(device).float()
                gt_seg2 = batch[frame_id + 1]['sem_seg_ori'].to(device).float()

                rgb1 = batch[frame_id]['rgb'].to(device).float()
                rgb2 = batch[frame_id + 1]['rgb'].to(device).float()

                flow = batch[frame_id]['flow'].to(device)
                r_flow = batch[frame_id]['r_flow'].to(device)

                idx1 = select_mask(mask1, gt_seg1)
                idx2 = select_mask(mask2, gt_seg2)

                # Build FB masks
                mask_fw = warp_w_opflow(rgb1, flow)
                mask_fw_bw = warp_w_opflow(rgb1 + mask_fw, r_flow)
                mask_fw_all = mask_fw + mask_fw_bw

                mask_bw = warp_w_opflow(rgb2, r_flow)
                mask_bw_fw = warp_w_opflow(rgb2 + mask_bw, flow)
                mask_bw_all = mask_bw + mask_bw_fw

                # Build occlusion masks
                mag_sq_fw = length_sq(mask_fw) + length_sq(mask_fw_bw) 
                mag_sq_bw = length_sq(mask_bw) + length_sq(mask_bw_fw)
                occ_thresh_fw =  0.01 * mag_sq_fw + 0.5
                occ_thresh_bw =  0.01 * mag_sq_bw + 0.5
                occ_fw = (length_sq(mask_fw_all) > occ_thresh_fw).float()
                occ_bw = (length_sq(mask_bw_all) > occ_thresh_bw).float()

                # loss += self.loss_fn(mask_fw, mask2[idx2])
                # loss += self.loss_fn(mask_bw, mask1[idx1])
                loss += self.loss_fn(mask_fw * (1 - occ_fw), mask2[idx2] * (1 - occ_fw))
                loss += self.loss_fn(mask_bw * (1 - occ_bw), mask1[idx1] * (1 - occ_bw))
                # loss += charbonnier_loss(mask_fw_all, (1 - occ_fw))
                # loss += charbonnier_loss(mask_bw_all, (1 - occ_bw))
        return loss