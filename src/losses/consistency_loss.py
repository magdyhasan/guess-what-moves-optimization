
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

def select_mask(mask, gt_seg):
    mask = F.interpolate(mask[None], size=(gt_seg.shape[-2], gt_seg.shape[-1]))
    mask_iou = iou(mask, gt_seg)
    return int(mask_iou.max(dim=1)[1])
    

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
    # align corner True or False shows similar results
    mask = F.grid_sample(mask, grid, mode='bilinear')
    return mask.squeeze()

class ConsistencyLoss:
    def __init__(self, cfg, model):
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.device=model.device
        self.cfg = cfg

    def __call__(self, sample, masks_softmaxed, masks_softmaxed2, flow, r_flow, gt_segs, gt_segs2):
        return self.lossSSL(sample, masks_softmaxed, masks_softmaxed2, flow, r_flow, gt_segs, gt_segs2)

    def loss(self, sample, masks_softmaxed, masks_softmaxed2, flows, r_flows, gt_segs, gt_segs2):
        loss = 0
        for mask_softmaxed, mask_softmaxed2, flow, r_flow, gt_seg, gt_seg2 in zip(masks_softmaxed, masks_softmaxed2, flows, r_flows, gt_segs, gt_segs2):
            idx1 = select_mask(mask_softmaxed, gt_seg)
            idx2 = select_mask(mask_softmaxed2, gt_seg2)
            mask_forward = warp_w_opflow(mask_softmaxed[idx1][None], flow.permute(1, 2, 0))
            mask_backward = warp_w_opflow(mask_softmaxed2[idx2][None], r_flow.permute(1, 2, 0))
            loss += self.loss_fn(mask_forward, mask_softmaxed2[idx2])
            loss += self.loss_fn(mask_backward, mask_softmaxed[idx1])
        return loss
    
    def loss2(self, sample, masks_softmaxed, masks_softmaxed2, flows, r_flows, gt_segs, gt_segs2):
        loss = 0
        for mask_softmaxed, mask_softmaxed2, flow, r_flow, gt_seg, gt_seg2 in zip(masks_softmaxed, masks_softmaxed2, flows, r_flows, gt_segs, gt_segs2):
            mask_forward = warp_w_opflow(mask_softmaxed, flow.permute(1, 2, 0))
            mask_backward = warp_w_opflow(mask_softmaxed2, r_flow.permute(1, 2, 0))
            loss += self.loss_fn(mask_forward, mask_softmaxed2)
            # loss += self.loss_fn(mask_backward, mask_softmaxed)
        return loss
    
    def lossSSL(self, sample, masks_softmaxed, masks_softmaxed2, flows, r_flows, gt_segs, gt_segs2):
        loss = 0
        for mask_softmaxed, mask_softmaxed2, flow, r_flow, gt_seg, gt_seg2 in zip(masks_softmaxed, masks_softmaxed2, flows, r_flows, gt_segs, gt_segs2):
            idx1 = select_mask(mask_softmaxed, gt_seg)
            idx2 = select_mask(mask_softmaxed2, gt_seg2)
            mask_forward = warp_w_opflow(mask_softmaxed[idx1], flow)
            mask_backward = warp_w_opflow(mask_softmaxed2[idx2], r_flow)
            loss += self.loss_fn(mask_forward, mask_softmaxed2[idx2])
            loss += self.loss_fn(mask_backward, mask_softmaxed[idx1])
        return loss
