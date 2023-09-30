
import torch
import torch.nn.functional as F
from torch import nn
import utils
import wandb

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
    # align corner True or False shows similar results
    mask = F.grid_sample(mask, grid, mode='bilinear')
    return mask.squeeze()

class SSLConsistencyLoss:
    def __init__(self, cfg, model):
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.device=model.device
        self.cfg = cfg

    def __call__(self, sample, masks_softmaxed, device, iteration):
        return self.lossSSL(sample, masks_softmaxed, device, iteration)

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
    
    def lossSSLALL(self, sample, masks_softmaxed, device):
        loss = 0
        for batch, vid_masks_softmaxed in zip(sample, masks_softmaxed):
            for frame_id in range(len(batch) - 1):
                mask1 = vid_masks_softmaxed[frame_id][0]
                mask2 = vid_masks_softmaxed[frame_id + 1][0]
                
                flow = batch[frame_id]['flow'].to(device)
                r_flow = batch[frame_id]['r_flow'].to(device)

                for k in range(mask1.shape[0]):
                    mask_forward = warp_w_opflow(mask1[k], flow)
                    mask_backward = warp_w_opflow(mask2[k], r_flow)

                    loss += self.loss_fn(mask_forward, mask2[k])
                    loss += self.loss_fn(mask_backward, mask1[k])
        return loss
    
    def lossSSL(self, sample, masks_softmaxed, device, iteration):
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

                mask_forward = warp_w_opflow(mask1[idx1], flow)
                mask_backward = warp_w_opflow(mask2[idx2], r_flow)

                if batch[frame_id]['frame_id'] == 16:
                    wandb_table = wandb.Table(columns=['image', 'GT', 'prediction', 'prediction_refined'])
                    wandb_rgb1 = wandb.Image(batch[frame_id]['original_rgb'])
                    wandb_gt1 = wandb.Image(gt_seg1)
                    wandb_mask1 = wandb.Image(F.interpolate(mask1[idx1][None][None], size=(gt_seg1.shape[-2], gt_seg1.shape[-1]))[0][0])
                    wandb_mask1_refined = wandb.Image(F.interpolate(mask_backward[None][None], size=(gt_seg1.shape[-2], gt_seg1.shape[-1]))[0][0])
                    wandb_table.add_data(wandb_rgb1, wandb_gt1, wandb_mask1, wandb_mask1_refined)

                    wandb_rgb2 = wandb.Image(batch[frame_id + 1]['rgb'])
                    wandb_gt2 = wandb.Image(gt_seg2)
                    wandb_mask2 = wandb.Image(F.interpolate(mask2[idx2][None][None], size=(gt_seg1.shape[-2], gt_seg1.shape[-1]))[0][0])
                    wandb_mask2_refined = wandb.Image(F.interpolate(mask_forward[None][None], size=(gt_seg1.shape[-2], gt_seg1.shape[-1]))[0][0])
                    wandb_table.add_data(wandb_rgb2, wandb_gt2, wandb_mask2, wandb_mask2_refined)

                    wandb.log({'prediction_images/' + str(iteration): wandb_table})

                # mask_forward.shape [128, 224]
                loss += self.loss_fn(mask_forward, mask2[idx2])
                loss += self.loss_fn(mask_backward, mask1[idx1])
        return loss

        for mask_softmaxed, mask_softmaxed2, flow, r_flow, gt_seg, gt_seg2 in zip(masks_softmaxed, masks_softmaxed2, flows, r_flows, gt_segs, gt_segs2):
            idx1 = select_mask(mask_softmaxed, gt_seg)
            idx2 = select_mask(mask_softmaxed2, gt_seg2)
            mask_forward = warp_w_opflow(mask_softmaxed[idx1], flow)
            mask_backward = warp_w_opflow( [idx2], r_flow)
            loss += self.loss_fn(mask_forward, mask_softmaxed2[idx2])
            loss += self.loss_fn(mask_backward, mask_softmaxed[idx1])
        return loss
