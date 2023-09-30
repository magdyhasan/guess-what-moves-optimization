import logging
logging.disable(logging.CRITICAL)

import os
import torch
from pathlib import Path
from types import SimpleNamespace
from detectron2.checkpoint import DetectionCheckpointer

import config
import utils as ut
from eval_utils import eval_unsupmf_masks
from mask_former_trainer import setup, Trainer
from datasets.flow_ssl_vid_detectron_single import FlowSSLVIDDetectron
import torch.nn.functional as F
import wandb

torch.cuda.set_device('cuda:0')

from argparse import ArgumentParser
import argparse



def get_argparse_args():
    parser = ArgumentParser()
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--use_wandb', dest='wandb_sweep_mode', action='store_false')  # for sweep
    parser.add_argument('--config-file', type=str,
                        default='configs/maskformer/maskformer_R50_bs16_160k_dino.yaml')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

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
    # flow[:, :, 0] = flow[:, :, 0] / (w-1)
    # flow[:, :, 1] = flow[:, :, 1] / (h-1)
    flow = flow.unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, h, w))
    grid = grid + 2*flow

    # mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    # align corner True or False shows similar results
    mask = F.grid_sample(mask, grid, mode='bilinear')
    return mask.squeeze()

def warp_frame_with_optical_flow(prev_frame, flow):
    # Compute the grid of new pixel positions based on the optical flow
    batch_size, num_channels, width, height = prev_frame.size()
    
    # Generate a grid of pixel coordinates
    grid_x, grid_y = torch.meshgrid(torch.arange(0, width), torch.arange(0, height))
    
    # Reshape the grid to match the input dimensions
    grid_x = grid_x.to(prev_frame.device).unsqueeze(0).expand([batch_size, -1, -1])
    grid_y = grid_y.to(prev_frame.device).unsqueeze(0).expand([batch_size, -1, -1])
    
    # Calculate the new pixel positions by adding the flow vectors
    new_x = grid_x + flow[:, 0, :, :]
    new_y = grid_y + flow[:, 1, :, :]
    
    # Stack the new pixel positions
    new_positions = torch.stack((new_x, new_y), dim=3)
    
    # Use grid_sample to interpolate pixel values from the previous frame
    warped_frame = F.grid_sample(prev_frame, new_positions)
    
    return warped_frame

def load_model_cfg(args):
    
    cfg = setup(args)
    random_state = ut.random_state.PytorchRNGState(seed=cfg.SEED).to(torch.device(cfg.MODEL.DEVICE))

    model = Trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model,
                                         random_state=random_state,
                                         save_dir=os.path.join(cfg.OUTPUT_DIR, '../..', 'checkpoints'))

    checkpoint_path = '../outputs/stv2/20230705_004149_te/checkpoints/checkpoint_0019999.pth'
    checkpoint = checkpointer.resume_or_load(checkpoint_path, resume=True)
    model.eval()
    
    return model, cfg


def main(args):
    # model, cfg = load_model_cfg('../checkpoints/STv2', "STv2")
    model, cfg = load_model_cfg(args)
    # model, cfg = load_model_cfg('../checkpoints/FBMS', "FBMS")
    

    img_dir = '/SegTrackv2/JPEGImages'
    gt_dir = '/SegTrackv2/annotations'

    val_flow_dir = '/SegTrackv2/Flows_gap1/'
    val_data_dir = [val_flow_dir, img_dir, gt_dir]

    root_path_str = '/storage/user/mahmo/temp/ssl-vos/data'
    root_path = Path(f"/{root_path_str.lstrip('/').rstrip('/')}")
    val_data_dir = [root_path / path.lstrip('/').rstrip('/') for path in val_data_dir]
    samples = FlowSSLVIDDetectron(data_dir=val_data_dir,
                                    resolution=(128, 224),
                                    pair_list=[1, 2, -1, -2],
                                    val_seq=['soldier'],
                                    size_divisibility=32,
                                    flow_clip=float('inf'),
                                    norm=False)
    for i in range(len(samples)):
        sample = samples[i]
        preds = model.forward_base([sample], keys=["rgb"], get_eval=True)
        masks_raw = torch.stack([x['sem_seg'] for x in preds], 0)
        masks_softmaxed = torch.softmax(masks_raw, dim=1)

        device = masks_raw.device

        gt_seg = samples[i]['sem_seg_ori'].to(device).float()

        idx = select_mask(masks_softmaxed[0], gt_seg)

        samples[i]['masks_softmaxed'] = masks_softmaxed[:, idx, :]

    
    print('IOU at {}: {}\n'.format(-1, eval_unsupmf_masks(samples)))

    steps = 10
    ra = 0.8
    wnn = 0
    for st in range(steps):
        for i in range(len(samples) - 1):
            mask1 = samples[i]['masks_softmaxed'][0]
            mask2 = samples[i + 1]['masks_softmaxed'][0]

            device = mask1.device

            flow = samples[i]['flow'].to(device)
            r_flow = samples[i]['r_flow'].to(device)

            mask_forward = warp_w_opflow(mask1, flow)
            mask_backward = warp_w_opflow(mask2, r_flow)

            # mask_forward = warp_frame_with_optical_flow(mask1[None, None], flow.permute(2, 0, 1)[None])[0][0]
            # mask_backward = warp_frame_with_optical_flow(mask2[None, None], r_flow.permute(2, 0, 1)[None])[0][0]

            samples[i + 1]['masks_softmaxed_forward'] = mask_forward
            samples[i]['masks_softmaxed_backward'] = mask_backward
        
        for i in range(len(samples)):
            org = samples[i]['masks_softmaxed'][0]
            if i == 0:
                refined = org * ra + (1-ra) * samples[i]['masks_softmaxed_backward'] 
            elif i == len(samples) -1:
                refined = org * ra + (1-ra) * samples[i]['masks_softmaxed_forward']
            else:
                refined = org * ra + (1-ra) * (samples[i]['masks_softmaxed_forward'] + samples[i]['masks_softmaxed_backward'])
            
            samples[i]['masks_softmaxed_refined'] = refined[None, :]


        # wandb_table = wandb.Table(columns=['image', 'GT', 'prediction', 'prediction_refined', 'masks_softmaxed_forward', 'masks_softmaxed_backward'])
        wandb_table = wandb.Table(columns=['image', 'GT', 'prediction', 'prediction_refined'])
        for i in range(len(samples)):
            device = samples[i]['masks_softmaxed_refined'].device
            gt_seg = samples[i]['sem_seg_ori'].to(device).float()
            flow = samples[i]['flow'].to(device)

            if samples[i]['frame_id'] in [16, 17]:
                wandb_rgb1 = wandb.Image(samples[i]['original_rgb'])
                wandb_gt1 = wandb.Image(gt_seg)
                wandb_mask1 = wandb.Image(F.interpolate(samples[i]['masks_softmaxed'][None], size=(gt_seg.shape[-2], gt_seg.shape[-1]))[0][0])
                wandb_mask1_refined = wandb.Image(F.interpolate(samples[i]['masks_softmaxed_refined'][None], size=(gt_seg.shape[-2], gt_seg.shape[-1]))[0][0])
                
                flow_ed = F.interpolate(flow.permute(2, 0, 1)[None][None], size=(2, gt_seg.shape[-2], gt_seg.shape[-1]))[0][0]
                gt_warpped = warp_frame_with_optical_flow(gt_seg[None][None], flow_ed[None])[0][0]
                # wandb_mask_forward = wandb.Image(F.interpolate(samples[i]['masks_softmaxed_forward'][None][None], size=(gt_seg.shape[-2], gt_seg.shape[-1]))[0][0])
                # wandb_mask_forward = wandb.Image(F.interpolate(gt_warpped[None][None], size=(gt_seg.shape[-2], gt_seg.shape[-1]))[0][0])
                # wandb_mask_backward = wandb.Image(F.interpolate(samples[i]['masks_softmaxed_backward'][None][None], size=(gt_seg.shape[-2], gt_seg.shape[-1]))[0][0])

                wandb_table.add_data(wandb_rgb1, wandb_gt1, wandb_mask1, wandb_mask1_refined)

        wandb.log({'images/' + str(wnn): wandb_table})
        wnn += 1

        for i in range(len(samples)):
            samples[i]['masks_softmaxed'] = samples[i]['masks_softmaxed_refined']

        print('IOU at {}: {}\n'.format(st, eval_unsupmf_masks(samples)))

if __name__ == "__main__":
    wandb.init("GWMOV1OFF")
    args = get_argparse_args().parse_args()
    if args.resume_path:
        args.config_file = "/".join(args.resume_path.split('/')[:-2]) + '/config.yaml'
        print(args.config_file)
    main(args)
    wandb.finish()