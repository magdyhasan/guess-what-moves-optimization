import logging
logging.disable(logging.CRITICAL)

import os
import torch
from pathlib import Path
from types import SimpleNamespace
from detectron2.checkpoint import DetectionCheckpointer

import config
import utils as ut
from eval_utils_old import eval_unsupmf
from mask_former_trainer import setup, Trainer

torch.cuda.set_device('cuda:0')

def load_model_cfg(ckpt_path, dataset=None):
    experiment = Path('../outputs/') / ckpt_path
    
    # args = SimpleNamespace(config_file=str(experiment / 'config.yaml'), opts=[], wandb_sweep_mode=False, resume_path=str(experiment / 'checkpoints/checkpoint_best.pth'), eval_only=True)  # better way
    args = SimpleNamespace(config_file='configs/maskformer/maskformer_R50_bs16_160k_dino.yaml', opts=["GWM.DATASET", dataset], wandb_sweep_mode=False, resume_path=str(experiment / 'checkpoints/checkpoint_best.pth'), eval_only=True)
    cfg = setup(args)
    random_state = ut.random_state.PytorchRNGState(seed=cfg.SEED).to(torch.device(cfg.MODEL.DEVICE))

    model = Trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model,
                                         random_state=random_state,
                                         save_dir=os.path.join(cfg.OUTPUT_DIR, '../..', 'checkpoints'))

    checkpoint_path = str(experiment / 'checkpoints/checkpoint_best.pth')
    checkpoint = checkpointer.resume_or_load(checkpoint_path, resume=False)
    model.eval()
    
    return model, cfg


def main():
    # model, cfg = load_model_cfg('../checkpoints/STv2', "STv2")
    model, cfg = load_model_cfg('../checkpoints/STv2', "STv2")
    # model, cfg = load_model_cfg('../checkpoints/FBMS', "FBMS")
    _, val_loader = config.loaders(cfg)
    iou = eval_unsupmf(cfg, val_loader, model, criterion=None)
    print(iou)

if __name__ == "__main__":
    main()