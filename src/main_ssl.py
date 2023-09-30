import determinism  # noqa

determinism.i_do_nothing_but_dont_remove_me_otherwise_things_break()  # noqa

import argparse
import bisect
import copy
import os
import sys
import time
from argparse import ArgumentParser

import torch
import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import PeriodicCheckpointer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
import losses
import utils
from eval_utils import eval_unsupmf, get_unsup_image_viz, get_vis_header
from eval_utils_old import eval_unsupmf as eval_out
from mask_former_trainer import setup, Trainer
from mask_former.modeling.transformer.transformer_predictor import TransformerPredictor
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init




logger = utils.log.getLogger('gwm')

def freeze(module, set=False):
    for param in module.parameters():
        param.requires_grad = set


def main(args):
    cfg = setup(args)
    logger.info(f"Called as {' '.join(sys.argv)}")
    logger.info(f'Output dir {cfg.OUTPUT_DIR}')

    random_state = utils.random_state.PytorchRNGState(seed=cfg.SEED).to(torch.device(cfg.MODEL.DEVICE))
    random_state.seed_everything()
    utils.log.checkpoint_code(cfg.OUTPUT_DIR)

    if not cfg.SKIP_TB:
        writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        writer = None

    # initialize model
    model = Trainer.build_model(cfg)
    optimizer = Trainer.build_optimizer(cfg, model)
    scheduler = Trainer.build_lr_scheduler(cfg, optimizer)

    logger.info(f'Optimiser is {type(optimizer)}')


    checkpointer = DetectionCheckpointer(model,
                                         save_dir=os.path.join(cfg.OUTPUT_DIR, 'checkpoints'),
                                         random_state=random_state,
                                         optimizer=optimizer,
                                         scheduler=scheduler)
    periodic_checkpointer = PeriodicCheckpointer(checkpointer=checkpointer,
                                                 period=cfg.SOLVER.CHECKPOINT_PERIOD,
                                                 max_iter=cfg.SOLVER.MAX_ITER,
                                                 max_to_keep=None if cfg.FLAGS.KEEP_ALL else 5,
                                                 file_prefix='checkpoint')
    checkpoint = checkpointer.resume_or_load(args.resume_path, resume=args.resume_path is not None)
    iteration = 0 if args.resume_path is None else checkpoint['iteration']

    train_loader, val_loader = config.loaders(cfg, val_seq=['soldier'])

    # overfit single batch for debug
    # sample = next(iter(loader))

    criterions = {
        'SSLConsistency': (losses.SSLConsistencyLoss(cfg, model), .5, lambda x: 1)}

    criterion = losses.CriterionDict(criterions)

    if len(train_loader.dataset) == 0:
        logger.error("Training dataset: empty")
        sys.exit(0)

    logger.info(
        f'Start of training: dataset {cfg.GWM.DATASET},'
        f' train {len(train_loader.dataset)}, val {len(val_loader.dataset)},'
        f' device {model.device}, keys {cfg.GWM.SAMPLE_KEYS}, '
        f'multiple flows {cfg.GWM.USE_MULT_FLOW}')

    iou_best = 0
    timestart = time.time()
    dilate_kernel = torch.ones((2, 2), device=model.device)
    optimizer = Trainer.build_optimizer(cfg, model)

    # model.sem_seg_head.predictor = TransformerPredictor(
    #             cfg,
    #             768,
    #             num_queries = 1,
    #             mask_classification=False,
    #         ).to(torch.device(cfg.MODEL.DEVICE))

    model.mask_merger = Conv2d(2, 1, kernel_size=1).to(torch.device(cfg.MODEL.DEVICE))
    weight_init.c2_xavier_fill(model.mask_merger)
    total_iter = cfg.TOTAL_ITER if cfg.TOTAL_ITER else cfg.SOLVER.MAX_ITER  # early stop
    with torch.autograd.set_detect_anomaly(cfg.DEBUG) and \
         tqdm(initial=iteration, total=total_iter, disable=utils.environment.is_slurm()) as pbar:
        # eval_out(cfg=cfg, val_loader=val_loader, model=model, criterion=criterion,
        #                 writer=None, writer_iteration=iteration + 1, use_wandb=cfg.WANDB.ENABLE)
        while iteration < total_iter + 7000:
            for sample in train_loader:

                freeze(model, set=False)
                # freeze(model.sem_seg_head.predictor, set=True)
                freeze(model.sem_seg_head, set=True)
                freeze(model.mask_merger, set=True)

                # sample = [e for s in sample for e in s]
                raw_sem_seg = False

                preds = model.forward_base_vid(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True, raw_sem_seg=raw_sem_seg)

                masks_softmaxed = []
                for pred in preds:
                    masks_softmaxed.append([torch.softmax(f, dim=1) for f in pred])
                
                total_losses = []
                log_dicts = []

                loss, log_dict = criterion(sample, masks_softmaxed, model.device, iteration - 19999)

                total_losses.append(loss)
                log_dicts.append(log_dict)

                loss_ws = cfg.GWM.LOSS_MULT.HEIR_W
                total_w = float(sum(loss_ws[:len(total_losses)]))
                log_dict = {}
                if len(total_losses) == 1:
                    log_dict = log_dicts[0]
                    loss = total_losses[0]
                else:
                    loss = 0
                    for i, (tl, w, ld) in enumerate(zip(total_losses, loss_ws, log_dicts)):
                        for k, v in ld.items():
                            log_dict[f'{k}_{i}'] = v * w / total_w
                        loss += tl * w / total_w

                train_log_dict = {f'train/{k}': v for k, v in log_dict.items()}
                del log_dict
                train_log_dict['train/learning_rate'] = optimizer.param_groups[-1]['lr']
                train_log_dict['train/loss_total'] = loss.item()


                optimizer.zero_grad()


                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update()

                # Sanity check for RNG state
                if (iteration + 1) % 1000 == 0 or iteration + 1 in {1, 50}:
                    logger.info(
                        f'Iteration {iteration + 1}. RNG outputs {utils.random_state.get_randstate_magic_numbers(model.device)}')

                if cfg.DEBUG or (iteration + 1) % 100 == 0:
                    logger.info(
                        f'Iteration: {iteration + 1}, time: {time.time() - timestart:.01f}s, loss: {loss.item():.02f}.')

                    for k, v in train_log_dict.items():
                        if writer:
                            writer.add_scalar(k, v, iteration + 1)

                    if cfg.WANDB.ENABLE:
                        wandb.log(train_log_dict, step=iteration + 1)

                if (iteration + 1) % 50 == 0 or (iteration + 1) in [1, 50, 500]:
                    model.eval()
                    if cfg.WANDB.ENABLE and (iteration + 1) % 2500 == 0:
                        image_viz = get_unsup_image_viz(model, cfg, sample)
                        wandb.log({'train/viz': wandb.Image(image_viz.float())}, step=iteration + 1)

                    if iou := eval_unsupmf(cfg=cfg, val_loader=val_loader, model=model, criterion=criterion,
                                           writer=None, writer_iteration=iteration + 1, use_wandb=cfg.WANDB.ENABLE):
                        if cfg.SOLVER.CHECKPOINT_PERIOD and iou > iou_best:
                            iou_best = iou
                            if not args.wandb_sweep_mode:
                                checkpointer.save(name='checkpoint_best', iteration=iteration + 1, loss=loss,
                                                  iou=iou_best)
                            logger.info(f'New best IoU {iou_best:.02f} after iteration {iteration + 1}')
                            # eval_out(cfg=cfg, val_loader=val_loader, model=model, criterion=criterion,
                            #                writer=None, writer_iteration=iteration + 1, use_wandb=cfg.WANDB.ENABLE)
                        wandb.log({'eval/IoU': iou}, step=iteration + 1)
                        if cfg.WANDB.ENABLE:
                            wandb.log({'eval/IoU_best': iou_best}, step=iteration + 1)
                        if writer:
                            writer.add_scalar('eval/IoU_best', iou_best, iteration + 1)


                    model.train()

                periodic_checkpointer.step(iteration=iteration + 1, loss=loss)

                iteration += 1
                timestart = time.time()


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


if __name__ == "__main__":
    wandb_en = True
    if wandb_en:
        wandb.init("GWMOV1")
    args = get_argparse_args().parse_args()
    if args.resume_path:
        args.config_file = "/".join(args.resume_path.split('/')[:-2]) + '/config.yaml'
        print(args.config_file)
    main(args)
    if wandb_en:
        wandb.finish()
