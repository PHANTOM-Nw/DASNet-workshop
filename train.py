import torch
import time
import math
import wandb
import os
import numpy as np
from contextlib import nullcontext
import utils
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from dasnet.data.das import DASTrainDataset
from dasnet.model.maskrcnn import maskrcnn_resnet50_fpn_dasnet, MaskRCNNPredictor
from dasnet.model.dasnet import build_model
import logging
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches


allowed_bins = {
    1: (10, 20),
    2: (10, 20),
    3: (10, 20),
    4: (10, 20),
    5: (5, 20),
    6: (15, 20),
    7: (-2, -1),
    8: (5, 20),
    9: (5, 20),
    10: (-2, -1),
}

min_keep_bins = {
    1: 3,
    2: 3,
    3: 2,
    4: -1,
    5: 3,
    6: -1,
    7: -1,
    8: -1,
    9: -1,
    10: -1,
}

@torch.no_grad()
def visualize_dasnet_batch(model, images, targets, args, epoch, batch_idx, max_samples=1, dt=0.01, dx=5.1*2):
    """
    plot train: 3-channel data + label(bbox/mask/class) + predictions
    """
    if not utils.is_main_process():
        return

    model.eval()
    os.makedirs(args.figure_dir, exist_ok=True)

    num_show = min(len(images), max_samples)
    outputs = model(images[:num_show])
    model.train()
    saved_paths = []

    for i in range(num_show):
        img = images[i].detach().cpu().permute(1, 2, 0).numpy()  # (H,W,3)
        H, W, _ = img.shape
        extent = (0, W * dt, 0, H * dx / 1e3)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Channel 1", "Channel 2", "Channel 3", "Targets vs Predictions"]

        for c in range(3):
            ax = axes[c]
            vstd = np.std(img[:, :, c]) * 2
            ax.imshow(img[:, :, c] - np.mean(img[:, :, c]), cmap="seismic",
                      vmin=-vstd, vmax=vstd, extent=extent, aspect="auto", origin='lower')
            ax.set_title(titles[c])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Distance (km)")

        ax = axes[3]
        ax.imshow(np.mean(img, axis=2) - np.mean(img), cmap="gray",
                  extent=extent, aspect="auto", alpha=0.6, origin='lower')
        ax.set_title(titles[3])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (km)")
        if "boxes" in targets[i]:
            boxes = targets[i]["boxes"].cpu().numpy()
            labels = targets[i].get("labels", torch.zeros(len(boxes))).cpu().numpy()
            masks = targets[i].get("masks", None)
            att_masks = targets[i].get("attention_masks", None)
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1*dx/1e3, y1*dt),
                                         (x2-x1)*dx/1e3, (y2-y1)*dt,
                                         linewidth=1.5, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1*dx/1e3, y1*dt - 0.02, f"GT:{int(labels[j])}", color='lime', fontsize=7, weight='bold')
                if masks is not None:
                    mask = masks[j].cpu().numpy()
                    ax.contour(mask, colors='lime', alpha=0.4,
                               levels=[0.5], extent=extent, linewidths=1.0)

                if att_masks is not None:
                    att_mask = att_masks[j].cpu().numpy()
                    ax.contour(att_mask, colors='yellow', alpha=0.4,
                               levels=[0.5], extent=extent, linewidths=1.0)


        pred = outputs[i]
        if "boxes" in pred:
            pboxes = pred["boxes"].detach().cpu().numpy()
            plabels = pred.get("labels", torch.zeros(len(pboxes))).cpu().numpy()
            pscores = pred.get("scores", torch.zeros(len(pboxes))).cpu().numpy()
            pmasks = pred.get("masks", None)

            for j, box in enumerate(pboxes):
                if pscores[j] < 0.5:
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1*dx/1e3, y1*dt),
                                         (x2-x1)*dx/1e3, (y2-y1)*dt,
                                         linewidth=1.2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1*dx/1e3, y2*dt - 0.02,
                        f"P:{int(plabels[j])} ({pscores[j]:.2f})",
                        color='red', fontsize=7, weight='bold')
                if pmasks is not None:
                    mask = pmasks[j, 0].detach().cpu().numpy()
                    ax.contour(mask, colors='red', alpha=0.5,
                               levels=[0.5], extent=extent, linewidths=1.0)

        fig.tight_layout()
        save_path = os.path.join(args.figure_dir, f"epoch{epoch:02d}_batch{batch_idx:04d}_sample{i:02d}.png")
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(save_path)

    return saved_paths


logger = logging.getLogger("DASNet")

# def evaluate(model, data_loader, scaler, args, epoch=0, total_samples=1):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

#     loss_keys = ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']
#     for key in loss_keys:
#         metric_logger.add_meter(key, utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

#     header = "Eval:"
#     processed_samples = 0

#     with torch.inference_mode():
#         for images, targets in metric_logger.log_every(data_loader, args.print_freq, header):
#             images = [img.to(args.device) for img in images]
#             targets = [{k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#             batch_size = len(images)
#             metric_logger.update(loss=losses.item())

#             for key in loss_keys:
#                 if key in loss_dict:
#                     metric_logger.update(**{key: loss_dict[key].item()})

#             processed_samples += batch_size
#             if processed_samples >= total_samples:
#                 break

#     metric_logger.synchronize_between_processes()
#     print(f"Test Loss: {metric_logger.loss.global_avg:.6f}")

#     if args.wandb and utils.is_main_process():
#         wandb.log({
#             "test/loss": metric_logger.loss.global_avg,
#             "test/epoch": epoch,
#             **{f"test/{key}": metric_logger.meters[key].global_avg for key in loss_keys if key in metric_logger.meters}
#         })

#     del images, targets, loss_dict

#     return metric_logger
def evaluate(model, data_loader, scaler, args, epoch=0, total_samples=1, save_last_k=3):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    loss_keys = ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']
    for key in loss_keys:
        metric_logger.add_meter(key, utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Eval:"
    processed_samples = 0

    last_batches = []

    with torch.inference_mode():
        for images, targets in metric_logger.log_every(data_loader, args.print_freq, header):

            images = [img.to(args.device) for img in images]
            targets = [{k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            metric_logger.update(loss=losses.item())

            for key in loss_keys:
                if key in loss_dict:
                    metric_logger.update(**{key: loss_dict[key].item()})

            if len(last_batches) >= save_last_k:
                last_batches.pop(0)
            # last_batches.append((images, targets))
            last_batches.append(
                (
                    [img.detach().cpu() for img in images],
                    [{k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()} for t in targets]
                )
            )

            processed_samples += len(images)
            if processed_samples >= total_samples:
                break

    metric_logger.synchronize_between_processes()
    print(f"Test Loss: {metric_logger.loss.global_avg:.6f}")

    if args.wandb and utils.is_main_process():
        wandb.log({
            "test/loss": metric_logger.loss.global_avg,
            "test/epoch": epoch,
            **{f"test/{key}": metric_logger.meters[key].global_avg for key in loss_keys if key in metric_logger.meters}
        })

    del images, targets, loss_dict

    return metric_logger, last_batches

def train_one_epoch(
    model, 
    optimizer, 
    lr_scheduler, 
    data_loader,
    model_ema,
    epoch, 
    accumulation_steps,
    scaler, 
    total_samples,
    args
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # Mask R-CNN loss components
    loss_keys = ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']
    for key in loss_keys:
        metric_logger.add_meter(key, utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = f"Epoch: [{epoch}]"
    ctx = (
        nullcontext() 
        if args.device in ["cpu", "mps"] 
        else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    )

    processed_samples = 0
    optimizer.zero_grad()
    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        # meta = {
        #     "data": [img.to(args.device) for img in images],
        #     "targets": [{k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # }
        targets = [{k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        images = [img.to(args.device) for img in images]
        with ctx:
            # loss_dict = model(meta)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or processed_samples + len(images) >= total_samples:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_value)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            lr_scheduler.step()

        if model_ema is not None and batch_idx % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        # batch_size = len(meta["data"])
        metric_logger.update(loss=losses.item(), lr=optimizer.param_groups[0]["lr"])

        # if args.model in ["dasnet"]:...   
        for key in loss_keys:
            if key in loss_dict:
                metric_logger.update(**{key: loss_dict[key].item()})

        # clear CUDA caches
        # if (batch_idx + 1) % 50 == 0:
        #     torch.cuda.empty_cache()

        if args.wandb and utils.is_main_process():
            log_data = {
                "train/loss": losses.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
                "train/batch": batch_idx,
            }

            # if args.model in ["dasnet"]:...  
            for key in loss_keys:
                if key in loss_dict:
                    log_data[f"train/{key}"] = loss_dict[key].item()
            wandb.log(log_data)

        processed_samples += len(images)
        if processed_samples >= total_samples:
            break


def collate_fn(batch):
    return tuple(zip(*batch))


def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        # return max_lr * (it+1) / warmup_steps + min_lr
        return (max_lr - min_lr) * it / warmup_steps + min_lr
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
        figure_dir = os.path.join(args.output_dir, "figures")
        args.figure_dir = figure_dir
        utils.mkdir(figure_dir)

    utils.init_distributed_mode(args)
    print(args)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
    else:
        rank, world_size = 0, 1

    torch.manual_seed(402 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(402 + rank)

    device = torch.device(args.device)
    dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=((dtype == "float16") & torch.cuda.is_available()))
    args.dtype, args.ptdtype = dtype, ptdtype

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # if args.model in ["dasnet"]:...
    dataset = DASTrainDataset(
        ann_path=args.anno_path,
        root_dir=args.data_path,
        allowed_bins=allowed_bins,
        min_keep_bins=min_keep_bins,
        resize_scale=0.5,
        synthetic_noise=True,
        enable_vflip=True,
    )
    dataset_test = DASTrainDataset(
        ann_path=args.val_anno_path,
        root_dir=args.val_data_path,
        resize_scale=0.5,
        enable_vflip=False,
        synthetic_noise=False,
        enable_stack_event=False
    )

    train_sampler = DistributedSampler(dataset) if args.distributed else None
    test_sampler = DistributedSampler(dataset_test, shuffle=False) if args.distributed else None

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # build model
    model = build_model( # eqnet.models.__dict__[args.model].build_model(
        model_name="maskrcnn_resnet50_selectable_fpn",
        num_classes=1+10,
        pretrained=False,
        pretrained_backbone=True,
        trainable_backbone_layers=3,
        target_layer='P4',
        box_roi_pool_size=7,
        mask_roi_pool_size=(42, 42),
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        box_positive_fraction=0.25,
        max_size=6000,
        min_size=1422
    )
    logger.info("Model:\n{}".format(model))

    print("Model:\n{}".format(model))

    total_params = 0
    print("Trainable parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {param.shape}, {num_params}")
    print(f"Total number of trainable parameters: {total_params}")

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer = optim.AdamW(model.parameters(), lr=1.0, weight_decay=args.weight_decay)

    # lr_scheduler
    iters_per_epoch = len(data_loader)
    warmup_steps = args.lr_warmup_epochs * iters_per_epoch
    max_lr = args.lr
    min_lr = args.lr * args.lr_min_ratio
    max_steps = args.epochs * iters_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda it: get_lr(it, max_lr, min_lr, warmup_steps, max_steps)
    )

    # load checkpoint
    if args.resume:
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed training from epoch {args.start_epoch}")

    # initialize WandB
    if args.wandb and utils.is_main_process():
        wandb.init(project="DASNet", config=vars(args))
        wandb.watch(model, log="all", log_freq=args.print_freq)

    # training loop
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tmp_time = time.time()
        train_one_epoch(
            model, 
            optimizer, 
            lr_scheduler, 
            data_loader, 
            None, 
            epoch, 
            args.accumulation_steps, 
            scaler, 
            len(dataset), 
            args
        )
        print(f"Training time of epoch {epoch} of rank {rank}: {time.time() - tmp_time:.3f}")

        test_metric_logger, last_batches = evaluate(model, data_loader_test, scaler, args, epoch, total_samples=len(dataset_test))

        if utils.is_main_process():
            all_saved_paths = []

            for i, (images, targets) in enumerate(last_batches):
                saved = visualize_dasnet_batch(
                    model,
                    images,
                    targets,
                    args,
                    epoch,
                    batch_idx=i
                )
                all_saved_paths.extend(saved)

            if args.wandb:
                wandb.log({
                    f"test/visualizations_epoch_{epoch}": [
                        wandb.Image(p, caption=os.path.basename(p))
                        for p in all_saved_paths
                    ]
                })

        # save checkpoint
        if args.output_dir and utils.is_main_process():
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

            if test_metric_logger.loss.global_avg < best_loss:
                best_loss = test_metric_logger.loss.global_avg
                torch.save(checkpoint, os.path.join(args.output_dir, "model_best.pth"))

                if args.wandb:
                    best_model = wandb.Artifact("best_model", type="model")
                    best_model.add_file(os.path.join(args.output_dir, "model_best.pth"))
                    wandb.log_artifact(best_model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training completed in {total_time_str}")

    if args.wandb and utils.is_main_process():
        wandb.finish()


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Mask R-CNN Training for DASNet", add_help=add_help)

    # Dataset parameters
    parser.add_argument("--data-path", default="./data", type=str, help="Path to the training dataset")
    parser.add_argument("--anno-path", default="./annotations/train.json", type=str, help="Path to the training annotation file")
    parser.add_argument("--val-data-path", default="./data", type=str, help="Path to the validation dataset")
    parser.add_argument("--val-anno-path", default="./annotations/val.json", type=str, help="Path to the validation annotation file")

    # Training parameters
    parser.add_argument("--device", default="cuda", type=str, help="Device to use (cuda or cpu)")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="Batch size per GPU (total batch size = batch_size * number of GPUs)")
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="Total number of training epochs")
    parser.add_argument("--workers", default=4, type=int, metavar="N", help="Number of data loading workers")

    # Optimizer and learning rate settings
    parser.add_argument("--opt", default="adamw", type=str, help="Optimizer type")
    parser.add_argument("--lr", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay (L2 regularization)")
    parser.add_argument("--lr-warmup-epochs", default=1, type=int, help="Number of warm-up epochs for learning rate")
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str, help="Learning rate scheduler type")
    parser.add_argument("--lr-min-ratio", default=0.1, type=float, help="Minimum learning rate ratio relative to the initial LR")

    # Training configurations
    parser.add_argument("--accumulation-steps", default=1, type=int, help="Number of steps for gradient accumulation")
    parser.add_argument("--clip-value", default=None, type=float, help="Gradient clipping threshold")

    # WandB logging
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", default="DASNet", type=str, help="WandB project name")
    parser.add_argument("--wandb-name", default=None, type=str, help="WandB run name")

    # Checkpointing
    parser.add_argument("--output-dir", default="./output", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to a checkpoint file")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="Starting epoch number")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")

    # Distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--sync-bn", action="store_true", help="Use synchronized batch normalization")
    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed training processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="URL used for setting up distributed training")

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)