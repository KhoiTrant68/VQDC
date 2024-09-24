import argparse
import os
import sys
from typing import Dict

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import wandb
from models.utils_models import (Scheduler_LinearWarmup,
                                 Scheduler_LinearWarmup_CosineDecay)
from utils.logger import CaptionImageLogger, SetupCallback
from utils.utils_modules import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="Paths to base configuration files. These configurations are loaded from left to right. "
        "Parameters can be overwritten or added with command-line options in the format `--key value`.",
        default=[],
    )
    parser.add_argument(
        "-r",
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Resume training from a checkpoint. Provide the path to the checkpoint directory or a specific "
        "checkpoint file. If not specified, the script attempts to resume from the most recent checkpoint.",
    )
    parser.add_argument(
        "--loss_with_epoch",
        type=bool,
        default=True,
        help="If True, calculate and log loss for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="If set, enables logging with available experiment trackers (e.g., Wandb, TensorBoard).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for saving the trained model.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="If specified, training will be performed on the CPU.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Specify the mixed precision mode during training. Options are 'no', 'fp16', 'bf16', and 'fp8'.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Define the maximum number of epochs for training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="1",
        help="Specify checkpointing frequency. Can be an integer (save every n steps) or 'epoch' to save at the end of each epoch.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logging",
        help="Project directory for logging.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="feat",
        help="Mode of model. feature or entropy.",
    )
    parser.add_argument(
        "--batch_frequency",
        type=int,
        default=50,
        help="Log images every n batches.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=16,
        help="Maximum number of images to log.",
    )
    return parser


def training_function(config: Dict, args: argparse.Namespace):
    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.with_tracking else None,
        project_dir=args.project_dir if args.with_tracking else None,
    )

    if args.with_tracking:
        run = os.path.splitext(os.path.basename(__file__))[0]
        config_dict = OmegaConf.to_container(config, resolve=True)
        accelerator.init_trackers(run, config_dict)
        print('\n args.project_dir ',  args.project_dir)
        print('\n config ',  config)

        wandb.init(project=args.project_dir, config=config)

    data = instantiate_from_config(config.data)
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    model = instantiate_from_config(config.model).to(accelerator.device)
    lr = config.scheduler.base_learning_rate

    optimizer_ae = optim.Adam(
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.quantize.parameters())
        + list(model.quant_conv.parameters())
        + list(model.post_quant_conv.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
    )
    optimizer_disc = optim.Adam(
        model.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
    )

    ngpu = torch.cuda.device_count()
    steps_per_epoch = len(train_dataloader) // max(ngpu, 1)
    training_steps = steps_per_epoch * args.max_epochs
    warmup_steps = int(steps_per_epoch * config.scheduler.warmup_epochs_ratio)
    min_learning_rate = config.model.get("min_learning_rate", 0.0)

    scheduler_type = config["scheduler"]["scheduler_type"]
    if scheduler_type == "linear-warmup":
        scheduler_class = Scheduler_LinearWarmup
        scheduler_args = {"warmup_steps": warmup_steps}
    elif scheduler_type == "linear-warmup_cosine-decay":
        scheduler_class = Scheduler_LinearWarmup_CosineDecay
        scheduler_args = {
            "warmup_steps": warmup_steps,
            "max_steps": training_steps,
            "multipler_min": min_learning_rate / lr,
        }
    else:
        raise NotImplementedError(f"Scheduler type {scheduler_type} not implemented.")

    scheduler_ae = LambdaLR(optimizer_ae, scheduler_class(**scheduler_args))
    scheduler_disc = LambdaLR(optimizer_disc, scheduler_class(**scheduler_args))

    (
        model,
        optimizer_ae,
        optimizer_disc,
        train_dataloader,
        val_dataloader,
        scheduler_ae,
        scheduler_disc,
    ) = accelerator.prepare(
        model,
        optimizer_ae,
        optimizer_disc,
        train_dataloader,
        val_dataloader,
        scheduler_ae,
        scheduler_disc,
    )

    starting_epoch, overall_step = 0, 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            checkpoint_name = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            checkpoint_name = dirs[-1]

        if "epoch" in checkpoint_name:
            starting_epoch = int(checkpoint_name.split("_")[1]) + 1
        else:
            overall_step = int(checkpoint_name.split("_")[1])
            starting_epoch = overall_step // len(train_dataloader)
            overall_step %= len(train_dataloader)

    best_val_loss = float("inf")

    # Initialize callbacks
    setup_callback = SetupCallback(
        resume=args.resume_from_checkpoint,
        now="",  # You might want to set a timestamp here if needed
        logdir=accelerator.project_dir,
        ckptdir=os.path.join(accelerator.project_dir, "checkpoints"),
        cfgdir=os.path.join(accelerator.project_dir, "configs"),
        config=config,
        argv_content=sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
    )
    image_logger = CaptionImageLogger(
        batch_frequency=args.batch_frequency,
        max_images=args.max_images,
    )

    # Call setup callback at the beginning of training
    setup_callback.on_training_start()

    for epoch in tqdm(range(starting_epoch, args.max_epochs), desc="Epoch"):
        model.train()

        if args.with_tracking:
            total_loss = 0

        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc="Training", leave=False)
        ):
            x = model.get_input(batch, model.image_key).to(accelerator.device)
            if args.mode == "feat":
                xrec, qloss, indices, gate = model(x)
            else:
                xrec, qloss, indices, gate, x_entropy = model(x)

            ratio = indices.sum() / (indices.numel())

            aeloss, log_dict_ae = model.calculate_loss(
                x, xrec, qloss, epoch, optimizer_idx=0, gate=gate
            )
            aeloss = aeloss.detach()
            aeloss.requires_grad = True
            with torch.autograd.set_detect_anomaly(True):
                accelerator.backward(aeloss)
                optimizer_ae.step()
                optimizer_ae.zero_grad()
                scheduler_ae.step()

            discloss, log_dict_disc = model.calculate_loss(
                x, xrec, qloss, epoch, optimizer_idx=1, gate=gate
            )
            accelerator.backward(discloss)
            optimizer_disc.step()
            optimizer_disc.zero_grad()
            scheduler_disc.step()

            if args.with_tracking:
                total_loss = aeloss + discloss
                accelerator.log(
                    {
                        "train_aeloss": aeloss.item(),
                        "train_discloss": discloss.item(),
                        "train_fine_ratio": ratio.item(),
                        "train_loss": total_loss.item(),
                        **{
                            k: v.item() if v.numel() == 1 else v.tolist()
                            for k, v in log_dict_ae.items()
                        },
                        **{
                            k: v.item() if v.numel() == 1 else v.tolist()
                            for k, v in log_dict_disc.items()
                        },
                    },
                    step=overall_step,
                )
                wandb.log(
                    {
                        "train_aeloss": aeloss.item(),
                        "train_discloss": discloss.item(),
                        "train_fine_ratio": ratio.item(),
                        "train_loss": total_loss.item(),
                        **{
                            k: v.item() if v.numel() == 1 else v.tolist()
                            for k, v in log_dict_ae.items()
                        },
                        **{
                            k: v.item() if v.numel() == 1 else v.tolist()
                            for k, v in log_dict_disc.items()
                        },
                    },
                    step=overall_step,
                )

            if batch_idx % args.batch_frequency == 0:
                image_logger.log_img(
                    model,
                    batch,
                    batch_idx,
                    split="train",
                    mode=args.mode,
                    epoch=epoch,
                    step=overall_step,
                    accelerator=accelerator,
                )

            overall_step += 1

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_dataloader, desc="Validation", leave=False)
            ):
                x = model.get_input(batch, model.image_key).to(accelerator.device)
                if args.mode == "feat":
                    xrec, qloss, indices, gate = model(x)
                else:
                    xrec, qloss, indices, gate, x_entropy = model(x)
                ratio = indices.sum() / (indices.numel())

                aeloss, log_dict_ae = model.calculate_loss(
                    x, xrec, qloss, epoch, optimizer_idx=0, gate=gate
                )
                discloss, log_dict_disc = model.calculate_loss(
                    x, xrec, qloss, epoch, optimizer_idx=1, gate=gate
                )

                if args.with_tracking:
                    total_val_loss = aeloss + discloss
                    accelerator.log(
                        {
                            "val_aeloss": aeloss.item(),
                            "val_discloss": discloss.item(),
                            "val_fine_ratio": ratio.item(),
                            "val_loss": total_val_loss.item(),
                            **{
                                f"val_{k}": v.item() if v.numel() == 1 else v.tolist()
                                for k, v in log_dict_ae.items()
                            },
                            **{
                                f"val_{k}": v.item() if v.numel() == 1 else v.tolist()
                                for k, v in log_dict_disc.items()
                            },
                        },
                        step=overall_step,
                    )
                    wandb.log(
                        {
                            "val_aeloss": aeloss.item(),
                            "val_discloss": discloss.item(),
                            "val_fine_ratio": ratio.item(),
                            "val_loss": total_val_loss.item(),
                            **{
                                f"val_{k}": v.item() if v.numel() == 1 else v.tolist()
                                for k, v in log_dict_ae.items()
                            },
                            **{
                                f"val_{k}": v.item() if v.numel() == 1 else v.tolist()
                                for k, v in log_dict_disc.items()
                            },
                        },
                        step=overall_step,
                    )

                if batch_idx % args.batch_frequency == 0:
                    image_logger.log_img(
                        model,
                        batch,
                        batch_idx,
                        split="val",
                        mode=args.mode,
                        epoch=epoch,
                        step=overall_step,
                        accelerator=accelerator,
                    )

                total_val_loss += aeloss.item() + discloss.item()

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            print(
                f"\n New best validation loss: {best_val_loss:.4f}, saving checkpoint..."
            )
            accelerator.save_state(
                os.path.join(args.output_dir, "best_checkpoint.ckpt"),
                safe_serialization=False,
            )

    if args.with_tracking:
        accelerator.end_training()
        wandb.finish()


if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    training_function(config=config, args=args)
