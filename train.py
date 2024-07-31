import warnings
import os
import glob
import argparse
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from accelerate import Accelerator
from omegaconf import OmegaConf

from utils.utils_modules import instantiate_from_config

import pytz
import datetime


def get_current_time():
    return datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime(
        "%m-%dT%H-%M-%S"
    )


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs",
        default=[],
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=2021,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-l",
        "--logtype",
        type=str,
        default="tensorboard",
        choices=["wandb", "tensorboard"],
        help="log type",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project",
        default="DynamicVectorQuantization",
    )
    parser.add_argument(
        "--save_n",
        default=3,
        type=int,
        help="save top-n with monitor or save every n epochs without monitor",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    return parser


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_to_tensorboard(writer, log_stats, epoch):
    for key, value in log_stats.items():
        writer.add_scalar(key, value, epoch)


def main(args: argparse.Namespace):
    # Initialize accelerator early for better device management
    accelerator = Accelerator()

    set_seed(args.seed)

    # Simplify logdir and checkpoint handling
    if args.resume:
        if not os.path.exists(args.resume):
            raise ValueError(f"Cannot find {args.resume}")
        if os.path.isfile(args.resume):
            logdir = os.path.dirname(args.resume)
            ckpt = args.resume
        else:
            logdir = args.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        args.resume_from_checkpoint = ckpt
        args.base = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml"))) + args.base
        nowname = os.path.basename(logdir)
    else:
        basename = (
            os.path.splitext(os.path.basename(args.base[0]))[0] if args.base else ""
        )
        nowname = (
            f"{get_current_time()}_{basename}{f'_{args.postfix}' if args.postfix else ''}"
        )
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")

    # Configuration loading
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # Model and Data Loading
    model = instantiate_from_config(config.model)
    data_module = instantiate_from_config(config.data)
    train_dataloader, val_dataloader = data_module.setup_dataloaders()  

    # Optimizer and Accelerator Preparation
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Training Loop
    if accelerator.is_main_process and args.logtype == "tensorboard":
        tb_writer = SummaryWriter(logdir)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader, disable=not accelerator.is_local_main_process):
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs["loss"]

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * len(batch)

        train_loss /= len(train_dataloader.dataset)
        accelerator.print(
            f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}"
        )

        if accelerator.is_main_process:
            log_stats = {"train_loss": train_loss}
            if args.logtype == "tensorboard":
                log_to_tensorboard(tb_writer, log_stats, epoch)

    # Save Model
    if accelerator.is_main_process:
        torch.save(model.state_dict(), os.path.join(ckptdir, "model.pt"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print("Current Workspace:", os.getcwd())
    print("Using Configs:", args.base)
    main(args)