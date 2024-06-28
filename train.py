import os
import glob
import argparse
import datetime


import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils_modules import instantiate_from_config
from models.utils_models import scheduler_linear_warmup, scheduler_linear_warmup_cosine_decay


import pytz
Shanghai = pytz.timezone("Asia/Shanghai")
now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M-%S")

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
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
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
        "-s", "--seed", type=int, default=2021, help="seed for seed_everything"
    )
    parser.add_argument(
        "-f", "--postfix", type=str, default="", help="post-postfix for default name"
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
        default="wandb",
        nargs="?",
        help="log type",
        choices=["wandb", "tensorboard"],
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
        "--save_n", default=3, type=int, help="save top-n with monitor or save every n epochs without monitor"
    )

    return parser

def main(config):
    # create accelerator
    accelerator = Accelerator()

    # model
    model = instantiate_from_config(config.model)

    # data
    data = instantiate_from_config(config.data)
    # data._instantiate_datasets()

    train_dataloader = DataLoader(data.get_dataloader(split='train'), batch_size=config.data.params.batch_size, shuffle=True)
    val_dataloader = DataLoader(data.get_dataloader(split='validation'), batch_size=config.data.params.batch_size, shuffle=False)

    # optimizer
    lr = config.model.base_learning_rate
    optimizer_ae = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.quantize.parameters())
        + list(model.quant_conv.parameters())
        + list(model.post_quant_conv.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
    )
    optimizer_disc = torch.optim.Adam(
        model.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
    )

    

    # scheduler
    if config.model.scheduler_type == "linear-warmup":
        scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
            optimizer_ae, scheduler_linear_warmup(config.warmup_steps)
        )
        scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
            optimizer_disc, scheduler_linear_warmup(config.warmup_steps)
        )
    elif config.model.scheduler_type == "linear-warmup_cosine-decay":
        multipler_min = config.model.min_learning_rate / config.model.base_learning_rate
        scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
            optimizer_ae,
            scheduler_linear_warmup_cosine_decay(
                warmup_steps=config.model.warmup_epochs,
                max_steps=1000,
                multipler_min=multipler_min,
                step=0
            ),
        )
        scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
            optimizer_disc,
            scheduler_linear_warmup_cosine_decay(
                warmup_steps=config.model.warmup_epochs,
                max_steps=1000,
                multipler_min=multipler_min,
                step=0
            ),
        )
    else:
        raise NotImplementedError()

    # prepare with accelerator
    (
        model,
        optimizer_ae,
        optimizer_disc,
        scheduler_ae,
        scheduler_disc,
        train_dataloader,
        val_dataloader,
    ) = accelerator.prepare(
        model,
        optimizer_ae,
        optimizer_disc,
        scheduler_ae,
        scheduler_disc,
        train_dataloader,
        val_dataloader,
    )

    # training loop
    for epoch in range(config.model.max_epoch):
        # train
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer_ae.zero_grad()
            optimizer_disc.zero_grad()

            aeloss, discloss = model.training_step(batch, batch_idx)

            accelerator.backward(aeloss)
            optimizer_ae.step()
            scheduler_ae.step()

            accelerator.backward(discloss)
            optimizer_disc.step()
            scheduler_disc.step()

            train_loss += aeloss.item() + discloss.item()

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                aeloss, discloss = model.validation_step(batch, batch_idx)
                val_loss += aeloss.item() + discloss.item()

        # log
        print(
            f"Epoch {epoch+1}/{config.model.max_epoch}, Train Loss: {train_loss/len(train_dataloader)}, Val Loss: {val_loss/len(val_dataloader)}"
        )

        # save checkpoint
        if accelerator.is_main_process and (epoch + 1) % config.model.save_every_n_epochs == 0:
            accelerator.save(model.state_dict(), f"{ckptdir}/epoch_{epoch+1}.ckpt")

if __name__ == "__main__":
    # parse arguments
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    print("Current Workspace: ", str(os.getcwd()))
    print("Using Configs: {}".format(opt.base))

    # configure path
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:  # resume from checkpoint
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index("logs") + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:  # resume from logdir
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs") + 1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        if opt.postfix != "":
            nowname = now + name + "_" + opt.postfix
        else:
            nowname = now + name
        logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # configure config
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # set seed
    torch.manual_seed(2024)

    # create directories
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    # save config
    OmegaConf.save(config, os.path.join(cfgdir, "config.yaml"))

    # run main function
    main(config)


import pytz
Shanghai = pytz.timezone("Asia/Shanghai")
now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M-%S")

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
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
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
        "-s", "--seed", type=int, default=2021, help="seed for seed_everything"
    )
    parser.add_argument(
        "-f", "--postfix", type=str, default="", help="post-postfix for default name"
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
        default="wandb",
        nargs="?",
        help="log type",
        choices=["wandb", "tensorboard"],
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
        "--save_n", default=3, type=int, help="save top-n with monitor or save every n epochs without monitor"
    )

    return parser

def main(config):
    # create accelerator
    accelerator = Accelerator()

    # model
    model = instantiate_from_config(config.model)

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    train_dataloader = DataLoader(data._train_dataset, batch_size=config.data.params.batch_size, shuffle=True)
    val_dataloader = DataLoader(data._val_dataset, batch_size=config.data.params.batch_size, shuffle=False)

    # optimizer
    lr = config.model.learning_rate
    optimizer_ae = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.quantize.parameters())
        + list(model.quant_conv.parameters())
        + list(model.post_quant_conv.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
    )
    optimizer_disc = torch.optim.Adam(
        model.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
    )

    # scheduler
    if config.model.scheduler_type == "linear-warmup":
        scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
            optimizer_ae, scheduler_linear_warmup(config.model.warmup_steps)
        )
        scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
            optimizer_disc, scheduler_linear_warmup(config.model.warmup_steps)
        )
    elif config.model.scheduler_type == "linear-warmup_cosine-decay":
        multipler_min = config.model.min_learning_rate / config.model.learning_rate
        scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
            optimizer_ae,
            scheduler_linear_warmup_cosine_decay(
                warmup_steps=config.model.warmup_steps,
                max_steps=model.training_steps,
                multipler_min=multipler_min,
            ),
        )
        scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
            optimizer_disc,
            scheduler_linear_warmup_cosine_decay(
                warmup_steps=config.model.warmup_steps,
                max_steps=model.training_steps,
                multipler_min=multipler_min,
            ),
        )
    else:
        raise NotImplementedError()

    # prepare with accelerator
    (
        model,
        optimizer_ae,
        optimizer_disc,
        scheduler_ae,
        scheduler_disc,
        train_dataloader,
        val_dataloader,
    ) = accelerator.prepare(
        model,
        optimizer_ae,
        optimizer_disc,
        scheduler_ae,
        scheduler_disc,
        train_dataloader,
        val_dataloader,
    )

    # training loop
    for epoch in range(config.model.max_epoch):
        # train
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer_ae.zero_grad()
            optimizer_disc.zero_grad()

            aeloss, discloss = model.training_step(batch, batch_idx)

            accelerator.backward(aeloss)
            optimizer_ae.step()
            scheduler_ae.step()

            accelerator.backward(discloss)
            optimizer_disc.step()
            scheduler_disc.step()

            train_loss += aeloss.item() + discloss.item()

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                aeloss, discloss = model.validation_step(batch, batch_idx)
                val_loss += aeloss.item() + discloss.item()

        # log
        print(
            f"Epoch {epoch+1}/{config.model.max_epoch}, Train Loss: {train_loss/len(train_dataloader)}, Val Loss: {val_loss/len(val_dataloader)}"
        )

        # save checkpoint
        if accelerator.is_main_process and (epoch + 1) % config.model.save_every_n_epochs == 0:
            accelerator.save(model.state_dict(), f"{ckptdir}/epoch_{epoch+1}.ckpt")

if __name__ == "__main__":
    # parse arguments
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    print("Current Workspace: ", str(os.getcwd()))
    print("Using Configs: {}".format(opt.base))

    # configure path
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:  # resume from checkpoint
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index("logs") + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:  # resume from logdir
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs") + 1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        if opt.postfix != "":
            nowname = now + name + "_" + opt.postfix
        else:
            nowname = now + name
        logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # configure config
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # set seed
    torch.manual_seed(config.seed)

    # create directories
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    # save config
    OmegaConf.save(config, os.path.join(cfgdir, "config.yaml"))

    # run main function
    main(config)