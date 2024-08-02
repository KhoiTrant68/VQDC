import datetime
import os
import sys

sys.path.append(os.getcwd())

import warnings

warnings.filterwarnings("ignore")

import argparse
import glob

# Torch and Accelerate
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.utils_modules import instantiate_from_config

# Wandb
import wandb

now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")


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
        help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.",
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
        "-e", 
        "--max_epochs", 
        type=int, 
        default=100, 
        help="Number of epochs to train for"
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
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="my_project", help="Name of the wandb project")
    parser.add_argument("--wandb_entity", type=str, default="my_username", help="Your wandb username") 

    return parser

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace):

    # Initialize accelerator
    accelerator = Accelerator()

    # Seed everything
    set_seed(args.seed)

     # resume from checkpoint or logdir
    if args.name and args.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if args.resume:  # resume from checkpoint
        if not os.path.exists(args.resume):
            raise ValueError("Cannot find {}".format(args.resume))
        if os.path.isfile(args.resume):
            paths = args.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = args.resume
        else:  # resume from logdir
            assert os.path.isdir(args.resume), args.resume
            logdir = args.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        args.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        args.base = base_configs+args.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if args.name:
            name = "_" + args.name
        elif args.base:
            cfg_fname = os.path.split(args.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        if args.postfix != "":
            nowname = now + name + "_" + args.postfix
        else:
            nowname = now + name
        logdir = os.path.join("logs", nowname)
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
   
    # Configuration loading 
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # Model, Data, Optimizer Instantiation
    model = instantiate_from_config(config.model)
    data_module = instantiate_from_config(config.data)
    data_module.prepare_data()  # Prepare data if needed

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # Assuming you have an optimizer defined in your config
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.model.base_learning_rate)

    # Prepare model, optimizer, and dataloader with Accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
   
    # Configure learning rate scheduler if needed
    # scheduler = ...  # Instantiate scheduler based on your config
    # scheduler = accelerator.prepare(scheduler)  # Prepare the scheduler 

    # Initialize Wandb 
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project, 
            entity=args.wandb_entity, 
            name=nowname,  
            config=config,   # Log your config
            save_code=True   # Optional: Save your code to wandb
        )
        wandb.watch(model)   # Watch model gradients and parameters

    # Training loop
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.autocast():  # Enable mixed precision if available
                # Forward pass, loss calculation, etc. 
                outputs = model(**batch)
                loss = outputs['loss']

            # Backward pass and optimization 
            accelerator.backward(loss)
            optimizer.step()
            # scheduler.step()  # Update lr scheduler 
            optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
       
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                with accelerator.autocast():  
                    outputs = model(**batch)
                    loss = outputs["loss"]

                val_loss += loss.item() 

        val_loss /= len(val_dataloader)

        # Logging to Wandb
        if accelerator.is_main_process:
            wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss}, step=epoch)

            # ... Log any other metrics, images, etc.  using `wandb.log(...)` ...

    # ... (Save checkpoints, finalize Wandb run if needed) ...

    wandb.finish()

if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print("Current Workspace: ", str(os.getcwd()))
    print("Using Configs: {}".format(args.base))

    main(args)