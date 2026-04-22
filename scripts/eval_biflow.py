import os
import torch
import argparse
import copy
from glob import glob
from pathlib import Path
import sys

from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
print(f"Using root directory: {ROOT_DIR}")

from loaders.dataloader_egotraj import EgoTrajDataset, seq_collate_egotraj

from utils.config import Config
from utils.utils import set_random_seed, log_config_to_file

from models.flow_matching_biflow import BiFlowMatcher
from models.backbone_biflow import BiFlowModel
from trainer.biflow_trainer import BiFlowTrainer


def parse_config():
    """
    Parse the command line arguments and return the configuration options.
    """

    parser = argparse.ArgumentParser()

    # Basic configuration
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory to the checkpoint to load the model from.",
    )
    parser.add_argument("--cfg", default="auto", type=str, help="Config file path")
    parser.add_argument(
        "--exp",
        default="tbd_eval",
        type=str,
        help="Experiment description for each run, name of the saving folder.",
    )
    # Data configuration
    parser.add_argument(
        "--fold_name",
        default="tbd",
        type=str,
        help="Fold name for the experiment.",
    )
    parser.add_argument(
        "--data_source",
        default="original",
        type=str,
        choices=["original", "original_bal"],
        help="Data source: 'original' for EgoTraj-TBD, 'original_bal' for T2FPV-ETH.",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Override the batch size in the config file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory where the data is stored. Auto-set based on fold_name if not specified.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use.",
    )
    parser.add_argument(
        "--data_norm",
        default="min_max",
        choices=["min_max", "original"],
        help="Normalization method for the data.",
    )
    parser.add_argument(
        "--rotate",
        type=bool,
        default=True,
        help="Whether to rotate the trajectories in the dataset",
    )
    parser.add_argument(
        "--rotate_time_frame",
        type=int,
        default=6,
        help="Index of time frames to rotate the trajectories.",
    )

    # Reproducibility configuration
    parser.add_argument(
        "--fix_random_seed",
        action="store_true",
        default=False,
        help="fix random seed for reproducibility",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9527,
        help="Set the random seed to split the testing set for training evaluation.",
    )

    ### FM parameters ###
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=100,        help="Number of sampling timesteps for the FlowMatcher.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lin_poly",        choices=["euler", "lin_poly"],
        help="Solver for the FlowMatcher.",
    )
    parser.add_argument(
        "--lin_poly_p",
        type=int,
        default=5,        help="Degree of the polynomial in the linear stage.",
    )
    parser.add_argument(
        "--lin_poly_long_step",
        type=int,
        default=1000,
        help="Number of steps to mimic slope in the linear stage.",
    )
    ### FM parameters ###

    ### Eval Model ###
    parser.add_argument(
        "--mode",
        type=str,
        default="75",
        help="Evaluation mode: epoch number (e.g., '75'), 'best', or 'last'.",
    )
    parser.add_argument(
        "--save_for_vis",
        default=False,
        action="store_true",
        help="Save predictions for visualization.",
    )

    return parser.parse_args()


def init_basics(args):
    """
    Init the basic configurations for the experiment.
    """

    """Load the config file"""
    result_dir = args.ckpt_dir
    if args.cfg == "auto":
        yml_ls = glob(result_dir + "/*.yml")
        print(f"result_dir: {result_dir}")
        assert (
            len(yml_ls) >= 1
        ), "At least one config file should be found in the directory."
        yml_path = [f for f in yml_ls if "_updated.yml" in os.path.basename(f)][0]
        args.cfg = yml_path
    cfg = Config(args.cfg, f"{args.exp}", train_mode=False)

    tag = "_"
    if args.fold_name != "tbd":
        tag += f"{args.fold_name}_"

    ### Update data versions ###
    if args.data_source == "original":
        tag += "orig_"
    elif args.data_source == "gt_matching":
        tag += "gt_mat_"
    elif args.data_source == "occ_rep":
        tag += "occ_rep_"
    elif args.data_source == "original_bal":
        tag += "orig_bal_"
    else:
        raise ValueError(f"Invalid data source: {args.data_source}")

    ### Update FM parameters ###
    def _update_fm_params(args, cfg, tag):
        if cfg.denoising_method == "fm":
            cfg.sampling_steps = args.sampling_steps
            cfg.solver = args.solver

            if args.solver == "euler":
                solver_tag_ = args.solver
            elif args.solver == "lin_poly":
                cfg.lin_poly_p = args.lin_poly_p
                cfg.lin_poly_long_step = args.lin_poly_long_step
                solver_tag_ = (
                    f"lin_poly_p{args.lin_poly_p}_long{args.lin_poly_long_step}"
                )

            fm_tag_ = f"FM_S{cfg.sampling_steps}_{solver_tag_}"
            tag += fm_tag_
            cfg.solver_tag = fm_tag_

        return cfg, tag

    cfg, tag = _update_fm_params(args, cfg, tag)

    def _update_optimization_params(args, cfg, tag):
        if args.batch_size is not None:
            # override the batch size
            cfg.train_batch_size = args.batch_size
            cfg.val_batch_size = args.batch_size
            cfg.test_batch_size = args.batch_size
        return cfg, tag

    cfg, tag = _update_optimization_params(args, cfg, tag)

    ### voila, create the saving directory ###
    tag += "_test_set"

    tag += f"_{args.mode}"
    tag = tag.replace("__", "_")
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = cfg.create_dirs(tag_suffix=tag)

    """fix random seed"""
    if args.fix_random_seed:
        set_random_seed(args.seed)

    """set up tensorboard and text log"""
    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, "../tb_eval"))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)

    """print the config file"""
    log_config_to_file(cfg.yml_dict, logger=logger)
    return cfg, logger, tb_log


def build_data_loader(cfg, args, mode="train"):
    """
    Build the data loader for the NBA dataset.
    """

    def build_loader(cfg, args, split, batch_size, shuffle):
        dset = EgoTrajDataset(
            cfg=cfg,
            split=split,
            data_dir=args.data_dir,
            rotate_time_frame=args.rotate_time_frame,
            type=args.data_source,
            source=args.fold_name,
        )
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=seq_collate_egotraj,
            pin_memory=True,
        )
        return loader

    loader_dict = {}
    if mode == "eval":
        split_list = ["test"]
        batch_size_list = [cfg.test_batch_size]
        suffle_list = [False]
    elif mode == "train":
        split_list = ["train", "val", "test"]
        batch_size_list = [
            cfg.train_batch_size,
            cfg.val_batch_size,
            cfg.test_batch_size,
        ]
        suffle_list = [True, False, False]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    for split, batch_size, shuffle in zip(split_list, batch_size_list, suffle_list):
        loader = build_loader(cfg, args, split, batch_size, shuffle)
        loader_dict[split] = loader

    if mode == "eval":
        return loader_dict["test"]
    elif mode == "train":
        return loader_dict["train"], loader_dict["val"], loader_dict["test"]


def build_network(cfg, args, logger):
    """
    Build the network for the denoising model.
    """
    model = BiFlowModel(
        model_config=cfg.MODEL,
        logger=logger,
        config=cfg,
    )

    if cfg.denoising_method == "fm":
        denoiser = BiFlowMatcher(
            cfg,
            model,
            logger=logger,
        )
    else:
        raise NotImplementedError(
            f"Denoising method [{cfg.denoising_method}] is not implemented yet."
        )

    return denoiser


def main():
    """Main function to evaluate the BiFlow model."""

    args = parse_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Auto-set data_dir based on fold_name if not specified
    if args.data_dir is None:
        if args.fold_name == "tbd":
            args.data_dir = "./data/egotraj"
        else:
            args.data_dir = "./data/t2fpv"

    assert args.ckpt_dir is not None, "Must specify --ckpt_dir for evaluation."

    # Parse mode: "best", "last", or integer epoch number
    eval_mode = args.mode
    if eval_mode not in ("best", "last"):
        eval_mode = int(eval_mode)

    cfg, logger, tb_log = init_basics(args)
    cfg.K_LIST = [1, 5, 10, 20]

    if cfg.get("fut_traj_min", None) is None:
        # Old checkpoint without norm params in yml — must compute from train data
        _train_loader, _val_loader, test_loader = build_data_loader(cfg, args, mode="train")
    else:
        test_loader = build_data_loader(cfg, args, mode="eval")

    denoiser = build_network(cfg, args, logger)

    trainer = BiFlowTrainer(
        cfg=cfg,
        denoiser=denoiser,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        ema_update_every=1,
    )

    trainer.test(mode=eval_mode, save_for_vis=args.save_for_vis)
    torch.cuda.empty_cache()
    print("--------------------------------")
    print("ckpt_dir: ", args.ckpt_dir)
    print("data_source: ", args.data_source)
    print("fold_name: ", args.fold_name)
    print("solver: ", args.solver)
    print("--------------------------------")


if __name__ == "__main__":
    main()
