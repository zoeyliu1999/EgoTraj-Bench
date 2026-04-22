import copy
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import torch
import argparse

from tensorboardX import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
print(f"Using root directory: {ROOT_DIR}")

from utils.config import Config
from utils.utils import back_up_code_git, set_random_seed, log_config_to_file
from utils.dataset_config import FOLD_CONFIG

from models.flow_matching_biflow import BiFlowMatcher
from models.backbone_biflow import BiFlowModel
from trainer.biflow_trainer import BiFlowTrainer

from loaders.dataloader_egotraj import EgoTrajDataset, seq_collate_egotraj


def parse_config():
    """
    Parse the command line arguments and return the configuration options.
    """

    parser = argparse.ArgumentParser()

    # Basic configuration
    parser.add_argument(
        "--cfg",
        default="cfg/biflow_k20.yml",
        type=str,
        help="Config file path",
    )
    parser.add_argument(
        "--exp",
        default="train",
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
        "--epochs",
        default=None,
        type=int,
        help="Override the number of epochs in the config file.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
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
        "--overfit",
        default=False,
        action="store_true",
        help="Overfit the testing set by setting it to the same entries as the training set.",
    )
    parser.add_argument(
        "--checkpt_freq",
        default=1,
        type=int,
        help="Override the checkpt_freq in the config file.",
    )
    parser.add_argument(
        "--max_num_ckpts",
        default=5,
        type=int,
        help="Override the max_num_ckpts in the config file.",
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
    # resume training
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Whether to resume training from a checkpoint.",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="checkpoint_last",
        help="Name of the checkpoint to resume training from.",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=150,
        help="Start epoch for the resume training.",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=-1,
        help="Early stop for the training. -1 means no early stop.",
    )

    # Reproducibility configuration
    parser.add_argument(
        "--fix_random_seed",
        type=bool,
        default=True,
        help="fix random seed for reproducibility",
    )
    parser.add_argument("--seed", type=int, default=9527, help="Set the random seed.")

    ### FM parameters ###
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=10,
        help="Number of sampling timesteps for the FlowMatcher.",
    )

    # time scheduler during training
    parser.add_argument(
        "--t_schedule",
        type=str,
        choices=["uniform", "logit_normal"],
        default="logit_normal",
        help="Time schedule for the FlowMatcher.",
    )
    parser.add_argument(
        "--logit_norm_mean",
        default=-0.5,
        type=float,
        help="Mean for the logit normal distribution.",
    )
    parser.add_argument(
        "--logit_norm_std",
        default=1.5,
        type=float,
        help="Standard deviation for the logit normal distribution.",
    )

    parser.add_argument(
        "--fm_wrapper",
        type=str,
        default="direct",
        choices=["direct", "velocity", "precond"],
        help="Wrapper for the FlowMatcher.",
    )
    parser.add_argument(
        "--fm_rew_sqrt",
        default=False,
        action="store_true",
        help="Whether to apply square root to the reweighting factor.",
    )
    parser.add_argument(
        "--fm_in_scaling",
        type=bool,
        default=True,
        help="Whether to scale the input to the FlowMatcher.",
    )
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=0.13,
        help="Standard deviation for the data.",
    )
    # input dropout / masking rate
    parser.add_argument(
        "--drop_method",
        default="emb",
        type=str,
        choices=["None", "input", "emb"],
        help="Dropout method for the FlowMatcher.",
    )
    parser.add_argument(
        "--drop_logi_k",
        default=20.0,
        type=float,
        help="Logistic growth rate for masking rate at different timesteps.",
    )
    parser.add_argument(
        "--drop_logi_m",
        default=0.5,
        type=float,
        help="Logistic midpoint for masking rate at different timesteps.",
    )
    ### FM parameters ###

    ### Architecture configuration ###
    parser.add_argument(
        "--use_pre_norm",
        default=False,
        action="store_true",
        help="Where to normalize the input trajectories in the Transformer Encoders.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Overwrite the number of layers in the config file.",
    )
    parser.add_argument(
        "--dropout",
        default=None,
        type=float,
        help="Overwrite the dropout rate in the config file.",
    )
    ### Architecture configuration ###

    ### General denoising objective configuration ###
    parser.add_argument(
        "--tied_noise",
        type=bool,
        default=True,
        help="Whether to use tied noise for the denoiser.",
    )
    ### General denoising objective configuration ###

    ### Regression loss configuration ###
    parser.add_argument(
        "--loss_nn_mode",
        type=str,
        default="agent",
        choices=["agent", "scene"],
        help="Whether to use the agent-wise or scene-wise NN loss.",
    )
    parser.add_argument(
        "--loss_reg_reduction",
        type=str,
        default="sum",
        choices=["mean", "sum"],
        help="Reduction method for the regression loss.",
    )
    ### Regression loss configuration ###

    ### Classification loss configuration ###
    parser.add_argument(
        "--loss_cls_weight",
        type=float,
        default=1.0,
        help="Weight for the classification loss.",
    )
    ### Classification loss configuration ###

    ### Optimization configuration ###
    parser.add_argument(
        "--init_lr",
        type=float,
        default=1e-4,
        help="Override the peak learning rate in the config file.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Override the weight decay in the config file.",
    )
    ### Optimization configuration ###

    ### Refiner configuration ###
    parser.add_argument(
        "--module_layers",
        type=int,
        default=1,
        help="Number of layers for the module.",
    )
    parser.add_argument(
        "--loss_branch_past_weight",
        type=float,
        default=0.5,
        help="Weight for the branch past loss.",
    )
    parser.add_argument(
        "--approx_t_std",
        type=float,
        default=5e-2,
        help="Standard deviation for the approximate time in past branch.",
    )
    parser.add_argument(
        "--use_mask",
        type=bool,
        default=True,
        help="Whether to use the mask.",
    )
    parser.add_argument(
        "--use_imputation",
        type=bool,
        default=True,
        help="Whether to use the imputation.",
    )

    ### Refiner configuration ###

    ### anchor configuration ###
    parser.add_argument(
        "--use_anchor",
        type=bool,
        default=True,
        help="Whether to use the anchor.",
    )
    parser.add_argument(
        "--use_hist_cond",
        type=bool,
        default=True,
        help="Whether to use the historical conditioning.",
    )
    ### anchor configuration ###

    ### Fuser configuration ###
    parser.add_argument(
        "--fuser_name",
        type=str,
        default="SharedFuser",
        choices=["SharedFuser"],
        help="Fuser module to use.",
    )
    parser.add_argument(
        "--max_num_agents",
        type=int,
        default=None,
        help="Max number of agents per scene. Auto-set based on fold_name if not specified.",
    )
    ### Fuser configuration ###

    return parser.parse_args()


def init_basics(args, tag_prefix=None):
    """
    Init the basic configurations for the experiment.
    """

    """Load the config file"""
    cfg = Config(args.cfg, f"{args.exp}")

    # Keep K-list behavior from existing launch conventions.
    if tag_prefix is not None and "k5" in tag_prefix:
        cfg.K_LIST = [1, 5]
    else:
        cfg.K_LIST = [1, 3, 5, 20]
    cfg.USE_CLEAN_HIST = bool(tag_prefix is not None and "cln" in tag_prefix)

    # Sequential config overrides (flattened, no nested update helpers).
    if args.data_source not in {"original", "original_bal"}:
        raise ValueError(f"Invalid data source: {args.data_source}")

    # Sync fold_name from args into cfg so the log reflects the real data fold.
    # Without this, cfg.fold_name stays at the yaml default ('eth') even when
    # hotel/zara1/zara2 are passed via --fold_name, causing a misleading log entry.
    cfg.fold_name = args.fold_name

    cfg.MODEL.USE_ANCHOR = args.use_anchor
    cfg.MODEL.USE_HIST_COND = args.use_hist_cond
    cfg.OPTIMIZATION.LOSS_WEIGHTS["branch_past"] = args.loss_branch_past_weight
    cfg.MODEL.USE_MASK = args.use_mask
    cfg.MODEL.USE_IMPUTATION = args.use_imputation
    cfg.approx_t_std = args.approx_t_std

    if args.fuser_name != "SharedFuser":
        raise NotImplementedError(f"Fuser [{args.fuser_name}] is not implemented yet.")
    cfg.MODEL.FUSER_NAME = args.fuser_name

    if cfg.denoising_method == "fm":
        cfg.sigma_data = args.sigma_data
        cfg.sampling_steps = args.sampling_steps
        cfg.t_schedule = args.t_schedule
        if args.t_schedule == "logit_normal":
            cfg.logit_norm_mean = args.logit_norm_mean
            cfg.logit_norm_std = args.logit_norm_std
        cfg.fm_wrapper = args.fm_wrapper
        cfg.fm_rew_sqrt = args.fm_rew_sqrt
        cfg.fm_in_scaling = args.fm_in_scaling
        if (
            args.drop_method is not None
            and args.drop_logi_k is not None
            and args.drop_logi_m is not None
        ):
            cfg.drop_method = args.drop_method
            cfg.drop_logi_k = args.drop_logi_k
            cfg.drop_logi_m = args.drop_logi_m

    cfg.MODEL.USE_PRE_NORM = args.use_pre_norm
    cfg.MODEL.NUM_LAYERS = args.num_layers
    cfg.MODEL.DROPOUT = args.dropout
    if args.num_layers is not None:
        cfg.MODEL.CONTEXT_ENCODER.NUM_ATTN_LAYERS = args.num_layers
        cfg.MODEL.MOTION_DECODER.NUM_DECODER_BLOCKS = args.num_layers
    if args.dropout is not None:
        cfg.MODEL.CONTEXT_ENCODER.DROPOUT_OF_ATTN = args.dropout
        cfg.MODEL.MOTION_DECODER.DROPOUT_OF_ATTN = args.dropout

    cfg.tied_noise = args.tied_noise
    cfg.LOSS_NN_MODE = args.loss_nn_mode
    cfg.LOSS_REG_REDUCTION = args.loss_reg_reduction

    cfg.MODEL.CONTEXT_ENCODER.AGENTS = args.max_num_agents
    cfg.rotate = args.rotate
    if args.rotate:
        cfg.rotate_time_frame = args.rotate_time_frame
    cfg.data_norm = args.data_norm

    if args.init_lr is not None:
        cfg.OPTIMIZATION.LR = args.init_lr
    if args.weight_decay is not None:
        cfg.OPTIMIZATION.WEIGHT_DECAY = args.weight_decay
    cfg.OPTIMIZATION.LOSS_WEIGHTS["cls"] = args.loss_cls_weight
    if args.epochs is not None:
        cfg.OPTIMIZATION.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.train_batch_size = args.batch_size
        cfg.val_batch_size = args.batch_size
        cfg.test_batch_size = args.batch_size * 2
    if args.checkpt_freq is not None:
        cfg.checkpt_freq = args.checkpt_freq
    cfg.max_num_ckpts = args.max_num_ckpts

    cfg.RESUME.resume = args.resume
    cfg.RESUME.ckpt_name = args.ckpt_name
    cfg.RESUME.start_epoch = args.start_epoch
    cfg.RESUME.early_stop = args.early_stop

    # Keep tag compact and stable for release experiments.
    k_value = cfg.K_LIST[-1] if len(cfg.K_LIST) > 0 else cfg.MODEL.NUM_PROPOSED_QUERY
    data_source_tag = {
        "original": "orig",
        "original_bal": "orig_bal",
    }[args.data_source]
    tag_parts = [
        tag_prefix or "run",
        args.fold_name,
        data_source_tag,
        cfg.denoising_method.upper(),
        cfg.MODEL.FUSER_NAME,
        f"K{k_value}",
        f"EP{cfg.OPTIMIZATION.NUM_EPOCHS}",
        f"BS{cfg.train_batch_size}",
        f"LR{cfg.OPTIMIZATION.LR}",
    ]
    tag = "_" + "_".join(str(x) for x in tag_parts)

    ### viola, create the saving directory ###
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = cfg.create_dirs(tag_suffix=tag)

    """fix random seed"""
    if args.fix_random_seed:
        set_random_seed(args.seed)

    """set up tensorboard and text log"""
    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, "../tb"))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)

    """back up the code"""
    back_up_code_git(cfg, logger=logger)

    """print the config file"""
    log_config_to_file(cfg.yml_dict, logger=logger)
    return cfg, logger, tb_log


def build_data_loader(cfg, args, mode="train"):
    """
    Build the data loader for the NBA dataset.
    """

    DatasetClass = EgoTrajDataset
    collate_fn = seq_collate_egotraj

    def build_loader(cfg, args, split, batch_size, shuffle):
        dset = DatasetClass(
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
            collate_fn=collate_fn,
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
    """Main function to train the BiFlow model."""

    args = parse_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Auto-set data_dir and max_num_agents based on fold_name if not specified
    if args.fold_name not in FOLD_CONFIG:
        raise ValueError(
            f"Invalid fold name: '{args.fold_name}'. "
            f"Expected one of: {list(FOLD_CONFIG.keys())}"
        )
    fold_cfg = FOLD_CONFIG[args.fold_name]
    if args.data_dir is None:
        args.data_dir = fold_cfg["data_dir"]
    if args.max_num_agents is None:
        args.max_num_agents = fold_cfg["max_num_agents"]

    tag_prefix = "v1"
    cfg, logger, tb_log = init_basics(args, tag_prefix=tag_prefix)
    train_loader, val_loader, test_loader = build_data_loader(cfg, args)
    cfg.save_updated_yml()  # re-save with normalization stats from dataset

    # Re-seed before model init so weight initialization is deterministic
    # regardless of how many random ops data loading consumed
    if args.fix_random_seed:
        set_random_seed(args.seed)

    denoiser = build_network(cfg, args, logger)

    trainer = BiFlowTrainer(
        cfg=cfg,
        denoiser=denoiser,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        ema_update_every=1,
    )

    trainer.train()


if __name__ == "__main__":
    main()
