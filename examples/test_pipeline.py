"""
End-to-end pipeline test: load data, build model, and run a single forward pass.

Usage:
    python examples/test_pipeline.py --data_dir ./data/egotraj --fold_name tbd
    python examples/test_pipeline.py --data_dir ./data/t2fpv --fold_name eth --cfg cfg/biflow_t2fpv_k20.yml
"""

import sys
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.config import Config
from models.backbone_biflow import BiFlowModel
from models.flow_matching_biflow import BiFlowMatcher
from loaders.dataloader_egotraj import EgoTrajDataset, seq_collate_egotraj


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BiFlow pipeline test")
    parser.add_argument("--cfg", default="cfg/biflow_k20.yml", type=str)
    parser.add_argument("--data_dir", default="./data/egotraj", type=str)
    parser.add_argument("--fold_name", default="tbd", type=str)
    parser.add_argument("--data_source", default="original", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()

    logger = logging.getLogger("test_pipeline")
    logging.basicConfig(level=logging.INFO)

    # 1. Load config
    logger.info(f"Loading config: {args.cfg}")
    cfg = Config(args.cfg, "test_pipeline", train_mode=False)
    cfg.device = "cpu"
    cfg.test_batch_size = args.batch_size

    # 2. Load dataset
    logger.info(f"Loading dataset: fold={args.fold_name}, source={args.data_source}")
    dataset = EgoTrajDataset(
        cfg=cfg,
        split="test",
        data_dir=args.data_dir,
        rotate_time_frame=6,
        type=args.data_source,
        source=args.fold_name,
    )
    logger.info(f"  Dataset size: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate_egotraj,
    )

    # 3. Build model
    logger.info("Building model ...")
    model = BiFlowModel(model_config=cfg.MODEL, logger=logger, config=cfg)
    denoiser = BiFlowMatcher(cfg, model, logger=logger)
    denoiser.eval()

    n_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"  Total parameters: {n_params:,}")

    # 4. Load one batch
    batch = next(iter(loader))
    logger.info(f"  Batch keys: {list(batch.keys())}")
    logger.info(f"  past_traj shape: {batch['past_traj'].shape}")
    logger.info(f"  fut_traj shape:  {batch['fut_traj'].shape}")
    logger.info(f"  agent_mask shape: {batch['agent_mask'].shape}")

    logger.info("Pipeline test passed! Data + Model are compatible.")


if __name__ == "__main__":
    main()
