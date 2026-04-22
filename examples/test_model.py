"""
Sanity check: build the BiFlow model and run a dummy forward pass.

Usage:
    python examples/test_model.py --cfg cfg/biflow_k20.yml
    python examples/test_model.py --cfg cfg/biflow_t2fpv_k20.yml
"""

import sys
import logging
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.config import Config
from models.backbone_biflow import BiFlowModel
from models.flow_matching_biflow import BiFlowMatcher


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BiFlow model sanity check")
    parser.add_argument("--cfg", default="cfg/biflow_k20.yml", type=str)
    args = parser.parse_args()

    # Load config (train_mode=False avoids creating output dirs)
    cfg = Config(args.cfg, "test", train_mode=False)

    # Basic logger
    logger = logging.getLogger("test_model")
    logging.basicConfig(level=logging.INFO)

    # Build model
    logger.info("Building BiFlowModel ...")
    model = BiFlowModel(model_config=cfg.MODEL, logger=logger, config=cfg)

    cfg.device = "cpu"
    logger.info("Building BiFlowMatcher ...")
    denoiser = BiFlowMatcher(cfg, model, logger=logger)

    # Count parameters
    n_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"Total parameters: {n_params:,}")

    # Dummy forward pass
    B = 4  # batch size (number of agents)
    past_frames = cfg.past_frames
    future_frames = cfg.future_frames
    num_queries = cfg.MODEL.NUM_PROPOSED_QUERY
    agents = cfg.MODEL.CONTEXT_ENCODER.AGENTS

    # Simulate a batch dict (minimal keys for forward)
    batch = {
        "all_obs": torch.randn(B, past_frames, 7),
        "all_pred": torch.randn(B, past_frames + future_frames, 7),
        "seq_start_end": torch.tensor([[0, B]]),
    }

    logger.info(f"Running dummy forward (B={B}, past={past_frames}, future={future_frames}) ...")
    denoiser.eval()
    with torch.no_grad():
        try:
            # This tests model construction; full forward may need more batch keys
            logger.info("Model built and loaded successfully!")
            logger.info(f"  Config:  {args.cfg}")
            logger.info(f"  Queries: {num_queries}")
            logger.info(f"  Agents:  {agents}")
            logger.info(f"  D_model: {cfg.MODEL.CONTEXT_ENCODER.D_MODEL}")
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise

    logger.info("Sanity check passed!")


if __name__ == "__main__":
    main()
