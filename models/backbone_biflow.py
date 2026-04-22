import numpy as np

import torch
import torch.nn as nn
from .context_encoder import build_context_encoder
from .motion_decoder import build_decoder
from .feature_fuser import build_feature_fuser
from .utils.common_layers import build_mlps
from einops import repeat, rearrange
from models.utils.contextual_scorer import AnchorHead


class BiFlowModel(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.config = config
        self.past_frames = self.config.get("past_frames", 8)

        use_pre_norm = self.model_cfg.get("USE_PRE_NORM", False)
        assert not use_pre_norm, "Pre-norm is not supported in this model"

        self.use_mask = self.model_cfg.get("USE_MASK", False)
        self.use_imputation = self.model_cfg.get("USE_IMPUTATION", False)

        self.use_anchor = self.model_cfg.get("USE_ANCHOR", False)
        self.use_hist_cond = self.model_cfg.get("USE_HIST_COND", False)

        if self.use_anchor:
            self.anchor_head = AnchorHead(self.dim)

        self.context_encoder = build_context_encoder(
            self.model_cfg.CONTEXT_ENCODER, use_pre_norm
        )

        self.feature_fuser = build_feature_fuser(self.config)

        self.motion_decoder_fut = build_decoder(
            self.model_cfg.MOTION_DECODER, use_pre_norm, use_anchor=self.use_anchor
        )

        self.motion_decoder_pst = build_decoder(
            self.model_cfg.MOTION_DECODER, use_pre_norm
        )

        dim_decoder = self.model_cfg.MOTION_DECODER.D_MODEL
        self.reg_head_fut = build_mlps(
            c_in=dim_decoder,
            mlp_channels=self.model_cfg.REGRESSION_MLPS,
            ret_before_act=True,
            without_norm=True,
        )
        self.cls_head_fut = build_mlps(
            c_in=dim_decoder,
            mlp_channels=self.model_cfg.CLASSIFICATION_MLPS,
            ret_before_act=True,
            without_norm=True,
        )

        self.reg_head_pst = build_mlps(
            c_in=dim_decoder,
            mlp_channels=self.model_cfg.RECONSTRUCTION_MLPS,
            ret_before_act=True,
            without_norm=True,
        )
        self.cls_head_pst = build_mlps(
            c_in=dim_decoder,
            mlp_channels=self.model_cfg.CLASSIFICATION_MLPS,
            ret_before_act=True,
            without_norm=True,
        )

        pst_frozen = self.config.OPTIMIZATION.LOSS_WEIGHTS["branch_past"] < 1e-6
        if pst_frozen:
            for p in self.motion_decoder_pst.parameters():
                p.requires_grad = False
            for p in self.cls_head_pst.parameters():
                p.requires_grad = False
            for p in self.reg_head_pst.parameters():
                p.requires_grad = False

        # print out the number of parameters
        params_encoder = sum(p.numel() for p in self.context_encoder.parameters())
        params_fuser = sum(p.numel() for p in self.feature_fuser.parameters())
        params_decoder_fut = sum(
            p.numel() for p in self.motion_decoder_fut.parameters()
        )
        params_decoder_pst = sum(
            p.numel() for p in self.motion_decoder_pst.parameters()
        )
        params_total = sum(p.numel() for p in self.parameters())
        params_other = (
            params_total
            - params_encoder
            - params_decoder_fut
            - params_decoder_pst
            - params_fuser
        )
        logger.info(
            "Total parameters: {:,}, Encoder: {:,}, Fuser: {:,}, Decoder_fut: {:,}, Decoder_pst: {:,}, Other: {:,}".format(
                params_total,
                params_encoder,
                params_fuser,
                params_decoder_fut,
                params_decoder_pst,
                params_other,
            )
        )

    def forward(self, y_t_in, t_fut, x_t_in, t_pst, x_data):
        """
        y: noisy vector
        x_data: data dict containing the following keys:
            - past_traj: past trajectory
            - future_traj: future trajectory
            - future_traj_vel: future trajectory velocity
            - trajectory mask: [it may exist]
            - batch_size: batch size
            - indexes: exist when we aim to perform IMLE
        time: denoising time step
        """
        ### NBA assertions
        assert y_t_in.shape[-1] == 24, "y shape is not correct"
        assert x_t_in.shape[-1] == 16, "x shape is not correct"

        B, K, A, _ = y_t_in.shape

        ### history condition
        past_traj = x_data["past_traj_original_scale"]  # [B, A, P, 6]
        past_traj_mask = x_data["past_traj_valid"]  # [B, A, P]
        past_traj = (
            past_traj
            if self.use_imputation
            else past_traj * past_traj_mask.unsqueeze(-1)
        )
        if self.config.get("use_ablation_dataset", False):
            agent_mask = torch.ones((B, A), device=past_traj.device)
        else:
            agent_mask = x_data["agent_mask"]  # [B, A]

        agent_score = None

        # concat valid mask -> change the in_dim of encoder
        concat_past_traj_mask = (
            past_traj_mask.unsqueeze(-1)
            if self.use_mask
            else torch.zeros_like(past_traj_mask).unsqueeze(-1)
        )
        # TODO: feature encoder to handle frame_score
        concat_frame_score = torch.zeros_like(past_traj_mask).unsqueeze(
            -1
        )  # [B, A, P, 1]
        concat_list = [past_traj, concat_past_traj_mask, concat_frame_score]
        past_traj = torch.cat(concat_list, dim=-1)

        ### context encoder
        encoder_out = self.context_encoder(past_traj, agent_mask, agent_score)
        encoder_out_batch = repeat(encoder_out, "b a d -> b k a d", k=K)

        ### anchor head
        anchor_agent, anchor_scene = None, None
        if self.use_anchor:
            anchor_agent, anchor_scene = self.anchor_head(
                encoder_out, agent_mask, agent_score
            )

        ### init embeddings
        if not self.use_hist_cond:
            x_cond = torch.zeros_like(encoder_out_batch)
        else:
            x_cond = encoder_out_batch
        y_query_token, x_query_token, y_t_emb, x_t_emb = self.feature_fuser(
            y_t_in,
            t_fut,
            encoder_out_batch,
            x_t_in,
            t_pst,
            x_cond,
            agent_mask,
            anchor_agent,
            anchor_scene,
        )

        readout_token_fut = self.motion_decoder_fut(
            y_query_token, y_t_emb, agent_mask, anchor_agent, anchor_scene
        )
        readout_token_pst = self.motion_decoder_pst(x_query_token, x_t_emb, agent_mask)

        ### readout layers
        denoiser_y = self.reg_head_fut(readout_token_fut)  # [B, K, A, F * D]
        denoiser_cls_fut = self.cls_head_fut(readout_token_fut).squeeze(-1)  # [B, K, A]
        denoiser_x = self.reg_head_pst(readout_token_pst)  # [B, K, A, P * D]
        denoiser_cls_pst = self.cls_head_pst(readout_token_pst).squeeze(-1)  # [B, K, A]

        return denoiser_x, denoiser_cls_pst, denoiser_y, denoiser_cls_fut
