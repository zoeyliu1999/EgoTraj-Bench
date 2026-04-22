import copy
import torch.nn as nn
import torch
from einops import rearrange, repeat

from models.utils.common_layers import modulate


class MotionDecoderScore(nn.Module):
    """both scene and score"""

    def __init__(self, config, use_pre_norm, use_adaln=True, use_anchor=False):
        super().__init__()
        self.num_blocks = config.get("NUM_DECODER_BLOCKS", 2)
        self.self_attn_K = nn.ModuleList([])
        self.self_attn_A = nn.ModuleList([])
        template_encoder = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            dropout=config.get("DROPOUT_OF_ATTN", 0.1),
            nhead=config.NUM_ATTN_HEAD,
            dim_feedforward=config.D_MODEL * 4,
            norm_first=use_pre_norm,
            batch_first=True,
        )
        self.use_adaln = use_adaln
        self.dim = config.D_MODEL
        self.use_anchor = use_anchor

        if use_adaln:
            template_adaln = nn.Sequential(
                nn.SiLU(), nn.Linear(config.D_MODEL, 2 * config.D_MODEL, bias=True)
            )

            self.t_adaLN = nn.ModuleList([])

        if use_anchor:
            template_ach_adaln = nn.Sequential(
                nn.LayerNorm(config.D_MODEL),
                nn.SiLU(),
                nn.Linear(config.D_MODEL, 2 * config.D_MODEL, bias=True),
            )
            self.ach_adaLN = nn.ModuleList([])

        for _ in range(self.num_blocks):
            self.self_attn_K.append(copy.deepcopy(template_encoder))
            self.self_attn_A.append(copy.deepcopy(template_encoder))

            if use_adaln:
                self.t_adaLN.append(copy.deepcopy(template_adaln))

                # zero initialization parameters of adaln
                nn.init.constant_(self.t_adaLN[-1][-1].weight, 0)
                nn.init.constant_(self.t_adaLN[-1][-1].bias, 0)

            if use_anchor:
                self.ach_adaLN.append(copy.deepcopy(template_ach_adaln))

                # zero initialization parameters of adaln
                nn.init.constant_(self.ach_adaLN[-1][-1].weight, 0)
                nn.init.constant_(self.ach_adaLN[-1][-1].bias, 0)

    def forward(
        self,
        query_token,
        time_emb=None,
        agent_mask=None,
        anchor_agent=None,
        anchor_scene=None,
    ):
        """
        @param query_token: [B, K, A, D]
        @param time_emb: [B, D]
        @param agent_mask: [B, A]
        @param anchor_agent: [B, A, D]
        @param anchor_scene: [B, D]
        """
        B, K, A = query_token.shape[:3]
        cur_query = query_token

        if self.use_anchor:
            if anchor_agent is None:
                anchor_agent = torch.zeros(B, A, self.dim, device=query_token.device)
            if anchor_scene is None:
                anchor_scene = torch.zeros(B, self.dim, device=query_token.device)

            anchor_agent = anchor_agent * agent_mask.unsqueeze(-1)
            anchor_agent = repeat(anchor_agent, "b a d -> b k a d", k=K)
            anchor_scene = repeat(anchor_scene, "b d -> b k a d", k=K, a=A)
            anchor_ = anchor_agent + anchor_scene

        agent_mask = repeat(agent_mask, "b a -> (b k) a", k=K)

        for i in range(self.num_blocks):
            if self.use_adaln:
                # time modulation
                shift, scale = self.t_adaLN[i](time_emb).chunk(2, dim=-1)
                cur_query = modulate(cur_query, shift, scale)  # [B, K, A, D]

            # K-to-K self-attention
            cur_query = rearrange(cur_query, "b k a d -> (b a) k d")
            cur_query = self.self_attn_K[i](cur_query)

            # A-to-A self-attention, add the agent mask
            cur_query = rearrange(cur_query, "(b a) k d -> (b k) a d", b=B)
            if self.use_anchor:
                cur_query = rearrange(cur_query, "(b k) a d -> b k a d", b=B)
                shift, scale = self.ach_adaLN[i](anchor_).chunk(2, dim=-1)
                cur_query = modulate(cur_query, shift, scale)  # [B, K, A, D]
                cur_query = rearrange(cur_query, "b k a d -> (b k) a d", b=B)

            cur_query = self.self_attn_A[i](
                cur_query, src_key_padding_mask=(agent_mask == 0)
            )

            # reshape
            cur_query = rearrange(cur_query, "(b k) a d -> b k a d", b=B)

        return cur_query
