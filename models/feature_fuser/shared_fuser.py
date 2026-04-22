import torch
import torch.nn as nn

from einops import repeat, rearrange

from models.utils.common_layers import SinusoidalPosEmb


class SharedFuser(nn.Module):
    def __init__(self, config):
        super(SharedFuser, self).__init__()
        self.config = config
        self.past_frames = self.config.get("past_frames", 8)
        self.future_frames = self.config.get("future_frames", 12)

        self.model_cfg = config.MODEL
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL

        ###### shared layers ######
        ### serves the purpose of positional encoding
        self.motion_query_embedding = nn.Embedding(
            self.model_cfg.NUM_PROPOSED_QUERY, self.dim
        )
        self.agent_order_embedding = nn.Embedding(
            self.model_cfg.CONTEXT_ENCODER.AGENTS, self.dim
        )
        self.post_pe_cat_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        # time embedding
        time_dim = self.dim * 1
        sinu_pos_emb = SinusoidalPosEmb(self.dim, theta=10000)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        ###### separate branches ######

        self.branch_embedding = nn.Embedding(2, self.dim)

        self.branch_pst = CondAttnFuserS(self.config, self.past_frames * 2)
        self.branch_fut = CondAttnFuserS(self.config, self.future_frames * 2)

        pst_frozen = self.config.OPTIMIZATION.LOSS_WEIGHTS["branch_past"] < 1e-6
        if pst_frozen:
            for p in self.branch_pst.parameters():
                p.requires_grad = False

    def forward(
        self,
        y_t_in,
        t_fut,
        y_t_cond,
        x_t_in,
        t_pst,
        x_t_cond,
        agent_mask,
        anchor_agent,
        anchor_scene,
    ):
        B, K, A, _ = y_t_in.shape
        t_fut_ = t_fut
        t_pst_ = t_pst

        if self.config.denoising_method == "fm":
            t_fut_scale = t_fut * 1000.0
            t_pst_scale = t_pst * 1000.0
        else:
            t_fut_scale = t_fut
            t_pst_scale = t_pst

        x_t_emb_ = self.time_mlp(t_pst_scale)
        y_t_emb_ = self.time_mlp(t_fut_scale)

        branch_id_emb = self.branch_embedding(
            torch.tensor([0, 1], device=y_t_in.device)
        )
        x_t_emb = x_t_emb_ + branch_id_emb[0].unsqueeze(0)
        y_t_emb = y_t_emb_ + branch_id_emb[1].unsqueeze(0)

        x_t_emb_batch = repeat(x_t_emb, "b d -> b k a d", b=B, k=K, a=A)
        y_t_emb_batch = repeat(y_t_emb, "b d -> b k a d", b=B, k=K, a=A)

        k_pe = self.motion_query_embedding(torch.arange(K, device=y_t_in.device))
        k_pe_batch = repeat(k_pe, "k d -> b k a d", b=B, a=A)

        a_pe = self.agent_order_embedding(torch.arange(A, device=y_t_in.device))
        a_pe_batch = repeat(a_pe, "a d -> b k a d", b=B, k=K)

        ### saparate modules
        agent_mask_batch = repeat(agent_mask, "b a -> b k a", k=K)
        agent_mask_batch = rearrange(agent_mask_batch, "b k a -> (b k) a")

        emb_fusion_pst = self.branch_pst(
            x_t_in,
            x_t_cond,
            t_pst_,
            agent_mask_batch,
            x_t_emb_batch,
            k_pe_batch,
            a_pe_batch,
            None,
            None,
        )
        emb_fusion_fut = self.branch_fut(
            y_t_in,
            y_t_cond,
            t_fut_,
            agent_mask_batch,
            y_t_emb_batch,
            k_pe_batch,
            a_pe_batch,
            anchor_agent,
            anchor_scene,
        )

        query_token_pst = self.post_pe_cat_mlp(emb_fusion_pst + k_pe_batch + a_pe_batch)
        query_token_fut = self.post_pe_cat_mlp(emb_fusion_fut + k_pe_batch + a_pe_batch)

        return query_token_fut, query_token_pst, y_t_emb_, x_t_emb_


class CondAttnFuserS(nn.Module):
    def __init__(self, config, in_dim):
        super(CondAttnFuserS, self).__init__()
        self.model_cfg = config.MODEL
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        time_dim = self.dim * 1
        self.config = config

        self.noisy_y_mlp = nn.Sequential(
            nn.Linear(in_dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        dropout_ = self.model_cfg.MOTION_DECODER.DROPOUT_OF_ATTN
        self.noisy_y_attn_k = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=4,
            dim_feedforward=self.dim * 4,
            dropout=dropout_,
            batch_first=True,
        )
        self.noisy_y_attn_a = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=4,
            dim_feedforward=self.dim * 4,
            dropout=dropout_,
            batch_first=True,
        )

        dim_decoder = self.model_cfg.MOTION_DECODER.D_MODEL

        self.use_anchor = self.model_cfg.get("USE_ANCHOR", False)
        fuse_dim = time_dim + self.dim * 2 + self.dim * int(self.use_anchor)
        self.init_emb_fusion_mlp = nn.Sequential(
            nn.Linear(fuse_dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dim_decoder),
        )

    def forward(
        self,
        y,
        y_cond,
        time_,
        agent_mask_batch,
        t_emb_batch,
        k_pe_batch,
        a_pe_batch,
        anchor_agent,
        anchor_scene,
    ):
        B, K, A, _ = y.shape

        y_emb = self.noisy_y_mlp(y)
        y_emb = y_emb + k_pe_batch + a_pe_batch

        y_emb_k = rearrange(y_emb, "b k a d -> (b a) k d")
        y_emb_k = self.noisy_y_attn_k(y_emb_k)
        y_emb = rearrange(y_emb_k, "(b a) k d -> b k a d", b=B, a=A)

        y_emb_a = rearrange(y_emb, "b k a d -> (b k) a d")
        y_emb_a = self.noisy_y_attn_a(
            y_emb_a, src_key_padding_mask=(agent_mask_batch == 0)
        )
        y_emb = rearrange(y_emb_a, "(b k) a d -> b k a d", b=B, k=K)

        if self.training and self.config.get("drop_method", None) == "emb":
            assert (
                self.config.get("drop_logi_k", None) is not None
                and self.config.get("drop_logi_m", None) is not None
            )
            m, k = self.config.drop_logi_m, self.config.drop_logi_k
            p_m = 1 / (1 + torch.exp(-k * (time_ - m)))
            p_m = p_m[:, None, None, None]
            y_emb = y_emb.masked_fill(torch.rand_like(p_m) < p_m, 0.0)

        concat_list = [y_cond, y_emb, t_emb_batch]

        if self.use_anchor:
            if anchor_agent is None:
                anchor_agent = torch.zeros(B, A, self.dim, device=y_emb.device)
            if anchor_scene is None:
                anchor_scene = torch.zeros(B, self.dim, device=y_emb.device)

            agent_mask = rearrange(agent_mask_batch, "(b k) a -> b k a", k=K)
            anchor_agent = repeat(
                anchor_agent, "b a d -> b k a d", k=K
            ) * agent_mask.unsqueeze(-1)
            anchor_scene = repeat(anchor_scene, "b d -> b k a d", k=K, a=A)
            anchor_ = anchor_agent + anchor_scene
            concat_list.append(anchor_)

        emb_fusion = self.init_emb_fusion_mlp(torch.cat(concat_list, dim=-1))
        return emb_fusion
