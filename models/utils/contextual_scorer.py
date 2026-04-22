import torch.nn as nn


class AnchorHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.agent_anchor = nn.Sequential(
            nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.scene_pool = nn.Sequential(
            nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def forward(self, encoder_feat, agent_mask, agent_score):
        if agent_score is not None:
            encoder_feat = encoder_feat * agent_score.unsqueeze(-1)
        agent_anchor = self.agent_anchor(encoder_feat)

        mask = agent_mask.float()
        denom = mask.sum(dim=1, keepdim=True)
        assert denom.min() > 0, "No valid agents"
        pooled = (encoder_feat * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
        scene_anchor = self.scene_pool(pooled)  # [B, D]

        return agent_anchor, scene_anchor
