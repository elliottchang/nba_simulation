"""Factorized possession model.

One shared player encoder feeds four heads that mirror how a possession
actually unfolds:

    ball-ender  P(which of the 5 offensive players ends the chance)
    action      P(2pt / 3pt / turnover / off_foul / drawn_foul | ball-ender)
    shot        P(make / miss / andone / shooting_foul | shot attempt)
    rebound     P(offensive board | miss)  +  P(which player gets it)

Player tokens = stat-anchored: a linear projection of the (z-scored) stat
vector and live stint/fatigue features, plus a small learned residual
embedding shared across seasons, plus a season embedding. Matchup structure
enters through cross-attention of the ball-ender token over the five
defender tokens — interactions are learned between latent *types*, which is
what three seasons of data can support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def mlp(in_dim, hidden, out_dim, dropout=config.DROPOUT):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class PlayerEncoder(nn.Module):
    def __init__(self, num_players, num_seasons):
        super().__init__()
        d = config.PLAYER_TOKEN_DIM
        self.stat_proj = nn.Linear(config.STAT_DIM + 2, d)   # stats + stint/cum
        self.player_emb = nn.Embedding(num_players, config.PLAYER_EMB_DIM, padding_idx=0)
        self.season_emb = nn.Embedding(num_seasons, config.SEASON_EMB_DIM)
        self.merge = nn.Linear(d + config.PLAYER_EMB_DIM + config.SEASON_EMB_DIM, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, idx, feat, season_idx):
        # idx: (B, 5)  feat: (B, 5, STAT_DIM+2)  season_idx: (B,)
        s = self.season_emb(season_idx)                       # (B, S)
        s = s.unsqueeze(1).expand(-1, idx.shape[1], -1)       # (B, 5, S)
        x = torch.cat([self.stat_proj(feat), self.player_emb(idx), s], dim=-1)
        return self.norm(self.merge(x))                       # (B, 5, D)


class PossessionModelV2(nn.Module):
    def __init__(self, num_players, num_seasons):
        super().__init__()
        d = config.PLAYER_TOKEN_DIM
        c = config.CONTEXT_DIM
        h = config.HIDDEN_DIM

        self.encoder = PlayerEncoder(num_players, num_seasons)
        self.matchup_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)

        # ball-ender: score each offensive token against pooled context
        self.ender_score = mlp(d + 2 * d + c, h, 1)
        # action: ball-ender token + pools + matchup vector + context
        self.action_head = mlp(2 * d + 2 * d + c, h, len(config.ACTIONS))
        # shot result: same inputs + is_three flag
        self.shot_head = mlp(2 * d + 2 * d + c + 1, h, len(config.SHOT_RESULTS))
        # oreb probability: pools + is_three + context
        self.oreb_head = mlp(2 * d + c + 1, h, 1)
        # rebounder: token + own-side pool (side-agnostic, side enters via flag)
        self.reb_score = mlp(d + d + 1, h, 1)

    # ------------------------------------------------------------ pieces
    def encode(self, batch):
        off = self.encoder(batch["off_idx"], batch["off_feat"], batch["season_idx"])
        deff = self.encoder(batch["def_idx"], batch["def_feat"], batch["season_idx"])
        return off, deff

    def ender_logits(self, off, deff, ctx):
        off_pool, def_pool = off.mean(1), deff.mean(1)
        pooled = torch.cat([off_pool, def_pool, ctx], dim=-1)          # (B, 2D+C)
        pooled = pooled.unsqueeze(1).expand(-1, 5, -1)                 # (B, 5, 2D+C)
        return self.ender_score(torch.cat([off, pooled], dim=-1)).squeeze(-1)  # (B, 5)

    def _ender_inputs(self, off, deff, ctx, slot):
        B = off.shape[0]
        ender = off[torch.arange(B, device=off.device), slot.clamp(min=0)]  # (B, D)
        # rows without an attributed ball-ender use the lineup-mean token
        ender = torch.where((slot >= 0).unsqueeze(-1), ender, off.mean(1))
        matchup, _ = self.matchup_attn(ender.unsqueeze(1), deff, deff)
        matchup = matchup.squeeze(1)
        return torch.cat([ender, matchup, off.mean(1), deff.mean(1), ctx], dim=-1)

    def action_logits(self, off, deff, ctx, slot):
        return self.action_head(self._ender_inputs(off, deff, ctx, slot))

    def shot_logits(self, off, deff, ctx, slot, is_three):
        x = self._ender_inputs(off, deff, ctx, slot)
        return self.shot_head(torch.cat([x, is_three.unsqueeze(-1)], dim=-1))

    def oreb_logit(self, off, deff, ctx, is_three):
        x = torch.cat([off.mean(1), deff.mean(1), ctx, is_three.unsqueeze(-1)], dim=-1)
        return self.oreb_head(x).squeeze(-1)

    def rebounder_logits(self, side_tokens, is_offensive):
        pool = side_tokens.mean(1, keepdim=True).expand(-1, 5, -1)
        flag = is_offensive.view(-1, 1, 1).expand(-1, 5, 1).float()
        return self.reb_score(torch.cat([side_tokens, pool, flag], dim=-1)).squeeze(-1)

    # ------------------------------------------------------------ training loss
    def loss(self, batch):
        off, deff = self.encode(batch)
        ctx = batch["ctx"]
        losses, metrics = {}, {}

        # ball-ender
        mask = batch["slot"] >= 0
        if mask.any():
            logits = self.ender_logits(off, deff, ctx)[mask]
            losses["ender"] = F.cross_entropy(logits, batch["slot"][mask])
            metrics["ender_acc"] = (logits.argmax(-1) == batch["slot"][mask]).float().mean()

        # action: all rows (slot -1 rows are handled with a lineup-mean token,
        # which keeps un-attributed drawn fouls in the action distribution)
        logits = self.action_logits(off, deff, ctx, batch["slot"])
        losses["action"] = F.cross_entropy(logits, batch["action"])

        # shot result
        smask = mask & (batch["shot"] >= 0)
        if smask.any():
            is_three = (batch["action"] == 1).float()
            logits = self.shot_logits(off, deff, ctx, batch["slot"], is_three)[smask]
            losses["shot"] = F.cross_entropy(logits, batch["shot"][smask])

        # rebound
        omask = batch["oreb"] >= 0
        if omask.any():
            is_three = (batch["action"] == 1).float()
            logit = self.oreb_logit(off, deff, ctx, is_three)[omask]
            losses["oreb"] = F.binary_cross_entropy_with_logits(
                logit, batch["oreb"][omask].float())

        rmask = batch["reb_slot"] >= 0
        if rmask.any():
            side = torch.where(batch["reb_side"].view(-1, 1, 1).bool(), off, deff)
            logits = self.rebounder_logits(side, batch["reb_side"])[rmask]
            losses["rebounder"] = F.cross_entropy(logits, batch["reb_slot"][rmask])

        total = sum(losses.values())
        return total, {k: v.item() for k, v in losses.items()}, metrics


def make_optimizer(model: PossessionModelV2):
    """Embeddings get heavy weight decay; everything else light."""
    emb_params, other_params = [], []
    for name, p in model.named_parameters():
        (emb_params if "emb" in name else other_params).append(p)
    return torch.optim.AdamW([
        {"params": emb_params, "weight_decay": config.EMB_WEIGHT_DECAY},
        {"params": other_params, "weight_decay": 1e-5},
    ], lr=config.LR)
