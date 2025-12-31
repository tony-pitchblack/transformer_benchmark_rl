import math
import typing as tp

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class MLPBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: tp.Sequence[int],
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_dim, *list(hidden_dims), out_dim]
        layers: tp.List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        y = self.net(x.reshape(-1, x.shape[-1]))
        return y.reshape(*leading, y.shape[-1])


def _soft_update_(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_tgt, p_src in zip(target.parameters(), source.parameters(), strict=True):
            p_tgt.data.mul_(1.0 - tau).add_(p_src.data, alpha=tau)


class CRRLightningModule(LightningModule):
    def __init__(
        self,
        encoder_torch_model: tp.Any,
        actor_hidden_dims: tp.Sequence[int],
        critic_hidden_dims: tp.Sequence[int],
        dropout: float,
        n_negatives: int,
        gamma: float,
        beta: float,
        tau: float,
        m_adv: int,
        m_td: int,
        lr_actor: float,
        lr_critic: float,
        weight_decay: float = 0.0,
        max_advantage: float = 20.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_torch_model"])

        self.encoder = encoder_torch_model
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # For compatibility with src.models.transformers.trainer.RecallCallback
        self.torch_model = self.encoder
        self.item_embs = self.encoder.item_model.get_all_embeddings().detach()

        self.actor_hidden_dims = list(actor_hidden_dims)
        self.critic_hidden_dims = list(critic_hidden_dims)
        self.dropout = float(dropout)

        self.n_negatives = int(n_negatives)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.tau = float(tau)
        self.m_adv = int(m_adv)
        self.m_td = int(m_td)
        self.lr_actor = float(lr_actor)
        self.lr_critic = float(lr_critic)
        self.weight_decay = float(weight_decay)
        self.max_advantage = float(max_advantage)

        self.automatic_optimization = False

        with torch.no_grad():
            dim = int(self.encoder.item_model.get_all_embeddings().shape[1])

        self.actor = MLPBackbone(
            in_dim=dim,
            hidden_dims=self.actor_hidden_dims,
            out_dim=dim,
            dropout=self.dropout,
        )
        self.critic = MLPBackbone(
            in_dim=2 * dim,
            hidden_dims=self.critic_hidden_dims,
            out_dim=1,
            dropout=self.dropout,
        )
        self.actor_tgt = MLPBackbone(
            in_dim=dim,
            hidden_dims=self.actor_hidden_dims,
            out_dim=dim,
            dropout=self.dropout,
        )
        self.critic_tgt = MLPBackbone(
            in_dim=2 * dim,
            hidden_dims=self.critic_hidden_dims,
            out_dim=1,
            dropout=self.dropout,
        )
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.actor_tgt.parameters():
            p.requires_grad = False
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

    def validation_step(  # type: ignore[override]
        self, batch: tp.Dict[str, torch.Tensor], batch_idx: int
    ) -> tp.Dict[str, torch.Tensor]:
        x = batch["x"]
        item_embs = self.encoder.item_model.get_all_embeddings().detach()
        self.item_embs = item_embs

        state_embs = self._encode_states(batch, item_embs)
        lengths = (x != 0).sum(dim=1).clamp_min(1) - 1
        last_state = state_embs[torch.arange(x.shape[0], device=x.device), lengths]
        policy_emb = self.actor(last_state)
        logits = policy_emb @ item_embs.to(policy_emb.device).T
        return {"logits": logits}

    def configure_optimizers(self):
        opt_actor = torch.optim.AdamW(
            self.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay
        )
        opt_critic = torch.optim.AdamW(
            self.critic.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay
        )
        return [opt_actor, opt_critic]

    @staticmethod
    def _sample_negatives(
        shape: tp.Tuple[int, ...],
        n_items: int,
        n_item_extra_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.randint(
            low=n_item_extra_tokens, high=n_items, size=shape, device=device
        )

    def _encode_states(
        self, batch: tp.Dict[str, torch.Tensor], item_embs: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder.encode_sessions(batch, item_embs)

    @staticmethod
    def _aligned_reward_for_action(reward: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.roll(reward, shifts=-1, dims=1)
        r[:, -1] = 0.0
        r = r.masked_fill(y == 0, 0.0)
        return r

    def _q(
        self, critic: MLPBackbone, state_embs: torch.Tensor, action_embs: torch.Tensor
    ) -> torch.Tensor:
        q = critic(torch.cat([state_embs, action_embs], dim=-1)).squeeze(-1)
        return q

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        x = batch["x"]
        y = batch["y"]
        reward = batch["reward"]
        valid = y != 0

        item_embs = self.encoder.item_model.get_all_embeddings().detach()
        n_items = int(item_embs.shape[0])
        n_item_extra_tokens = int(
            getattr(self.encoder.item_model, "n_item_extra_tokens", 0)
            or getattr(self.encoder, "n_item_extra_tokens", 0)
            or 0
        )
        if n_item_extra_tokens <= 0 and hasattr(self.encoder.item_model, "n_items"):
            n_item_extra_tokens = max(0, int(item_embs.shape[0]) - int(self.encoder.item_model.n_items))

        state_embs = self._encode_states(batch, item_embs)

        reward_a = self._aligned_reward_for_action(reward, y)

        neg = self._sample_negatives(
            shape=(y.shape[0], y.shape[1], self.n_negatives),
            n_items=n_items,
            n_item_extra_tokens=n_item_extra_tokens,
            device=y.device,
        )
        cand = torch.cat([y.unsqueeze(-1), neg], dim=-1)  # [B,L,C]
        cand_embs = item_embs[cand]  # [B,L,C,D]

        policy_embs = self.actor(state_embs)  # [B,L,D]
        logits = torch.einsum("bld,blcd->blc", policy_embs, cand_embs)
        logp = F.log_softmax(logits, dim=-1)[..., 0]

        act_embs = item_embs[y]  # [B,L,D]
        q_sa = self._q(self.critic, state_embs, act_embs)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            c = probs.shape[-1]
            m = min(self.m_adv, c)
            flat_probs = probs.reshape(-1, c)
            sampled_idx = torch.multinomial(flat_probs, num_samples=m, replacement=True)
            sampled_idx = sampled_idx.reshape(*probs.shape[:-1], m)  # [B,L,m]
            sampled_embs = torch.gather(
                cand_embs,
                dim=-2,
                index=sampled_idx.unsqueeze(-1).expand(-1, -1, -1, cand_embs.shape[-1]),
            )  # [B,L,m,D]
            q_samples = self._q(self.critic, state_embs.unsqueeze(-2).expand_as(sampled_embs), sampled_embs)
            baseline = q_samples.mean(dim=-1)
            adv = (q_sa - baseline).clamp(min=-self.max_advantage, max=self.max_advantage)
            weight = torch.exp(adv / max(self.beta, 1e-8))

        actor_loss = -(logp * weight).masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)

        state_next = torch.roll(state_embs, shifts=-1, dims=1)
        y_next = torch.roll(y, shifts=-1, dims=1)
        valid_next = y_next != 0

        neg_next = self._sample_negatives(
            shape=(y_next.shape[0], y_next.shape[1], self.n_negatives),
            n_items=n_items,
            n_item_extra_tokens=n_item_extra_tokens,
            device=y.device,
        )
        cand_next = torch.cat([y_next.unsqueeze(-1), neg_next], dim=-1)
        cand_next_embs = item_embs[cand_next]

        with torch.no_grad():
            pol_next = self.actor_tgt(state_next)
            logits_next = torch.einsum("bld,blcd->blc", pol_next, cand_next_embs)
            probs_next = F.softmax(logits_next, dim=-1)
            c2 = probs_next.shape[-1]
            m2 = min(self.m_td, c2)
            flat_probs2 = probs_next.reshape(-1, c2)
            sampled_idx2 = torch.multinomial(flat_probs2, num_samples=m2, replacement=True)
            sampled_idx2 = sampled_idx2.reshape(*probs_next.shape[:-1], m2)
            sampled_next_embs = torch.gather(
                cand_next_embs,
                dim=-2,
                index=sampled_idx2.unsqueeze(-1).expand(-1, -1, -1, cand_next_embs.shape[-1]),
            )
            q_next = self._q(
                self.critic_tgt,
                state_next.unsqueeze(-2).expand_as(sampled_next_embs),
                sampled_next_embs,
            )
            v_next = q_next.mean(dim=-1)
            v_next = v_next.masked_fill(~valid_next, 0.0)
            target = reward_a + self.gamma * v_next
            target = target.masked_fill(~valid, 0.0)

        if int(valid.sum()) == 0:
            critic_loss = torch.tensor(0.0, device=y.device)
        else:
            critic_loss = F.mse_loss(q_sa.masked_select(valid), target.masked_select(valid))

        opt_actor, opt_critic = self.optimizers()  # type: ignore[assignment]
        opt_critic.zero_grad()
        self.manual_backward(critic_loss)
        opt_critic.step()

        opt_actor.zero_grad()
        self.manual_backward(actor_loss)
        opt_actor.step()

        _soft_update_(self.actor_tgt, self.actor, self.tau)
        _soft_update_(self.critic_tgt, self.critic, self.tau)

        self.log("train/actor_loss", actor_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/critic_loss", critic_loss, on_step=False, on_epoch=True, prog_bar=False)
        return actor_loss + critic_loss


