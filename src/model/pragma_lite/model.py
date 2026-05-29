from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PragmaLiteConfig:
    vocab_size: int
    d_model: int = 192
    n_heads: int = 3
    n_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 4096


class PragmaLite(nn.Module):
    def __init__(self, cfg: PragmaLiteConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.dropout = nn.Dropout(cfg.dropout)

        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.cls_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_mlm_logits: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        if attention_mask is None:
            key_padding_mask = None
        else:
            key_padding_mask = attention_mask == 0
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if return_mlm_logits:
            return self.mlm_head(h)
        return h

    def mlm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlm_head(hidden_states)

    def cls_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        usr = hidden_states[:, 0, :]
        return self.cls_head(usr).squeeze(-1)


class PragmaLiteModel(nn.Module):
    def __init__(self, config: object | None = None, **kwargs: int | float | bool) -> None:
        super().__init__()
        if config is None:
            config = type("Config", (), kwargs)()
        elif kwargs:
            merged = dict(getattr(config, "__dict__", {}))
            merged.update(kwargs)
            config = type("Config", (), merged)()

        self.config = config
        self.d_model = int(getattr(config, "d_model", 192))
        self.vocab_size = int(getattr(config, "vocab_size", 4096))
        self.max_event_tokens = int(getattr(config, "max_event_tokens", 24))
        self.max_events = int(getattr(config, "max_events", 512))
        n_heads = int(getattr(config, "n_heads", 4))
        dropout = float(getattr(config, "dropout", 0.1))
        event_layers = int(getattr(config, "event_layers", 1))
        history_layers = int(getattr(config, "history_layers", 1))

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.event_token_pos = nn.Embedding(self.max_event_tokens, self.d_model)
        self.history_pos = nn.Embedding(self.max_events + 1, self.d_model)
        self.calendar_proj = nn.Linear(3, self.d_model)
        self.time_proj = nn.Sequential(nn.Linear(1, self.d_model), nn.Tanh())

        event_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=int(getattr(config, "d_ffn", self.d_model * 4)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(event_layer, num_layers=event_layers)

        history_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=int(getattr(config, "d_ffn", self.d_model * 4)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(history_layer, num_layers=history_layers)
        self.profile_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.mlm_head = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.vocab_size),
        )

    def _encode_profile(self, profile_input_ids: torch.Tensor) -> torch.Tensor:
        profile_hidden = self.token_emb(profile_input_ids)
        profile_pooled = profile_hidden.mean(dim=1)
        return self.profile_encoder(profile_pooled)

    def encode_events(
        self, event_input_ids: torch.Tensor, calendar_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, num_events, num_tokens = event_input_ids.shape
        token_hidden = self.token_emb(event_input_ids)
        pos_ids = torch.arange(num_tokens, device=event_input_ids.device)
        token_hidden = token_hidden + self.event_token_pos(pos_ids).view(1, 1, num_tokens, self.d_model)

        flat_hidden = token_hidden.view(batch_size * num_events, num_tokens, self.d_model)
        encoded = self.event_encoder(flat_hidden)
        pooled = encoded.mean(dim=1).view(batch_size, num_events, self.d_model)

        if calendar_features is not None:
            pooled = pooled + self.calendar_proj(calendar_features.to(pooled.dtype))
        return pooled

    def forward(
        self,
        profile_input_ids: torch.Tensor,
        event_input_ids: torch.Tensor,
        event_times: torch.Tensor,
        calendar_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        profile_embedding = self._encode_profile(profile_input_ids)
        event_embeddings = self.encode_events(event_input_ids, calendar_features=calendar_features)

        num_events = event_embeddings.size(1)
        history_pos = torch.arange(num_events, device=event_embeddings.device)
        history_input = event_embeddings + self.history_pos(history_pos).view(1, num_events, self.d_model)
        history_input = history_input + self.time_proj(event_times.unsqueeze(-1).to(history_input.dtype))
        history_hidden = self.history_encoder(history_input)
        history_embedding = history_hidden.mean(dim=1)

        record_embedding = self.fusion(torch.cat([profile_embedding, history_embedding], dim=-1))
        return {
            "record_embedding": record_embedding,
            "profile_embedding": profile_embedding,
            "event_embeddings": event_embeddings,
            "history_embedding": history_embedding,
        }
