from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class PragmaLiteConfig:
    vocab_size: int
    d_model: int = 192
    n_heads: int = 3
    n_layers: int = 4
    d_ffn: int | None = None
    dropout: float = 0.1
    max_seq_len: int = 4096
    max_profile_tokens: int = 200
    max_event_tokens: int = 24
    max_events: int = 512
    profile_layers: int = 1
    event_layers: int | None = None
    history_layers: int | None = None
    rope_base: float = 10000.0
    use_time_encoding: bool = True
    use_profile_encoder: bool = True

    def __post_init__(self) -> None:
        if self.d_ffn is None:
            self.d_ffn = self.d_model * 4
        if self.event_layers is None:
            self.event_layers = self.n_layers
        if self.history_layers is None:
            self.history_layers = self.n_layers


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("...n,d->...nd", positions.float(), self.inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        return emb.cos(), emb.sin()


class PragmaAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rope: bool, rope_base: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if positions is None:
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
            cos, sin = self.rope(positions)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            valid = attention_mask.bool()
            scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(out)


class PragmaEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float, use_rope: bool, rope_base: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PragmaAttention(d_model, n_heads, dropout, use_rope=use_rope, rope_base=rope_base)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x), attention_mask=attention_mask, positions=positions))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class PragmaEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float,
        use_rope: bool,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PragmaEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    use_rope=use_rope,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.use_rope = use_rope

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, positions=positions)
        return self.norm(x)


class PragmaLiteModel(nn.Module):
    def __init__(self, config: object | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            cfg = PragmaLiteConfig(**kwargs)
        elif kwargs:
            merged = dict(getattr(config, "__dict__", {}))
            merged.update(kwargs)
            cfg = PragmaLiteConfig(**merged)
        elif isinstance(config, PragmaLiteConfig):
            cfg = config
        else:
            cfg = PragmaLiteConfig(**getattr(config, "__dict__", {}))

        self.cfg = cfg
        self.config = cfg
        self.d_model = cfg.d_model
        self.vocab_size = cfg.vocab_size
        self.max_profile_tokens = cfg.max_profile_tokens
        self.max_event_tokens = cfg.max_event_tokens
        self.max_events = cfg.max_events
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.profile_cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.profile_encoder = PragmaEncoder(
            num_layers=cfg.profile_layers,
            d_model=self.d_model,
            n_heads=cfg.n_heads,
            d_ffn=cfg.d_ffn,
            dropout=cfg.dropout,
            use_rope=True,
            rope_base=cfg.rope_base,
        )
        self.event_encoder = PragmaEncoder(
            num_layers=cfg.event_layers,
            d_model=self.d_model,
            n_heads=cfg.n_heads,
            d_ffn=cfg.d_ffn,
            dropout=cfg.dropout,
            use_rope=False,
            rope_base=cfg.rope_base,
        )
        self.history_encoder = PragmaEncoder(
            num_layers=cfg.history_layers,
            d_model=self.d_model,
            n_heads=cfg.n_heads,
            d_ffn=cfg.d_ffn,
            dropout=cfg.dropout,
            use_rope=True,
            rope_base=cfg.rope_base,
        )
        self.calendar_proj = nn.Linear(3, self.d_model)
        self.time_proj = nn.Sequential(nn.Linear(1, self.d_model), nn.Tanh())
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
        self.cls_head = nn.Linear(self.d_model, 1)
        nn.init.normal_(self.profile_cls, std=0.02)

    def _encode_profile(
        self,
        profile_input_ids: torch.Tensor,
        profile_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens = profile_input_ids.shape
        token_hidden = self.dropout(self.token_emb(profile_input_ids))
        cls_token = self.profile_cls.expand(batch_size, -1, -1)
        profile_hidden = torch.cat([cls_token, token_hidden], dim=1)
        if profile_attention_mask is None:
            mask = torch.ones((batch_size, num_tokens + 1), dtype=torch.bool, device=profile_input_ids.device)
        else:
            mask = torch.cat(
                [
                    torch.ones((batch_size, 1), dtype=torch.bool, device=profile_input_ids.device),
                    profile_attention_mask.bool(),
                ],
                dim=1,
            )
        encoded = self.profile_encoder(profile_hidden, attention_mask=mask)
        return encoded[:, 0, :], encoded[:, 1:, :]

    def encode_events(
        self,
        event_input_ids: torch.Tensor,
        calendar_features: torch.Tensor | None = None,
        event_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_events, num_tokens = event_input_ids.shape
        if num_events == 0:
            return self.token_emb.weight.new_zeros((batch_size, 0, self.d_model))
        token_hidden = self.token_emb(event_input_ids)
        flat_hidden = token_hidden.view(batch_size * num_events, num_tokens, self.d_model)
        flat_mask = None
        if event_attention_mask is not None:
            flat_mask = event_attention_mask.view(batch_size * num_events, num_tokens)
        encoded = self.event_encoder(flat_hidden, attention_mask=flat_mask)
        pooled = encoded[:, 0, :]
        pooled = pooled.view(batch_size, num_events, self.d_model)

        if calendar_features is not None:
            pooled = pooled + self.calendar_proj(calendar_features.to(pooled.dtype))
        return pooled

    def _encode_event_tokens(
        self,
        event_input_ids: torch.Tensor,
        event_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_events, num_tokens = event_input_ids.shape
        if num_events == 0:
            empty_hidden = self.token_emb.weight.new_zeros((batch_size, 0, max(num_tokens - 1, 0), self.d_model))
            empty_embed = self.token_emb.weight.new_zeros((batch_size, 0, self.d_model))
            return empty_hidden, empty_embed

        token_hidden = self.dropout(self.token_emb(event_input_ids))
        flat_hidden = token_hidden.view(batch_size * num_events, num_tokens, self.d_model)
        flat_mask = None
        if event_attention_mask is not None:
            flat_mask = event_attention_mask.view(batch_size * num_events, num_tokens)
        encoded = self.event_encoder(flat_hidden, attention_mask=flat_mask)
        encoded = encoded.view(batch_size, num_events, num_tokens, self.d_model)
        local_hidden = encoded[:, :, 1:, :]
        pooled = encoded[:, :, 0, :]
        return local_hidden, pooled

    def _encode_history(
        self,
        profile_embedding: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_tokens = torch.cat([profile_embedding.unsqueeze(1), event_embeddings], dim=1)
        if self.cfg.use_time_encoding and event_times is not None and event_embeddings.size(1) > 0:
            time_bias = self.time_proj(event_times.unsqueeze(-1).to(history_tokens.dtype))
            history_tokens[:, 1:, :] = history_tokens[:, 1:, :] + time_bias
        history_mask = torch.ones(history_tokens.shape[:2], dtype=torch.bool, device=history_tokens.device)
        history_hidden = self.history_encoder(history_tokens, attention_mask=history_mask)
        return history_hidden[:, 0, :], history_hidden[:, 1:, :]

    def _mlm_logits(
        self,
        local_context: torch.Tensor,
        event_context: torch.Tensor,
        user_context: torch.Tensor,
    ) -> torch.Tensor:
        user_context = user_context.unsqueeze(1).expand_as(local_context)
        fused = torch.cat([local_context, event_context, user_context], dim=-1)
        return self.mlm_head(fused)

    def forward(
        self,
        profile_input_ids: torch.Tensor,
        event_input_ids: torch.Tensor,
        event_times: torch.Tensor | None = None,
        calendar_features: torch.Tensor | None = None,
        profile_attention_mask: torch.Tensor | None = None,
        event_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.cfg.use_profile_encoder:
            profile_embedding, profile_hidden = self._encode_profile(
                profile_input_ids,
                profile_attention_mask=profile_attention_mask,
            )
        else:
            token_hidden = self.dropout(self.token_emb(profile_input_ids))
            if profile_attention_mask is None:
                profile_embedding = token_hidden.mean(dim=1)
            else:
                weights = profile_attention_mask.to(token_hidden.dtype).unsqueeze(-1)
                profile_embedding = (token_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            profile_hidden = token_hidden
        event_token_hidden, event_embeddings = self._encode_event_tokens(
            event_input_ids,
            event_attention_mask=event_attention_mask,
        )
        if calendar_features is not None and event_embeddings.size(1) > 0:
            event_embeddings = event_embeddings + self.calendar_proj(calendar_features.to(event_embeddings.dtype))
        history_embedding, history_event_hidden = self._encode_history(
            profile_embedding=profile_embedding,
            event_embeddings=event_embeddings,
            event_times=event_times,
        )
        record_embedding = self.fusion(torch.cat([profile_embedding, history_embedding], dim=-1))
        return {
            "record_embedding": record_embedding,
            "profile_embedding": profile_embedding,
            "event_embeddings": event_embeddings,
            "history_embedding": history_embedding,
            "zh_usr": history_embedding,
            "zh_evt": history_event_hidden,
            "history_event_hidden": history_event_hidden,
            "profile_hidden": profile_hidden,
            "event_token_hidden": event_token_hidden,
        }

    def cls_logits(self, hidden_states: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(hidden_states, dict):
            hidden = hidden_states["record_embedding"]
        else:
            hidden = hidden_states[:, 0, :]
        return self.cls_head(hidden).squeeze(-1)


class PragmaLite(PragmaLiteModel):
    def __init__(self, cfg: PragmaLiteConfig) -> None:
        super().__init__(cfg)
        self.usr_id = 3
        self.evt_id = 4

    def _split_flat_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        device = input_ids.device
        profile_ids = torch.zeros((bsz, self.cfg.max_profile_tokens), dtype=torch.long, device=device)
        profile_mask = torch.zeros((bsz, self.cfg.max_profile_tokens), dtype=torch.bool, device=device)
        event_ids = torch.zeros(
            (bsz, self.cfg.max_events, self.cfg.max_event_tokens),
            dtype=torch.long,
            device=device,
        )
        event_mask = torch.zeros(
            (bsz, self.cfg.max_events, self.cfg.max_event_tokens),
            dtype=torch.bool,
            device=device,
        )
        token_event_index = torch.full((bsz, seq_len), -1, dtype=torch.long, device=device)
        token_is_valid = attention_mask.bool()

        for batch_idx in range(bsz):
            valid_len = int(token_is_valid[batch_idx].sum().item())
            flat = input_ids[batch_idx, :valid_len].tolist()
            if not flat:
                continue
            evt_positions = [idx for idx, token in enumerate(flat) if token == self.evt_id]
            profile_tokens = flat[1 : evt_positions[0]] if evt_positions else flat[1:]
            profile_tokens = profile_tokens[: self.cfg.max_profile_tokens]
            if profile_tokens:
                profile_ids[batch_idx, : len(profile_tokens)] = torch.tensor(profile_tokens, device=device)
                profile_mask[batch_idx, : len(profile_tokens)] = True

            for event_idx, start in enumerate(evt_positions[: self.cfg.max_events]):
                stop = evt_positions[event_idx + 1] if event_idx + 1 < len(evt_positions) else valid_len
                tokens = flat[start:stop][: self.cfg.max_event_tokens]
                if not tokens:
                    continue
                event_ids[batch_idx, event_idx, : len(tokens)] = torch.tensor(tokens, device=device)
                event_mask[batch_idx, event_idx, : len(tokens)] = True
                token_event_index[batch_idx, start : start + len(tokens)] = event_idx

        return {
            "profile_input_ids": profile_ids,
            "profile_attention_mask": profile_mask,
            "event_input_ids": event_ids,
            "event_attention_mask": event_mask,
            "token_event_index": token_event_index,
            "token_is_valid": token_is_valid,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_mlm_logits: bool = False,
    ) -> torch.Tensor:
        pieces = self._split_flat_inputs(input_ids, attention_mask=attention_mask)
        outputs = super().forward(
            profile_input_ids=pieces["profile_input_ids"],
            event_input_ids=pieces["event_input_ids"],
            event_times=None,
            calendar_features=None,
            profile_attention_mask=pieces["profile_attention_mask"],
            event_attention_mask=pieces["event_attention_mask"],
        )

        bsz, seq_len = input_ids.shape
        hidden = self.token_emb.weight.new_zeros((bsz, seq_len, self.d_model))
        hidden[:, 0, :] = outputs["record_embedding"]
        local_context = hidden.clone()
        event_context = hidden.clone()

        profile_hidden = outputs["profile_hidden"]
        if profile_hidden.size(1) > 0:
            profile_len = profile_hidden.size(1)
            local_context[:, 1 : 1 + profile_len, :] = profile_hidden
            event_context[:, 1 : 1 + profile_len, :] = outputs["profile_embedding"].unsqueeze(1)

        event_token_hidden = outputs["event_token_hidden"]
        history_event_hidden = outputs["history_event_hidden"]
        token_event_index = pieces["token_event_index"]
        for batch_idx in range(bsz):
            valid_positions = pieces["token_is_valid"][batch_idx].nonzero(as_tuple=False).flatten()
            for pos in valid_positions.tolist():
                event_idx = int(token_event_index[batch_idx, pos].item())
                if event_idx < 0:
                    continue
                event_positions = (token_event_index[batch_idx] == event_idx).nonzero(as_tuple=False).flatten()
                token_offset = int((event_positions == pos).nonzero(as_tuple=False).item())
                if token_offset == 0:
                    local_context[batch_idx, pos, :] = outputs["event_embeddings"][batch_idx, event_idx, :]
                elif token_offset - 1 < event_token_hidden.size(2):
                    local_context[batch_idx, pos, :] = event_token_hidden[batch_idx, event_idx, token_offset - 1, :]
                event_context[batch_idx, pos, :] = history_event_hidden[batch_idx, event_idx, :]

        if return_mlm_logits:
            return self._mlm_logits(local_context, event_context, outputs["record_embedding"])

        hidden[:, 1:, :] = local_context[:, 1:, :]
        return hidden
