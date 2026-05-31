from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.tokenizer.vocab import SPECIAL_TOKENS


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
    calendar_mlp: bool = True
    calendar_hidden_dim: int | None = None
    tie_mlm_to_embedding: bool = True
    use_additive_time_proj: bool = True
    use_history_order_emb: bool = True

    def __post_init__(self) -> None:
        if self.d_ffn is None:
            self.d_ffn = self.d_model * 4
        if self.event_layers is None:
            self.event_layers = self.n_layers
        if self.history_layers is None:
            self.history_layers = self.n_layers
        if self.calendar_hidden_dim is None:
            self.calendar_hidden_dim = self.d_model


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
        mask: torch.Tensor | None = None,
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

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool()[:, None, None, :], torch.finfo(scores.dtype).min)
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
        mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x), mask=mask, positions=positions))
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
        mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, positions=positions)
        return self.norm(x)


class KeyValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_value_pos: int, dropout: float) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.value_pos_emb = nn.Embedding(max_value_pos, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        value_pos: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.token_emb(key_ids)
            + self.token_emb(value_ids)
            + self.value_pos_emb(value_pos.clamp_min(0))
        )
        return self.dropout(hidden)


def _canonicalize_grad_layout(grad: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tuple(grad.shape) != tuple(reference.shape) or tuple(grad.stride()) == tuple(reference.stride()):
        return grad
    aligned = torch.empty_like(reference)
    aligned.copy_(grad)
    return aligned


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
        self.usr_token_id = SPECIAL_TOKENS.index("[USR]")
        self.evt_token_id = SPECIAL_TOKENS.index("[EVT]")

        self.kv_embedding = KeyValueEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_value_pos=max(cfg.max_profile_tokens, cfg.max_event_tokens) + 1,
            dropout=cfg.dropout,
        )
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
        self.history_order_emb = nn.Embedding(cfg.max_events + 1, self.d_model)
        if cfg.calendar_mlp:
            self.calendar_proj = nn.Sequential(
                nn.Linear(6, cfg.calendar_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.calendar_hidden_dim, self.d_model),
            )
        else:
            self.calendar_proj = nn.Linear(6, self.d_model)
        self.time_proj = nn.Sequential(nn.Linear(1, self.d_model), nn.Tanh())
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.mlm_head = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.mlm_bias = nn.Parameter(torch.zeros(self.vocab_size)) if cfg.tie_mlm_to_embedding else None
        self.mlm_out = None if cfg.tie_mlm_to_embedding else nn.Linear(self.d_model, self.vocab_size)
        self.cls_head = nn.Linear(self.d_model, 1)

    def _shared_special_token_embedding(self, token_id: int, batch_size: int) -> torch.Tensor:
        token_ids = torch.full(
            (batch_size, 1),
            int(token_id),
            dtype=torch.long,
            device=self.kv_embedding.token_emb.weight.device,
        )
        return self.kv_embedding.token_emb(token_ids)

    def _prepend_positions(self, values: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if values is None:
            return None
        zeros = torch.zeros((batch_size, 1), dtype=values.dtype, device=device)
        return torch.cat([zeros, values], dim=1)

    def _encode_profile(
        self,
        profile_key_ids: torch.Tensor,
        profile_value_ids: torch.Tensor,
        profile_value_pos: torch.Tensor,
        profile_time: torch.Tensor | None = None,
        profile_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = profile_key_ids.shape
        token_hidden = self.kv_embedding(profile_key_ids, profile_value_ids, profile_value_pos)
        if self.cfg.use_additive_time_proj and profile_time is not None:
            token_hidden = token_hidden + self.time_proj(profile_time.unsqueeze(-1).to(token_hidden.dtype))
        cls_token = self._shared_special_token_embedding(self.usr_token_id, batch_size).to(token_hidden.dtype)
        hidden = torch.cat([cls_token, token_hidden], dim=1)
        if profile_mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=hidden.device), profile_mask.bool()], dim=1)
        positions = self._prepend_positions(profile_time, batch_size=batch_size, device=hidden.device)
        encoded = self.profile_encoder(hidden, mask=mask, positions=positions)
        return encoded[:, 0, :], encoded[:, 1:, :]

    def encode_events(
        self,
        event_key_ids: torch.Tensor,
        event_value_ids: torch.Tensor,
        event_value_pos: torch.Tensor,
        event_token_mask: torch.Tensor,
        calendar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_events, num_tokens = event_key_ids.shape
        if num_events == 0:
            return event_key_ids.new_zeros((batch_size, 0, self.d_model), dtype=torch.float32)
        token_hidden = self.kv_embedding(event_key_ids, event_value_ids, event_value_pos)
        flat_hidden = token_hidden.view(batch_size * num_events, num_tokens, self.d_model)
        flat_cls = self._shared_special_token_embedding(self.evt_token_id, batch_size * num_events).to(token_hidden.dtype)
        flat_hidden = torch.cat([flat_cls, flat_hidden], dim=1)
        flat_mask = torch.cat(
            [
                torch.ones((batch_size, num_events, 1), dtype=torch.bool, device=event_key_ids.device),
                event_token_mask.bool(),
            ],
            dim=2,
        ).view(batch_size * num_events, num_tokens + 1)
        encoded = self.event_encoder(flat_hidden, mask=flat_mask)
        pooled = encoded[:, 0, :].view(batch_size, num_events, self.d_model)
        if calendar_features is not None:
            pooled = pooled + self.calendar_proj(calendar_features.to(pooled.dtype))
        return pooled

    def _encode_event_tokens(
        self,
        event_key_ids: torch.Tensor,
        event_value_ids: torch.Tensor,
        event_value_pos: torch.Tensor,
        event_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_events, num_tokens = event_key_ids.shape
        if num_events == 0:
            empty_hidden = event_key_ids.new_zeros((batch_size, 0, num_tokens, self.d_model), dtype=torch.float32)
            empty_embed = event_key_ids.new_zeros((batch_size, 0, self.d_model), dtype=torch.float32)
            return empty_hidden, empty_embed
        token_hidden = self.kv_embedding(event_key_ids, event_value_ids, event_value_pos)
        flat_hidden = token_hidden.view(batch_size * num_events, num_tokens, self.d_model)
        flat_cls = self._shared_special_token_embedding(self.evt_token_id, batch_size * num_events).to(token_hidden.dtype)
        flat_hidden = torch.cat([flat_cls, flat_hidden], dim=1)
        flat_mask = torch.cat(
            [
                torch.ones((batch_size, num_events, 1), dtype=torch.bool, device=event_key_ids.device),
                event_token_mask.bool(),
            ],
            dim=2,
        ).view(batch_size * num_events, num_tokens + 1)
        encoded = self.event_encoder(flat_hidden, mask=flat_mask)
        encoded = encoded.view(batch_size, num_events, num_tokens + 1, self.d_model)
        return encoded[:, :, 1:, :], encoded[:, :, 0, :]

    def _encode_history(
        self,
        profile_embedding: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_time: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_tokens = torch.cat([profile_embedding.unsqueeze(1), event_embeddings], dim=1)
        batch_size = history_tokens.size(0)
        if self.cfg.use_history_order_emb and event_embeddings.size(1) > 0:
            order = torch.arange(1, event_embeddings.size(1) + 1, device=history_tokens.device)
            history_tokens[:, 1:, :] = history_tokens[:, 1:, :] + self.history_order_emb(order)[None, :, :]
        if self.cfg.use_additive_time_proj and event_time is not None and event_embeddings.size(1) > 0:
            history_tokens[:, 1:, :] = history_tokens[:, 1:, :] + self.time_proj(
                event_time.unsqueeze(-1).to(history_tokens.dtype)
            )
        if event_mask is None:
            history_mask = torch.ones(history_tokens.shape[:2], dtype=torch.bool, device=history_tokens.device)
        else:
            history_mask = torch.cat(
                [torch.ones((batch_size, 1), dtype=torch.bool, device=history_tokens.device), event_mask.bool()],
                dim=1,
            )
        positions = self._prepend_positions(event_time, batch_size=batch_size, device=history_tokens.device)
        history_hidden = self.history_encoder(history_tokens, mask=history_mask, positions=positions)
        return history_hidden[:, 0, :], history_hidden[:, 1:, :]

    def _mlm_logits(
        self,
        local_context: torch.Tensor,
        event_context: torch.Tensor,
        user_context: torch.Tensor,
    ) -> torch.Tensor:
        if local_context.ndim == 4:
            user_context = user_context.unsqueeze(1).unsqueeze(2).expand_as(local_context)
        elif local_context.ndim == 3:
            user_context = user_context.unsqueeze(1).expand_as(local_context)
        else:
            raise ValueError(f"Unsupported local_context rank: {local_context.ndim}")
        fused = torch.cat([local_context, event_context, user_context], dim=-1)
        hidden = self.mlm_head(fused)
        if self.cfg.tie_mlm_to_embedding:
            return F.linear(hidden, self.kv_embedding.token_emb.weight, self.mlm_bias)
        if self.mlm_out is None:
            raise RuntimeError("mlm_out must be initialized when tie_mlm_to_embedding is disabled")
        return self.mlm_out(hidden)

    def forward(
        self,
        profile_key_ids: torch.Tensor,
        profile_value_ids: torch.Tensor,
        profile_value_pos: torch.Tensor,
        profile_time: torch.Tensor | None = None,
        profile_mask: torch.Tensor | None = None,
        event_key_ids: torch.Tensor | None = None,
        event_value_ids: torch.Tensor | None = None,
        event_value_pos: torch.Tensor | None = None,
        event_token_mask: torch.Tensor | None = None,
        event_time: torch.Tensor | None = None,
        calendar_features: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
        return_mlm_logits: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if event_key_ids is None or event_value_ids is None or event_value_pos is None or event_token_mask is None:
            raise ValueError("Structured event tensors are required: event_key_ids, event_value_ids, event_value_pos, event_token_mask")
        profile_embedding, profile_hidden = self._encode_profile(
            profile_key_ids=profile_key_ids,
            profile_value_ids=profile_value_ids,
            profile_value_pos=profile_value_pos,
            profile_time=profile_time,
            profile_mask=profile_mask,
        )
        event_token_hidden, event_embeddings = self._encode_event_tokens(
            event_key_ids=event_key_ids,
            event_value_ids=event_value_ids,
            event_value_pos=event_value_pos,
            event_token_mask=event_token_mask,
        )
        if calendar_features is not None:
            event_embeddings = event_embeddings + self.calendar_proj(calendar_features.to(event_embeddings.dtype))
        history_embedding, history_event_hidden = self._encode_history(
            profile_embedding=profile_embedding,
            event_embeddings=event_embeddings,
            event_time=event_time,
            event_mask=event_mask,
        )
        record_embedding = self.fusion(torch.cat([profile_embedding, history_embedding], dim=-1))
        outputs = {
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
        if return_mlm_logits:
            event_context = history_event_hidden.unsqueeze(2).expand_as(event_token_hidden)
            return self._mlm_logits(event_token_hidden, event_context, history_embedding)
        return outputs

    def cls_logits(self, hidden_states: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = hidden_states["record_embedding"] if isinstance(hidden_states, dict) else hidden_states
        return self.cls_head(hidden).squeeze(-1)
