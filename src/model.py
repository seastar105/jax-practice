from typing import Optional

import flax.nnx as nnx
import jax.numpy as jnp
from einops import einsum, rearrange

from src.config import TransformerConfig

ACTIVATION_MAP = {
    "relu": nnx.relu,
    "gelu": nnx.gelu,
    "silu": nnx.silu,
    "relu_squared": lambda x: nnx.relu(x) ** 2,
}

# ff_dropout -> Dropout in FeedForward, fc1 -> act -> dropout -> fc2
# attention_dropout -> Dropout in Attention, qkv -> attention -> dropout -> out -> proj
# residual_dropout -> Dropout per sub-block, x -> norm -> attention -> residual + dropout(x) -> norm -> ff -> residual + dropout(x)
KERNEL_INIT = nnx.initializers.normal(stddev=0.02)


class GLUFeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        rngs: nnx.Rngs,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        self.fc1 = nnx.Linear(dim, 2 * intermediate_dim, use_bias=use_bias, rngs=rngs, kernel_init=KERNEL_INIT)
        self.act = ACTIVATION_MAP[activation]
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.fc2 = nnx.Linear(intermediate_dim, dim, use_bias=use_bias, rngs=rngs, kernel_init=KERNEL_INIT)

    def __call__(self, x):
        x, gate = jnp.split(self.fc1(x), 2, axis=-1)
        x = self.act(gate) * x
        return self.fc2(self.dropout(x))


class FeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        rngs: nnx.Rngs,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        self.fc1 = nnx.Linear(dim, intermediate_dim, use_bias=use_bias, rngs=rngs, kernel_init=KERNEL_INIT)
        self.act = ACTIVATION_MAP[activation]
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.fc2 = nnx.Linear(intermediate_dim, dim, use_bias=use_bias, rngs=rngs, kernel_init=KERNEL_INIT)

    def __call__(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class RotaryEmbedding(nnx.Module):
    def __init__(self, dim, max_timescale: float = 10000.0):
        self.dim = dim
        half_dim = dim // 2
        fraction = 2 * jnp.arange(0, half_dim) / self.dim
        self.timescale = nnx.Variable(max_timescale**fraction)

    def __call__(self, x, position_ids):
        position = position_ids[:, jnp.newaxis, :, jnp.newaxis]
        sinusoid_inp = position / self.timescale
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        first_half, second_half = jnp.split(x, 2, axis=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return jnp.concatenate((first_part, second_part), axis=-1)


def scaled_dot_product_attention(q, k, v, mask: Optional[jnp.array] = None, dropout: Optional[nnx.Dropout] = None):
    # q, k, v: (..., seq_len, dim)
    scale = 1 / jnp.sqrt(q.shape[-1])

    logits = einsum(q, k, "b ... q d, b ... k d -> b ... q k") * scale
    if mask is not None:
        mask_value = -jnp.finfo(logits.dtype).max
        logits = jnp.where(mask, mask_value, logits)

    attn = nnx.softmax(logits, axis=-1)
    if dropout is not None:
        attn = dropout(attn)
    return einsum(attn, v, "b ... q k, b ... k d -> b ... q d")


class CausalSelfAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        dropout: float = 0.0,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_class: nnx.Module = nnx.LayerNorm,
        pos_emb: str = "abs",
    ):
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim_head = dim // num_heads

        self.q_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs, kernel_init=KERNEL_INIT)
        self.k_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs, kernel_init=KERNEL_INIT)
        self.v_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs, kernel_init=KERNEL_INIT)
        self.out_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs, kernel_init=KERNEL_INIT)

        self.attn_dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.qk_norm = qk_norm

        if self.qk_norm:
            self.q_norm = norm_class(self.dim_head, rngs=rngs)
            self.k_norm = norm_class(self.dim_head, rngs=rngs)

        self.pos_emb = pos_emb
        if self.pos_emb == "rotary":
            self.rotary_emb = RotaryEmbedding(self.dim_head)

    def __call__(self, x, position_ids):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.pos_emb == "rotary":
            q = self.rotary_emb(q, position_ids)
            k = self.rotary_emb(k, position_ids)

        seq_len = q.shape[-2]
        mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), 1)

        attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.attn_dropout)
        return self.out_proj(rearrange(attn, "... h n d -> ... n (h d)"))


class Block(nnx.Module):
    def __init__(
        self,
        dim: int,
        dim_ff: int,
        num_heads: int,
        rngs: nnx.Rngs,
        pos_emb: str = "abs",
        ff_class: str = "vanilla",
        ff_activation: str = "gelu",
        ff_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        use_bias: bool = False,
        norm_class: str = "layernorm",
        qk_norm: bool = False,
    ):
        norm_class = nnx.LayerNorm if norm_class == "layernorm" else nnx.RMSNorm
        ff_class = GLUFeedForward if ff_class == "glu" else FeedForward

        self.attn_norm = norm_class(dim, rngs=rngs)
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            rngs,
            dropout=attention_dropout,
            use_bias=use_bias,
            qk_norm=qk_norm,
            norm_class=norm_class,
            pos_emb=pos_emb,
        )

        self.ff_norm = norm_class(dim, rngs=rngs)
        self.ff = ff_class(dim, dim_ff, rngs, activation=ff_activation, dropout=ff_dropout, use_bias=use_bias)

        self.dropout = nnx.Dropout(rate=residual_dropout, rngs=rngs)

    def __call__(self, x, position_ids):
        x = x + self.dropout(self.attn(self.attn_norm(x), position_ids))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class Transformer(nnx.Module):
    def __init__(
        self,
        config: TransformerConfig,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.token_emb = nnx.Embed(config.vocab_size, config.dim, rngs=rngs, embedding_init=KERNEL_INIT)
        self.pos_emb = (
            nnx.Embed(config.context_length, config.dim, rngs=rngs, embedding_init=KERNEL_INIT)
            if config.pos_emb == "abs"
            else None
        )
        self.dropout = nnx.Dropout(rate=config.residual_dropout, rngs=rngs)

        self.blocks = [
            Block(
                config.dim,
                config.dim_ff,
                config.num_heads,
                rngs,
                config.pos_emb,
                config.ff_class,
                config.ff_activation,
                config.ff_dropout,
                config.attention_dropout,
                config.residual_dropout,
                config.use_bias,
                config.norm_class,
                config.qk_norm,
            )
            for _ in range(config.num_layers)
        ]

        norm_class = nnx.LayerNorm if config.norm_class == "layernorm" else nnx.RMSNorm
        self.final_norm = norm_class(config.dim, rngs=rngs)

        self.tie_embedding = config.tie_embedding
        if self.tie_embedding:
            self.lm_head = None
        else:
            self.lm_head = nnx.Linear(
                config.dim, config.vocab_size, use_bias=config.use_bias, rngs=rngs, kernel_init=KERNEL_INIT
            )

        self.use_remat = config.use_remat

    def __call__(self, x: jnp.array, position_ids: Optional[jnp.array] = None):
        embs = self.token_emb(x)
        if position_ids is not None:
            assert position_ids.shape[-1] == x.shape[-1], "position_ids must have the same shape as x"
        else:
            position_ids = jnp.arange(x.shape[-1]).reshape(1, -1)
        if self.pos_emb is not None:
            embs = embs + self.pos_emb(position_ids)
        embs = self.dropout(embs)

        for block in self.blocks:
            if self.use_remat:
                embs = nnx.remat(block.__call__)(embs, position_ids)
            else:
                embs = block(embs, position_ids)
        embs = self.final_norm(embs)

        if self.tie_embedding:
            logits = einsum(embs, self.token_emb.embedding, "b t d, v d -> b t v")
        else:
            logits = self.lm_head(embs)
        return logits
