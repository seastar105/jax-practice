from typing import Optional

import flax.nnx as nnx
import jax.numpy as jnp
from einops import einsum, rearrange

ACTIVATION_MAP = {
    "relu": nnx.relu,
    "gelu": nnx.gelu,
    "silu": nnx.silu,
}

# ff_dropout -> Dropout in FeedForward, fc1 -> act -> dropout -> fc2
# attention_dropout -> Dropout in Attention, qkv -> attention -> dropout -> out -> proj
# residual_dropout -> Dropout per sub-block, x -> norm -> attention -> residual + dropout(x) -> norm -> ff -> residual + dropout(x)


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
        self.fc1 = nnx.Linear(dim, 2 * intermediate_dim, use_bias=use_bias, rngs=rngs)
        self.act = ACTIVATION_MAP[activation]
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.fc2 = nnx.Linear(intermediate_dim, dim, use_bias=use_bias, rngs=rngs)

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
        self.fc1 = nnx.Linear(dim, intermediate_dim, use_bias=use_bias, rngs=rngs)
        self.act = ACTIVATION_MAP[activation]
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.fc2 = nnx.Linear(intermediate_dim, dim, use_bias=use_bias, rngs=rngs)

    def __call__(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


def scaled_dot_product_attention(q, k, v, mask: Optional[jnp.array] = None, dropout: Optional[nnx.Dropout] = None):
    # q, k, v: (..., seq_len, dim)
    # mask logits where mask is False
    scale = 1 / jnp.sqrt(q.shape[-1])

    logits = einsum(q, k, "b ... q d, b ... k d -> b ... q k") * scale
    if mask is not None:
        mask_value = -jnp.finfo(logits.dtype).max
        logits = jnp.where(mask, logits, mask_value)

    attn = nnx.softmax(logits, axis=-1)
    if dropout is not None:
        attn = dropout(attn)
    return einsum(attn, v, "b ... q k, b ... k d -> b ... q d")


class CausalSelfAttention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs, dropout: float = 0.0, use_bias: bool = False):
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim_head = dim // num_heads

        self.q_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs)

        self.attn_dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

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
        ff_activation: str = "gelu",
        ff_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        use_bias: bool = False,
        norm_class: str = "rmsnorm",
        use_glu: bool = False,
    ):
        norm_class = nnx.LayerNorm if norm_class == "layernorm" else nnx.RMSNorm
        ff_class = GLUFeedForward if use_glu else FeedForward

        self.attn_norm = norm_class(dim, rngs=rngs)
        self.attn = CausalSelfAttention(dim, num_heads, rngs, dropout=attention_dropout, use_bias=use_bias)

        self.ff_norm = norm_class(dim, rngs=rngs)
        self.ff = ff_class(dim, dim_ff, rngs, activation=ff_activation, dropout=ff_dropout, use_bias=use_bias)

        self.dropout = nnx.Dropout(rate=residual_dropout, rngs=rngs)

    def __call__(self, x):
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class Transformer(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        dim_ff: int,
        num_heads: int,
        rngs: nnx.Rngs,
        context_length: int,
        ff_activation: str = "gelu",
        ff_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        use_bias: bool = False,
        norm_class: str = "rmsnorm",
        use_glu: bool = False,
    ):
        self.token_emb = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.pos_emb = nnx.Embed(context_length, dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=residual_dropout, rngs=rngs)

        blocks = [
            Block(
                dim,
                dim_ff,
                num_heads,
                rngs,
                ff_activation,
                ff_dropout,
                attention_dropout,
                residual_dropout,
                use_bias,
                norm_class,
                use_glu,
            )
            for _ in range(num_layers)
        ]
        self.blocks = nnx.Sequential(*blocks)

        norm_class = nnx.LayerNorm if norm_class == "layernorm" else nnx.RMSNorm
        self.final_norm = norm_class(dim, rngs=rngs)

        self.lm_head = nnx.Linear(dim, vocab_size, use_bias=use_bias, rngs=rngs)

    def __call__(self, x: jnp.array, position_ids: Optional[jnp.array] = None):
        embs = self.token_emb(x)
        if position_ids is not None:
            assert position_ids.shape[-1] == x.shape[-1], "position_ids must have the same shape as x"
        else:
            position_ids = jnp.arange(x.shape[-1])
        embs = self.dropout(embs + self.pos_emb(position_ids))
        embs = self.blocks(embs)
        logits = self.lm_head(self.final_norm(embs))
        return logits
