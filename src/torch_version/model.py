import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from src.bbpe import Tokenizer


def gelu(x: torch.Tensor):
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))


ACTIVATION_MAP = {
    "gelu": gelu,
}


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.view(-1, logits.size(-1))  # (N, C)
    targets = targets.view(-1)  # (N,)

    max_elems = logits.max(dim=-1).values
    logits = logits - max_elems.unsqueeze(-1)

    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1)).unsqueeze(-1)
    loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return loss.mean()


def softmax(logits: torch.Tensor, dim: int = -1):
    logits = logits - torch.max(logits, dim=dim, keepdim=True)[0]
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
):
    # query: (B, ..., Q, D)
    # key: (B, ..., K, D)
    # value: (B, ..., K, D_v)

    # scale first
    scale = query.size(-1) ** -0.5
    logits = torch.einsum("b ... q d, b ... k d -> b ... q k", query, key)
    logits = logits * scale

    if mask is not None:
        mask_value = -torch.finfo(logits.dtype).max
        logits = logits.masked_fill(mask, mask_value)

    attn_weights = softmax(logits, dim=-1)

    if dropout > 0.0:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout)

    return torch.einsum("b ... q k, b ... k d -> b ... q d", attn_weights, value)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon

    def rms(self, x: torch.Tensor):
        # (B, T, D) -> (B, T)
        return torch.sqrt(torch.mean(x**2, dim=-1) + self.epsilon)

    def forward(self, x: torch.Tensor):
        return x / self.rms(x).unsqueeze(-1) * self.weight.view(1, 1, -1)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.epsilon) + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, activation: str = "gelu"):
        super().__init__()

        self.proj1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.act = ACTIVATION_MAP[activation]
        self.proj2 = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.proj2(self.act(self.proj1(x)))


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_pdrop: float = 0.0, context_length: int = 1024):
        super().__init__()
        assert dim % num_heads == 0, f"dim must be divisible by num_heads, {dim} % {num_heads} != 0"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_pdrop = attn_pdrop

        self.num_heads = num_heads
        self.dim_head = dim // num_heads

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(), persistent=False
        )

    def get_mask(self, x: torch.Tensor):
        seq_len = x.shape[-2]

        if seq_len > self.mask.size(0):
            self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), persistent=False)
            self.mask = self.mask.to(x.device)

        return self.mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor):
        # (B, T, D) -> (B, H, T, D // H)
        B, T, D = x.size()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, T, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.dim_head).transpose(1, 2)

        attn = scaled_dot_product_attention(
            q, k, v, mask=self.get_mask(x), dropout=self.attn_pdrop if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(attn)


class Block(nn.Module):
    # Pre-norm
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_ff: int,
        attn_pdrop: float = 0.0,
        residual_pdrop: float = 0.0,
        epsilon: float = 1e-5,
        context_length: int = 1024,
        activation: str = "gelu",
        norm_class: str = "rms_norm",
        norm_type: str = "pre_norm",
        parallel_block: bool = False,
    ):
        super().__init__()
        if norm_class == "rms_norm":
            norm_class = partial(RMSNorm, dim=dim, epsilon=epsilon)
        elif norm_class == "layer_norm":
            norm_class = partial(LayerNorm, dim=dim, epsilon=epsilon)
        elif norm_class == "none":
            norm_class = nn.Identity

        self.attn_norm = norm_class()
        self.attn = CausalMultiHeadAttention(dim, num_heads, attn_pdrop=attn_pdrop, context_length=context_length)
        self.ff_norm = norm_class()
        self.ff = FeedForward(dim, dim_ff, activation=activation)

        self.parallel_block = parallel_block
        self.norm_type = norm_type

        self.dropout = nn.Dropout(residual_pdrop)

    def forward(self, x: torch.Tensor):
        if self.parallel_block:
            x = x + self.dropout(self.attn(self.attn_norm(x))) + self.dropout(self.ff(self.ff_norm(x)))
        else:
            if self.norm_type == "pre_norm":
                x = x + self.dropout(self.attn(self.attn_norm(x)))
                x = x + self.dropout(self.ff(self.ff_norm(x)))
            elif self.norm_type == "post_norm":
                x = self.attn_norm(x + self.dropout(self.attn(x)))
                x = self.ff_norm(x + self.dropout(self.ff(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        context_length: int,
        dim: int,
        num_heads: int,
        dim_ff: int,
        attn_pdrop: float = 0.0,
        residual_pdrop: float = 0.0,
        epsilon: float = 1e-5,
        activation: str = "gelu",
        norm_class: str = "rms_norm",  # rms_norm, layer_norm, none
        norm_type: str = "pre_norm",  # pre_norm, post_norm
        parallel_block: bool = False,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(context_length, dim)
        self.dropout = nn.Dropout(residual_pdrop)

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                Block(
                    dim,
                    num_heads,
                    dim_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                    epsilon=epsilon,
                    context_length=context_length,
                    activation=activation,
                    norm_class=norm_class,
                    norm_type=norm_type,
                    parallel_block=parallel_block,
                )
            )

        self.layers = nn.ModuleList(self.layers)

        self.final_norm = RMSNorm(dim, epsilon=epsilon)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.final_norm(x))


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    device = next(model.parameters()).device
    input_ids = torch.LongTensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    cnt = 0
    output = []
    while cnt < max_new_tokens:
        logits = model(input_ids)
        logits = logits[0, -1, :]  # (V,)
        logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            # Create mask for tokens where cumulative probability exceeds top_p
            sorted_indices_to_remove = cum_probs > top_p
            # Shift the mask to ensure at least one token (the highest probability one) is kept
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            probs[sorted_indices[sorted_indices_to_remove]] = 0.0
            probs = probs / probs.sum()

        next_token = torch.multinomial(probs, num_samples=1).item()
        input_ids = torch.cat([input_ids, torch.LongTensor([[next_token]]).to(device)], dim=-1)
        output.append(next_token)
        if next_token == eos_id:
            break
        cnt += 1
    return tokenizer.decode(output)


def estimate_flops(
    vocab_size: int, context_length: int, num_layers: int, dim: int, num_heads: int, intermediate_dim: int
):
    # Assume batch size is 1, and only matmul
    # Attention
    def get_matmul_flops(N, M, K):
        return 2 * N * M * K

    attn_proj_flops = 4 * get_matmul_flops(context_length, dim, dim)  # Projection: (T, D) x (D, D)
    attn_proj_flops *= num_layers

    attn_flops = 0
    attn_flops += get_matmul_flops(context_length, dim, context_length)  # Attention: (T, D) x (D, T), QK^T
    attn_flops += get_matmul_flops(context_length, context_length, dim)  # Attention: (T, T) x (T, D), attn_weights x V
    attn_flops *= num_layers

    ff_flops = 2 * get_matmul_flops(context_length, dim, intermediate_dim)  # (T, D) x (D, D_ff)
    ff_flops *= num_layers

    lm_head_flops = get_matmul_flops(context_length, dim, vocab_size)

    total_flops = attn_proj_flops + attn_flops + ff_flops + lm_head_flops
    print(f"Total FLOPs: {total_flops/1e12:.2f}T")
    print(f"Attn proj FLOPs: {attn_proj_flops/1e12:.2f}T ({attn_proj_flops/total_flops * 100:.2f}%)")
    print(f"Attn FLOPs: {attn_flops/1e12:.2f}T ({attn_flops/total_flops * 100:.2f}%)")
    print(f"FF FLOPs: {ff_flops/1e12:.2f}T ({ff_flops/total_flops * 100:.2f}%)")
    print(f"LM head FLOPs: {lm_head_flops/1e12:.2f}T ({lm_head_flops/total_flops * 100:.2f}%)")


def calculate_parameters(
    vocab_size: int, context_length: int, num_layers: int, dim: int, num_heads: int, intermediate_dim: int
):
    # Token embedding
    token_emb_params = vocab_size * dim
    # Position embedding
    pos_emb_params = context_length * dim
    # Transformer Block
    block_params = 2 * dim + 4 * dim * dim + 2 * dim * intermediate_dim  # rmsnorm + attn + ff
    block_params *= num_layers
    # final norm
    final_norm_params = dim
    # LM head
    lm_head_params = dim * vocab_size

    num_params = token_emb_params + pos_emb_params + block_params + lm_head_params + final_norm_params

    # check
    model = TransformerLM(vocab_size, num_layers, context_length, dim, num_heads, intermediate_dim)
    actual_params = sum(p.numel() for p in model.parameters())
    diff = num_params - actual_params
    print(f"Diff: {diff}")
    print(f"Num params: {num_params}")
    print(f"Token emb: {token_emb_params/1e6:.2f}M ({token_emb_params/num_params * 100:.2f}%)")
    print(f"Pos emb: {pos_emb_params/1e6:.2f}M ({pos_emb_params/num_params * 100:.2f}%)")
    print(f"Block: {block_params/1e6:.2f}M ({block_params/num_params * 100:.2f}%)")
    print(f"Final norm: {final_norm_params/1e6:.2f}M ({final_norm_params/num_params * 100:.2f}%)")
    print(f"LM head: {lm_head_params/1e6:.2f}M ({lm_head_params/num_params * 100:.2f}%)")


if __name__ == "__main__":
    vocab_size = 50257
    context_length = 4096
    names = ["small", "medium", "large", "xlarge"]
    num_layers = [12, 24, 36, 48]
    dim = [768, 1024, 1280, 1600]
    num_heads = [12, 16, 20, 25]
    intermediate_dim = [3072, 4096, 5120, 6400]

    for name, nl, d, nh, idim in zip(names, num_layers, dim, num_heads, intermediate_dim):
        print(f"Model: {name}")
        print("======================================")
        estimate_flops(vocab_size, context_length, nl, d, nh, idim)
        calculate_parameters(vocab_size, context_length, nl, d, nh, idim)
        print("======================================")
