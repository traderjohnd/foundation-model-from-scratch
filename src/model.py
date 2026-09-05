"""Canonical decoder-only Transformer architecture for the project.

Extracted from the verified Notebook 03 implementation without changing the
architecture contract. Notebook 03 remains the implementation/audit record;
this module makes the validated model reusable by the training pipeline.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 16_384
CONTEXT_LENGTH = 512
DROPOUT = 0.10
INIT_STD = 0.02
MODEL_SEED = 42

@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    context_length: int
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = DROPOUT

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

MODEL_CONFIGS = {
    "A": ModelConfig("Model A", VOCAB_SIZE, CONTEXT_LENGTH, 4, 256, 4, 704),
    "B": ModelConfig("Model B", VOCAB_SIZE, CONTEXT_LENGTH, 6, 384, 6, 1024),
    "C": ModelConfig("Model C", VOCAB_SIZE, CONTEXT_LENGTH, 8, 512, 8, 1360),
}

def analytical_parameter_count(cfg: ModelConfig) -> dict:
    embeddings = cfg.vocab_size * cfg.d_model
    attention_per_layer = 4 * cfg.d_model**2
    swiglu_per_layer = 3 * cfg.d_model * cfg.d_ff
    norms_per_layer = 2 * cfg.d_model
    block_per_layer = attention_per_layer + swiglu_per_layer + norms_per_layer
    transformer_blocks = cfg.n_layers * block_per_layer
    final_norm = cfg.d_model
    total = embeddings + transformer_blocks + final_norm
    return {
        "embeddings": embeddings,
        "attention_per_layer": attention_per_layer,
        "swiglu_per_layer": swiglu_per_layer,
        "norms_per_layer": norms_per_layer,
        "block_per_layer": block_per_layer,
        "transformer_blocks": transformer_blocks,
        "final_norm": final_norm,
        "total": total,
    }

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x_float * torch.rsqrt(mean_square + self.eps)
        return (x_normalized * self.weight).to(dtype=input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        pair_dims = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = base ** (-pair_dims / head_dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    @staticmethod
    def _apply_rotation(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)

    def forward(self, q, k):
        seq_len = q.size(-2)
        cos = self.cos_cached[:seq_len].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        sin = self.sin_cached[:seq_len].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        return self._apply_rotation(q, cos, sin), self._apply_rotation(k, cos, sin)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.max_seq_len = cfg.context_length
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, self.max_seq_len)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask[None, None, :, :], persistent=False)

    def _split_heads(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(self, x, return_attention=False):
        _, t, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        q, k = self.rope(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :t, :t]
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        attention = self.attn_dropout(attention)
        context = torch.matmul(attention, v)
        output = self.resid_dropout(self.out_proj(self._merge_heads(context)))
        return (output, attention) if return_attention else output

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class DecoderOnlyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.size(1) > self.cfg.context_length:
            raise ValueError("Sequence length exceeds configured context length.")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

def initialize_model_weights(model, seed=MODEL_SEED, base_std=INIT_STD):
    torch.manual_seed(seed)
    cfg = model.cfg
    nn.init.normal_(model.token_embedding.weight, mean=0.0, std=base_std)
    residual_std = base_std / math.sqrt(2 * cfg.n_layers)
    for block in model.blocks:
        nn.init.ones_(block.attn_norm.weight)
        nn.init.ones_(block.ffn_norm.weight)
        nn.init.normal_(block.attn.q_proj.weight, 0.0, base_std)
        nn.init.normal_(block.attn.k_proj.weight, 0.0, base_std)
        nn.init.normal_(block.attn.v_proj.weight, 0.0, base_std)
        nn.init.normal_(block.attn.out_proj.weight, 0.0, residual_std)
        nn.init.normal_(block.ffn.gate_proj.weight, 0.0, base_std)
        nn.init.normal_(block.ffn.up_proj.weight, 0.0, base_std)
        nn.init.normal_(block.ffn.down_proj.weight, 0.0, residual_std)
    nn.init.ones_(model.final_norm.weight)
