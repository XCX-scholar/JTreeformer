import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from utils.config import ModelConfig


# --- Helper Functions & Classes ---
def get_position_encoding(position: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    """
    Generates sinusoidal positional encodings.

    Args:
        position (torch.Tensor): Tensor of shape (batch_size, seq_len)
                                 containing positions.
        hidden_dim (int): The dimensionality of the encoding.

    Returns:
        torch.Tensor: The positional encoding tensor of shape
                      (batch_size, seq_len, hidden_dim).
    """
    if hidden_dim % 2 != 0:
        raise ValueError(f"hidden_dim must be even, but got {hidden_dim}")

    device = position.device
    pe = torch.zeros(*position.shape, hidden_dim, device=device)

    inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_dim, 2, device=device).float() / hidden_dim))

    pos_enc_a = torch.sin(position.unsqueeze(-1) * inv_freq)
    pos_enc_b = torch.cos(position.unsqueeze(-1) * inv_freq)

    pe[..., 0::2] = pos_enc_a
    pe[..., 1::2] = pos_enc_b
    return pe


class WeightInitializer:
    """
    Applies DeepNet-style weight initialization to model parameters.
    """
    def __init__(self, beta: float):
        self.beta = beta

    def __call__(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            is_special_layer = any(name in module.__class__.__name__.lower() for name in ["v_proj", "out_proj", "ff2"])
            gain = self.beta if is_special_layer else 1.0
            nn.init.xavier_normal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, MultiHeadAttention):
            nn.init.xavier_uniform_(module.q_proj.weight, gain=1.0)
            nn.init.xavier_uniform_(module.k_proj.weight, gain=1.0)
            nn.init.xavier_uniform_(module.v_proj.weight, gain=self.beta)
            nn.init.xavier_uniform_(module.out_proj.weight, gain=self.beta)
# --- Core Modules ---

class NodeFeaturizer(nn.Module):
    """
    Constructs node features by summing various configurable embeddings.
    """

    def __init__(self, config: ModelConfig, is_decoder: bool = False):
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder
        self.hidden_dim = config.hidden_dim_decoder if is_decoder else config.hidden_dim_encoder

        self.node_type_embedding = nn.Embedding(
            config.num_node_type + 3, self.hidden_dim, padding_idx=0
        )
        if config.use_hs_embedding:
            self.hs_embedding = nn.Embedding(
                config.max_hs + 1, self.hidden_dim, padding_idx=0
            )
        if config.use_layer_embedding:
            self.layer_number_embedding = nn.Embedding(
                config.max_layer_num + 1, self.hidden_dim, padding_idx=0
            )
        if not is_decoder and config.use_degree_embedding:
            self.degree_embedding = nn.Embedding(
                config.max_degree + 1, self.hidden_dim, padding_idx=0
            )
        self.virtual_token_embedding = nn.Embedding(1, self.hidden_dim)

    def forward(
            self,
            node_type: torch.Tensor,
            hs: torch.Tensor,
            layer_number: torch.Tensor,
            parent_pos: torch.Tensor,
            degree: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = node_type.size(0)
        x = self.node_type_embedding(node_type)
        x += get_position_encoding(parent_pos.clamp(min=0), self.hidden_dim)

        if self.config.use_hs_embedding:
            x += self.hs_embedding(hs)
        if self.config.use_layer_embedding:
            x += self.layer_number_embedding(layer_number)
        if not self.is_decoder and self.config.use_degree_embedding:
            if degree is None:
                raise ValueError("Degree must be provided for encoder.")
            x += self.degree_embedding(degree)

        virtual_token = self.virtual_token_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.cat([virtual_token, x], dim=1)


class AttentionBias(nn.Module):
    """
    Computes a learnable additive bias for attention based on adjacency.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.adjacency_bias = nn.Embedding(2, num_heads, padding_idx=0)
        self.virtual_token_bias = nn.Parameter(torch.zeros(1, num_heads, 1))
        nn.init.xavier_uniform_(self.virtual_token_bias)

    def forward(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = adj_matrix.shape
        graph_bias = self.adjacency_bias(adj_matrix.long()).permute(0, 3, 1, 2)
        full_bias = F.pad(graph_bias, (1, 0, 1, 0))
        full_bias[:, :, 0, :] += self.virtual_token_bias
        full_bias[:, :, :, 0] += self.virtual_token_bias
        return full_bias


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with support for attention bias, causal masking,
    and KV-caching for efficient autoregressive decoding.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def _reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_bias: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            causal_mask: bool = False,
            use_kv_cache: bool = False,
    ) -> torch.Tensor:

        batch_size, tgt_len, _ = query.shape

        # --- KV Caching Logic ---
        if use_kv_cache:
            if self.k_cache is not None:
                # We are in generation mode, `query` is the current token
                # `key` and `value` are also for the current token
                # Concatenate new K,V with cached K,V
                k = torch.cat([self.k_cache, self.k_proj(key)], dim=1)
                v = torch.cat([self.v_cache, self.v_proj(value)], dim=1)
            else:
                # First step of generation
                k = self.k_proj(key)
                v = self.v_proj(value)
            # Update cache
            self.k_cache = k.detach()
            self.v_cache = v.detach()
        else:
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = self.q_proj(query)

        # --- Reshape for Multi-head Attention ---
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Attention Score Calculation ---
        q = q * self.scaling
        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        # --- Apply Masks and Biases ---
        if attn_bias is not None:
            attn_weights += attn_bias

        if causal_mask:
            mask = torch.triu(torch.ones(*attn_weights.shape[-2:], device=query.device, dtype=torch.bool), diagonal=1)
            attn_weights.masked_fill_(mask, float("-inf"))

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # --- Final Attention Output ---
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.hidden_dim)

        return self.out_proj(attn_output)


class DAGCNConv(MessagePassing):
    """
    Directed Acyclic Graph Convolution (DAGCN) layer.
    Formula: H' = (I + θ * (I - D_in⁻¹ * A)) * H * W
    The edge_index is assumed to be directed from parent to child.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.theta = nn.Parameter(torch.Tensor([1.0]))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        nn.init.ones_(self.theta)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        in_degree = degree(col, x.size(0), dtype=x.dtype)
        in_degree_inv = in_degree.pow(-1)
        in_degree_inv[in_degree_inv == float('inf')] = 0
        norm = in_degree_inv[col]

        propagated_message = self.propagate(edge_index, x=x, norm=norm)

        identity_part = x
        dag_part = identity_part - propagated_message

        out = identity_part + self.theta * dag_part
        return self.lin(out)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j
