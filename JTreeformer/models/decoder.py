import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utils.config import VAEConfig
from models.common_layers import MultiHeadAttention, DAGCNConv, WeightInitializer

class DecoderLayer(nn.Module):
    """
    A single layer of the Graph Transformer Decoder.

    This layer is similar to the EncoderLayer but uses causal self-attention
    and a Directed Acyclic Graph Convolution (DAGCN) for its graph-based branch.
    """
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.alpha = config.decoder_alpha

        # Causal Self-Attention sub-layer
        self.self_attn = MultiHeadAttention(
            hidden_dim=config.hidden_dim_decoder,
            num_heads=config.num_head_decoder,
            dropout_rate=config.dropout_rate if config.dropout else 0.0
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim_decoder)

        if config.use_graph_conv:
            self.dagcn = DAGCNConv(config.hidden_dim_decoder, config.hidden_dim_decoder)
            self.fusion_linear = nn.Linear(config.hidden_dim_decoder * 2, config.hidden_dim_decoder)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim_decoder, config.expand_dim_decoder),
            nn.ReLU(),
            nn.Linear(config.expand_dim_decoder, config.hidden_dim_decoder)
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim_decoder)
        self.dropout = nn.Dropout(config.dropout_rate if config.dropout else 0.0)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            attn_bias: torch.Tensor,
            padding_mask: torch.Tensor,
            use_kv_cache: bool
    ) -> torch.Tensor:
        residual = x

        x_attn = self.self_attn(
            query=x, key=x, value=x,
            attn_bias=attn_bias,
            key_padding_mask=padding_mask,
            causal_mask=True,
            use_kv_cache=use_kv_cache
        )

        if self.config.use_graph_conv:
            # DAGCN operates on nodes without virtual token
            node_features_no_cls = x[:, 1:, :].reshape(-1, x.size(-1))

            # Filter padding before DAGCN
            # print(node_features_no_cls.shape, padding_mask.shape)
            is_pad_node = padding_mask[:, 1:].flatten()
            active_nodes_mask = ~is_pad_node

            # print(node_features_no_cls.shape, edge_index)
            dag_features = torch.zeros_like(node_features_no_cls)
            if active_nodes_mask.any():
                dag_features[active_nodes_mask] = F.relu(self.dagcn(
                    node_features_no_cls[active_nodes_mask], edge_index
                ))

            dag_features = dag_features.view(x.size(0), -1, x.size(-1))
            x_dag = F.pad(dag_features, (0, 0, 1, 0))

            x_fused = self.fusion_linear(torch.cat([x_attn, x_dag], dim=-1))
            x = self.norm1(residual + self.alpha * self.dropout(x_fused))
        else:
            x = self.norm1(residual + self.alpha * self.dropout(x_attn))

        # --- Sub-layer 2: Feed-Forward Network ---
        residual = x
        x_ffn = self.ffn(x)
        x = self.norm2(residual + self.alpha * self.dropout(x_ffn))

        return x


class Decoder(nn.Module):
    """
    The complete JTreeformer Decoder module.
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers_decoder)])
        self.final_norm = nn.LayerNorm(config.hidden_dim_decoder)

        # Output head for node type prediction
        self.node_output_proj = nn.Linear(config.hidden_dim_decoder, config.num_node_type + 3)

        # Cross-attention mechanism for relation `r_i` prediction
        self.relation_cross_attn = MultiHeadAttention(
            hidden_dim=config.hidden_dim_decoder,
            num_heads=config.num_head_decoder,
            dropout_rate=0.0
        )
        self.relation_output_proj = nn.Linear(config.hidden_dim_decoder, config.max_layer_num)

        self.apply(WeightInitializer(beta=config.decoder_beta))

    def reset_kv_cache(self):
        """Resets the KV-cache in all attention layers."""
        for layer in self.layers:
            layer.self_attn._reset_cache()

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            attn_bias: torch.Tensor,
            padding_mask: torch.Tensor,
            use_kv_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input features for the decoder sequence.
            edge_index (torch.Tensor): Edge index for the graph structure.
            attn_bias (torch.Tensor): Precomputed attention bias.
            padding_mask (torch.Tensor): Mask for padding tokens.
            use_kv_cache (bool): Whether to use KV-caching for inference.

        Returns:
            A tuple of (node_logits, relation_logits).
        """
        x = self.final_norm(x)  # Pre-LN

        for i, layer in enumerate(self.layers):
            x = layer(
                x=x,
                edge_index=edge_index,
                attn_bias=attn_bias if not use_kv_cache else attn_bias[:, :, -1:, :],
                padding_mask=padding_mask,
                use_kv_cache=use_kv_cache
            )

        # --- Node Type Prediction ---
        # We predict the next node for all positions
        node_logits = self.node_output_proj(x)

        # --- Relation Prediction ---
        # Query: hidden state of the previous token (v_{i-1})
        # Key/Value: context of all previous tokens (v_0, ..., v_{i-1})
        # We only need to predict relations for non-virtual nodes
        query = x  # Use h_{i-1} to predict r_i
        key_value = x
        relation_padding_mask = padding_mask if padding_mask is not None else None

        relation_context = self.relation_cross_attn(
            query=query, key=key_value, value=key_value,
            key_padding_mask=relation_padding_mask,
            causal_mask=True  # Can only attend to past
        )
        relation_logits = self.relation_output_proj(relation_context)

        return node_logits, relation_logits
