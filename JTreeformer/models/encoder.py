import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch

from utils.config import VAEConfig
from models.common_layers import MultiHeadAttention, WeightInitializer


class EncoderLayer(nn.Module):
    """
    A single layer of the Graph Transformer Encoder.

    This layer consists of a multi-head self-attention mechanism running in
    parallel with a Graph Convolutional Network (GCN) layer. Their outputs are
    fused, followed by a feed-forward network. Residual connections and layer
    normalization are applied around both sub-layers, scaled by the DeepNet factor `alpha`.
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.alpha = config.encoder_alpha

        self.self_attn = MultiHeadAttention(
            hidden_dim=config.hidden_dim_encoder,
            num_heads=config.num_head_encoder,
            dropout_rate=config.dropout_rate if config.dropout else 0.0
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim_encoder)

        if config.use_graph_conv:
            self.gcn = GCNConv(config.hidden_dim_encoder, config.hidden_dim_encoder)
            self.fusion_linear = nn.Linear(config.hidden_dim_encoder * 2, config.hidden_dim_encoder)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim_encoder, config.expand_dim_encoder),
            nn.ReLU(),
            nn.Linear(config.expand_dim_encoder, config.hidden_dim_encoder)
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim_encoder)
        self.dropout = nn.Dropout(config.dropout_rate if config.dropout else 0.0)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch_vec: torch.Tensor,
            attn_bias: torch.Tensor,
            padding_mask: torch.Tensor
    ) -> torch.Tensor:
        # --- Sub-layer 1: Self-Attention and GCN ---
        residual = x

        x_attn = self.self_attn(
            query=x, key=x, value=x,
            attn_bias=attn_bias,
            key_padding_mask=padding_mask
        )

        if self.config.use_graph_conv:
            # GCN part - needs to operate on unbatched nodes
            # We remove the virtual token for GCN processing
            num_graphs = batch_vec.max().item() + 1
            node_features_no_cls = x[:, 1:, :].reshape(-1, x.size(-1))

            # Filter out padding nodes before GCN
            is_pad_node = padding_mask[:, :-1].flatten()
            active_nodes_mask = ~is_pad_node

            # GCN operates only on active nodes
            gcn_features = torch.zeros_like(node_features_no_cls)
            if active_nodes_mask.any():
                gcn_features[active_nodes_mask] = F.relu(self.gcn(
                    node_features_no_cls[active_nodes_mask], edge_index
                ))

            gcn_features = gcn_features.view(num_graphs, -1, x.size(-1))

            # Re-add a placeholder for the virtual token
            x_gcn = F.pad(gcn_features, (0, 0, 1, 0))

            # Fuse and apply residual connection
            x_fused = self.fusion_linear(torch.cat([x_attn, x_gcn], dim=-1))
            x = self.norm1(residual + self.alpha * self.dropout(x_fused))
        else:
            x = self.norm1(residual + self.alpha * self.dropout(x_attn))

        # --- Sub-layer 2: Feed-Forward Network ---
        residual = x
        x_ffn = self.ffn(x)
        x = self.norm2(residual + self.alpha * self.dropout(x_ffn))

        return x


class Encoder(nn.Module):
    """
    The complete JTreeformer Encoder module.
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers_encoder)])
        self.final_norm = nn.LayerNorm(config.hidden_dim_encoder)

        self.apply(WeightInitializer(beta=config.encoder_beta))

    def forward(self, pyg_batch: Batch, attn_bias: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pyg_batch (Batch): A PyG Batch object containing batched node features `x`.
            attn_bias (torch.Tensor): Precomputed attention bias.
            padding_mask (torch.Tensor): Mask for padding tokens.

        Returns:
            torch.Tensor: The final hidden states of all nodes.
                          Shape: (batch_size, seq_len + 1, hidden_dim).
        """
        x, edge_index, batch_vec = pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch

        x = self.final_norm(x)

        for layer in self.layers:
            x = layer(
                x=x,
                edge_index=edge_index,
                batch_vec=batch_vec,
                attn_bias=attn_bias,
                padding_mask=padding_mask
            )

        return x
