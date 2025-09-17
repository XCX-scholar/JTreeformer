import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict
from utils.config import VAEConfig
from models.common_layers import NodeFeaturizer, AttentionBias
from models.encoder import Encoder
from models.decoder import Decoder
from jtnn_utils.chemutils import get_mol


class JTreeformer(nn.Module):
    """
    The main Tree-Structured Variational Autoencoder (VAE) model.
    Includes auxiliary property prediction and dynamic `hs` calculation for decoding.
    """

    def __init__(self, config: VAEConfig, vocab: Dict[str, int]):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}  # For decoding

        self.encoder_featurizer = NodeFeaturizer(config, is_decoder=False)
        self.decoder_featurizer = NodeFeaturizer(config, is_decoder=True)

        self.encoder_attn_bias = AttentionBias(config.num_head_encoder)
        self.decoder_attn_bias = AttentionBias(config.num_head_decoder)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        if config.is_vae:
            self.mean_proj = nn.Linear(config.hidden_dim_encoder, config.latent_dim)
            self.logvar_proj = nn.Linear(config.hidden_dim_encoder, config.latent_dim)
        else:
            self.latent_proj = nn.Linear(config.hidden_dim_encoder, config.latent_dim)

        self.latent_to_decoder_proj = nn.Linear(config.latent_dim, config.hidden_dim_decoder)

        self.latent_fusion_proj = nn.Linear(
            config.hidden_dim_decoder + config.latent_dim,
            config.hidden_dim_decoder
        )

        # Auxiliary head for property prediction
        if self.config.predict_properties:
            self.property_predictor = nn.Sequential(
                nn.Linear(config.hidden_dim_encoder, config.hidden_dim_encoder // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim_encoder // 2, 3)  # 3 properties: w, logp, tpsa
            )

    def _get_latent_representation(self, encoder_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_graph_embedding = encoder_output[:, 0, :]
        output = {}
        if self.config.is_vae:
            output["mean"] = self.mean_proj(global_graph_embedding)
            output["logvar"] = self.logvar_proj(global_graph_embedding)
        else:
            output["z"] = self.latent_proj(global_graph_embedding)

        if self.config.predict_properties:
            output["property_preds"] = self.property_predictor(global_graph_embedding)

        return output

    def forward(self, pyg_batch: Batch) -> Dict[str, torch.Tensor]:
        # --- Encoder Pass ---
        encoder_features = self.encoder_featurizer(
            node_type=pyg_batch.x_dense,
            hs=pyg_batch.hs_dense,
            layer_number=pyg_batch.layer_number_dense,
            parent_pos=pyg_batch.parent_pos_dense,
            degree=pyg_batch.degree_dense
        )
        encoder_bias = self.encoder_attn_bias(pyg_batch.adj_dense)
        enc_padding_mask = F.pad(pyg_batch.padding_mask, (1, 0), value=False)

        encoder_output = self.encoder(
            pyg_batch=Batch(x=encoder_features, edge_index=pyg_batch.edge_index, batch=pyg_batch.batch),
            attn_bias=encoder_bias,
            padding_mask=enc_padding_mask
        )

        # --- Latent Space and Property Prediction ---
        latent_dict = self._get_latent_representation(encoder_output)

        if self.config.is_vae:
            std = torch.exp(0.5 * latent_dict["logvar"])
            eps = torch.randn_like(std)
            z = latent_dict["mean"] + eps * std
        else:
            z = latent_dict["z"]

        # --- Decoder Pass ---
        decoder_features = self.decoder_featurizer(
            node_type=pyg_batch.x_dense, hs=pyg_batch.hs_dense,
            layer_number=pyg_batch.layer_number_dense, parent_pos=pyg_batch.parent_pos_dense
        )
        latent_as_start_token = self.latent_to_decoder_proj(z)
        decoder_features[:, 0, :] = latent_as_start_token
        z_broadcasted = z.unsqueeze(1).expand(-1, decoder_features.size(1), -1)
        fused_features = self.latent_fusion_proj(
            torch.cat([decoder_features, z_broadcasted], dim=-1)
        )

        decoder_bias = self.decoder_attn_bias(pyg_batch.adj_dense)

        node_logits, relation_logits = self.decoder(
            x=fused_features,
            edge_index=pyg_batch.edge_index,
            attn_bias=decoder_bias,
            padding_mask=enc_padding_mask
        )

        output = {
            "node_logits": node_logits,
            "relation_logits": relation_logits,
            **latent_dict
        }
        return output

    def decode(self, z: torch.Tensor, max_len: int, stop_token_id: int):
        self.eval()
        self.decoder.reset_kv_cache()
        batch_size = z.size(0)
        device = z.device

        current_features = self.latent_to_decoder_proj(z).unsqueeze(1)

        gen_node_indices = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        paths = [[-1] for _ in range(batch_size)]

        adj_matrices = torch.zeros(batch_size, max_len, max_len, device=device)
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            for step in range(max_len):
                attn_bias = self.decoder_attn_bias(adj_matrices[:, :step, :step])

                z_broadcasted = z.unsqueeze(1).expand(-1, current_features.size(1), -1)
                fused_features = self.latent_fusion_proj(
                    torch.cat([current_features, z_broadcasted], dim=-1)
                )

                padding_mask = torch.full((batch_size,current_features.size(1)), 0, dtype=torch.bool, device=device)
                edge_index_list = []
                current_seq_len = fused_features.size(1)
                num_nodes_per_graph = current_seq_len - 1
                if num_nodes_per_graph > 0:
                    node_offset = 0
                    for i in range(batch_size):
                        adj_i = adj_matrices[i, :current_seq_len, :current_seq_len]
                        edge_index_i = adj_i.nonzero(as_tuple=False).t()
                        if edge_index_i.numel() > 0:
                            edge_index_list.append(edge_index_i + node_offset)
                        node_offset += num_nodes_per_graph

                if len(edge_index_list) > 0:
                    edge_index = torch.cat(edge_index_list, dim=1)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

                node_logits, relation_logits = self.decoder(
                    x=fused_features, edge_index=edge_index,
                    attn_bias=attn_bias, padding_mask=padding_mask, use_kv_cache=True
                )

                node_logits[:, -1, 0] = -float('inf')
                next_node_id = node_logits[:, -1, 1:].argmax(dim=-1)

                next_node_id = torch.where(finished_sequences, stop_token_id, next_node_id)
                gen_node_indices[:, step] = next_node_id
                finished_sequences.logical_or_(next_node_id == stop_token_id)

                if finished_sequences.all(): break

                # --- Find argmax within the valid range of relations ---
                predicted_relation_list = []
                for i in range(batch_size):
                    if finished_sequences[i]:
                        predicted_relation_list.append(torch.tensor(0, device=device))
                        continue

                    max_r = len(paths[i]) - 1
                    valid_logits = relation_logits[i, -1, :max_r + 1]
                    best_r = valid_logits.argmax(dim=-1)
                    predicted_relation_list.append(best_r)
                predicted_relation = torch.stack(predicted_relation_list)

                next_node_features_list = []
                for i in range(batch_size):
                    if finished_sequences[i]:
                        next_node_features_list.append(torch.zeros(1, 1, self.config.hidden_dim_decoder, device=device))
                        continue

                    r = predicted_relation[i].item()
                    parent_path_pos = len(paths[i]) - r - 1
                    parent_node_idx = paths[i][parent_path_pos]

                    new_node_idx = step
                    if step > 0:
                        adj_matrices[i, parent_node_idx, new_node_idx] = 1
                        adj_matrices[i, new_node_idx, parent_node_idx] = 1

                    paths[i] = paths[i][:parent_path_pos + 1] + [new_node_idx]

                    node_smiles = self.inv_vocab.get(next_node_id[i].item(), "")
                    mol = get_mol(node_smiles)
                    hs = sum(a.GetTotalNumHs() for a in mol.GetAtoms()) if mol else 0

                    next_node_features = self.decoder_featurizer(
                        node_type=next_node_id[i].view(1, 1),
                        hs=torch.tensor([[hs]], device=device),
                        layer_number=torch.tensor([[len(paths[i]) - 1]], device=device),
                        parent_pos=torch.tensor([[parent_node_idx]], device=device)
                    )
                    next_node_features_list.append(next_node_features[:, 1:])

                current_features = torch.cat(next_node_features_list, dim=0)

        return gen_node_indices


if __name__ == '__main__':
    config = VAEConfig()
    vocab = {
        "<pad>": 0, "C": 1, "N": 2, "O": 3, "c1ccccc1": 4,
        "(": 5, ")": 6, "=": 7, "#": 8, "[C@@H]": 9,
        "<stop>": 10
    }
    config.vocab_size = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}

    print("Instantiating JTreeformer model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JTreeformer(config, vocab).to(device)
    print("Model instantiated successfully.")

    batch_size = 2
    max_decode_len = 15
    stop_token_id = vocab["<stop>"]

    z = torch.randn(batch_size, config.latent_dim).to(device)

    print(f"\nTesting decode method with batch_size={batch_size} and max_len={max_decode_len}...")

    model.eval()
    with torch.no_grad():
        decoded_indices = model.decode(z, max_len=max_decode_len, stop_token_id=stop_token_id)

    print("Decode method finished.")

    print("\n--- Decoded Results ---")
    for i in range(batch_size):
        sequence_indices = decoded_indices[i].tolist()
        print(sequence_indices)

        if stop_token_id in sequence_indices:
            stop_idx = sequence_indices.index(stop_token_id)
            sequence_indices = sequence_indices[:stop_idx]

        smiles_sequence = [inv_vocab.get(idx, "<unk>") for idx in sequence_indices]

        print(f"Sample {i + 1}: {' '.join(smiles_sequence)}")

    print("\nTest script finished successfully.")
