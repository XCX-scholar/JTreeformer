from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """
    Configuration for the JTreeformer model, dataset, and training.

    This dataclass centralizes all hyperparameters and ablation study flags,
    making experiments reproducible and easy to configure.
    """
    # --- Model Architecture ---
    num_layers_encoder: int = 2
    num_layers_decoder: int = 2
    hidden_dim_encoder: int = 32
    expand_dim_encoder: int = 128
    hidden_dim_decoder: int = 32
    expand_dim_decoder: int = 128
    num_head_encoder: int = 4
    num_head_decoder: int = 4
    latent_dim: int = 64

    # --- VAE/AE Switch ---
    is_vae: bool = True

    # --- Data & Vocabulary ---
    num_node_type: int = 4622  # Based on original vocab
    max_hs: int = 50
    max_degree: int = 20
    max_layer_num: int = 200

    # --- Ablation Flags: Features ---
    use_hs_embedding: bool = True
    use_layer_embedding: bool = True
    use_degree_embedding: bool = True  # Encoder-only

    # --- Ablation Flags: Model Components ---
    use_graph_conv: bool = True

    # --- Ablation Flags: Auxiliary Losses ---
    predict_properties: bool = True

    # --- Dropout ---
    dropout: bool = True
    dropout_rate: float = 0.1

    # --- DeepNet Initialization Parameters ---
    # These are calculated in __post_init__
    # encoder-decoder architecture.
    encoder_alpha: float = field(init=False)
    encoder_beta: float = field(init=False)
    decoder_alpha: float = field(init=False)
    decoder_beta: float = field(init=False)

    # --- Device ---
    device: str = "cuda:0"

    def __post_init__(self):
        """
        Calculates the DeepNet scaling factors after the object is initialized.
        """
        # Specific formula for the encoder
        self.encoder_alpha = 0.81 * (((self.num_layers_encoder + 2) ** 4) * self.num_layers_decoder) ** (1 / 16)
        self.encoder_beta = 0.87 * (((self.num_layers_encoder + 2) ** 4) * self.num_layers_decoder) ** (-1 / 16)

        # Specific formula for the decoder
        self.decoder_alpha = (2 * self.num_layers_decoder + 1) ** (1 / 4)
        self.decoder_beta = (8 * self.num_layers_decoder + 4) ** (-1 / 4)

