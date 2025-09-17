import torch
import torch.nn as nn

class MultiTargetPredictor(nn.Module):
    """
    A model to predict multiple continuous target values from a latent vector,
    with flexible configuration.

    It uses a shared MLP backbone and a dictionary of 2-layer MLP output heads,
    one for each named target.
    """

    def __init__(self, latent_dim, hidden_dim, num_layers, targets_config):
        """
        Args:
            latent_dim (int): The dimension of the input latent vector.
            hidden_dim (int): The dimension of the hidden layers.
            num_layers (int): The number of layers in the shared MLP backbone.
            targets_config (dict): A dictionary where keys are target names (str)
                                   and values are the dimensions of each target (int).
                                   Example: {'property_A': 1, 'property_B': 3}
        """
        super().__init__()
        self.targets_config = targets_config

        layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.shared_mlp = nn.Sequential(*layers)

        self.output_heads = nn.ModuleDict()
        for target_name, target_dim in targets_config.items():
            self.output_heads[target_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, target_dim)
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape [B, latent_dim].

        Returns:
            dict: A dictionary where keys are target names and values are the
                  predicted tensors for each target.
        """
        shared_features = self.shared_mlp(x)
        predictions = {
            name: head(shared_features) for name, head in self.output_heads.items()
        }

        return predictions
