import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    Encodes a scalar timestep 't' into a vector embedding using sine and cosine functions
    of different frequencies. This allows the model to understand the position in the
    diffusion process.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        Args:
            t (torch.Tensor): A tensor of shape [batch_size] with timesteps.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, embed_dim].
        """
        device = t.device
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DenoisingMLPBlock(nn.Module):
    """
    A single block of the MLP, which includes two linear layers, an activation,
    and the addition of the time embedding. This structure allows the time
    information to influence the processing at each stage.
    """
    def __init__(self, input_dim, output_dim, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.time_proj = nn.Linear(time_embed_dim, output_dim)

    def forward(self, x, t_emb):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            t_emb (torch.Tensor): Time embedding of shape [batch_size, time_embed_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        h = self.linear1(x)
        h = h + self.time_proj(t_emb)

        h = self.activation(h)
        h = self.linear2(h)
        return h


class NoisePredictorMLP(nn.Module):
    """
    The main denoising network. It's an MLP with residual connections and time
    embeddings injected into each block.
    """

    def __init__(self, latent_dim, time_embed_dim=128, hidden_dim=512, num_layers=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_net = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DenoisingMLPBlock(hidden_dim, hidden_dim, time_embed_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, noisy_latent_z, time_t):
        """
        Args:
            noisy_latent_z (torch.Tensor): The noisy input vector of shape [batch_size, latent_dim].
            time_t (torch.Tensor): A tensor of timesteps of shape [batch_size].

        Returns:
            torch.Tensor: The predicted noise of shape [batch_size, latent_dim].
        """
        t_emb = self.time_embed_net(time_t)
        h = self.input_proj(noisy_latent_z)
        for block in self.blocks:
            h = h + block(h, t_emb)
        predicted_noise = self.output_proj(h)
        return predicted_noise

if __name__ == '__main__':
    latent_dimension = 64
    batch_size = 4
    model = NoisePredictorMLP(latent_dim=latent_dimension)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    dummy_noisy_latents = torch.randn(batch_size, latent_dimension)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,))
    predicted_noise = model(dummy_noisy_latents, dummy_timesteps)
    print(f"\nInput noisy latent shape: {dummy_noisy_latents.shape}")
    print(f"Input timesteps shape:    {dummy_timesteps.shape}")
    print(f"Output predicted noise shape: {predicted_noise.shape}")
    assert predicted_noise.shape == dummy_noisy_latents.shape
    print("\nSuccessfully executed a forward pass!")
