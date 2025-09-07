import torch
import torch.nn.functional as F
from tqdm import tqdm
from noise_predictor import NoisePredictorMLP
def get_noise_schedule(schedule_name: str, timesteps: int, **kwargs):
    """
    Generates a variance schedule for the diffusion process.

    Args:
        schedule_name (str): The name of the schedule ('linear', 'cosine').
        timesteps (int): The total number of diffusion steps (T).

    Returns:
        torch.Tensor: A tensor of betas of shape [timesteps].
    """
    if schedule_name == "linear":
        beta_start = kwargs.get('beta_start', 0.0001)
        beta_end = kwargs.get('beta_end', 0.02)
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule_name == "cosine":
        s = kwargs.get('s', 0.008)
        t = torch.linspace(0, timesteps, timesteps + 1)
        f_t = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule_name}")


class DiffusionModel:
    def __init__(self, noise_predictor: NoisePredictorMLP, timesteps: int = 2000, schedule_name: str = 'cosine'):
        self.timesteps = timesteps
        self.noise_predictor = noise_predictor

        # --- Pre-calculate diffusion constants based on the schedule ---
        self.betas = get_noise_schedule(schedule_name, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def _extract(self, a, t, x_shape):
        """Extracts coefficients for a batch of timesteps t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """
        The forward diffusion process. Adds noise to the data.

        Args:
            x_start (torch.Tensor): The initial clean data (z_0), shape [B, D].
            t (torch.Tensor): Timesteps for each sample in the batch, shape [B].
            noise (torch.Tensor, optional): The noise to add. If None, generated randomly.

        Returns:
            torch.Tensor: The noisy data at the given timesteps (x_t).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _predict_x0_from_noise(self, x_t, t, noise):
        """Predicts the original clean sample (x_0) from a noisy sample (x_t) and predicted noise."""
        return (
                self._extract(1. / self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def p_sample(self, x_t, t, t_prev, eta=0.0):
        """
        A single DDIM sampling step to reverse the diffusion process.
        DDPM is a special case of DDIM where eta = 1.0.

        Args:
            x_t (torch.Tensor): The current noisy sample [B, D].
            t (torch.Tensor): The current timestep [B].
            t_prev (torch.Tensor): The previous timestep [B].
            eta (float): The DDIM eta parameter. 0.0 for deterministic DDIM, 1.0 for DDPM-like stochasticity.

        Returns:
            torch.Tensor: The denoised sample for the previous timestep (x_{t-1}).
        """
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        pred_noise = self.noise_predictor(x_t, t)

        pred_x0 = self._predict_x0_from_noise(x_t, t, pred_noise)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        sigma = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * pred_noise
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt

        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

    @torch.no_grad()
    def sample(self, shape, device, num_inference_steps=50, eta=0.0):
        """
        The full sampling loop to generate new data from pure noise.

        Args:
            shape (tuple): The shape of the output tensor (e.g., [batch_size, latent_dim]).
            device: The device to run the sampling on.
            num_inference_steps (int): The number of denoising steps. Fewer steps is faster.
            eta (float): 0.0 for DDIM, 1.0 for DDPM.

        Returns:
            torch.Tensor: The generated clean sample.
        """
        x = torch.randn(shape, device=device)
        inference_timestep_sequence = torch.linspace(self.timesteps - 1, 0, num_inference_steps).long().to(device)

        for i in tqdm(range(len(inference_timestep_sequence)), desc="Sampling"):
            t = torch.full((shape[0],), inference_timestep_sequence[i], device=device, dtype=torch.long)
            t_prev = torch.full((shape[0],),
                                inference_timestep_sequence[i + 1] if i + 1 < len(inference_timestep_sequence) else 0,
                                device=device, dtype=torch.long)
            x = self.p_sample(x, t, t_prev, eta)

        return x


if __name__ == '__main__':
    latent_dimension = 64
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_predictor_model = NoisePredictorMLP(latent_dim=latent_dimension).to(device)
    diffusion = DiffusionModel(noise_predictor_model, timesteps=1000, schedule_name='cosine')
    print("--- Testing Forward Process (q_sample) ---")
    dummy_clean_latents = torch.randn(batch_size, latent_dimension, device=device)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,), device=device)

    noisy_latents = diffusion.q_sample(dummy_clean_latents, dummy_timesteps)
    print(f"Clean latents shape:  {dummy_clean_latents.shape}")
    print(f"Noisy latents shape:  {noisy_latents.shape}")
    assert noisy_latents.shape == dummy_clean_latents.shape
    print("Forward process test successful.\n")
    print("--- Testing Inference (sample) with DDIM ---")
    generated_samples_ddim = diffusion.sample(
        shape=(batch_size, latent_dimension),
        device=device,
        num_inference_steps=50,
        eta=0.0
    )
    print(f"Generated DDIM samples shape: {generated_samples_ddim.shape}")
    assert generated_samples_ddim.shape == (batch_size, latent_dimension)
    print("\n--- Testing Inference (sample) with DDPM ---")
    generated_samples_ddpm = diffusion.sample(
        shape=(batch_size, latent_dimension),
        device=device,
        num_inference_steps=1000,
        eta=1.0
    )
    print(f"Generated DDPM samples shape: {generated_samples_ddpm.shape}")
    assert generated_samples_ddpm.shape == (batch_size, latent_dimension)

    print("\nInference tests successful!")
