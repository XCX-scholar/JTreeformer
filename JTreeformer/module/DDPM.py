from module.SkipNet import SkipNet
import torch
import math
import torch.nn.functional as F
import torch.nn as nn

# refer to https://huggingface.co/blog/annotated-diffusion

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - len(t.shape))))

def cosine_beta_schedule(t, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = t + 1
    x = torch.linspace(0, t, steps)
    alphas_cumprod = torch.cos(((x / t) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(t):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, t)

def quadratic_beta_schedule(t):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, t) ** 2

def sigmoid_beta_schedule(t):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, t)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Model.

    Args:
        latent_space_dim (int): Dimension of the latent space.
        expand_factor (int): Expansion factor for hidden layers in SkipNet.
        time_embedding_dim (int): Dimension of time embeddings.
        num_block (int): Number of blocks in SkipNet.
        dropout (bool): Whether to use dropout.
        dropout_rate (float): Dropout rate.
        noise_schedule (str): Noise schedule type ('linear', 'cosine', etc.).
        num_sample_steps (int): Number of sampling steps.
        device (str/torch.device): Device to use for training (e.g., 'cuda:0', 'cpu').

    """
    def __init__(
        self,
        latent_space_dim=512,
        expand_factor=4,
        time_embedding_dim=512,
        num_block=6,
        dropout=True,
        dropout_rate=0.2,
        Init_params=True,
        noise_schedule='cosine',
        num_sample_steps=2000,
        device = "cuda:0"
    ):

        super(GaussianDiffusion, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.num_block = num_block
        self.expand_factor=expand_factor
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.Init_params = Init_params
        self.device = torch.device(device)
        self.noise_net = SkipNet(
            in_dim=latent_space_dim,
            out_dim=latent_space_dim,
            hidden_dim=expand_factor*latent_space_dim,
            time_embedding_dim=time_embedding_dim,
            num_block=num_block,
            dropout=True,
            dropout_rate=0.2,
            Init_params=True,
            device = device
        )

        # image dimensions

        if noise_schedule == 'linear':
            self.schedule = linear_beta_schedule
        elif noise_schedule == 'cosine':
            self.schedule = cosine_beta_schedule
        elif noise_schedule == 'quadratic':
            self.schedule = quadratic_beta_schedule
        elif noise_schedule == 'quadratic':
            self.schedule = quadratic_beta_schedule
        elif noise_schedule == 'sigmoid':
            self.schedule = sigmoid_beta_schedule
        # elif noise_schedule == 'learned':
        #     log_snr_max = linear_log_snr(torch.tensor([0])).item()
        #     log_snr_min = linear_log_snr(torch.tensor([1])).item()
        #
        #     self.log_snr = learned_noise_schedule(
        #         log_snr_max = log_snr_max,
        #         log_snr_min = log_snr_min,
        #         hidden_dim = learned_schedule_net_hidden_dim,
        #         frac_gradient = learned_noise_schedule_frac_gradient
        #     )

        self.num_sample_steps = num_sample_steps

        self.betas = self.schedule(t=self.num_sample_steps).to(self.device)
        self.alphas = (1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))


    def denoising_sample(self, x, t, t_minus_one,eta=0.2,grad_scale=1,condition_gradient=None):
        """
        Denoising step (DDIM).

        Args:
            x (torch.Tensor): Current sample.
            t (torch.Tensor): Current timestep.
            t_minus_one (torch.Tensor): Previous timestep.
            eta (float): DDIM eta parameter.
            grad_scale (float): Scaling factor for condition gradient.
            condition_gradient (torch.Tensor, optional): Gradient of the condition.

        Returns:
            torch.Tensor: Updated sample.
        """
        t_sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape).to(self.device)
        t_minus_one_sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod,t_minus_one, x.shape).to(self.device)
        t_sqrt_alphas_cumprod=extract(self.sqrt_alphas_cumprod, t, x.shape).to(self.device)
        t_minus_one_sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod,t_minus_one, x.shape).to(self.device)
        t_alphas_cumprod = extract(self.alphas_cumprod, t, x.shape).to(self.device)
        t_minus_one_alphas_cumprod = extract(self.alphas_cumprod, t_minus_one, x.shape).to(self.device)

        pred_noise=self.noise_net(x,t)
        t_theta=eta*t_minus_one_sqrt_one_minus_alphas_cumprod/t_sqrt_one_minus_alphas_cumprod*torch.sqrt(1-t_alphas_cumprod/t_minus_one_alphas_cumprod)
        pred_x = t_minus_one_sqrt_alphas_cumprod/t_sqrt_alphas_cumprod*(x-t_sqrt_one_minus_alphas_cumprod*pred_noise)+torch.sqrt(torch.square(t_minus_one_sqrt_one_minus_alphas_cumprod)-torch.square(t_theta))*pred_noise+t_theta*torch.randn_like(pred_noise,device=self.device)
        if condition_gradient is not None:
            pred_x+=grad_scale*condition_gradient

        return pred_x

    # def denoising_sampling_loop(self,shape):
    #     predict_sample = torch.randn(shape, device=self.device)
    #     for i in tqdm(range(self.num_sample_steps), total=self.num_sample_steps):
    #         predict_sample = self.denoising_sample(predict_sample,torch.full((shape[0],),i,dtype=torch.long),i)

    def diffusion_sampling(self,x_start, times, noise):
        """
        Forward diffusion process.

        Args:
            x_start (torch.Tensor): Starting sample.
            times (torch.Tensor): Timesteps.
            noise (torch.Tensor): Noise tensor.

        Returns:
            torch.Tensor: Noisy sample at the given timesteps.
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, times, x_start.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, times, x_start.shape).to(self.device)

        return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).to(self.device)
