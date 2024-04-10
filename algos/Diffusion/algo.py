from math import sqrt
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from einops import rearrange, reduce
from torch import Tensor

from tqdm import tqdm

from utils.params import (extract,
                          extract_2,
                          BetaSchedule)
from algos.U_Net.u_net_std import UNetModel #https://github.com/aik2mlj/polyffusion
    
#--------- Diffusion -------------
class Diffusion(nn.Module):
    """
    Algorythm of Diffusion model(DDPM) 

    params
    ------
    cfg: object
        config
    device: torch.device
        using device
    """
    def __init__(self, cfg, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.num_steps = cfg.train.num_steps #num steps
        #parameters related α, β
        self.beta_schedule = BetaSchedule(cfg.env.beta["start"],
                                                    cfg.env.beta["end"],
                                                    cfg.train.num_steps,
                                                    device) #β
        self.beta = self.beta_schedule.betas #β
        self.alpha = 1.0 - self.beta #α
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0) #𝛼¯
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod) #√𝛼¯

        #sampling parameters
        with torch.no_grad():
            alpha_cumprod_prev = torch.cat([self.alpha_cumprod.new_tensor([1.]), self.alpha_cumprod[:-1]]) #𝛼¯_t-1
            self.sqrt_1m_alpha_cumprod = (1. - self.alpha_cumprod)**.5 #√(1-𝛼¯)
            self.sqrt_recip_alpha_cumprod = self.alpha_cumprod**-.5 #√(1/𝛼¯)
            self.sqrt_recip_m1_alpha_cumprod = (1 / self.alpha_cumprod - 1)**.5 #√(1/(1-𝛼¯))
            variance = self.beta * (1. - alpha_cumprod_prev) / (1. - self.alpha_cumprod)
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            self.mean_x0_coef = self.beta * (alpha_cumprod_prev**.5) / (1. - self.alpha_cumprod)
            self.mean_xt_coef = (1. - alpha_cumprod_prev) * ((1 - self.beta)** 0.5) / (1. - self.alpha_cumprod)

        
        #setting U-Net
        self.u_net_config = cfg.env.u_net
        self.net = UNetModel(
            self.u_net_config["channel_init"],
            self.u_net_config["channel_init"],
            self.u_net_config["channel"],
            self.u_net_config["num_blocks"],
            self.u_net_config["attention_levels"],
            self.u_net_config["multiple_layer"],
            self.u_net_config["n_head"],
            self.u_net_config["tf_layer"],
            self.u_net_config["d_cond"]
                         ).to(device)
    
    def forward(self, x, is_train=True, cond=None):
        """
        x: Tensor(batch_size, channel*modalities, length, pitch)
        is_train: bool 
        cond: Tensor(batch_size, 1, d_cond) condition
        """
        #obtarin parameters
        batch_size = x.shape[0] 
        t = torch.randint(0, self.cfg.train.num_steps, (batch_size,), device=self.device, dtype=torch.long)#step information
        #obtain parameter at t
        alpha_cumprod_t = extract_2(self.alpha_cumprod, t, batch_size, self.device) #𝛼¯_t
        sqrt_alpha_cumprod_t = extract_2(self.sqrt_alpha_cumprod, t, batch_size, self.device) # √𝛼¯_t

        #add noise (q(x_t|x_0)
        noise = torch.randn_like(x)
        mean = sqrt_alpha_cumprod_t * x
        var = 1.0 - alpha_cumprod_t
        x_t = mean + (var**0.5) * noise

        #predict noise
        if cond is None:
            cond = -torch.ones((batch_size, 1, self.u_net_config["d_cond"]), device=self.device)
        cond = cond.to(x_t.dtype)
        noise_pred = self.net(x_t, t, cond)

        #caluculate loss
        loss = F.mse_loss(noise, noise_pred)

        return loss
    
    @torch.no_grad()
    def forward_val(self, x, is_train=False, cond=None):
        """
        x: Tensor(batch_size, channel*modalities, length, pitch)
        is_train: bool
        cond: Tensor(batch_size, 1, d_cond) condition
        """
        #obtain parameteres
        batch_size = x.shape[0] 
        t = torch.randint(0, self.cfg.train.num_steps, (batch_size,), device=self.device, dtype=torch.long)#step information
        #t時点の情報を獲得
        alpha_cumprod_t = extract_2(self.alpha_cumprod, t, batch_size, self.device) #𝛼¯_t
        sqrt_alpha_cumprod_t = extract_2(self.sqrt_alpha_cumprod, t, batch_size, self.device) # √𝛼¯_t

        #add noise q(x_t|x_0)
        noise = torch.randn_like(x)
        mean = sqrt_alpha_cumprod_t * x
        var = 1.0 - alpha_cumprod_t
        x_t = mean + (var**0.5) * noise

        #predict noise
        if cond is None:
            cond = -torch.ones((batch_size, 1, self.u_net_config["d_cond"]), device=self.device)
        cond = cond.to(x_t.dtype)
        noise_pred = self.net(x_t, t, cond)

        #caluculate loss
        loss = F.mse_loss(noise, noise_pred)

        return loss
    
    @torch.no_grad()
    def sampler(self, x, cond=None):
        """
        sampling
        x: Tensor(batch_size, channel*modalities, length, pitch)
        """
        #obtain parameters
        batch_size = x.shape[0]
        if cond is None:
            cond = -torch.ones((batch_size, 1, self.u_net_config["d_cond"]), device=self.device)
        cond = cond.to(x.dtype)

        #denoising sampling
        bar = tqdm(total=self.num_steps)
        for step in reversed(range(0, self.num_steps)):
            #parameter at t
            t = x.new_full((batch_size, ), step, dtype=torch.long)
            sqrt_recip_alpha_cumprod = x.new_full(
                (batch_size, 1, 1, 1), self.sqrt_recip_alpha_cumprod[step]
            ) #√(1/𝛼¯_t)
            sqrt_recip_m1_alpha_cumprod = x.new_full(
                (batch_size, 1, 1, 1), self.sqrt_recip_m1_alpha_cumprod[step]
            ) #√(1/(1-𝛼¯_t))

            #denoising
            e_t = self.net(x, t, cond)
            x0 = sqrt_recip_alpha_cumprod * x - sqrt_recip_m1_alpha_cumprod * e_t
            mean_x0_coef = x.new_full((batch_size, 1, 1, 1), self.mean_x0_coef[step])
            mean_xt_coef = x.new_full((batch_size, 1, 1, 1), self.mean_xt_coef[step])
            mean = mean_x0_coef * x0 + mean_xt_coef * x
            log_var = x.new_full((batch_size, 1, 1, 1), self.log_var[step])
            #not add noise at t=0
            if step == 0:
                noise = 0
            else:
                noise = torch.randn(x.shape, device=self.device)
            x_prev = mean + (0.5 * log_var).exp() * noise
            x = x_prev
            bar.update(1)
        
        return x_prev
    
    @torch.no_grad()
    def sampler_inference(self, x, sample, mask, cond=None):
        """
        sampling for accompainment and arranging
        x: Tensor(batch_size, channel*modalities, length, pitch)
        sample: Tensor(batch_size, channel*modalities, length, pitch)
        mask: Tensor(batch_size, channel*modalities, length, pitch)
        """
        #obtain parameters
        batch_size = x.shape[0]
        if cond is None:
            cond = -torch.ones((batch_size, 1, self.u_net_config["d_cond"]), device=self.device)
        cond = cond.to(x.dtype)

        #denoising sampling
        bar = tqdm(total=self.num_steps)
        for step in reversed(range(0, self.num_steps)):
            #parameter at t
            t = x.new_full((batch_size, ), step, dtype=torch.long)
            sqrt_recip_alpha_cumprod = x.new_full(
                (batch_size, 1, 1, 1), self.sqrt_recip_alpha_cumprod[step]
            ) #√(1/𝛼¯_t)
            sqrt_recip_m1_alpha_cumprod = x.new_full(
                (batch_size, 1, 1, 1), self.sqrt_recip_m1_alpha_cumprod[step]
            ) #√(1/(1-𝛼¯_t))
            sqert_alpha_cumprod = x.new_full(
                (batch_size, 1, 1, 1), self.sqrt_alpha_cumprod[step]
            ) #√𝛼¯_t
            sqrt_1m_alpha_cumprod = x.new_full(
                (batch_size, 1, 1, 1), self.sqrt_1m_alpha_cumprod[step]
            ) #√(1-𝛼¯_t)

            #denoise
            e_t = self.net(x, t, cond)
            x0 = sqrt_recip_alpha_cumprod * x - sqrt_recip_m1_alpha_cumprod * e_t
            mean_x0_coef = x.new_full((batch_size, 1, 1, 1), self.mean_x0_coef[step])
            mean_xt_coef = x.new_full((batch_size, 1, 1, 1), self.mean_xt_coef[step])
            mean = mean_x0_coef * x0 + mean_xt_coef * x
            log_var = x.new_full((batch_size, 1, 1, 1), self.log_var[step])
            #not add noise at t=0
            if step == 0:
                noise = 0
                noise_sample = 0
            else:
                noise = torch.randn(x.shape, device=self.device)
                noise_sample = torch.randn(sample.shape, device=self.device)
            x_prev = mean + (0.5 * log_var).exp() * noise

            #generate sample
            y = sqert_alpha_cumprod * sample + sqrt_1m_alpha_cumprod * noise_sample
            x_prev = mask * y + (1 - mask) * x_prev
            x = x_prev
            bar.update(1)
        
        return x_prev
            


    
    #save parameters
    def get_state_dict(self):
        return self.state_dict()
    
    #load parameters
    def load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

