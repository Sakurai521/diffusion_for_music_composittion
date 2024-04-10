import torch
import numpy as np
import copy
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from einops import rearrange, reduce, repeat

from algos.Diffusion.algo import Diffusion
from algos.CLIP.algo import CLIP

class Model_Base(nn.Module):
    """
    setting model

    params
    ------
    cfg: object
        config
    device: torch.device
        using device

    attributes
    ----------
    cfg: object
    device: torch.device
    diffusion: object
        diffusion model
    text_encoder: object
        text encoder model
    """
    def __init__(self, cfg, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.init_models()

    def init_models(self):
        #Diffusion model
        self.diffusion = Diffusion(cfg=self.cfg, device=self.device)
        #text encoder
        if self.cfg.env.text_encoder == "CLIP" and self.cfg.train.text_condition:
            print("Use CLIP")
            self.text_encoder = CLIP(cfg=self.cfg, device=self.device)
            if self.cfg.env.CLIP["parameter_path"] is not None:
                model_path = self.cfg.env.CLIP["parameter_path"]
                print(f"Load CLIP model from {model_path}")
                param = np.load(model_path, allow_pickle=True).item()
                self.text_encoder.model.load_state_dict(param["model"])
        else:
            print("Not use text encoder")
            self.text_encoder = None

        self.optimizer = optim.Adam(self.diffusion.parameters(), lr=self.cfg.train.learning_rate)
    
    def forward(self, input, is_train=True):
        """
        params
        ------
        input: dict[torch.tensor]
        is_train: bool
        
        return
        ------
        loss_info: dict[torch.tensor]
            loss
        """
        if is_train:
            self.train()
            self.diffusion.train()
            self.optimizer.zero_grad()
        else:
            self.eval()
            self.diffusion.eval()
        loss_info = dict()
        #formatting input data
        hidden = []
        modality_key = []
        for key in self.cfg.train.input_names:
            hidden.append(input[key])
            modality_key.append(key) 
        modalities = len(hidden) #num of modalities
        if modalities == 1:
            hidden = hidden[0]
        else:
            hidden = torch.cat(hidden, dim=1) #(b, c*m, t, p)
        #text encoding
        text_enc = None
        if self.cfg.train.text_condition:
            #encoding with probability
            if np.random.rand() < self.cfg.env.p_cond:
                text = input["text"]
                text_enc = self.text_encoder.text_encoding(text)

        #Diffusion
        if is_train:
            d_loss = self.diffusion(hidden, is_train=is_train, cond=text_enc)
        else:
            d_loss = self.diffusion.forward_val(hidden, is_train=is_train, cond=text_enc)
        loss_info["diffusion"] = d_loss

        # Update model parameters
        if is_train:
            d_loss.backward()
            self.optimizer.step()


        return loss_info
    
    def sampling(self, test_data=None):
        """
        sampling

        params
        ------
        test_data: dict
        """
        #generate noise
        sampler = self.diffusion.sampler
        #number of samples
        noise = torch.randn([self.cfg.train.sampling_size, 
                            self.cfg.env.u_net["channel_init"],
                            self.cfg.train.data_length, 
                            88], device=self.device)
        #sampling
        #text encoding
        if self.cfg.train.text_condition:
            text_idx = ["folk song", "jazz song", "classical song"]
            text = []
            #choice genre randomly
            for i in range(self.cfg.train.sampling_size):
                idx = np.random.randint(0, 3)
                text.append(text_idx[idx])
            text_enc = self.text_encoder.text_encoding(text)
        else:
            text_enc = None
        sampled_h = sampler(noise, cond=text_enc)
        output = dict()
        for i, key in enumerate(self.cfg.train.input_names):
            hidden_tmp = sampled_h[:, 2*i:2*(i+1), :, :]
            output[key] = hidden_tmp.to("cpu")
        output["text"] = text

        return output
    
    def sampling_inference(self, test_data, text=None):
        """
        sampling for inference other part

        params
        ------
        test_data: dict
        """
        #generate noise
        sampler = self.diffusion.sampler_inference
        #number of samples
        self.cfg.train.sampling_size = len(test_data["right"])
        noise = torch.randn([self.cfg.train.sampling_size, 
                            self.cfg.env.u_net["channel_init"],
                            self.cfg.train.data_length, 
                            88], device=self.device)
        #sampling
        mask_1 = torch.ones_like(test_data["right"], device=self.device)
        mask_0 = torch.zeros_like(test_data["left"], device=self.device)
        mask = torch.cat([mask_1, mask_0], dim=1).to(self.device)
        samples = torch.cat([test_data["right"], test_data["left"]], dim=1).to(self.device)
        #text encoding
        if self.cfg.train.text_condition:
            text_idx = ["folk song", "jazz song", "classical song"]
            text = []
            #choice genre randomly
            for i in range(self.cfg.train.sampling_size):
                idx = np.random.randint(0, 3)
                text.append(text_idx[idx])
            text_enc = self.text_encoder.text_encoding(text)
        else:
            text_enc = None
        sampled_h = sampler(noise, samples, mask, cond=text_enc)
        output = dict()
        for i, key in enumerate(self.cfg.train.input_names):
            hidden_tmp = sampled_h[:, 2*i:2*(i+1), :, :]
            output[key] = hidden_tmp.to("cpu")
        output["text"] = text

        return output
        