from typing import List, Dict
from PIL import Image
import torch
from torch import nn, optim
import glob
import os
import pandas as pd
import json
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler

class CLIP:
    """
    Fine-tuning CLIP

    params
    ------
    cfg: object
    device: torch.device
    """
    def __init__(self,
                 cfg,
                 device
                 ):
        self.cfg = cfg
        self.device = device
        print(f"Use {self.cfg.env.CLIP['pre_trained_model']} pre-train model")
        self.model, self.preprosess = clip.load(self.cfg.env.CLIP['pre_trained_model'], 
                                                device=device, 
                                                jit=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.pretrain.learning_rate)
        self.image_loss = nn.CrossEntropyLoss()
        self.text_loss = nn.CrossEntropyLoss()
        
    def make_clipdata(self,
                      image_path: List[str],
                      text: List[str]
                      ):
        """
        Make CLIP data

        params
        ------
        image_path: list[str] 
        text: list[str]
        
        return
        ------
        dataset: torch.utils.data.TensorDataset
        """
        #前処理
        image_conv = []
        text_conv = []
        for i in range(len(image_path)):
            image = Image.open(image_path[i])
            image_conv.append(self.preprosess(image).unsqueeze(0).to(self.device))
        image_conv = torch.cat(image_conv, dim=0)
        text_conv = clip.tokenize(text).to(self.device)
        dataset = torch.utils.data.TensorDataset(image_conv, text_conv)

        return dataset

    def pretrain(self,
                 data,
                 is_train=True):
        """
        pretraining CLIP

        params
        ------
        data: torch.utils.data.TensorDataset
        is_train: bool

        return
        ------
        loss_info: dict loss
        """
        if is_train:
            self.optimizer.zero_grad()
            self.model.train()
        else:
            self.model.eval()
        image = data[0].to(self.device)
        text = data[1].to(self.device)
        #training
        logits_per_image, logits_per_text = self.model(image, text)
        ground_truth = torch.arange(self.cfg.pretrain.batch_size).to(self.device)
        image_loss = self.image_loss(logits_per_image, ground_truth)
        text_loss = self.text_loss(logits_per_text, ground_truth)
        loss = (image_loss + text_loss)/2
        if is_train:
            loss.backward()
            if self.device == "cpu":
                self.optimizer.step()
            else:
                for p in self.model.parameters():
                    p.data = p.data.float() 
                    p.grad.data = p.grad.data.float() 
                self.optimizer.step()
                clip.model.convert_weights(self.model)
        loss_info = {"image": image_loss, "text": text_loss, "CLIP": loss}

        return loss_info
    
    def text_encoding(self,
                      text: List[str]
                      ):
        """
        Text encoding

        params
        ------
        text: list[str]

        return
        ------
        text_enc: torch.tensor encoded text
        """
        text_token= clip.tokenize(text).to(self.device)
        text_enc= self.model.encode_text(text_token).unsqueeze(1)

        return text_enc
    
    #save parameters
    def get_state_dict(self):
        return self.model.state_dict()
    
    #load parameters
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
