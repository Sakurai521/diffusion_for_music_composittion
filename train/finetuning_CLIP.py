import sys
import os
from pathlib import Path
module_path = os.path.join(Path().resolve(), '../')
sys.path.append(module_path)
os.environ['PYTHONPATH'] = module_path

import hydra
from omegaconf import DictConfig, OmegaConf
from utils.dataset import Memory_Dataset


import copy
from tqdm import tqdm
import torch
import random
import numpy as np
import wandb

from utils.logger import setup_experiment
from algos.CLIP.algo import CLIP

def run(cfg):
    """
    diffusionによる学習を行う
    params
    ------
    cfg: object
        config.yalmで設定した値
    """
    # ---------- setup experiment ----------
    cwd, results_dir, device = setup_experiment(cfg)
    
    # ---------- 学習 ----------
    #load train data
    D = Memory_Dataset(cfg, cwd, cfg.train.train_data_path[0], 
                       cfg.pretrain.train_image_data_path[0], 
                       device=device)
    #load validation data
    D_val = Memory_Dataset(cfg, cwd, cfg.train.validation_data_path[0], 
                           cfg.pretrain.validation_image_data_path[0], 
                           device=device)
    #initialize model
    model = CLIP(cfg, device)
    train_data = model.make_clipdata(D.image_path, D.texts)
    validation_data = model.make_clipdata(D_val.image_path, D_val.texts)
    train_data = torch.utils.data.DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=True)
    validation_data = torch.utils.data.DataLoader(validation_data, batch_size=cfg.train.batch_size, shuffle=True)

    #parameters for saving loss
    key_list = ["image", "text", "CLIP"]
    loss_train = dict()
    loss_val = dict()
    for key in key_list:
        loss_train[key] = []
        loss_val[key] = []

    #train
    for itr in tqdm(range(0, cfg.pretrain.train_iteration)):
        #train data
        for i, data in enumerate(train_data):
            if i == len(train_data)-1:
                break
            loss_info = model.pretrain(data, is_train=True)
            for key in loss_train.keys():
                if i == 0:
                    loss_train[key].append(0)
                loss_train[key][-1] += loss_info[key].item()/(len(train_data)-1)
        #validation data
        if (itr+1)%cfg.pretrain.validation_interval == 0:
            for i, data in enumerate(validation_data):
                if i == len(validation_data)-1:
                    break
                loss_info = model.pretrain(data, is_train=True)
                for key in loss_val.keys():
                    if i == 0:
                        loss_val[key].append(0)
                    loss_val[key][-1] += loss_info[key].item()/(len(validation_data)-1)
        else:
            for key in loss_val.keys():
                loss_val[key].append(None)

        #save to wandb
        if cfg.main.wandb:
            for key in loss_train.keys():
                wandb.log(data={f"loss_{key}/train": loss_train[key][-1]}, step=itr)
            for key in loss_val.keys():
                if loss_val[key][-1] is not None:
                    wandb.log(data={f"loss_{key}/validation": loss_val[key][-1]}, step=itr)
        
        #save model
        if (itr+1)%cfg.pretrain.checkpoint_interval == 0:
            #save mmodel parameters
            params = dict()
            params["model"] = model.get_state_dict()
            params["optimizer"] = model.optimizer.state_dict()
            #save loss
            save_loss = dict()
            save_loss["train"] = loss_train
            save_loss["validation"] = loss_val
            params["loss"] = save_loss
            os.makedirs(f"{results_dir}/model", exist_ok=True)
            torch.save(params, f"{results_dir}/model/model_{itr+1}.pt")
        


@hydra.main(config_path="config", config_name="config")
def main(cfg_raw : DictConfig) -> None:
    _cfg = copy.deepcopy(cfg_raw)
    _cfg.main.experiment_name = "finetuning_CLIP"
    _cfg.main.tags = "CLIP"
    _cfg.train.text_condition = True
    run(_cfg)

if __name__=="__main__":
    main()