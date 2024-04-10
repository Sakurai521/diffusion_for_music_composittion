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
import numpy as np
import wandb

from utils.logger import setup_experiment
from algos.main import Model_Base

def run(cfg):
    """
    params
    ------
    cfg: object
        config.yalm
    """
    # ---------- setup experiment ----------
    cwd, results_dir, device = setup_experiment(cfg)
    
    # ---------- train ----------
    #input train data
    D = Memory_Dataset(cfg, cwd, cfg.train.train_data_path[0], device=device)
    #input validation data
    D_val = Memory_Dataset(cfg, cwd, cfg.train.validation_data_path[0], device=device)
    #initialize model
    model = Model_Base(cfg, device)

    #parameters for saving loss
    loss_train = dict()
    loss_train["diffusion"] = []
    loss_val = dict()
    loss_val["diffusion"] = []

    #input pretrain model
    itr_start = 0
    if cfg.train.model_path is not None:
        print(f"load model from {cfg.train.model_path}")
        params = torch.load(cfg.train.model_path, map_location=device)
        model.load_state_dict(params["model"])
        model.optimizer.load_state_dict(params["optimizer"])
        #input loss
        loss_train = params["loss"]["train"]
        loss_val = params["loss"]["validation"]
        itr_start = len(loss_train["diffusion"])
        print(f"itr_start: {itr_start}")
        print(f"itr_end: {cfg.train.train_iteration}")
        #save loss to wandb
        if cfg.main.wandb:
            for i in range(itr_start):
                for key in loss_train.keys():
                    wandb.log(data={f"loss_{key}/train": loss_train[key][i]}, step=i)
                for key in loss_val.keys():
                    if loss_val[key][i] is not None:
                        wandb.log(data={f"loss_{key}/validation": loss_val[key][i]}, step=i)

    #train
    for itr in tqdm(range(itr_start, cfg.train.train_iteration)):
        #train data
        train_batch = D.split_batch()
        loss_train["diffusion"].append(0)
        for i in range(len(train_batch)):
            loss_info = model(train_batch[i], is_train=True)
            for key in loss_info.keys():
                loss_train[key][-1] += loss_info[key].item()/len(train_batch)
        #validation data
        loss_val["diffusion"].append(0)
        if (itr+1)%cfg.train.validation_interval == 0:
            val_batch = D_val.split_batch()
            for i in range(len(val_batch)):
                loss_info = model(val_batch[i], is_train=False)
                for key in loss_info.keys():
                    loss_val[key][-1] += loss_info[key].item()/len(val_batch)
        else:
            loss_val["diffusion"][-1] = None

        #save to wandb
        if cfg.main.wandb:
            for key in loss_train.keys():
                wandb.log(data={f"loss_{key}/train": loss_train[key][-1]}, step=itr)
            for key in loss_val.keys():
                if loss_val[key][-1] is not None:
                    wandb.log(data={f"loss_{key}/validation": loss_val[key][-1]}, step=itr)
        
        #save model
        if (itr+1)%cfg.train.checkpoint_interval == 0:
            #モデルのパラメータの保存
            params = dict()
            params["model"] = model.state_dict()
            params["optimizer"] = model.optimizer.state_dict()
            #lossの保存
            save_loss = dict()
            save_loss["train"] = loss_train
            save_loss["validation"] = loss_val
            params["loss"] = save_loss
            os.makedirs(f"{results_dir}/model", exist_ok=True)
            torch.save(params, f"{results_dir}/model/model_{itr+1}.pt")
        
        
@hydra.main(config_path="config", config_name="config")
def main(cfg_raw : DictConfig) -> None:
    _cfg = copy.deepcopy(cfg_raw)
    run(_cfg)

if __name__=="__main__":
    main()

