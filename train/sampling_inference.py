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
    
    # ---------- sampling ----------
    #load test data
    D_test = Memory_Dataset(cfg, cwd, cfg.train.test_data_path[0], device=device)
    #initialize model
    model = Model_Base(cfg, device)

    #parameters for saving loss
    loss_train = dict()
    loss_train["diffusion"] = []
    loss_val = dict()
    loss_val["diffusion"] = []

    #load model
    itr_start = 0
    if cfg.train.model_path is not None:
        print(f"sampling from {cfg.train.model_path}")
        params = torch.load(cfg.train.model_path, map_location=device)
        model.load_state_dict(params["model"])
        model.optimizer.load_state_dict(params["optimizer"])
        #load iteration
        if params["loss"] is not None:
            loss_train = params["loss"]["train"]
            loss_val = params["loss"]["validation"]
            itr_start = len(loss_train["diffusion"])
        print(f"itr_start: {itr_start}")
        #sampling
        print("sampling")
        x = model.sampling_inference(D_test.inputs, text=D_test.texts)
        os.makedirs(f"{results_dir}/sample", exist_ok=True)
        np.save(f"{results_dir}/sample/sample_{itr_start}.npy", x)

    else:
        NotImplementedError("model_path is None")
        

@hydra.main(config_path="config", config_name="config")
def main(cfg_raw : DictConfig) -> None:
    _cfg = copy.deepcopy(cfg_raw)
    _cfg.main.experiment_name = "sampling_inference"
    _cfg.main.wandb = False
    run(_cfg)

if __name__=="__main__":
    main()