import os

import numpy as np
import random
import glob
import torch

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf

def np_to_tensor(data, device=torch.device("cpu")):
    """
    convert numpy data to tensor data

    params
    ------
    data: numpy.ndarray
        numpy data
    device: torch.device
        using device

    return
    -------
    data: tensor
        tensor data
    """
    if not torch.is_tensor(data):
        return torch.tensor(data, dtype=torch.float32, device=device)
    else:
        return data

class Memory_Dataset:
    """
    setting dataset

    params
    ------
    cfg: object
        config
    cwd: str
        current folder path
    dataset_path: str
    image_path: str
    device: torch.device
        using device
        
    attributions
    -------------
    cfg: object
    cwd: str
    dataset_path: str
    file_names: list
    inputs: dict[tensor]
    image_path: list
    texts: list
    device: torch.device
    """
    def __init__(self,
                 cfg,
                 cwd,
                 dataset_path,
                 image_path=None,
                 device=torch.device("cpu")
                 ):
        self.cfg = cfg
        self.cwd = cwd
        self.dataset_path = dataset_path
        self.device = device
        self.file_names = self.get_file_names()
        #initialize input data
        self.inputs = dict()
        for name in self.cfg.train.input_names:
            self.inputs[name] = torch.empty(
                (len(self.file_names), *self.cfg.train.input_shapes[name]),
                dtype=torch.float32
            )
        self.image_path = []
        if self.cfg.train.text_condition and image_path is not None:
            self.image_path = sorted(glob.glob(os.path.join(self.cwd, image_path, "*.jpg")))
        self.texts = [[]for i in range(len(self.file_names))]
        self.load_dataset()


    #------load dataset------
    def get_file_names(self):
        """
        obtain file names
        returns
        -------
        file_names: list
            ファイル名
        """
        dataset_dir = os.path.join(self.cwd, self.dataset_path)
        extention = "*.npy"
        if not os.path.exists(dataset_dir): 
            raise NotImplementedError(f"{dataset_dir} is not exist")
        #loading
        file_names = sorted(glob.glob(os.path.join(dataset_dir, extention)))

        return file_names
    
    def load_dataset(self):
        print("find %d npy files!" % len(self.file_names))
        #load input data
        for idx, file_name in tqdm(enumerate(self.file_names), desc="load dataset"):
            data = np.load(file_name, allow_pickle=True).item()
            for name in self.cfg.train.input_names:
                self.inputs[name][idx] = np_to_tensor(data[name], device=self.device)
            if self.cfg.train.text_condition:
                self.texts[idx] = data["text"]

    def split_batch(self):
        """
        split data into batch

        returns
        -------
        batches: list[dict]
            batch data
        """
        #obtain data length
        data_length = len(self.file_names)
        batch_size = self.cfg.train.batch_size
        if data_length < batch_size:
            print("num of train or validation data is not over batch size")
            raise NotImplementedError
        #split data into batch
        batch_nums = data_length//batch_size
        batches = list()
        batch_list = [i for i in range(data_length)]
        random.shuffle(batch_list)
        for i in range(batch_nums):
            batch_index = batch_list[i*batch_size:(i+1)*batch_size]
            batches.append(dict())
            for name in self.cfg.train.input_names:
                batches[i][name] = torch.empty(
                (batch_size, *self.cfg.train.input_shapes[name]),
                dtype=torch.float32
                ).to(self.device)
                for j, idx in enumerate(batch_index):
                    batches[i][name][j] = self.inputs[name][idx]
            if self.cfg.train.text_condition:
                batches[i]["text"] = list()
                for j, idx in enumerate(batch_index):
                    batches[i]["text"].append(self.texts[idx])

        return batches






