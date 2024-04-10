import torch
import numpy as np

original_path = "pretrained/polyfusion/weights_best.pt"
original = torch.load(original_path, map_location={'cuda:0': 'cpu'})

path = "results/folksong_16note_8bar_polyffusion/2023-11-23/run_6/model/model_1.npy"
params = np.load(path, allow_pickle=True).item()
print(original["model"].keys())