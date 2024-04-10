# How to Training
You can fine-tuning text encoding model(CLIP) by running ```train/finetuning_CLIP.py```<br>
You can train model by running ```train/train.py```<br>
```
$ python train/finetuning_CLIP.py #fine-tuning CLIP
$ python train/train.py #train
```
If you change train and fine-tuning setting, change config file in ```train/config```<br>

# How to Sampling
You can sampling by running ```train/sampling.py```<br>
You can infer accompainment by running ```train/sampling_inference.py```<br>
```
$ python train/sampling.py #sampling
$ python train/sampling_inference.py #infer accompainment
```
If you convert result of sampling or inference, run ```dataset/npy_to_midi.ipynb```


