**TransRefine++**


This GitHub repository contains the training code for the paper  "** TransRefine++: Cross Transformer augmented Feature Refinement for Zero-Shot Remote Sensing Scene Classification **". 

**The Framework for the TransRefine++ **

![](figs/TransRefine++-blockdg.png)

**Dependencies to run the TransRefine++**

The implementation of "TransRefine++" is based on Python 3.8.8 and [PyTorch](https://pytorch.org/) 1.8.0. To install all required dependencies:
```
$ pip install -r requirements.txt
```
Furthermore, we use [Weights & Biases](https://wandb.ai/site) (W&B) to do experiments. However, in this training code, we set the wandb mode="disabled" for the sake of user-friendly. Just install wandb: 
```
$ pip install wandb
```
**Optional:** To keep track and organize the results of experiments, set the wandb mode="online". To start quickly, you may need to follow the [online documentation](https://docs.wandb.ai/quickstart) of W&B. To run these codes, [sign up](https://app.wandb.ai/login?signup=true) an online account to track experiments or create a [local wandb server](https://hub.docker.com/r/wandb/local) using docker (recommended).

**Downloading datasets**

We trained the model "TransRefine++" on three prominent benchmark datasets of the zero-shot scene classification in remote sensing images: [UCM21](http://weegee.vision.ucmerced.edu/datasets/landuse.html), [AID30](https://captain-whu.github.io/AID/), NWPU45. Please download the NWPU dataset in your own way. To train the "TransRefine++", first, you should download these datasets. Then, use the following link to obtain super-resolution images. 

https://github.com/zsyOAOA/ResShift

Further, organize them as follows: 
```
.
├── data
│   ├── UCM/...
│   ├── AID/...
│   ├── NWPU/...
│
└── ···
```
Specifically, the super-resolution images of the UCM dataset are organized as follows: 
```
├── data
│   ├── UCM/runway/runway89.png
|   |__ UCM/river/river35.png
|   |__ UCM/sparseresidential/sparseresidential25.png
│   |__ ...
└── ···
```
**Generaring res101.mat and att_splits.mat files**

Please follow the instructions we have included below to get res101.mat and att_splits.mat files, and place them in TransRefine++/data/xlsa17/data/{dataset}/... 

Run the following codes one after the other.
```
$ cd splits_creation_process
```
```
$ python mataw.py
```
First of all, set the dataset path and run the mataw.py file. This code will generate {dataset}_img.csv and {dataset}_label.csv files.

```
>> mat_create 
```
This Matlab file creates res101.mat file with the help of {dataset}_img.csv and {dataset}_label.csv.

```
$ python dataset_create.py
```
```
>> merge_train_val
```
Initially, set the dataset path and run the dataset_create.py file. This code obtains att_splits.mat file, and merge-train_val.m mat file combines train and validation locations. 

Then, place **res101.mat** and **att_splits.mat** files into TransRefine++/data/xlsa17/data/{dataset}/...

Example for the UCM dataset: 
```
          TransRefinePP/data/xlsa17/data/UCM/res101.mat    
          TransRefinePP/data/xlsa17/data/UCM/att_splits.mat
```
**Extracting visual features**

Run the specified commands to extract the visual features of four datasets:

```
$ python preprocessing.py --dataset UCM --compression --device cuda:0 
$ python preprocessing.py --dataset AID --compression --device cuda:0
$ python preprocessing.py --dataset NWPU --compression --device cuda:0

Note: Adjust the "split_idx" value in the preprocessing.py file according to the dataset path if you get FileNotFoundError while running the above code.
```

**Training TransRefine++**

In `./wandb_config`, we provide parameter settings for UCM21, AID30, and NWPU45 for zero-shot learning tasks. 
Run the given commands to train the "TransRefine++" from scratch:

```
$ python train_TransRefine++_ucm.py   # UCM
$ python train_TransRefine++_aid.py   # AID
$ python train_TransRefine++_nwpu.py  # NWPU
```
