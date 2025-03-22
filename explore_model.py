# %%
import sys
import yaml
import wandb
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn

from datetime import datetime

from utils.utils_train import train_one_epoch, validation_step
from model.dataloader import SoccerNetCalibrationDataset, WorldCup2014Dataset, TSWorldCupDataset
from model.cls_hrnet import get_cls_net
from model.losses import MSELoss

from torchvision import transforms as v2


import matplotlib.pyplot as plt

import copy
# %%
import torchvision.transforms.functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = yaml.safe_load(open('config/hrnetv2_w48_nj22.yaml', 'r'))
model = get_cls_net(cfg, pretrained='./SV_kp', device=device)

# %%
loaded_state = torch.load('./SV_kp', map_location=device)

# %%


# Filter out classifier weights
filtered_dict = {k: v for k, v in loaded_state.items() 
                if model.state_dict()[k].shape == v.shape}
print(filtered_dict.keys())


# %%
# Update model with filtered weights
model_dict = model.state_dict()
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)


# %%
model.load_state_dict(loaded_state)
model.to(device)

# %%
