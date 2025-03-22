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
from model.dataloader import (
    SoccerNetCalibrationDataset,
    WorldCup2014Dataset,
    TSWorldCupDataset,
)
from model.cls_hrnet import get_cls_net
from model.losses import MSELoss

from torchvision import transforms as v2


import matplotlib.pyplot as plt

# %% [markdown]
# # check data loader

# %%
import copy
import torchvision.transforms.functional as f


def correct_labels(data):
    if "Goal left post left" in data.keys():
        data["Goal left post left "] = copy.deepcopy(data["Goal left post left"])
        del data["Goal left post left"]
    return data


class ToTensor(torch.nn.Module):
    def __call__(self, sample):
        image = sample["image"]

        return {"image": f.to_tensor(image).float(), "data": sample["data"]}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# %%
import json
from PIL import Image
from utils.utils_keypoints import KeypointsDB


img_name = "datasets/calibration-2023/valid/00000.jpg"
image = Image.open(img_name)
data = json.load(open(img_name.split(".")[0] + ".json"))

data = correct_labels(data)


no_transforms = v2.Compose([ToTensor()])

sample = no_transforms({"image": image, "data": data})
img_db = KeypointsDB(sample["data"], sample["image"])

# draw keypoint_world_coords_2D
for idx, kp in enumerate(img_db.keypoint_world_coords_2D):
    plt.scatter(kp[0], kp[1], c="red", s=10)
    plt.text(kp[0], kp[1], str(idx), color="red")
plt.show()

# %%


img_db.get_full_keypoints()

print(img_db.keypoints_final.keys())
print("all points: ", img_db.keypoints_final)
img_db.draw_keypoints()

# %%
img_db.draw_field()
# %%

data["Side line left"]


# %%
img_db.data.keys()

target, mask = img_db.get_tensor_w_mask()

# check target mask type shape and other info
print("target type: ", type(target))
print("target shape: ", target.shape)
print("mask type: ", type(mask))
print("mask shape: ", mask.shape)
print("mask: ", mask)

# target is (58, 270, 480)
# max target along the first dimension and plt the image
max_target = np.max(target, axis=0)
plt.imshow(max_target)
plt.show()

# %%
from model.transforms import transforms, no_transforms

validation_set = SoccerNetCalibrationDataset(
    "datasets/calibration-2023/", "valid", transform=no_transforms, main_cam_only=True
)
validation_loader = torch.utils.data.DataLoader(
    validation_set, num_workers=1, batch_size=1, shuffle=False
)

# %%
# get first batch
batch = next(iter(validation_loader))


# %%
len(batch)
# image, torch.from_numpy(target).float(), torch.from_numpy(mask).float()

type(batch[0])
# tensor to numpy
image = batch[0].numpy().squeeze().transpose(1, 2, 0)
# plot image
plt.imshow(image)
plt.show()

# show target
target = batch[1].numpy().squeeze()
print(target.shape)


# %%
# show 1 layer of target
print(target[12, :, :].shape)
plt.imshow(target[12, :, :])
plt.show()
plt.colorbar

target[12, :, :]
