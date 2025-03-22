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
from utils.utils_keypoints import court_definitions
from model.dataloader import (
    BasketballCourtDataset,
    SoccerNetCalibrationDataset,
    WorldCup2014Dataset,
    TSWorldCupDataset,
)
from model.cls_hrnet import get_cls_net
from model.losses import MSELoss

from torchvision import transforms as v2


import matplotlib.pyplot as plt

import pytest
import os


def test_basketball_court_dataloader():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = BasketballCourtDataset(
        xml_file=os.path.join(current_dir, "datasets/annotations.xml"),
        img_dir=os.path.join(current_dir, "datasets/"),
    )
    print(len(dataset))
    print(dataset.annotations)
    print(dataset.labels)

    for annotation in dataset.annotations:
        for label, points in annotation.items():
            if label == "arc":
                assert len(points) == 4


def test_court_keypoints():
    key_points_name, court_keypoints = court_definitions[
        "FIBA"
    ].get_keypoint_world_coords_2D()

    # plot court keypoints with key points name
    plt.figure(figsize=(10, 10))
    for name, point in zip(key_points_name, court_keypoints):
        plt.text(point[0], point[1], name, fontsize=12)
        plt.plot(point[0], point[1], "ro")
    # set x,y axis same sacle
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

    for name, point in zip(key_points_name, court_keypoints):
        print(f"{name}: {point}")

    assert False
