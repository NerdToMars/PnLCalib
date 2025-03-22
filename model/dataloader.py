import copy
import os
import sys
import glob
import json
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as f

from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image

from utils.utils_keypoints import KeypointsDB, BasketballCourtKeypointsDB
from utils.utils_keypointsWC import KeypointsWCDB

import xml.etree.ElementTree as ET
from model.transforms import no_transforms
from typing import Dict, List


class BasketballCourtDataset(Dataset):
    """
    PyTorch Dataset for loading basketball court images and keypoints from CVAT annotations.
    """

    def __init__(self, xml_file, img_dir, transform=None):
        """
        Args:
            xml_file (string): Path to the CVAT annotation XML file
            img_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.img_dir = img_dir
        self.transform = transform

        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get the list of all labels from the task definition
        self.labels = []
        for label in root.findall(".//task/labels/label"):
            label_name = label.find("name").text
            self.labels.append(label_name)

        # Create a dictionary to map label names to indices
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        # Initialize lists to store images and annotations
        self.images = []
        self.annotations = []
        self.ellipse_point_labels = ["arc"]
        self.image_db = None

        # Extract images and keypoints
        for image in root.findall(".//image"):
            img_id = int(image.get("id"))
            img_name = image.get("name")
            img_width = int(image.get("width"))
            img_height = int(image.get("height"))

            # Collect all points for this image with their labels
            image_points: Dict[str, List[Dict[str, float]]] = {}

            for points in image.findall(".//points"):
                label = points.get("label")
                coords = points.get("points")

                # Parse points coordinates
                if ";" in coords:  # Multiple points (like for 'arc')
                    for point in coords.split(";"):
                        x, y = map(float, point.split(","))
                        # nomalize x, y based on img_width, img_height
                        x = x / img_width
                        y = y / img_height
                        #
                        if label not in image_points:
                            image_points[label] = []
                        image_points[label].append({"x": x, "y": y})
                else:  # Single point
                    x, y = map(float, coords.split(","))
                    x = x / img_width
                    y = y / img_height
                    #
                    if label not in image_points:
                        image_points[label] = []
                    image_points[label].append({"x": x, "y": y})

            self.images.append(
                {
                    "id": img_id,
                    "name": img_name,
                    "width": img_width,
                    "height": img_height,
                }
            )

            self.annotations.append(image_points)

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an image and its keypoints by index.

        Returns:
            image: tensor of shape [3, H, W]

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = os.path.join(self.img_dir, self.images[idx]["name"])
        image = Image.open(img_path)
        # resize image to 960x540
        # image = image.resize((960, 540))

        # Get points for this image: Dict[str, List[Dict[str, float]]]
        points = self.annotations[idx]

        # Apply transforms to the image
        sample = self.transform({"image": image, "data": points})

        image_db = BasketballCourtKeypointsDB("NFHS", sample["image"], sample["data"])
        target, mask = image_db.get_tensor_w_mask()
        self.image_db = image_db
        # image_db.draw_keypoints(show_heatmap=True)

        return (
            sample["image"],
            torch.from_numpy(target).float(),
            torch.from_numpy(mask).float(),
        )


class SoccerNetCalibrationDataset(Dataset):
    def __init__(self, root_dir, split, transform, main_cam_only=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.match_info = json.load(open(root_dir + split + "/match_info.json"))

        self.files = self.get_image_files(rate=1)

        if main_cam_only:
            self.get_main_camera()

    def get_image_files(self, rate=3):
        files = glob.glob(os.path.join(self.root_dir + self.split, "*.jpg"))
        files.sort()
        if rate > 1:
            files = files[::rate]
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = Image.open(img_name)
        data = json.load(open(img_name.split(".")[0] + ".json"))
        data = self.correct_labels(data)

        # convert data to tensor
        sample = self.transform({"image": image, "data": data})

        img_db = KeypointsDB(sample["data"], sample["image"])
        target, mask = img_db.get_tensor_w_mask()
        image = sample["image"]

        return image, torch.from_numpy(target).float(), torch.from_numpy(mask).float()

    def get_main_camera(self):
        self.files = [
            file
            for file in self.files
            if int(self.match_info[file.split("/")[-1]]["ms_time"])
            == int(self.match_info[file.split("/")[-1]]["replay_time"])
        ]

    def correct_labels(self, data):
        if "Goal left post left" in data.keys():
            data["Goal left post left "] = copy.deepcopy(data["Goal left post left"])
            del data["Goal left post left"]

        return data


class WorldCup2014Dataset(Dataset):
    def __init__(self, root_dir, split, transform):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        assert self.split in ["train_val", "test"], f"unknown dataset type {self.split}"

        self.files = glob.glob(os.path.join(self.root_dir + self.split, "*.jpg"))
        self.homographies = glob.glob(
            os.path.join(self.root_dir + self.split, "*.homographyMatrix")
        )
        self.num_samples = len(self.files)

        self.files.sort()
        self.homographies.sort()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.get_image_by_index(idx)
        homography = self.get_homography_by_index(idx)
        img_db = KeypointsWCDB(image, homography, (960, 540))
        target, mask = img_db.get_tensor_w_mask()

        sample = self.transform({"image": image, "target": target, "mask": mask})

        return sample["image"], sample["target"], sample["mask"]

    def convert_homography_WC14GT_to_SN(self, H):
        T = np.eye(3)
        # T[0, -1] = -115 / 2
        # T[1, -1] = -74 / 2
        yard2meter = 0.9144
        S = np.eye(3)
        S[0, 0] = yard2meter
        S[1, 1] = yard2meter
        H_SN = S @ (T @ H)

        return H_SN

    def get_image_by_index(self, index):
        img_file = self.files[index]
        image = Image.open(img_file)
        return image

    def get_homography_by_index(self, index):
        homography_file = self.homographies[index]
        with open(homography_file, "r") as file:
            lines = file.readlines()
            matrix_elements = []
            for line in lines:
                matrix_elements.extend([float(element) for element in line.split()])
        homography = np.array(matrix_elements).reshape((3, 3))
        homography = self.convert_homography_WC14GT_to_SN(homography)
        homography = torch.from_numpy(homography)
        homography = homography / homography[2:3, 2:3]
        return homography


class TSWorldCupDataset(Dataset):
    def __init__(self, root_dir, split, transform):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        assert self.split in ["train", "test"], f"unknown dataset type {self.split}"

        self.files_txt = self.get_txt()

        self.files = self.get_jpg_files()
        self.homographies = self.get_homographies()
        self.num_samples = len(self.files)

        self.files.sort()
        self.homographies.sort()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.get_image_by_index(idx)
        homography = self.get_homography_by_index(idx)
        img_db = KeypointsWCDB(image, homography, (960, 540))
        target, mask = img_db.get_tensor_w_mask()

        sample = self.transform({"image": image, "target": target, "mask": mask})

        return sample["image"], sample["target"], sample["mask"]

    def get_txt(self):
        with open(self.root_dir + self.split + ".txt", "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines

    def get_jpg_files(self):
        all_jpg_files = []
        for dir in self.files_txt:
            full_dir = self.root_dir + "Dataset/80_95/" + dir
            jpg_files = []
            for file in os.listdir(full_dir):
                if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                    jpg_files.append(os.path.join(full_dir, file))

            all_jpg_files.extend(jpg_files)

        return all_jpg_files

    def get_homographies(self):
        all_homographies = []
        for dir in self.files_txt:
            full_dir = self.root_dir + "Annotations/80_95/" + dir
            homographies = []
            for file in os.listdir(full_dir):
                if file.lower().endswith(".npy"):
                    homographies.append(os.path.join(full_dir, file))

            all_homographies.extend(homographies)

        return all_homographies

    def convert_homography_WC14GT_to_SN(self, H):
        T = np.eye(3)
        # T[0, -1] = -115 / 2
        # T[1, -1] = -74 / 2
        yard2meter = 0.9144
        S = np.eye(3)
        S[0, 0] = yard2meter
        S[1, 1] = yard2meter
        H_SN = S @ (T @ H)

        return H_SN

    def get_image_by_index(self, index):
        img_file = self.files[index]
        image = Image.open(img_file)
        return image

    def get_homography_by_index(self, index):
        homography_file = self.homographies[index]
        homography = np.load(homography_file)
        homography = self.convert_homography_WC14GT_to_SN(homography)
        homography = torch.from_numpy(homography)
        homography = homography / homography[2:3, 2:3]
        return homography
