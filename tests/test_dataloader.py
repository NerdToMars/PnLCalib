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
from model.transforms import no_transforms
import cv2
import rerun as rr


def find_camera_pose(world_points, image_points, camera_matrix, dist_coeffs=None):
    """
    Find the rotation and translation between camera and world coordinates.

    Args:
        world_points: List of 3D points in world coordinates (mm)
        image_points: List of 2D points in image coordinates (pixels)
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients, default is None (no distortion)

    Returns:
        rotation_matrix: 3x3 rotation matrix
        rotation_vector: 3x1 rotation vector in Rodrigues form
        translation_vector: 3x1 translation vector
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))

    # Convert world points to numpy array with Z=0 (assuming points are on a plane)
    world_points_array = np.array(
        [[[point[0], point[1], 0] for point in world_points]], dtype=np.float32
    )

    # Convert image points to numpy array
    image_points_array = np.array(
        [[point["x"], point["y"]] for point in image_points], dtype=np.float32
    )

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        world_points_array, image_points_array, camera_matrix, dist_coeffs
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    print("success: ", success)
    print("rotation_vector: ", rotation_vector)
    print("translation_vector: ", translation_vector)

    return rotation_matrix, rotation_vector, translation_vector


def create_court_grid(width=1524, height=834, grid_step=50):
    """
    Create a grid of 3D points representing the basketball court.

    Args:
        width: Court width in mm (1524mm = half-court width)
        height: Court height in mm (834mm = distance from baseline to center)
        grid_step: Step size for the grid in mm

    Returns:
        Nx3 array of 3D points on the court plane (Z=0)
    """
    x_range = np.arange(-width, width + grid_step, grid_step)
    y_range = np.arange(-height, height + grid_step, grid_step)

    grid_points = []
    for x in x_range:
        for y in y_range:
            grid_points.append([x, y, 0])

    return np.array(grid_points, dtype=np.float32)


def visualize_with_rerun(
    court_points_3d,
    colors=None,
    camera_position=None,
    camera_rotation=None,
    camera_matrix=None,
):
    """
    Visualize the 3D point cloud using Rerun.

    Args:
        court_points_3d: Nx3 array of 3D points
        colors: Nx3 array of RGB colors for each point
        camera_position: Camera position in world coordinates
        camera_rotation: Camera rotation matrix
    """

    # Log the court points with colors
    if colors is not None:
        rr.log("world/court", rr.Points3D(court_points_3d, colors=colors))
    else:
        rr.log("world/court", rr.Points3D(court_points_3d))

    # Log camera position and orientation if available
    if camera_position is not None and camera_rotation is not None:
        # Convert rotation matrix to quaternion for Rerun
        import scipy.spatial.transform as transform

        rotation = transform.Rotation.from_matrix(camera_rotation)
        quaternion = rotation.as_quat()  # [x, y, z, w] format

        # Log camera transform using Rerun's expected format
        print("camera_position: ", camera_position)
        print("camera_rotation: ", camera_rotation)
        print("quaternion: ", quaternion)

        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=camera_position.flatten(),
                rotation=quaternion,  # Rerun expects quaternion in [x, y, z, w] format
            ),
        )

        # Add a visual indicator at the camera position
        camera_pos = camera_position.flatten()
        rr.log(
            "world/camera_position",
            rr.Points3D(
                positions=np.array([camera_pos]),
                colors=np.array([[255, 0, 0]]),
                sizes=np.array([20.0]),
            ),
        )

        # Also log the camera as a separate entity with transformation
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=camera_pos,
                rotation=quaternion,
                scale=np.array([1.0, 1.0, 1.0]),
            ),
        )

        # Add a camera frustum visualization
        if camera_matrix is not None:
            focal_scaled = (
                camera_matrix[0, 0] / 20.0
            )  # Scale down for better visualization
            rr.log(
                "world/camera/view",
                rr.Pinhole(focal_length=focal_scaled, width=1920, height=1080),
            )
        # rr.log(
        #     "world/camera",
        #     rr.Transform3D(transform=camera_transform),
        # )

        # Optional: Add a simple camera frustum visualization
        rr.log("world/camera", rr.Pinhole(focal_length=1000, width=1920, height=1080))

    # Create a 3D coordinate axes
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[100, 0, 0], [0, 100, 0], [0, 0, 100]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

    # Create a 3D coordinate axes at the origin
    rr.log(
        "world/origin",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[100, 0, 0], [0, 100, 0], [0, 0, 100]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

    print("Running Rerun viewer. Close the window to continue.")


def visualize_with_matplotlib(court_points_3d, colors=None, camera_position=None):
    """
    Alternative visualization using matplotlib (for environments without Rerun).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot court points
    if colors is not None:
        # Normalize colors to 0-1 range if they're in 0-255
        if np.max(colors) > 1:
            colors = colors / 255.0

        ax.scatter(
            court_points_3d[:, 0],
            court_points_3d[:, 1],
            court_points_3d[:, 2],
            c=colors,
            s=5,
            alpha=0.8,
        )
    else:
        ax.scatter(
            court_points_3d[:, 0],
            court_points_3d[:, 1],
            court_points_3d[:, 2],
            c="blue",
            s=5,
            alpha=0.8,
        )

    # Plot camera position if available
    if camera_position is not None:
        ax.scatter(
            camera_position[0],
            camera_position[1],
            camera_position[2],
            c="red",
            s=100,
            marker="o",
            label="Camera",
        )

    # Set labels and limits
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                court_points_3d[:, 0].max() - court_points_3d[:, 0].min(),
                court_points_3d[:, 1].max() - court_points_3d[:, 1].min(),
                court_points_3d[:, 2].max() - court_points_3d[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (court_points_3d[:, 0].max() + court_points_3d[:, 0].min()) * 0.5
    mid_y = (court_points_3d[:, 1].max() + court_points_3d[:, 1].min()) * 0.5
    mid_z = (court_points_3d[:, 2].max() + court_points_3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.title("Basketball Court 3D Point Cloud")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_basketball_court_dataloader():
    rr.init("Basketball Court 3D Visualization", spawn=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = BasketballCourtDataset(
        xml_file=os.path.join(current_dir, "datasets/annotations.xml"),
        img_dir=os.path.join(current_dir, "datasets/"),
        transform=no_transforms,
    )
    print(len(dataset))
    print(dataset.annotations)
    print(dataset.labels)

    for annotation in dataset.annotations:
        for label, points in annotation.items():
            if label == "arc":
                assert len(points) == 4

    img, heatmap, mask = dataset.__getitem__(1)
    print(img.shape)
    print(heatmap.shape)
    print("mask type: ", type(mask))
    print("mask shape: ", mask.shape)
    print("mask: ", mask)

    # target is (58, 270, 480)
    # max target along the first dimension and plt the image
    max_target = np.max(heatmap.cpu().numpy(), axis=0)
    # plt.imshow(max_target)
    # # plt.show()

    # dataset.image_db.draw_keypoints(show_heatmap=True)

    # get coorepondences
    real_world_points, image_points = dataset.image_db.get_correspondences()
    print(real_world_points)
    print(image_points)

    K_flat = [
        1911.8533121729813,
        -1.2057610757951638,
        1908.9765226798836,
        0.0,
        1909.2967489628422,
        1096.4811450057355,
        0.0,
        0.0,
        1.0,
    ]

    # Reshape into 3x3 matrix
    camera_matrix = np.array(
        [
            [K_flat[0], K_flat[1], K_flat[2]],
            [K_flat[3], K_flat[4], K_flat[5]],
            [K_flat[6], K_flat[7], K_flat[8]],
        ],
        dtype=np.float32,
    )

    # Find camera pose
    rotation_matrix, rotation_vector, translation_vector = find_camera_pose(
        real_world_points, image_points, camera_matrix
    )

    # Print results
    print("Rotation Vector (in Rodrigues form):")
    print(rotation_vector)
    print("\nRotation Matrix:")
    print(rotation_matrix)
    print("\nTranslation Vector (in mm):")
    print(translation_vector)

    # Create a court grid
    court_points_3d = create_court_grid(width=1524, height=834, grid_step=2)
    image = dataset.image_db.image.cpu().numpy()
    image = image.transpose(1, 2, 0)

    if image is None:
        colors = None
    else:
        # Project 3D points to image plane to get colors
        points_3d = court_points_3d.reshape(-1, 1, 3)
        zero_dist = np.zeros((4, 1), dtype=np.float32)
        image_points, _ = cv2.projectPoints(
            points_3d,
            rotation_vector,
            translation_vector,
            camera_matrix,
            zero_dist,
        )
        colors = []
        for point in image_points.reshape(-1, 2):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Get BGR color from image and convert to RGB
                color = image[y, x][::-1]  # Convert BGR to RGB
                colors.append(color)
            else:
                colors.append([128, 128, 128])  # Gray for points outside the image
        colors = np.array(colors)
    # Calculate camera position in world coordinates
    # R_inv * (-t) gives the camera position in world coordinates
    camera_position = -np.dot(rotation_matrix.T, translation_vector)

    # Visualize with Rerun if available, else use matplotlib
    try:
        print("Using Rerun for visualization...")
        visualize_with_rerun(
            court_points_3d, colors, camera_position, rotation_matrix.T, camera_matrix
        )
    except ImportError:
        print("Rerun not found, using matplotlib for visualization...")
        visualize_with_matplotlib(court_points_3d, colors, camera_position)

    plt.imshow(image)

    plt.show()
    assert False


def test_court_keypoints():
    key_points_name, court_keypoints = court_definitions[
        "FIBA"
    ].get_keypoint_world_coords_2D()

    # # plot court keypoints with key points name
    # plt.figure(figsize=(10, 10))
    # for name, point in zip(key_points_name, court_keypoints):
    #     plt.text(point[0], point[1], name, fontsize=12)
    #     plt.plot(point[0], point[1], "ro")
    # # set x,y axis same sacle
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.show()

    for name, point in zip(key_points_name, court_keypoints):
        print(f"{name}: {point}")
