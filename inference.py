import copy
import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Polygon

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict


lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
                [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                [[0., 37.66, -2.44], [0., 30.34, -2.44]],
                [[0., 37.66, 0.], [0., 37.66, -2.44]],
                [[0., 30.34, 0.], [0., 30.34, -2.44]],
                [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                [[105., 30.34, 0.], [105., 30.34, -2.44]],
                [[105., 37.66, 0.], [105., 37.66, -2.44]],
                [[52.5, 0., 0.], [52.5, 68, 0.]],
                [[0., 68., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [0., 68., 0.]],
                [[105., 0., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [105., 0., 0.]],
                [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
                [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                [[99.5, 24.84, 0.], [105., 24.84, 0.]]]


def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)

    return P


def inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("rgb_frame shape:", rgb_frame.shape)
    frame = Image.fromarray(rgb_frame)

    frame = f.to_tensor(frame).float().unsqueeze(0)
    _, _, h_original, w_original = frame.size()
    origin_frame = frame if frame.size()[-1] == 960 else transform2(frame)
    frame = origin_frame.to(device)
    b, c, h, w = frame.size()

    # resize rgb_frame to frame size
    rgb_frame = cv2.resize(rgb_frame, (w, h))

    with torch.no_grad():
        heatmaps = model(frame)
        heatmaps_l = model_l(frame)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])


    # draw line on frame to visualize
    visualize_line(rgb_frame, line_coords, True)

    visualize_keypoints(rgb_frame, kp_coords, True)

    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

    cam.update(kp_dict, lines_dict)
    final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)

    return final_params_dict


def project(frame, P):

    for line in lines_coords:
        w1 = line[0]
        w2 = line[1]
        i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
        i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
        i1 /= i1[-1]
        i2 /= i2[-1]
        frame = cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), (255, 0, 0), 3)

    r = 9.15
    pts1, pts2, pts3 = [], [], []
    base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(37, 143, 50):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts1.append([ipos[0], ipos[1]])

    base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(217, 323, 200):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts2.append([ipos[0], ipos[1]])

    base_pos = np.array([0, 0, 0., 0.])
    for ang in np.linspace(0, 360, 500):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts3.append([ipos[0], ipos[1]])

    XEllipse1 = np.array(pts1, np.int32)
    XEllipse2 = np.array(pts2, np.int32)
    XEllipse3 = np.array(pts3, np.int32)
    frame = cv2.polylines(frame, [XEllipse1], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse2], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse3], False, (255, 0, 0), 3)

    return frame


def process_input(input_path, input_type, model_kp, model_line, kp_threshold, line_threshold, pnl_refine,
                  save_path, display):

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)

    if input_type == 'video':
        cap = cv2.VideoCapture(input_path)
        if save_path != "":
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        pbar = tqdm(total=total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            final_params_dict = inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine)
            if final_params_dict is not None:
                P = projection_from_cam_params(final_params_dict)
                projected_frame = project(frame, P)
            else:
                projected_frame = frame
                
            if save_path != "":
                out.write(projected_frame)
    
            if display:
                cv2.imshow('Projected Frame', projected_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            pbar.update(1)

        cap.release()
        if save_path != "":
            out.release()
        cv2.destroyAllWindows()

    elif input_type == 'image':
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Unable to read the image {input_path}")
            return

        final_params_dict = inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine)
        if final_params_dict is not None:
            P = projection_from_cam_params(final_params_dict)
            projected_frame = project(frame, P)
        else:
            projected_frame = frame

        if save_path != "":
            cv2.imwrite(save_path, projected_frame)
        else:
            plt.imshow(cv2.cvtColor(projected_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


def visualize_line(input_frame, line_coords, visualize=False):
    """
    Visualize detected line endpoints as dots on the RGB frame
    Args:
        frame: RGB frame to draw on
        line_coords: Coordinates of detected lines: List[List[tensor]]
            Structure: [batch][line_index][2, 3] where each line has 
            [[x1, y1, conf1], [x2, y2, conf2]] format
        visualize: Boolean flag to enable/disable visualization
    Returns:
        frame: Frame with visualized dots if visualize=True, otherwise original frame
    """
    if not visualize:
        return input_frame
    
    height, width = input_frame.shape[:2]
    frame = input_frame.copy()
    print("line_coords shape:", np.array(line_coords).shape)
    
    # Assuming line_coords is a batch of line detections
    for batch_idx, batch in enumerate(line_coords):
        print(f"Processing batch {batch_idx}, contains {len(batch)} lines")
        for line_idx, line in enumerate(batch):
            if line is not None:
                # Convert tensor to numpy array
                line = line.cpu().numpy()
                print(f"Line {line_idx} coords:", line)
                
                # Extract start and end points
                # line shape is [2, 3] where each row is [x, y, confidence]
                x1, y1 = int(line[0][0]), int(line[0][1])  # Start point
                x2, y2 = int(line[1][0]), int(line[1][1])  # End point
                
                # Draw dots with labels
                frame = cv2.circle(frame, (x1, y1), radius=5, color=(255, 0, 0), thickness=-1)  # Start point (red)
                frame = cv2.circle(frame, (x2, y2), radius=5, color=(0, 0, 255), thickness=-1)  # End point (blue)
                
                # Add text labels
                frame = cv2.putText(frame, f'{line_idx}s', (x1+5, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 0, 0), 1)
                frame = cv2.putText(frame, f'{line_idx}e', (x2+5, y2+5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 1)
    
    cv2.imwrite("lines.png", frame)
    return frame


def visualize_keypoints(input_frame, kp_coords, visualize=False):
    """
    Visualize detected keypoints as dots on the RGB frame
    Args:
        input_frame: RGB frame to draw on
        kp_coords: Coordinates of detected keypoints
        visualize: Boolean flag to enable/disable visualization
    Returns:
        frame: Frame with visualized keypoints if visualize=True, otherwise original frame
    """
    if not visualize:
        return input_frame
    
    height, width = input_frame.shape[:2]
    frame = copy.deepcopy(input_frame)
    print("frame data type:", type(frame))

    # image tensor to numpy
    # frame = frame.cpu().numpy()
    print("frame shape:", frame.shape)

    # frame = frame.astype(np.uint8)
    
    try:
        print("kp_coords structure:", type(kp_coords), len(kp_coords))
        
        # Assuming kp_coords is a batch of keypoint detections
        for batch_idx, batch in enumerate(kp_coords):
            print(f"Processing keypoint batch {batch_idx}, contains {len(batch)} keypoints")
            for kp_idx, kp in enumerate(batch):
                if kp is not None:
                    # Convert tensor to numpy array if it's a tensor
                    if isinstance(kp, torch.Tensor):
                        kp = kp.cpu().numpy()
                    
                    print(f"Keypoint {kp_idx} coords shape:", kp.shape)
                    print(f"Keypoint {kp_idx} coords:", kp)
                    
                    # Handle different keypoint formats
                    if len(kp.shape) == 1:  # Simple [x, y, conf] format
                        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                    elif len(kp.shape) == 2:  # More complex format like [[x, y, conf]]
                        x, y, conf = int(kp[0][0]), int(kp[0][1]), kp[0][2]
                    else:
                        print(f"Unsupported keypoint format for keypoint {kp_idx}")
                        continue
                    
                    # Only draw keypoints with sufficient confidence
                    if conf > 0.01:  # Adjust threshold as needed
                        print(f"Drawing keypoint {kp_idx} at ({x}, {y}) with confidence {conf}")
                        # Draw keypoint as a green circle
                        cv2.circle(frame, (x, y), radius=7, color=(0, 255, 0), thickness=-1)
                        
                        # Add keypoint index as label
                        cv2.putText(frame, f'{kp_idx}', (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (255, 255, 255), 2)
                        # cv2.imwrite(f"keypoints{kp_idx}.png", frame)
    except Exception as e:
        print(f"Error in visualize_keypoints: {e}")
        import traceback
        traceback.print_exc()
    cv2.imwrite("keypoints.png", frame)
    # plt.figure(figsize=(12, 8))
    # plt.imshow(frame)
    # plt.axis('off')
    # plt.title("Detected Keypoints")
    # plt.show()
    return frame


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video or image and plot lines on each frame.")
    parser.add_argument("--weights_kp", type=str, help="Path to the model for keypoint inference.")
    parser.add_argument("--weights_line", type=str, help="Path to the model for line projection.")
    parser.add_argument("--kp_threshold", type=float, default=0.3434, help="Threshold for keypoint detection.")
    parser.add_argument("--line_threshold", type=float, default=0.7867, help="Threshold for line detection.")
    parser.add_argument("--pnl_refine", action="store_true", help="Enable PnL refinement module.")
    parser.add_argument("--device", type=str, default="cpu", help="CPU or CUDA device index")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or image file.")
    parser.add_argument("--input_type", type=str, choices=['video', 'image'], required=True,
                        help="Type of input: 'video' or 'image'.")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the processed video.")
    parser.add_argument("--display", action="store_true", help="Enable real-time display.")
    args = parser.parse_args()


    input_path = args.input_path
    input_type = args.input_type
    model_kp = args.weights_kp
    model_line = args.weights_line
    pnl_refine = args.pnl_refine
    save_path = args.save_path
    device = args.device
    display = args.display and input_type == 'video'
    kp_threshold = args.kp_threshold
    line_threshold = args.line_threshold

    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))

    loaded_state = torch.load(args.weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(args.weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    transform2 = T.Resize((540, 960))

    process_input(input_path, input_type, model_kp, model_line, kp_threshold, line_threshold, pnl_refine,
                  save_path, display)
