# %%
import numpy as np
import pytest
import os
import cv2
import matplotlib.pyplot as plt
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import onnxruntime
from typing import Tuple, List, Dict, Any


def remove_non_court(input_rgb, encoder_ort_session, decoder_ort_session):
    """
    Removes non-court pixels from an input RGB image using the SAM model.
    
    Args:
        input_rgb (np.ndarray): Input RGB image as a numpy array with shape (H, W, 3)
        ort_session: ONNX Runtime session with loaded SAM model
        
    Returns:
        np.ndarray: RGB image with non-court pixels masked out
    """
    # Check if input is valid
    if input_rgb is None or input_rgb.size == 0:
        print("Error: Invalid input image")
        return None
    
    # 1. Preprocess image for SAM
    preprocessed_image = preprocess_image(input_rgb)
    
    # 2. Get image embedding using SAM encoder
    image_embedding = get_image_embedding(preprocessed_image, encoder_ort_session)
    
    # 3. Generate prompts for court detection
    # We'll use both point prompts and box prompts for better accuracy
    prompts = []
    
    # plt
    plt.imshow(input_rgb)
    plt.show()

    prompts = {"points": np.array([[600, 140]])}
    # 4. Run SAM model with prompts to get masks
    masks = run_sam_prediction(input_rgb.shape, image_embedding, prompts, decoder_ort_session)


    
    # 5. Post-process masks to get the best court mask
    court_mask = post_process_masks(masks, input_rgb.shape[:2])
    
    # 6. Apply the mask to the original image
    masked_image = apply_mask(input_rgb, court_mask)
    
    # Optional: Visualize the result
    # visualize_result(input_rgb, masked_image, court_mask)
    
    return masked_image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for SAM model input.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        
    Returns:
        Preprocessed image ready for SAM encoder
    """
    # Resize if needed (SAM works best with input of size 1024x1024)
    target_size = 1024
    h, w = image.shape[:2]
    
    # Calculate resize ratio
    resize_ratio = min(target_size / h, target_size / w)
    new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)
    
    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Pad to square if needed
    padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded_image[:new_h, :new_w, :] = resized_image
    
    # Normalize pixel values to [0, 1]
    normalized_image = padded_image.astype(np.float32) / 255.0
    
    # Convert from RGB to BGR if needed and apply SAM normalization
    # SAM uses different normalization than typical PyTorch models
    pixel_mean = np.array([0.485, 0.456, 0.406])
    pixel_std = np.array([0.229, 0.224, 0.225])
    normalized_image = (normalized_image - pixel_mean) / pixel_std
    
    # plt
    plt.imshow(resized_image)
    plt.title('resized image')
    plt.show()
    # Transpose to (C, H, W) format
    transposed_image = normalized_image.transpose(2, 0, 1)
    
    # Add batch dimension
    batched_image = np.expand_dims(transposed_image, axis=0)
    
    return batched_image


def get_image_embedding(preprocessed_image: np.ndarray, ort_session) -> np.ndarray:
    """
    Get image embedding using the SAM encoder.
    
    Args:
        preprocessed_image: Preprocessed image as numpy array
        ort_session: ONNX Runtime session
        
    Returns:
        Image embedding from SAM encoder
    """
    # Get model input and output names
    
    # Run the model
    embedding = ort_session.run(None, {"images": preprocessed_image.astype(np.float32)})[0]
    
    return embedding

def run_sam_prediction(original_size, embedding: np.ndarray, prompts: Dict[str, Any], ort_session) -> List[np.ndarray]:
    """
    Run SAM prediction with the provided embedding and prompts.
    
    Args:
        embedding: Image embedding from SAM encoder
        prompts: Dictionary containing point and box prompts
        ort_session: ONNX Runtime session
    
    Returns:
        List of predicted masks
    """
    # Prepare inputs for the decoder
    # Format depends on the specific SAM ONNX model being used
    
    # Default values for other parameters
    input_dict = {
        'image_embeddings': embedding,
        'has_mask_input': np.array([0], dtype=np.float32),  # No previous mask
        'mask_input': np.zeros((1, 1, 256, 256), dtype=np.float32),  # Empty mask input
        'orig_im_size': np.array([original_size[1], original_size[0]], dtype=np.float32),
    }
    
    # Add point prompts if available
    if 'points' in prompts:
        points = prompts['points']
        onnx_coords = np.concatenate([points, np.array([[0.0,0.0]])], axis=0)[None, :,:].astype(np.float32)
        onnx_labels = np.concatenate([np.array([1]), np.array([-1])])[None, :].astype(np.float32)
        # import copy
        # coords = copy.deepcopy(onnx_coords).astype(float)
        # coords[..., 0] = coords[..., 0] * 

        input_dict['point_coords'] = onnx_coords
        input_dict['point_labels'] = onnx_labels



    
    # Get output from decoder
    try:
        # Adjust these based on your specific ONNX model's input/output names
        outputs = ort_session.run(None, input_dict)
        
        # Process the output masks
        masks = outputs[0]  # Assuming first output is masks
        scores = outputs[1]  # Assuming second output is scores
        print("second: ", scores)
        
        return masks
    except Exception as e:
        print(f"Error running SAM model: {e}")
        return []


def post_process_masks(masks: List[np.ndarray], original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Post-process SAM masks to get the best court mask.
    
    Args:
        masks: List of predicted masks from SAM
        original_shape: Original image shape (H, W)
        
    Returns:
        Best court mask at original image resolution
    """
    if masks is None or len(masks) == 0:
        # Return empty mask if no predictions
        return np.zeros(original_shape, dtype=np.uint8)
    
    # Select the mask with the highest score (first mask in sorted list)
    best_mask = masks[0].squeeze()

    print("best mask shape is: ", best_mask.shape)
    
    # Resize to original dimensions
    best_mask = best_mask.astype(np.uint8) * 255  # Convert to uint8
    best_mask = cv2.resize(best_mask, (original_shape[1], original_shape[0]))

    # plt
    plt.imshow(best_mask)
    plt.show()
    
    # Threshold to get binary mask
    _, binary_mask = cv2.threshold(best_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Further refine the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    
    return refined_mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to the image to keep only court pixels.
    
    Args:
        image: Original RGB image
        mask: Binary mask where 255 is court and 0 is non-court
        
    Returns:
        Masked RGB image with only court pixels
    """
    # Create a 3-channel mask
    mask_3channel = cv2.merge([mask, mask, mask])
    
    # Apply the mask
    masked_image = cv2.bitwise_and(image, mask_3channel)
    
    # Optional: Set non-court pixels to a specific color (e.g., black)
    # masked_image[mask_3channel == 0] = [0, 0, 0]  # Set to black
    
    return masked_image


def visualize_result(original: np.ndarray, masked: np.ndarray, mask: np.ndarray) -> None:
    """
    Visualize the original image, mask, and masked image
    
    Args:
        original: Original RGB image
        masked: Masked RGB image
        mask: Binary mask
    """
    # Create a figure with 3 subplots
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')
    
    # Mask
    plt.subplot(1, 3, 2)
    plt.title("Court Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    # Masked image
    plt.subplot(1, 3, 3)
    plt.title("Court Only")
    plt.imshow(masked)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
def test_ellipse_fit():
    # %%
    current_dir = os.path.dirname(os.path.abspath(__file__))
    right_img_path = os.path.join(current_dir, "right_img_remapped.png")
    right_img_full = cv2.imread(right_img_path)
    right_img = right_img_full[:1600, :2400]
    resize_ratio = 1
    w, h = (
        int(right_img.shape[0] * resize_ratio),
        int(right_img.shape[1] * resize_ratio),
    )
    print("target w,h is :", w, h)
    right_img = cv2.resize(right_img, (h, w))

    plt.imshow(right_img)
    plt.show()
    # %% mask
    de_onnx_model_path = os.path.join(current_dir, "../vit_l_decoder.onnx")
    en_onnx_model_path = os.path.join(current_dir, "../vit_l_encoder.onnx")

    decoder_ort_session = onnxruntime.InferenceSession(de_onnx_model_path)
    encoder_ort_session = onnxruntime.InferenceSession(en_onnx_model_path)
    right_img = remove_non_court(right_img, encoder_ort_session, decoder_ort_session)

    right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    # run same

    right_edge = cv2.Canny(right_img_gray, 100, 200)
    right_edge = cv2.resize(right_edge, (int(h * 0.4), int(w * 0.4)))
    # right_edge = right_edge[:100, :230]

    plt.imshow(right_edge)
    plt.show()

    # TODO: run hough line, remove all the line whose line length large than 25 pixel
    # Create a copy of the edge image for line removal
    edge_without_lines = right_edge

    # Run Hough line detection
    lines = cv2.HoughLinesP(
        right_edge,
        rho=2,
        theta=np.pi / 180,
        threshold=10,  # Threshold for line detection
        minLineLength=25,  # Min line length as specified in TODO
        maxLineGap=5,  # Max gap between line segments to connect them
    )

    print(f"detected {len(lines)} lines")

    # Create a blank image to draw the lines
    line_mask = np.zeros_like(right_edge)

    # If lines are detected
    line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line length
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > 25:
                # Draw the line on the mask with some thickness to ensure we remove it completely
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                line_count += 1

    print(f"Detected and removed {line_count} lines longer than 25 pixels")

    # Remove lines from edge image using the mask
    edge_without_lines[line_mask > 0] = 0
    # Create figure for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(right_edge, cmap="gray")
    axes[0, 0].set_title("Original Edge Image")
    # Display the line mask and the filtered edge image
    axes[0, 1].imshow(line_mask, cmap="gray")
    axes[0, 1].set_title("Line Mask")

    axes[0, 2].imshow(edge_without_lines, cmap="gray")
    axes[0, 2].set_title("Edge Image with Lines Removed")
    result = hough_ellipse(
        right_edge, accuracy=20, threshold=10, min_size=50, max_size=80
    )
    result.sort(order="accumulator")
    # Create a blank image to draw the ellipses
    ellipse_image = np.zeros_like(right_edge)

    if len(result) > 0:
        # Sort the ellipses by accumulator value (best fits first)
        result = result[::-1]
        print(f"Detected {len(result)} ellipses")
        # Draw the best ellipses
        for i, res in enumerate(result):  # Draw top 5 ellipses
            print("res: ", res)
            yc, xc, a, b, orientation = res[1:6]

            # Draw the ellipse
            try:
                rr, cc = ellipse_perimeter(
                    int(yc), int(xc), int(a), int(b), orientation
                )
                # Check bounds
                valid_indices = (
                    (rr >= 0)
                    & (rr < ellipse_image.shape[0])
                    & (cc >= 0)
                    & (cc < ellipse_image.shape[1])
                )
                if np.any(valid_indices):
                    rr = rr[valid_indices]
                    cc = cc[valid_indices]
                    ellipse_image[rr, cc] = 255
                    # Print ellipse parameters
                    print(
                        f"Ellipse {i + 1}: center=({xc:.1f}, {yc:.1f}), axes=({a:.1f}, {b:.1f}), "
                        f"orientation={orientation:.2f} rad, score={res[0]}"
                    )
            except Exception as e:
                print(f"Error drawing ellipse {i + 1}: {e}")
        # Show the detected ellipses
        axes[1, 1].imshow(ellipse_image, cmap="gray")
        axes[1, 1].set_title("Detected Ellipses")
    else:
        print("No ellipses detected.")
        axes[1, 1].imshow(np.zeros_like(right_edge), cmap="gray")
        axes[1, 1].set_title("No Ellipses Detected")
    print("ellipse fit result: ", result)
    # TODO: draw all ellipse
    plt.show()
    assert False


# %%
