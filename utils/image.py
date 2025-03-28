import cv2
import numpy as np
from typing import Optional, Tuple, List, Sequence

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders a list of 4 points representing a rectangle into
    top-left, top-right, bottom-right, bottom-left order.

    Args:
        pts (np.ndarray): A numpy array of shape (4, 2) containing the points.

    Returns:
        np.ndarray: The ordered points in shape (4, 2).
    """
    # Ensure input is a NumPy array
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts, dtype="float32")
        
    if pts.shape != (4, 2):
         raise ValueError(f"Input points must have shape (4, 2), but got {pts.shape}")

    rect = np.zeros((4, 2), dtype="float32")

    # Sum: top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference: top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Applies a perspective transform to an image based on 4 points.

    Args:
        image (np.ndarray): The input image (BGR or Grayscale).
        pts (np.ndarray): A numpy array of shape (4, 2) containing the points
                         defining the region to warp. Order doesn't strictly matter
                         as `order_points` will be called.

    Returns:
        Optional[np.ndarray]: The warped image, or None if points are invalid.
    """
    try:
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute the width of the new image (max of bottom and top edge widths)
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute the height of the new image (max of right and left edge heights)
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Ensure valid dimensions
        if maxWidth <= 0 or maxHeight <= 0:
            print("Warning: Invalid dimensions calculated for perspective transform.")
            return None

        # Define the destination points for the warped image (a rectangle)
        dst = np.array([
            [0, 0],                    # Top-left
            [maxWidth - 1, 0],         # Top-right
            [maxWidth - 1, maxHeight - 1], # Bottom-right
            [0, maxHeight - 1]         # Bottom-left
        ], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped

    except Exception as e:
        print(f"Error during four_point_transform: {e}")
        return None


def extract_object_from_mask(
    mask: np.ndarray,
    image: np.ndarray,
    threshold: float = 0.5,
    bilateral_params: Optional[Tuple[int, int, int]] = (11, 17, 17),
    median_ksize: Optional[int] = 5,
    approx_poly_epsilon_factor: float = 0.02,
    min_contour_area: float = 100.0
) -> Optional[np.ndarray]:
    """
    Extracts and perspective-warps an object from an image based on its
    segmentation mask. Assumes the object is roughly quadrilateral.

    Args:
        mask (np.ndarray): The segmentation mask corresponding to the image.
                           Can be float (0.0-1.0) or uint8 (0-255).
        image (np.ndarray): The original image (BGR format expected for output).
        threshold (float): Threshold value if the mask is float type. Defaults to 0.5.
        bilateral_params (Optional[Tuple[int, int, int]]): Parameters (d, sigmaColor, sigmaSpace)
                                                            for cv2.bilateralFilter. If None, filter is skipped.
                                                            Defaults to (11, 17, 17).
        median_ksize (Optional[int]): Kernel size for cv2.medianBlur. Must be odd.
                                      If None, filter is skipped. Defaults to 5.
        approx_poly_epsilon_factor (float): Factor to multiply arcLength by for cv2.approxPolyDP epsilon.
                                           Controls how strictly the contour is simplified. Defaults to 0.02.
        min_contour_area (float): Minimum area (in pixels) for a contour to be considered. Defaults to 100.0.

    Returns:
        Optional[np.ndarray]: The extracted and warped object in RGB format,
                              or None if no suitable object contour is found.
    """
    if mask is None or image is None:
        print("Error: Input mask or image is None.")
        return None
        
    if mask.shape[:2] != image.shape[:2]:
        print(f"Error: Mask shape {mask.shape[:2]} and image shape {image.shape[:2]} must match.")
        return None

    # --- 1. Preprocess Mask ---
    # Handle mask type and thresholding
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        # Assuming mask values are between 0.0 and 1.0
        processed_mask = (mask > threshold).astype(np.uint8) * 255
    elif mask.dtype == np.uint8:
        # Assuming mask values are 0 or 255
        processed_mask = mask.copy()
    else:
        print(f"Warning: Unsupported mask dtype {mask.dtype}. Attempting conversion.")
        try:
            processed_mask = mask.astype(np.uint8)
             # If original was bool, scale it
            if mask.dtype == bool:
                 processed_mask *= 255
        except Exception:
             print("Error: Could not convert mask to uint8.")
             return None
    
    # Ensure mask is single channel
    if len(processed_mask.shape) == 3 and processed_mask.shape[2] == 3:
        processed_mask = cv2.cvtColor(processed_mask, cv2.COLOR_BGR2GRAY)
    elif len(processed_mask.shape) == 3 and processed_mask.shape[2] > 1:
         processed_mask = processed_mask[:, :, 0] # Take the first channel if multi-channel but not 3

    # Apply optional filters
    if bilateral_params is not None:
        processed_mask = cv2.bilateralFilter(processed_mask, *bilateral_params)
    if median_ksize is not None and median_ksize > 1 and median_ksize % 2 == 1:
        processed_mask = cv2.medianBlur(processed_mask, median_ksize)

    # Check if mask is empty after preprocessing
    if cv2.countNonZero(processed_mask) == 0:
        print("Mask is empty after preprocessing.")
        return None

    # --- 2. Find Contours and Filter ---
    # Find contours on the potentially cleaned mask
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (descending) to prioritize larger objects
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_quadrilateral = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            # Since contours are sorted, we can stop early
            break 
            
        # Approximate the contour shape
        peri = cv2.arcLength(cnt, True)
        epsilon = approx_poly_epsilon_factor * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the approximation has 4 vertices (is a quadrilateral)
        if len(approx) == 4:
            # Found a quadrilateral meeting the area criteria.
            # Since contours are sorted by area, this is the largest one so far.
            largest_quadrilateral = approx
            max_area = area # Store its area just in case, though not strictly needed now
            break # Found the largest quadrilateral, no need to check smaller ones

    # --- 3. Apply Transform if Found ---
    if largest_quadrilateral is None:
        print("No suitable quadrilateral contour found.")
        return None
    else:
        # Reshape points for the transform function
        pts = largest_quadrilateral.reshape(4, 2)

        # Apply the perspective transform to the *original image*
        warped = four_point_transform(image, pts)

        if warped is not None:
            # Convert result to RGB before returning (OpenCV uses BGR)
            return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        else:
            print("Perspective transform failed.")
            return None

# Example Usage (requires mask and image loaded as numpy arrays):
# Assuming 'segmentation_mask' is (H, W) or (H, W, 1), float or uint8
# Assuming 'original_image' is (H, W, 3), BGR uint8
#
# warped_object = extract_object_from_mask(segmentation_mask, original_image)
#
# if warped_object is not None:
#     # Display or save warped_object (which is RGB)
#     cv2.imshow("Warped Object (RGB)", cv2.cvtColor(warped_object, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Could not extract object.")