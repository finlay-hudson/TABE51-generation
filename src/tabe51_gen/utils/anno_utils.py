import numpy as np


def get_bbox_from_binary_mask(binary_mask):
    # Find the non-zero coordinates
    non_zero_coords = np.argwhere(binary_mask)

    # If there are no non-zero pixels, return None
    if non_zero_coords.size == 0:
        return None

    # Get the bounding box coordinates
    min_row, min_col = np.min(non_zero_coords, axis=0)
    max_row, max_col = np.max(non_zero_coords, axis=0)

    # Return the bounding box as (min_row, min_col, max_row, max_col)
    return np.array([min_row, min_col, max_row, max_col]).astype(int)


def add_h_w_perc_to_bbox(bbox_xyxy, h_perc=10, w_perc=10):
    extended_bbox_xyxy = np.array(bbox_xyxy).copy()
    control_bbox_w = bbox_xyxy[2] - bbox_xyxy[0]
    control_bbox_h = bbox_xyxy[3] - bbox_xyxy[1]

    extended_bbox_xyxy[0] -= control_bbox_w * (w_perc / 100)
    extended_bbox_xyxy[1] -= control_bbox_h * (h_perc / 100)
    extended_bbox_xyxy[2] += control_bbox_w * (w_perc / 100)
    extended_bbox_xyxy[3] += control_bbox_h * (h_perc / 100)

    return extended_bbox_xyxy.tolist()
