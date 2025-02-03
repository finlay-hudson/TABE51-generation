import cv2
import numpy as np
from natsort import natsorted
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor


def load_images_from_dir(img_dir, exclude_patterns=None, resize_to=None, device=None, out_max=255, img_mean=None,
                         img_std=None):
    """
    :param img_dir (str).
    :param exclude_patterns (Optional[list]) Patterns of filename wishing to be excluded from loading
    :param resize_to (Optional[list]) Size to resize images to
    :param device (Optional[torch.device]) Device to put images on
    :param out_max (int) max value for pixel on output, 255 would mean uint8 dtype and 1 would mean float32 dtype
    :param img_mean (Optional(list)) mean norm values
    :param img_std (Optional(list)) mean std values
    :return frames (T, H, W, 3) array with uint8 [0, 255].
    """
    if out_max not in [1, 255]:
        raise NotImplementedError(f"Do not have implementation for max value of {out_max}")
    img_dir = Path(img_dir)
    img_files = list(img_dir.glob("*"))
    remove_files = []
    if exclude_patterns is not None:
        if not (isinstance(exclude_patterns, list)):
            exclude_patterns = [exclude_patterns]
        for img_fp in img_files:
            for pattern in exclude_patterns:
                if pattern in img_fp.name:
                    remove_files.append(img_fp)
                    break

    img_files = natsorted(img_files)
    frames = []
    loaded_filepaths = []
    unique_img_shapes = set()
    orig_im_shapes = []
    for fp in img_files:
        if fp in remove_files:
            continue
        img_pil = Image.open(fp)
        orig_im_shapes.append(img_pil.size)
        if resize_to is not None:
            img_pil = img_pil.resize(resize_to)
        img = pil_to_tensor(img_pil)

        frames.append(img)
        unique_img_shapes.add(img_pil.size)
        loaded_filepaths.append(fp)

    if not len(frames):
        raise ValueError(f"No frames found in {img_dir}")
    if not len(unique_img_shapes) == 1:
        raise ValueError("More than one unique shape found in directory, reshaping needed")
    frames = torch.stack(frames)
    if out_max == 1:
        frames = frames / 255.0

    if img_mean is not None:
        # normalize by mean
        frames -= torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    if img_std is not None:
        # normalize by std
        frames /= torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if device is not None:
        frames = frames.to(device)

    return frames, loaded_filepaths, list(unique_img_shapes)[0], orig_im_shapes


def load_frames_from_video(video_fn):
    assert Path(video_fn).exists(), f"No video file at {video_fn}"
    cap = cv2.VideoCapture(str(video_fn))

    # List to hold all frames
    frames = []

    # Loop through the video and read each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        # Append the frame to the list
        frames.append(frame)

    # Release the video capture object
    cap.release()

    return frames


def load_frames_from_video_efficient(video_fn: Path | str, target_fps: float = 10.0):
    """
    Generator function to load frames from a video file at a specified frame rate.

    Args:
        video_fn (pathlib.Path | str): Path to the video file.
        target_fps (float): Desired frames per second.

    Yields:
        frame: The next frame from the video at the specified FPS.
    """
    assert Path(video_fn).exists(), f"No video file at {video_fn}"
    cap = cv2.VideoCapture(str(video_fn))

    original_fps = cap.get(cv2.CAP_PROP_FPS)  # Get original FPS of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("ORIG FRAME COUNT", frame_count)
    time_per_frame_original = 1 / original_fps
    time_per_frame_target = 1 / target_fps

    try:
        current_frame_index = 0
        current_target_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available

            # Calculate the actual time of the current frame in the original video
            current_frame_time = current_frame_index * time_per_frame_original

            # Check if the current frame time is closest to the next target frame time
            if current_frame_time >= current_target_time:
                yield frame
                current_target_time += time_per_frame_target  # Move to the next target time

            current_frame_index += 1

    finally:
        cap.release()  # Ensure the video capture object is released


def crop_frame(frame: np.ndarray, target_ratio_width: int, target_ratio_height: int):
    original_height, original_width = frame.shape[:2]
    original_ratio = original_width / original_height
    target_ratio = target_ratio_width / target_ratio_height

    # Check if the original aspect ratio is wider or taller than the target
    if original_ratio > target_ratio:
        # Original is wider than target, crop width
        new_width = int(original_height * target_ratio)
        x_offset = (original_width - new_width) // 2
        cropped_frame = frame[:, x_offset:x_offset + new_width]
    else:
        # Original is taller than target, crop height
        new_height = int(original_width / target_ratio)
        y_offset = (original_height - new_height) // 2
        cropped_frame = frame[y_offset:y_offset + new_height, :]

    return cropped_frame


def crop_frames(frames: list, target_ratio_width: int, target_ratio_height: int):
    return [crop_frame(frame, target_ratio_width, target_ratio_height) for frame in frames]


def get_video_fps(video_fn):
    assert Path(video_fn).exists(), f"No video file at {video_fn}"

    return cv2.VideoCapture(str(video_fn)).get(cv2.CAP_PROP_FPS)


def down_sample_frame_fps(frames: list, original_fps: float, target_fps: float = 10.0):
    # return frames[::int(original_fps / target_fps)]
    # Calculate the time per frame for original and target fps
    time_per_frame_original = 1 / original_fps
    time_per_frame_target = 1 / target_fps

    # List to store downsampled frames
    downsampled_frames = []

    # Iterate through the original frames using the target time interval
    current_target_time = 0  # Starting time for the first target frame
    for i in range(len(frames)):
        # Calculate the actual time of the current frame in the original video
        current_frame_time = i * time_per_frame_original

        # Check if the current frame time is closest to the next target frame time
        if current_frame_time >= current_target_time:
            downsampled_frames.append(frames[i])  # Add the frame
            current_target_time += time_per_frame_target  # Move to the next target time

    # Convert the list of downsampled frames to a NumPy array
    return downsampled_frames
