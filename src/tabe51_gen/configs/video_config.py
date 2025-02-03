from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfigs:
    aspect_ratio: Optional[tuple[int, int]] = (16, 9)
    video_names: tuple = ("door_walkthrough_pt1.MOV", "door_walkthrough_pt2.MOV")
    data_name: Optional[str] = "door_walkthrough"
    sampled_fps: float = 15.0
    data_root_dir: Path = Path(__file__).parent.parent.parent.parent / "examples"
    raw_video_dir: str = "videos"
    downsampled_frame_dir: str = "downsampled_frames"
    seperated_frames_dir: str = "seperated_frames"
    reverse_in_front_frames: bool = False  # Set to true if you want the infront frames to play backwards
    # Set to true, if you want the final front frame to pause to elongate to length of behind frames, if they are longer
    allow_in_front_final_pause: bool = True
    # These next 2 keys are for line_up_frames.py and should be edited to the (start, stop) frames for the video
    # This can be multiple ranges if you want multiple clips infront
    in_front_frame_ranges: tuple[list[int, int]] = ([250, 312],)
    behind_frame_range: tuple[int, int] = (79, 141)
    ref_im_frame_num: int = 0  # This defaults to zero and currently no need to change it
    sam_checkpoint: Path = Path(__file__).parent.parent.parent.parent / "checkpoints" / "sam2" / "sam2_hiera_large.pt"
    bbox_clipped_frames_dir: str = "bbox_clipped_frames"
    segmented_frames_dir: str = "segmented_frames"
    # Set to true if you want to use mono depth to help the segmentation of the front frames
    use_mono_depth_for_front: bool = False
    # This is the minimum value for the mono depth to be considered a valid depth if use_mono_depth_for_front
    mono_depth_min_val: int = 100
    further_downsample: Optional[int] = None  # Add this if you want to further down sample the frame rate of the video
    composited_frames_dir: str = "composited"
