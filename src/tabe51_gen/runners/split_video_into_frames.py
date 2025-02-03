from omegaconf import OmegaConf
from pathlib import Path

from PIL import Image

from src.tabe51_gen.configs.video_config import DataConfigs
from src.tabe51_gen.utils.video_utils import load_frames_from_video_efficient, crop_frame

if __name__ == '__main__':
    config = OmegaConf.structured(DataConfigs())
    aspect_ratio = config.aspect_ratio
    video_names = config.video_names
    # If there are multiple videos names, the frames will be concatenated together in the output
    out_name = Path(video_names[0]).stem
    if config.data_name:
        out_name = config.data_name
    cnt = 0
    for video_name in video_names:
        out_dir = Path(config.data_root_dir) / config.downsampled_frame_dir / out_name
        out_dir.mkdir(exist_ok=True, parents=True)
        video_fn = Path(config.data_root_dir) / config.raw_video_dir / video_name
        for frame in load_frames_from_video_efficient(video_fn, config.sampled_fps):
            print("Processing frame", cnt)
            if aspect_ratio is not None:
                frame = crop_frame(frame, aspect_ratio[0], aspect_ratio[1])
            Image.fromarray(frame[..., ::-1]).save(out_dir / f"{cnt}.jpg")
            cnt += 1
