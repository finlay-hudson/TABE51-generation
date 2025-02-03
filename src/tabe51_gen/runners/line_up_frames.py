import json
from pathlib import Path
import shutil
from typing import Optional

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tkinter import messagebox

from src.tabe51_gen.configs.video_config import DataConfigs
from src.tabe51_gen.utils.vis_utils import add_text, add_border


def overlay_images(background: np.ndarray, overlay: np.ndarray, position: tuple=(0, 0), alpha: float=0.5):
    # Ensure overlay fits within the background
    y_offset, x_offset = position
    if y_offset + overlay.shape[0] > background.shape[0] or x_offset + overlay.shape[1] > background.shape[1]:
        raise ValueError("Overlay exceeds background dimensions.")

    # Blend the images
    blended = background.copy()  # Make a copy of the background
    blended[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = (
            (1 - alpha) * background[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]].astype(
        float) + alpha * overlay.astype(float)).astype(np.uint8)

    return blended


def _save(orig_frame_root: Path, ref_im: np.ndarray, backgrounds: list, foregrounds: list[list], bg_start_frame: int,
          fg_start_frames: list[int], out_root: Path, ref_im_frame_num: int, save_name: str):
    out_dir = out_root / save_name
    if out_dir.exists():
        response = messagebox.askyesno("Confirm Overwrite", f"'{out_dir}' already exists. Do you want to overwrite it?")
        if not response:
            raise ValueError(f"Name: {save_name} already has existing dir {out_dir}")
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir()

    Image.fromarray(ref_im).save(out_dir / "ref_im.jpg")

    behind_out = out_dir / f"behind_clip"
    behind_out.mkdir()
    for i, bg_im in enumerate(backgrounds):
        Image.fromarray(bg_im).save(behind_out / f"{bg_start_frame + i}.jpg")

    for i, foreground in enumerate(foregrounds):
        overlay_out = out_dir / f"in_front_clip_{i + 1}"
        overlay_out.mkdir()
        for j, fg_im in enumerate(foreground):
            Image.fromarray(fg_im).save(overlay_out / f"{fg_start_frames[i] + j}.jpg")

    with open(out_dir / "extra_info.json", "w") as f:
        json.dump({"orig_vid": str(orig_frame_root), "bg_start_frame": bg_start_frame,
                   "fg_start_frames": fg_start_frames, "ref_im_frame": ref_im_frame_num}, f, indent=4)

    return save_name


def image_viewer(ref_im: np.ndarray, backgrounds: list, fg_overlays: list[list], bg_start_frame: int,
                 fg_start_frames: list, orig_frame_root: Path, out_root: Optional[Path] = None, ref_im_frame_num=0,
                 save_name: str = "default"):
    index = 0
    print("Flick through images with left and right arrows, if happy with the frame alignment press 's' to save. If "
          "not press 'esc' and change values or in_front_frame_range and behind_frame_range in your video config")
    fg_overlay_lens = [len(overlay) for overlay in fg_overlays]
    while True:
        index = max(index, 0)
        if index >= max(len(backgrounds), *fg_overlay_lens):
            index = max(len(backgrounds), *fg_overlay_lens) - 1
        if index < len(backgrounds):
            back_img = backgrounds[index]
        else:
            back_img = np.zeros_like(backgrounds[0])

        anno_ref_img = add_text(ref_im.copy(), "Ref Img", bg_col=(255, 255, 255), scale=5, thickness=3)
        anno_back_img = add_text(back_img.copy(), f"Back img: {str(bg_start_frame + index)}", bg_col=(255, 255, 255),
                                 scale=5, thickness=3)
        front_imgs = []
        anno_front_imgs = []
        overlap_img = back_img.copy()
        for i, fg_overlay in enumerate(fg_overlays):
            if index < len(fg_overlay):
                front_img = fg_overlay[index]
            else:
                front_img = np.zeros_like(fg_overlay[0])
            front_imgs.append(front_img)
            anno_front_imgs.append(
                add_border(add_text(front_img.copy(), f"Front Img {i + 1}: {str(fg_start_frames[i] + index)}",
                                    bg_col=(255, 255, 255), scale=5, thickness=3), 10, (0, 0, 0)))
            overlap_img = overlay_images(overlap_img, front_img)

        overlap_img = add_text(overlap_img, "Overlap", bg_col=(255, 255, 255), scale=5, thickness=3)
        vis_img = np.concatenate([add_border(anno_ref_img, 10, (0, 0, 0)),
                                  add_border(anno_back_img, 10, (0, 0, 0)),
                                  *anno_front_imgs, add_border(overlap_img, 10, (0, 0, 0))],
                                 axis=1)[..., ::-1]
        # Display the image
        cv2.imshow("Image Viewer", cv2.resize(vis_img, (1920, 1080)))

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # Escape key to exit
            break
        elif key == 81:  # Left arrow key
            index = (index - 1)  # % len(image_files)  # Go to the previous image
        elif key == 83:  # Right arrow key
            index = (index + 1)  # % len(image_files)  # Go to the next image
        elif key == ord('s'):  # Press 's' to save
            if out_root is None:
                raise ValueError("Need an out dir to be able to save")
            save_name = _save(orig_frame_root, ref_im, backgrounds, fg_overlays, bg_start_frame, fg_start_frames,
                              out_root, ref_im_frame_num, save_name)
            print(f"Saved with name {save_name}")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = OmegaConf.structured(DataConfigs())
    video_name = config.data_name
    frame_root = config.data_root_dir / config.downsampled_frame_dir / video_name
    out_root = config.data_root_dir / config.seperated_frames_dir
    out_root.mkdir(exist_ok=True)

    ref_im_frame = np.array(Image.open(frame_root / f"{config.ref_im_frame_num}.jpg"))
    in_front_frames = []
    min_infront_frame_num = 1e9
    assert len(config.in_front_frame_ranges) and config.behind_frame_range
    for in_front_frame_range in config.in_front_frame_ranges:
        in_front_frames.append([np.array(Image.open(frame_root / f"{i}.jpg")) for i in range(*in_front_frame_range)])
    behind_frames = [np.array(Image.open(frame_root / f"{i}.jpg")) for i in range(*config.behind_frame_range)]

    if config.reverse_in_front_frames:
        in_front_frames = in_front_frames[::-1]

    if config.allow_in_front_final_pause:
        for i, in_front_frame in enumerate(in_front_frames):
            if len(in_front_frame) < len(behind_frames):
                in_front_frames[i] += [in_front_frame[-1] for _ in range(len(behind_frames) - len(in_front_frame))]

    image_viewer(ref_im_frame, behind_frames, in_front_frames, config.behind_frame_range[0],
                 [if_fr[0] for if_fr in config.in_front_frame_ranges], frame_root, out_root, config.ref_im_frame_num,
                 config.data_name)
