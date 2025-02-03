from dataclasses import dataclass, asdict

from enum import Enum
import json
from typing import Optional

from natsort import natsorted
from omegaconf import OmegaConf
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch

from src.tabe51_gen.configs.video_config import DataConfigs
from src.tabe51_gen.utils.monodepth_utils import MonoDepth, is_item_leaving_frame
from src.tabe51_gen.utils.vis_utils import add_text, overlay_mask_over_image

MONODEPTH_DEVICE = torch.device("cuda:0")


class OcclusionLevel(Enum):
    SEVERE_OCCLUSION = 0
    SLIGHT_OCCLUSION = 1
    NO_OCCLUSION = 2
    NONE = 3


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name  # Convert Enum to its value (string/int)
        return super().default(obj)


@dataclass
class OcclusionInfo:
    level: OcclusionLevel
    amount: Optional[float] = None


def _load_ims(frame_dir, n_infront_clips=1):
    ref_im = Image.open(frame_dir / "ref_im.jpg")
    bg_ims = [Image.open(bg_im) for bg_im in natsorted(list((frame_dir / "behind_clip").glob("*")))]
    fg_ims = []
    for i in range(n_infront_clips):
        fg_ims.append(
            [Image.open(fg_im) for fg_im in natsorted(list((frame_dir / f"in_front_clip_{i + 1}").glob("*")))])

    return ref_im, fg_ims, bg_ims


def process_ims(ims):
    processed_ims = []
    for im in ims:
        if isinstance(im, Image.Image):
            im = np.array(im)
        p_im = im.astype(np.float32) / 255
        if p_im.ndim == 2:
            p_im = p_im[:, :, np.newaxis]
        processed_ims.append(p_im)

    return processed_ims


def blend_multiple_foregrounds(im_bg, im_fgs, alpha_fgs, ref_im):
    """
    Blends multiple foreground images onto a background using their alpha masks.

    Parameters:
    - im_bg: Background image (H, W, C)
    - im_fgs: List of foreground images [(H, W, C), ...]
    - alpha_fgs: List of alpha masks [(H, W), ...]
    - ref_im: Reference image (H, W, C)

    Returns:
    - Blended image (H, W, C)
    """
    composite = im_bg.copy()
    alpha_accumulated = np.zeros_like(alpha_fgs[0])  # Initialize accumulated alpha

    for im_fg, alpha_fg in zip(im_fgs, alpha_fgs):
        # Compute corrected foreground color
        F_i = np.clip((im_fg - (1 - alpha_fg) * ref_im) / np.maximum(alpha_fg, 1e-6), 0, 1)

        # Blend using accumulated alpha (corrected)
        composite = composite * (1 - alpha_fg) + F_i * alpha_fg * (1 - alpha_accumulated)

        # Update accumulated alpha properly
        alpha_accumulated = alpha_accumulated + alpha_fg * (1 - alpha_accumulated)

    return composite


# def combine_ims(ref_im, im_fg, alpha_fg, im_bg):
#     F1 = np.clip((im_fg - (1 - alpha_fg) * ref_im) / np.maximum(alpha_fg, 1e-6), a_min=0, a_max=1)
#
#     return (1 - alpha_fg) * im_bg + alpha_fg * F1


def _load_masks(seg_dir, n_infront_clips=1):
    fg_alphas = []
    for i in range(n_infront_clips):
        name = "in_front_clip" + f"_{i + 1}"
        fg_alphas.append(
            [np.array(Image.open(alpha_fn).convert("L")) for alpha_fn in natsorted(list((seg_dir / name).glob("*")))])
        # for alpha_fn in natsorted(list((seg_dir / name).glob("*"))):
        #     if alpha_fn.stem in fg_as:
        #         fg_as[alpha_fn.stem] += np.array(Image.open(alpha_fn).convert("L"))
        #     else:
        #         fg_as[alpha_fn.stem] = np.array(Image.open(alpha_fn).convert("L"))
        # fg_alphas = np.clip(list(fg_as.values()), 0, 255)
    bg_alphas = [Image.open(alpha_fn).convert("L") for alpha_fn in natsorted(list((seg_dir / "behind_clip").glob("*")))]

    return fg_alphas, bg_alphas


def _get_occlusion_info(all_fg_alphas: np.ndarray, bg_alphas: np.ndarray):
    occl_infos = []
    occl_percs = (all_fg_alphas.astype(bool) & (np.array(bg_alphas).astype(bool))).sum(axis=(1, 2)) / (
        np.array(bg_alphas).astype(bool)).sum(axis=(1, 2))
    item_heading_out_of_frame = True
    for i in range(len(bg_alphas)):
        item_leaving_frame = is_item_leaving_frame(bg_alphas, i, prev_out_of_frame=item_heading_out_of_frame,
                                                   edge_of_scene_thresh=1)
        item_heading_out_of_frame = item_leaving_frame
        if item_leaving_frame:
            occl_infos.append(OcclusionInfo(level=OcclusionLevel.NONE, amount=None))
            continue
        else:
            if occl_percs[i] <= 0.005:
                occl_level = OcclusionLevel.NO_OCCLUSION
            elif 0.0 < occl_percs[i] < 0.5:
                occl_level = OcclusionLevel.SLIGHT_OCCLUSION
            else:
                occl_level = OcclusionLevel.SEVERE_OCCLUSION

            occl_infos.append(OcclusionInfo(level=occl_level, amount=occl_percs[i]))

    return occl_infos


def main(config, frame_dir, alpha_dir, out_dir, n_infront_clips):
    ref_im, fg_ims, bg_ims = _load_ims(frame_dir, n_infront_clips)
    fg_alphas, bg_alphas = _load_masks(alpha_dir, n_infront_clips)

    if config.further_downsample is not None:
        fg_ims = [fg_im[::config.further_downsample] for fg_im in fg_ims]
        bg_ims = bg_ims[::config.further_downsample]
        fg_alphas = [fg_alpha[::config.further_downsample] for fg_alpha in fg_alphas]
        bg_alphas = bg_alphas[::config.further_downsample]

    if config.use_mono_depth_for_front:
        monodepth_pipe = MonoDepth(device=MONODEPTH_DEVICE)
        monodepth_results_per_frames = [monodepth_pipe.run(fg_im) for fg_im in fg_ims]
        for i, monodepth_results in enumerate(monodepth_results_per_frames):
            for j, monodepth_res in enumerate(monodepth_results):
                np_depth = np.array(monodepth_res["depth"])
                np_depth[fg_alphas[i][j] == 0] = 0
                alpha = fg_alphas[i][j]
                alpha[np_depth < config.mono_depth_min_val] = 0
                fg_alphas[i][j] = alpha

    assert len(fg_ims)
    assert len(fg_ims) == len(fg_alphas)

    out_dir_debug_vis = out_dir / "debug_vis"
    out_dir_debug_vis.mkdir(exist_ok=True)
    out_dir_dataset = out_dir / "frames"
    out_dir_dataset.mkdir(exist_ok=True)
    out_dir_gt = out_dir / "gt_masks"
    out_dir_gt.mkdir(exist_ok=True)
    out_dir_vis_masks = out_dir / "visible_masks"
    out_dir_vis_masks.mkdir(exist_ok=True)

    all_fg_alphas = np.bitwise_or.reduce(fg_alphas, axis=0)

    assert len(all_fg_alphas) == len(bg_alphas)

    # Get occlusion levels
    occl_infos = _get_occlusion_info(all_fg_alphas, np.array(bg_alphas))

    # Make the composite images
    anno_occlusions = []
    # for i, (fg_im, fg_alpha, bg_im) in enumerate(zip(fg_ims, fg_alphas, bg_ims)):
    p_fg_ims = [process_ims(fg_im) for fg_im in fg_ims]
    p_fg_alphas = [process_ims(fg_alpha) for fg_alpha in fg_alphas]
    for i, bg_im in enumerate(bg_ims):
        bg_alpha_im = np.array(bg_alphas[i])
        p_ref_im, p_bg_im = process_ims([ref_im, bg_im])
        comb_im = blend_multiple_foregrounds(p_bg_im, [ims[i] for ims in p_fg_ims], [alp[i] for alp in p_fg_alphas],
                                             p_ref_im)
        comb_im = (comb_im * 255).astype(np.uint8)
        if bg_alpha_im.ndim == 3:
            bg_alpha_im = bg_alpha_im[..., 0]

        vis_mask = (bg_alpha_im & (all_fg_alphas[i] == 0) * 255).astype(np.uint8)
        Image.fromarray(comb_im).save(out_dir_dataset / f"{str(i).zfill(5)}.jpg")
        Image.fromarray(vis_mask).save(out_dir_vis_masks / f"{str(i).zfill(5)}.png")
        Image.fromarray(bg_alpha_im).save(out_dir_gt / f"{str(i).zfill(5)}.png")
        anno_occlusions.append(asdict(occl_infos[i]))

        overlay_mask = overlay_mask_over_image(comb_im, bg_alpha_im)
        add_text(overlay_mask, f"Occl: {occl_infos[i].level.name} - {occl_infos[i].amount:.3f}", (10, 10),
                 scale=3, thickness=2, bg_col=(255, 255, 255))
        Image.fromarray(overlay_mask).save(out_dir_debug_vis / f"{i}.jpg")

    with open(out_dir / "annos.json", "w") as f:
        json.dump({"occlusion": anno_occlusions}, f, cls=EnumEncoder, indent=4)


if __name__ == "__main__":
    config = OmegaConf.structured(DataConfigs())
    for clip_dir in tqdm(list((config.data_root_dir / config.segmented_frames_dir).glob("*"))):
        clip_name = clip_dir.stem
        frame_dir = config.data_root_dir / config.seperated_frames_dir / clip_name.split("-")[0]
        alpha_dir = config.data_root_dir / config.segmented_frames_dir / clip_name
        out_dir = config.data_root_dir / config.composited_frames_dir / clip_name
        out_dir.mkdir(exist_ok=True, parents=True)
        main(config, frame_dir, alpha_dir, out_dir, n_infront_clips=len(list(alpha_dir.glob("in_front_clip*"))))
