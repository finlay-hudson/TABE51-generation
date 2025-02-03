import json
from pathlib import Path
from omegaconf import OmegaConf

import cv2
import numpy as np
from PIL import Image

from src.tabe51_gen.configs.video_config import DataConfigs
from src.tabe51_gen.utils.seg_ui_tool import PointClicker, run_through_img_sam, get_sam2_predictor, vid_sam_img_mean, \
    vid_sam_img_std
from src.tabe51_gen.utils.video_utils import load_images_from_dir


def runner(config, frame_dir, clip_name, scene_name, sam_pred_img, sam_pred_video, force=False):
    cropped_frame_dir = config.data_root_dir / config.bbox_clipped_frames_dir / clip_name / scene_name
    out_dir = config.data_root_dir / config.segmented_frames_dir / clip_name / scene_name
    out_dir.mkdir(exist_ok=True, parents=True)

    vid_sam_frames, loaded_filepaths, img_shape, orig_im_sizes = load_images_from_dir(cropped_frame_dir,
                                                                                      exclude_patterns=[".json"],
                                                                                      resize_to=(
                                                                                          sam_pred_video.image_size,
                                                                                          sam_pred_video.image_size),
                                                                                      img_mean=vid_sam_img_mean,
                                                                                      img_std=vid_sam_img_std,
                                                                                      device=sam_pred_video.device,
                                                                                      out_max=1)

    inference_state = None
    first_seg_frame = -1
    vid_seg_masks = []
    for cnt, im_path in enumerate(loaded_filepaths):
        if (out_dir / f"{im_path.stem}.png").exists() and not force:
            continue
        print(im_path)
        full_size_img = cv2.imread(str(frame_dir / (im_path.stem + ".jpg")))
        img = cv2.imread(str(im_path))
        bbox_crop_info = im_path.with_suffix(".json")
        with open(bbox_crop_info, "r") as f:
            crop_bbox = json.load(f)["bbox"]
        if crop_bbox is None:
            # There is nothing to segment so just output an empty mask
            Image.fromarray(np.zeros_like(img)).save(out_dir / f"{im_path.stem}.png")
            continue
        if first_seg_frame < 0:
            print("Press Q if object not yet in frame")
            positive_points, negative_points, mode = PointClicker(img).start()
            if len(positive_points):
                sam_mask = run_through_img_sam(sam_pred_img, positive_points, negative_points, img)
                if mode == "auto":
                    assert sam_pred_video is not None, "Must init vid sam first"
                    assert vid_sam_frames is not None, "Need to pass in all the frames for video sam"
                    if inference_state is None:
                        inference_state = sam_pred_video.init_state(vid_sam_frames[cnt:], sam_pred_video.image_size,
                                                                    sam_pred_video.image_size,
                                                                    sam_pred_video.device)
                        _, _, out_mask_logits = sam_pred_video.add_new_mask(inference_state, 0, 1, sam_mask)
                        vid_seg_masks = [(out_mask_logits[0, 0] > 0.0).cpu().numpy() for _, _, out_mask_logits in
                                         sam_pred_video.propagate_in_video(inference_state)]  # [-1]
                    assert len(vid_seg_masks), "No masks generated"
                    first_seg_frame = cnt
            else:
                sam_mask = np.zeros_like(img)[..., 0].astype(np.uint8)
        else:
            sam_mask = (vid_seg_masks[cnt - first_seg_frame] * 255).astype(np.uint8)
        orig_size_mask = np.array(Image.fromarray(sam_mask).resize(orig_im_sizes[cnt], Image.NEAREST))

        x1, y1 = crop_bbox[0]
        x2, y2 = crop_bbox[1]
        full_img_mask = np.zeros_like(full_size_img)[..., 0]
        full_img_mask[y1:y2, x1:x2] = orig_size_mask
        Image.fromarray(full_img_mask).save(out_dir / f"{im_path.stem}.png")


def main():
    config = OmegaConf.structured(DataConfigs())
    sam_pred_video = get_sam2_predictor(single_image=False, checkpoint=config.sam_checkpoint)
    sam_pred_img = get_sam2_predictor(single_image=True, checkpoint=config.sam_checkpoint)

    scene_names = []
    for scene_type in ["behind_clip", "in_front_clip"]:
        frame_dir = config.data_root_dir / config.seperated_frames_dir / config.data_name
        if scene_type == "in_front_clip":
            num_infront = len(list(frame_dir.glob("in_front_clip*")))
            for n_infront in range(num_infront):
                scene_names.append(scene_type + f"_{n_infront + 1}")
        else:
            scene_names.append(scene_type)

    for scene_name in scene_names:
        print(f"Running for: {scene_name}")
        runner(config, frame_dir / scene_name, config.data_name, scene_name, sam_pred_img,
               sam_pred_video, config.bbox_clipped_frames_dir)


if __name__ == "__main__":
    main()
