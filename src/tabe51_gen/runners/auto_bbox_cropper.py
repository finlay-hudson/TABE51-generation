import json
from omegaconf import OmegaConf
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.tabe51_gen.configs.video_config import DataConfigs
from src.tabe51_gen.utils.anno_utils import add_h_w_perc_to_bbox, get_bbox_from_binary_mask
from src.tabe51_gen.utils.seg_ui_tool import BoundingBoxCropper, get_sam2_predictor, PointClicker, \
    run_through_img_sam, vid_sam_img_mean, vid_sam_img_std
from src.tabe51_gen.utils.video_utils import load_images_from_dir
from src.tabe51_gen.utils.vis_utils import convert_one_to_three_channel


def runner(ds_root, frame_dir, out_clip_name, scene_name, sam_pred_img, sam_pred_video, bbox_clipped_frame_name):
    out_dir = ds_root / bbox_clipped_frame_name / out_clip_name / scene_name
    out_dir.mkdir(exist_ok=True, parents=True)

    vid_sam_frames, loaded_filepaths, img_shape, orig_im_sizes = load_images_from_dir(frame_dir,
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
        out_file_im = out_dir / (im_path.stem + ".png")
        out_file_bbox = out_dir / (im_path.stem + ".json")
        if out_file_im.exists() and out_file_bbox.exists():
            continue
        img = cv2.imread(str(im_path))
        if first_seg_frame < 0:
            print("Press Q if object not yet in frame")
            positive_points, negative_points, mode = PointClicker(img).start()
            if len(positive_points) and mode == "auto":
                sam_mask = run_through_img_sam(sam_pred_img, positive_points, negative_points, img)
                assert sam_pred_video is not None, "Must init vid sam first"
                assert vid_sam_frames is not None, "Need to pass in all the frames for video sam"
                if inference_state is None:
                    inference_state = sam_pred_video.init_state(vid_sam_frames[cnt:], sam_pred_video.image_size,
                                                                sam_pred_video.image_size, sam_pred_video.device)
                    _, _, out_mask_logits = sam_pred_video.add_new_mask(inference_state, 0, 1, sam_mask)
                    vid_seg_masks = [(out_mask_logits[0, 0] > 0.0).cpu().numpy() for _, _, out_mask_logits in
                                     sam_pred_video.propagate_in_video(inference_state)]  # [-1]
                assert len(vid_seg_masks), "No masks generated"
                first_seg_frame = cnt
            elif mode == "manual":
                print("Click and drag to make the custom bounding box, press enter once done")
                cropper = BoundingBoxCropper(im_path)
                _, bounding_box = cropper.run()
                sam_mask = np.zeros_like(img)[..., 0]
                x1, y1 = bounding_box[0]
                x2, y2 = bounding_box[1]
                sam_mask[y1:y2, x1:x2] = 1
            else:
                sam_mask = np.zeros_like(img).astype(np.uint8)
        else:
            sam_mask = np.array(
                Image.fromarray(vid_seg_masks[cnt - first_seg_frame]).resize(orig_im_sizes[cnt - first_seg_frame],
                                                                             Image.NEAREST))

        if sam_mask.any():
            xyxy_bbox = get_bbox_from_binary_mask(sam_mask)

            s_xyxy_bbox = add_h_w_perc_to_bbox(xyxy_bbox, 25, 25)
            s_xyxy_bbox[0] = max(0, s_xyxy_bbox[0])
            s_xyxy_bbox[1] = max(0, s_xyxy_bbox[1])
            s_xyxy_bbox[2] = min(s_xyxy_bbox[2], img.shape[0])
            s_xyxy_bbox[3] = min(s_xyxy_bbox[3], img.shape[1])
            cropped_img = img[s_xyxy_bbox[0]: s_xyxy_bbox[2], s_xyxy_bbox[1]: s_xyxy_bbox[3]]
            correct_format_bbox = [[s_xyxy_bbox[1], s_xyxy_bbox[0]], [s_xyxy_bbox[3], s_xyxy_bbox[2]]]
        else:
            cropped_img = convert_one_to_three_channel(sam_mask) if sam_mask.ndim == 2 else sam_mask
            correct_format_bbox = None

        cv2.imwrite(str(out_file_im), cropped_img.astype(np.uint8))
        with open(str(out_file_bbox), "w") as f:
            json.dump({"bbox": correct_format_bbox}, f)


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
        runner(config.data_root_dir, frame_dir / scene_name, config.data_name, scene_name, sam_pred_img,
               sam_pred_video, config.bbox_clipped_frames_dir)


if __name__ == "__main__":
    main()
