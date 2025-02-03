import torch
from transformers import pipeline as tr_pipeline

from src.tabe51_gen.utils.anno_utils import get_bbox_from_binary_mask


class MonoDepth:
    def __init__(self, checkpoint="depth-anything/Depth-Anything-V2-base-hf", device=torch.device("cuda:0"),
                 offload_from_device=True):
        self.offload_from_device = offload_from_device
        self.device = device
        if self.offload_from_device:
            self.pipe = tr_pipeline("depth-estimation", model=checkpoint)
        else:
            self.pipe = tr_pipeline("depth-estimation", model=checkpoint, device=self.device)

    def run(self, pil_ims):
        if self.offload_from_device:
            self.pipe.device = self.device
            self.pipe.model.to(self.device)
        depth_output = self.pipe(pil_ims, device=self.device)
        if self.offload_from_device:
            self.pipe.device = torch.device("cpu")
            self.pipe.model.to(self.pipe.device)

        return depth_output


def is_item_leaving_frame(all_masks, frame_idx, prev_out_of_frame=False, edge_of_scene_thresh=10):
    if all_masks[frame_idx] is None or all_masks[frame_idx].sum() == 0:
        # We just have to trust the previous out of frame state
        return prev_out_of_frame
    mask_shape = all_masks[frame_idx].shape
    curr_bbox = get_bbox_from_binary_mask(all_masks[frame_idx])
    if (curr_bbox[0] <= 0 or curr_bbox[1] <= 0 or curr_bbox[2] >= (mask_shape[0] - edge_of_scene_thresh) or
            curr_bbox[3] >= (mask_shape[1] - edge_of_scene_thresh)):
        if prev_out_of_frame:
            return True
        if (frame_idx + 1) < len(all_masks):
            next_mask = all_masks[frame_idx + 1]
            if next_mask is None or next_mask.sum() == 0:
                # Have to assume it has gone out of the frame
                return True
            next_bbox = get_bbox_from_binary_mask(all_masks[frame_idx + 1])
            if (next_bbox[0] <= 0 or next_bbox[1] <= 0 or
                    next_bbox[2] >= (mask_shape[0] - edge_of_scene_thresh) or
                    next_bbox[3] >= (mask_shape[1] - edge_of_scene_thresh)):
                return True
            # Might have just touched the edge of the frame for one frame so just assume not

    return False
