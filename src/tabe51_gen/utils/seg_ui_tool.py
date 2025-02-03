from collections import OrderedDict
import types

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

vid_sam_img_mean = (0.485, 0.456, 0.406)
vid_sam_img_std = (0.229, 0.224, 0.225)


class PointClicker:
    def __init__(self, image: np.ndarray):
        # Initialize parameters and variables
        self.image = image.copy()
        self.positive_points = []
        self.negative_points = []
        self.win_name = "Image"

    def click_event(self, event, x, y, flags, param):
        # Handle mouse clicks: left for positive points, middle for negative points
        if event == cv2.EVENT_LBUTTONDOWN:
            self.positive_points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)  # Green dot for positive
            print(f"Positive point added at: ({x}, {y})")
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.negative_points.append((x, y))
            print(f"Negative point added at: ({x}, {y})")
            cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)  # Red dot for negative
        cv2.imshow(self.win_name, self.image)

    def start(self):
        # Display image and set up user interactions
        cv2.imshow(self.win_name, self.image)
        cv2.setMouseCallback(self.win_name, self.click_event)

        print("Click points; right click for positive point (item of interest), middle click for negative point "
              "(background). Negative points often dont need using but can help")

        print("Once all points are clicked: press 's' to start automatic sam video segmentation or 'm' for individual "
              "image segmentation.")

        # Main loop for capturing key events
        mode = None
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting segmentation tool.")
                break
            elif key == ord('s'):
                print("Starting automatic process...")
                mode = "auto"
                break
            elif key == ord('m'):
                print("Manual process...")
                mode = "manual"
                break

        cv2.destroyAllWindows()

        return self.positive_points, self.negative_points, mode


@torch.inference_mode()
def init_state(
        self,
        images,
        video_height,
        video_width,
        compute_device,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
):
    """Initialize an inference state."""
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Warm up the visual backbone and cache the image feature on frame 0
    self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def get_sam2_predictor(single_image=False, model_cfg="sam2_hiera_l.yaml", checkpoint="sam2_hiera_large.pt"):
    if single_image:
        return SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    pred = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")
    # Monkey patch the new init state method to allow for frames to just be fed to the method
    pred.init_state = types.MethodType(init_state, pred)

    return pred


def run_through_img_sam(sam_pred, positive_points: list, negative_points: list, img: np.ndarray):
    # Convert points to NumPy array
    positive_points_np = np.array(positive_points)
    negative_points_np = np.array(negative_points)

    # Prepare image for SAM model
    sam_pred.set_image(img)

    # Create input labels for SAM: 1 for positive, 0 for negative
    input_labels = np.array([1] * len(positive_points_np) + [0] * len(negative_points_np))

    # Concatenate positive and negative points
    if len(negative_points_np):
        input_points = np.concatenate([positive_points_np, negative_points_np], axis=0)
    else:
        input_points = positive_points_np

    # Perform prediction
    masks, scores, logits = sam_pred.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    mask = masks[np.argmax(scores)]

    return mask.astype(np.uint8) * 255


class BoundingBoxCropper:
    def __init__(self, image_path):
        self.img = cv2.imread(str(image_path))
        self.orig_img = self.img.copy()  # To reset the image if needed
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.crop_box = None
        self.cropped_img = None

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.img = self.orig_img.copy()  # Reset to original image to avoid trailing rectangles
                cv2.rectangle(self.img, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Image", self.img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.crop_box = (self.start_point, self.end_point)
            cv2.rectangle(self.img, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.imshow("Image", self.img)

    def crop_to_box(self):
        if self.crop_box:
            x1, y1 = self.crop_box[0]
            x2, y2 = self.crop_box[1]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            x_min, x_max = max(0, x_min), min(self.orig_img.shape[1], x_max)
            y_min, y_max = max(0, y_min), min(self.orig_img.shape[0], y_max)
            self.cropped_img = self.orig_img[y_min:y_max, x_min:x_max]
            return self.cropped_img, ((x_min, y_min), (x_max, y_max))

    def reset(self):
        """Resets the image and bounding box to allow a new selection."""
        self.img = self.orig_img.copy()
        self.start_point = None
        self.end_point = None
        self.crop_box = None
        self.cropped_img = None
        cv2.imshow("Image", self.img)

    def run(self):
        cv2.imshow("Image", self.img)
        cv2.setMouseCallback("Image", self.draw_rectangle)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):  # Press "R" to reset the bounding box
                print("Resetting bounding box...")
                self.reset()
            elif key == 13:  # Press "Enter" to finalize selection
                print("Enter key pressed. Finalizing selection...")
                break
            elif key == 27:  # Press "Esc" to exit without finalizing
                print("Escape key pressed. Exiting without finalizing.")
                self.crop_box = None  # Clear selection if Esc is pressed
                break

        cv2.destroyAllWindows()
        return self.crop_to_box()
