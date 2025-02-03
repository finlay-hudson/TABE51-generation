import cv2
import numpy as np


def overlay_mask_over_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    # Ensure mask and image have the same dimensions
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask dimensions do not match.")
    if not mask.any():
        return image
    if mask.max() > 1:
        mask = mask // 255
    overlay_mask = np.zeros_like(image)
    overlay_mask[mask == 1] = color

    anno_img = image.copy()
    anno_img[mask == 1] = cv2.addWeighted(image[mask == 1], 1 - alpha, overlay_mask[mask == 1], alpha, 0)

    return anno_img


def add_text(frame, txt, pos=(0, 0), font=cv2.FONT_HERSHEY_PLAIN, scale=1.0, thickness=1, txt_col=(0, 0, 0),
             bg_col=None):
    was_float = False
    if frame.dtype.kind == 'f':
        was_float = True
        frame = (frame * 255).astype(np.uint8)
    x, y = pos
    text_size, _ = cv2.getTextSize(txt, font, scale, thickness)
    text_w, text_h = text_size
    if bg_col is not None:
        cv2.rectangle(frame, (pos[0] - 5, pos[1] - 5), (x + text_w + 5, y + text_h + 5), bg_col, -1)
    cv2.putText(frame, txt, (x, int(y + text_h + 1)), font, scale, txt_col, thickness)

    if was_float:
        return frame / 255.0

    return frame


def optimal_grid(num_images):
    cols = round(np.sqrt(num_images))
    rows = np.ceil(num_images / cols)

    return int(rows), int(cols)


def create_mosaic(images, rows=None, cols=None):
    if rows is None or cols is None:
        rows, cols = optimal_grid(len(images))
    # Assuming all images are of the same size
    img_height, img_width, img_channels = images[0].shape

    # Create a blank canvas
    mosaic = np.zeros((rows * img_height, cols * img_width, img_channels), dtype=images[0].dtype)

    # Place each image in the mosaic
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        mosaic[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width, :] = img

    return mosaic


def add_border(f, border_ceof=3, col=(255, 255, 255)):
    return cv2.copyMakeBorder(f.copy(), border_ceof, border_ceof, border_ceof, border_ceof, cv2.BORDER_CONSTANT,
                              value=col)


def convert_one_to_three_channel(single_channel_image):
    if single_channel_image.ndim > 2:
        return single_channel_image
    return np.stack((single_channel_image,) * 3, axis=-1)
