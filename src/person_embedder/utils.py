import os
import sys

import torch.nn.functional as F


def resource_path(relative_path):
    """
    Get the absolute path to a resource file, considering different environments.

    Args:
        relative_path (str): The relative path to the resource file.

    Returns:
        str: The absolute path to the resource file.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = os.path.join(sys._MEIPASS, "src", "models")  # pylint: disable=duplicate-code,protected-access,no-member

    except Exception:  # pylint: disable=broad-exception-caught
        base_path = os.path.abspath(os.path.join("src", "models"))

    return os.path.join(base_path, relative_path)


def resize_for_padding(input_tensor, target_size):
    # Get the original dimensions
    height, width = input_tensor.shape[1:]
    if height == 0 or width == 0:
        raise ValueError(f"got input {input_tensor}")
    target_height, target_width = target_size

    # Calculate scaling factors to maintain aspect ratio
    scale_h = target_height / height
    scale_w = target_width / width
    scale = min(scale_h, scale_w)  # Use the smaller scale to preserve aspect ratio

    # Calculate new height and width
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Resize the image while preserving aspect ratio
    resized_image = F.interpolate(
        input_tensor.unsqueeze(0),  # Add batch dimension for resizing
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )

    # Now we need to pad the image to the target size
    return resized_image, new_height, new_width, target_height, target_width


def pad_image_to_target_size(resized_image, target_size):
    _, _, h, w = resized_image.shape
    target_height, target_width = target_size

    # Calculate padding for each side
    pad_h = target_height - h
    pad_w = target_width - w

    # Pad evenly or add extra pixels on one side if necessary
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad the image
    padded_image = F.pad(
        resized_image,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )

    return padded_image
