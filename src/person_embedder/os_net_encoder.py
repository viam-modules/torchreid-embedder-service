import os
import os.path as osp
import pickle
from collections import OrderedDict
from functools import partial
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as T
from viam.logging import getLogger

from src.person_embedder.osnet import osnet_ain_x1_0
from src.person_embedder.utils import (
    pad_image_to_target_size,
    resize_for_padding,
    resource_path,
)

LOGGER = getLogger(__name__)


OSNET_REPO = "osnet"


class OSNetFeatureEmbedder:
    def __init__(self, model_path: str = None):
        """
        Initialize the FeatureEncoder with a feature extractor model.

        :param model_name: The name of the model to use for feature extraction.
        :param model_path: The path to the pre-trained model file.
        :param device: The device to run the model on ('cpu' or 'cuda').
        """
        if torch.cuda.is_available():
            use_gpu = True
            self.device = torch.device("cuda")
        else:
            use_gpu = False
            self.device = torch.device("cpu")

        self.input_shape = (256, 128)
        model = osnet_ain_x1_0(
            num_classes=1000, loss="softmax", pretrained=False, use_gpu=use_gpu
        )
        model.eval()
        if model_path is None:
            LOGGER.info("No model path provided, using default model")
            model_path = resource_path(
                os.path.join(OSNET_REPO, "osnet_ain_ms_d_c.pth.tar")
            )
        else:
            LOGGER.info(f"Using model path: {model_path}")
        load_pretrained_weights(model, model_path)
        self.model = model.to(self.device)

        ##preprocessing
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]

        def gpu_compatible_transforms(tensor: torch.Tensor):
            normalize = T.Normalize(mean=pixel_mean, std=pixel_std)
            return normalize(tensor)

        self.preprocess = gpu_compatible_transforms

    def compute_features_on_single_cropped_image(self, img: torch.Tensor):
        """
        Compute a single feature vector for an image.
        """
        resized_image, _, _, _, _ = resize_for_padding(img, self.input_shape)
        padded_image = pad_image_to_target_size(resized_image, self.input_shape)
        preprocessed_image = self.preprocess(padded_image)
        with torch.no_grad():
            res = self.model(preprocessed_image)
        return res[0]

    # TODO: implement compute_features when the tracker is able to ask for batched inference
    #  def compute_features(
    #     self, img: ImageObject, detections: List[Detection]
    # ) -> List[np.ndarray]:
    #     device = img.float32_tensor.device
    #     image_height, image_width = img.float32_tensor.shape[
    #         1:
    #     ]  # Assuming CxHxW format

    #     # Stack all bounding boxes into a tensor (x1, y1, x2, y2)
    #     bboxes = torch.tensor(
    #         [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] for d in detections],
    #         device=device,
    #     )

    #     # Crop and resize images
    #     cropped_images = []
    #     for bbox in bboxes:
    #         x1, y1, x2, y2 = map(int, bbox)  # Ensure integer coordinates
    #         x1, y1 = max(0, x1), max(0, y1)  # Clip to image dimensions
    #         x2, y2 = min(image_width, x2), min(image_height, y2)

    #         cropped_image = img.float32_tensor[
    #             :, y1:y2, x1:x2
    #         ]  # Crop image (CxH_cropxW_crop)

    #         # Ensure the cropped region is valid
    #         if cropped_image.numel() == 0:
    #             raise ValueError(f"Invalid crop region: {bbox}")

    #         # Resize the cropped image

    #         resized_image, new_height, new_width, _, _ = resize_for_padding(
    #             cropped_image, self.input_shape
    #         )
    #         padded_image = pad_image_to_target_size(resized_image, self.input_shape)
    #         cropped_images.append(padded_image[0])

    #     # Stack all resized images into a batch
    #     cropped_batch = torch.stack(
    #         cropped_images, dim=0
    #     )  # Resulting shape: (B, C, H, W)

    #     # Normalize the batch
    #     cropped_batch = self.preprocess(cropped_batch)

    #     # Extract features
    #     with torch.no_grad():
    #         res = self.model(cropped_batch)
    #     return res


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
