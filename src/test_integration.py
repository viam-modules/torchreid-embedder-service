from typing import Dict

import numpy as np
import pytest
import torch
from google.protobuf.struct_pb2 import Struct
from PIL import Image
from viam.proto.app.robot import ServiceConfig

from src.person_embedder_service import PersonEmbedderService

WORKING_CONFIG_DICT = {}
CONFIG_WITH_MODEL_PATH = {"model_path": "./src/models/osnet/osnet_ain_ms_d_c.pth.tar"}
IMG_PATH = "./src/test/alex/alex_2.jpeg"


def get_config(config_dict: Dict) -> ServiceConfig:
    """returns a config populated with picture_directory and camera_name
    attributes.X

    Returns:``
        ServiceConfig: _description_
    """
    struct = Struct()
    struct.update(dictionary=config_dict)
    config = ServiceConfig(attributes=struct)
    return config


class TestTracker:
    @pytest.mark.asyncio
    async def test_infer_without_model_path(self):
        # Test detection from vision service
        service = PersonEmbedderService("test")
        config = get_config(WORKING_CONFIG_DICT)
        service.reconfigure(config, None)
        image_object = Image.open(IMG_PATH)
        # Convert PIL image to numpy array
        image_array = np.array(image_object, dtype=np.uint8)
        uint8_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        float32_tensor = uint8_tensor.to(dtype=torch.float32)
        input_array = float32_tensor.numpy()

        input_tensor = {"input": input_array}
        res = await service.infer(input_tensor)
        assert res["embedding"].shape == (512,)

    @pytest.mark.asyncio
    async def test_infer_with_model_path(self):
        # Test detection from vision service
        service = PersonEmbedderService("test")
        config = get_config(CONFIG_WITH_MODEL_PATH)
        service.reconfigure(config, None)
        image_object = Image.open(IMG_PATH)
        # Convert PIL image to numpy array
        image_array = np.array(image_object, dtype=np.uint8)
        uint8_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        float32_tensor = uint8_tensor.to(dtype=torch.float32)
        input_array = float32_tensor.numpy()

        input_tensor = {"input": input_array}
        res = await service.infer(input_tensor)
        assert res["embedding"].shape == (512,)

    @pytest.mark.asyncio
    async def test_embeddings_consistency_with_and_without_model_path(self):
        """Test that embeddings are identical whether using default model or explicit model path."""
        # Load the same image for both tests
        image_object = Image.open(IMG_PATH)
        image_array = np.array(image_object, dtype=np.uint8)
        uint8_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        float32_tensor = uint8_tensor.to(dtype=torch.float32)
        input_array = float32_tensor.numpy()
        input_tensor = {"input": input_array}

        # Test without model path (uses default)
        service_default = PersonEmbedderService("test_default")
        config_default = get_config(WORKING_CONFIG_DICT)
        service_default.reconfigure(config_default, None)
        result_default = await service_default.infer(input_tensor)
        embedding_default = result_default["embedding"]

        # Test with explicit model path
        service_explicit = PersonEmbedderService("test_explicit")
        config_explicit = get_config(CONFIG_WITH_MODEL_PATH)
        service_explicit.reconfigure(config_explicit, None)
        result_explicit = await service_explicit.infer(input_tensor)
        embedding_explicit = result_explicit["embedding"]

        # Assert that both embeddings are identical
        np.testing.assert_array_almost_equal(
            embedding_default,
            embedding_explicit,
            decimal=10,  # High precision comparison
            err_msg="Embeddings should be identical whether using default or explicit model path",
        )

        # Also verify shapes are the same
        assert embedding_default.shape == embedding_explicit.shape == (512,)


if __name__ == "__main__":
    # Run all tests with pytest
    pytest.main(
        ["-xvs", __file__]
    )  # verbose, stop after first failure, don't capture output
