from typing import Dict

import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct
from PIL import Image
from viam.proto.app.robot import ServiceConfig

from src.person_embedder_service import PersonEmbedderService

WORKING_CONFIG_DICT = {}
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
    async def test_infer(self):
        # Test detection from vision service
        service = PersonEmbedderService("test")
        service.reconfigure(WORKING_CONFIG_DICT, None)
        image_object = Image.open(IMG_PATH)
        # Convert PIL image to numpy array
        image_array = np.array(image_object)

        input_tensor = {"input": image_array}
        res = await service.infer(input_tensor)
        assert res is not None


if __name__ == "__main__":
    # Run all tests with pytest
    pytest.main(
        ["-xvs", __file__]
    )  # verbose, stop after first failure, don't capture output
