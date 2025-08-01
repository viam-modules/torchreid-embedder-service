"""
This module provides a Viam ML Model Service module
to perform person Re-Id tracking.
"""

from typing import (
    ClassVar,
    Dict,
    Mapping,
    Optional,
    Sequence,
)

import torch
from numpy.typing import NDArray
from typing_extensions import Self
from viam.logging import getLogger
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.proto.service.mlmodel import Metadata
from viam.resource.base import ResourceBase
from viam.resource.types import (
    Model,
    ModelFamily,
)
from viam.services.mlmodel import MLModel
from viam.utils import ValueTypes

from src.person_embedder.os_net_encoder import OSNetFeatureEmbedder

LOGGER = getLogger(__name__)


class PersonEmbedderService(MLModel, Reconfigurable):
    """PersonEmbedderService is a subclass a Viam MLModel Service"""

    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "mlmodel"), "torchreid-embedder-service"
    )

    def __init__(self, name: str):
        super().__init__(name=name)
        self.embedder: OSNetFeatureEmbedder = None

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """returns new ml model service"""
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        """Validate config and returns a list of dependencies."""
        return []

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        self.embedder = OSNetFeatureEmbedder()
        return

    async def infer(
        self,
        input_tensors: Dict[str, NDArray],
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, NDArray]:
        """Perform inference on the input tensors to generate person embeddings.

        Args:
            input_tensors: Dictionary containing input tensors with key "input"
            extra: Optional extra parameters
            timeout: Optional timeout for the operation

        Returns:
            Dictionary containing the embedding with key "embedding"
        """

        # Extract the cropped image from input tensors
        cropped_image = input_tensors["input"]
        uint8_tensor = torch.from_numpy(cropped_image).contiguous()  # -> to (C, H, W)
        float32_tensor = uint8_tensor.to(dtype=torch.float32)
        # Ensure the tensor is on the correct device (CPU/GPU)
        if hasattr(self.embedder, "device"):
            float32_tensor = float32_tensor.to(self.embedder.device)

        # Compute features using the OSNet encoder
        embedding = self.embedder.compute_features_on_single_cropped_image(
            float32_tensor
        )

        # Convert back to numpy array for return
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        return {"embedding": embedding}

    async def metadata(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Metadata:
        """Get the metadata (such as name, type, expected tensor/array shape, inputs, and outputs) associated with the ML model.

        ::

            my_mlmodel = MLModelClient.from_robot(robot=machine, name="my_mlmodel_service")

            metadata = await my_mlmodel.metadata()

        Returns:
            Metadata: The metadata

        For more information, see `ML model service <https://docs.viam.com/dev/reference/apis/services/ml/#metadata>`_.
        """
        return NotImplementedError
