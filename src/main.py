# pylint: disable=missing-module-docstring
import asyncio

from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.mlmodel import MLModel
from viam.services.vision import Vision

from src.person_embedder_service import PersonEmbedderService


async def main():
    """
    This function creates and starts a new module, after adding all desired
    resource models. Resource creators must be registered to the resource
    registry before the module adds the resource model.
    """
    Registry.register_resource_creator(
        MLModel.API,
        PersonEmbedderService.MODEL,
        ResourceCreatorRegistration(
            PersonEmbedderService.new_service,
            PersonEmbedderService.validate_config,
        ),
    )
    module = Module.from_args()

    module.add_model_from_registry(MLModel.API, PersonEmbedderService.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
