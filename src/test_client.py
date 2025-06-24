import asyncio

import numpy as np
from PIL import Image
from viam.robot.client import RobotClient
from viam.services.mlmodel import MLModelClient

IMG_PATH = "./src/test/alex/alex_2.jpeg"


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key="xxxxxxxxx",
        api_key_id="xxxxxxxxx",
    )
    return await RobotClient.at_address("xxxxxxxxx", opts)


async def main():
    machine = await connect()

    # print("Resources:")
    # print(machine.resource_names)
    image_object = Image.open(IMG_PATH)
    resized_image = image_object.resize((256, 256))
    # Convert PIL image to numpy array
    image_array = np.array(resized_image)

    input_tensor = {"input": image_array}

    # mlmodel-1
    mlmodel_1 = MLModelClient.from_robot(machine, "mlmodel-1")
    mlmodel_1_return_value = await mlmodel_1.infer(input_tensor)
    print(f"mlmodel-1 metadata return value: {mlmodel_1_return_value}")

    # Don't forget to close the machine when you're done!
    await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
