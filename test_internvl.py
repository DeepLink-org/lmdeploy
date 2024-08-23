from infer_ext.framework import lmdeploy_ext

import torch
import random

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image


if __name__ == '__main__': 
    torch.manual_seed(10)
    random.seed(10)

    pipe = pipeline('/data/models/InternVL-Chat-V1-5', backend_config = PytorchEngineConfig(tp=4, device_type='muxi', block_size=16,))

    #image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    image = load_image("/home/pujiang/zhousl/data/tiger.jpeg")
    image1 = load_image("/home/pujiang/zhousl/data/cat.jpg")

    prompts = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe this image'},
                {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
            ]
        }
    ]

    # warm up
    # response = pipe("How are you?")
    # print(response)

    # response = pipe(("describe the image:", image))
    # print(response)

    # question = ["How are you?"]
    # question = ["Please introduce Shanghai."]
    # test multi batch
    question = ["How are you?", "Please introduce Shanghai."]
    response = pipe(question)
    for idx, r in enumerate(response):
        print(f"batch_{idx}:")
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()

    # test image
    question = [("describe the image:", image)]
    response = pipe(question)
    for idx, r in enumerate(response):
        print(f"Q: {question[idx]}")
        # print(response)
        print(f"A: {r.text}")
        print()

    # question = ["How are you?", "How are you?"]
    # question = ["How are you?", "Please introduce Shanghai."]
    # response = pipe(question)
    # for idx, r in enumerate(response):
    #     print(f"Q: {question[idx]}")
    #     # print(response)
    #     print(f"A: {r.text}")
    #     print()

    # TODO: support session later
    # sess = pipe.chat("I am living in shanghai!")
    # print("############1: ", sess)
    # # import pdb; pdb.set_trace()
    # sess = pipe.chat("please introduce my city", session=sess)
    # print("############2: ", sess)
