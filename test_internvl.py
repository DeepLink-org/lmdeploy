# from infer_ext.framework import lmdeploy_ext

import torch
import random

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image


if __name__ == '__main__': 
    pipe = pipeline('/data/models/InternVL-Chat-V1-5', backend_config = PytorchEngineConfig(tp=4, device_type='muxi', block_size=16,))

    prompts = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe this image'},
                {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
            ]
        }
    ]

    # image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    image = load_image("/home/pujiang/zhousl/data/tiger.jpeg")
    
    # warm up
    response = pipe("How are you?")
    print(response.text)

    # test image
    response = pipe(("describe the image:", image), top_k=1)
    print(response.text)

    # test multi batch
    question = ["How are you?", "Please introduce Shanghai."]
    response = pipe(question)
    for idx, r in enumerate(response):
        print(f"batch_{idx}:")
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()

    # image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
    image = load_image("/home/pujiang/zhousl/data/human_pose.jpg")

    # test multi session
    sess = pipe.chat(("please describe this image.", image))
    print("session 1: ", sess)
    sess = pipe.chat('What is the woman doing?', session=sess)
    print("session 2: ", sess)
