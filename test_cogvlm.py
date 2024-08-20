from infer_ext.framework import lmdeploy_ext

import torch
import random

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image


if __name__ == '__main__': 
    torch.manual_seed(10)
    random.seed(10)
    # pipe = pipeline('/data/models/cogvlm2-llama3-chat-19B', backend_config = PytorchEngineConfig(tp=1, device_type='muxi', block_size=16, cache_max_entry_count=0.01))
    pipe = pipeline('/data/models/cogvlm2-llama3-chinese-chat-19b', backend_config = PytorchEngineConfig(tp=1, device_type='muxi', block_size=16, cache_max_entry_count=0.01))
    #image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    #response = pipe(('describe this image', image))
    #print(response)
    prompts = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe this image'},
                {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
            ]
        }
    ]

    # response = pipe(prompts, do_preprocess=True)
    image = load_image("/home/pujiang/zhousl/data/tiger.jpeg")
    image1 = load_image("/home/pujiang/zhousl/data/cat.jpg")

    response = pipe(('describe', image1), do_preprocess=True)
    print(response)
    response = pipe(('what is your name?'), do_preprocess=True)
    print(response)
    raise ValueError("test")

    response = pipe(('How are you?'), do_preprocess=True)
    print(response)
    raise ValueError("test")
