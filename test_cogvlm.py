import torch
import random

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image


if __name__ == '__main__': 
    pipe = pipeline('/data/models/cogvlm2-llama3-chinese-chat-19b', backend_config = PytorchEngineConfig(tp=1, device_type='maca', block_size=256, cache_max_entry_count=0.4))

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
    # print("warm up:")
    # response = pipe("How are you?", do_preprocess=True)
    # print(response.text)
    # print()

    # test image
    # print("test image:")
    # response = pipe(("describe the image:", image), top_k=1)
    # print(response.text)
    # print()

    # print("test multi batch:")
    # question = ["Please introduce Shanghai."]
    # question = ["What functions do you have?"]
    # question = ["How are you?", "Please introduce Shanghai."]
    # response = pipe(question, do_preprocess=True)
    # for idx, r in enumerate(response):
    #     print(f"Q: {question[idx]}")
    #     print(f"A: {r.text}")
    #     print()

    # image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
    image = load_image("/home/pujiang/zhousl/data/human_pose.jpg")
    
    # test multi session
    print("test multi session(only text):")
    # question = ("Please introduce Shanghai.")
    question = ("who are the president of USA?")
    sess = pipe.chat(question, top_k=1)
    print("session 1: ", sess)

    question = "please introduce his family."
    # question = "What's the key point of your description above?"
    sess = pipe.chat(question, session=sess, top_k=1)
    print("session 2: ", sess)
    print()
    
    # print("test multi session(text and image):")
    # question = "please describe this image:"
    # sess = pipe.chat((question, image))
    # print("session 1: ", sess)
    # question = "What's the key point of your description above?"
    # # question = "What are tho woman doing?"
    # sess = pipe.chat(question, session=sess)
    # print("session 2: ", sess)
    # print()