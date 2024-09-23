from infer_ext.framework import lmdeploy_ext

import time
import torch
import random

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN


def profile(round=6):
    # warm up
    response = pipe("How are you?")
    response = pipe(("describe the image:", image), top_k=1)
    
    prompt_tokens = 0
    completion_tokens = 0

    print("start profiling...")

    start_time = time.time()
    for i in range(round):
        response = pipe(("describe the image:", image), top_k=1)
        prompt_tokens += response.input_token_len
        completion_tokens += response.generate_token_len
    cost_time = time.time() - start_time

    print(f"number of prompt tokens: {prompt_tokens}.")
    print(f"number of completion tokens: {completion_tokens}.")
    print(f"token throughput (completion token): {(completion_tokens / cost_time):.3f} token/s.")
    print(f"token throughput (prompt + completion token): {((prompt_tokens + completion_tokens) / cost_time):.3f} token/s.")

    return


if __name__ == '__main__': 
    pipe = pipeline('/data/models/cogvlm2-llama3-chinese-chat-19b', model_name='cogvlm', backend_config=PytorchEngineConfig(tp=1, device_type='muxi', block_size=16, cache_max_entry_count=0.1))

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
    print("warm up:")
    question = "How are you?"
    response = pipe(question)
    print(f"Q: {question}")
    print(f"A: {response.text}")
    print()

    # test image
    print("test image:")
    question = ("describe the image:", image)
    start_time = time.time()
    response = pipe(question, top_k=1)
    cost_time = time.time() - start_time
    print(f"Q: {question}")
    print(f"A: {response.text}")
    print()
    prompt_tokens = response.input_token_len
    completion_tokens = response.generate_token_len
    print(f"number of prompt tokens: {prompt_tokens}.")
    print(f"number of completion tokens: {completion_tokens}.")
    print(f"token throughput (completion token): {(completion_tokens / cost_time):.3f} token/s.")
    print(f"token throughput (prompt + completion token): {((prompt_tokens + completion_tokens) / cost_time):.3f} token/s.")
    print()

    # test multi batch
    print("test multi batch:")
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
    print("test multi session:")
    question = ("please describe this image.", image)
    sess = pipe.chat(question)
    print("session 1: ", sess)

    question = "What is the woman doing?"
    sess = pipe.chat(question, session=sess)
    print("session 2: ", sess)
    print()

    image1 = load_image("/home/pujiang/zhousl/data/tiger.jpeg")
    image2 = load_image("/home/pujiang/zhousl/data/human_pose.jpg")
 
    # test multi iamge with multi batch
    print("test multi iamge with multi batch")
    question = [("please describe this image1.", image1), ("please describe this image2.", image2)]
    response = pipe(question)
    for idx, r in enumerate(response):
        print(f"batch_{idx}:")
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()

    # test multi iamge with multi session
    # TODO: solve aligned memory error
    # print("test multi iamge with multi session:")
    # question = [
    #    dict(role='user', content=[
    #       dict(type='text', text=f'<img>{IMAGE_TOKEN}{IMAGE_TOKEN}</img>\nDescribe the two images in detail.'),
    #       dict(type='image_data', image_data=dict(data=image1)),
    #       dict(type='image_data', image_data=dict(data=image2))
    #     ])
    # ]

    # response = pipe(question)
    # print(f"Q: Describe the two images in detail.")
    # print(f"A: {response.text}")
    # print()

    # question.append(dict(role='assistant', content=response.text))
    # question.append(dict(role='user', content='What are the similarities and differences between these two images.'))
    # response = pipe(question)

    # print(f"Q: What are the similarities and differences between these two images.")
    # print(f"A: {response.text}")
    # print()
