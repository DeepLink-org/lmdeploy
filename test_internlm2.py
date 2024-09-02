import torch
import random
import lmdeploy
from lmdeploy import PytorchEngineConfig

if __name__ == "__main__":
    pipe = lmdeploy.pipeline("/data/models/internlm2-chat-7b",
    # pipe = lmdeploy.pipeline("/data/models/internlm2_5-7b-chat",
                            backend_config = PytorchEngineConfig(tp=1,
                                                                 block_size=16,
                                                                 device_type='muxi',
                                                                 cache_max_entry_count=0.1))

    # warm up
    response = pipe("How are you?", do_preprocess=True)
    print(response.text)

    # test multi batch
    # question = ["How are you?"]
    # question = ["Please introduce Shanghai."]
    # question = ["What functions do you have?"]
    question = ["How are you?", "Please introduce Shanghai."]
    response = pipe(question, do_preprocess=True)
    for idx, r in enumerate(response):
        print(f"batch_{idx}:")
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()

    # test multi session
    sess = pipe.chat("I am living in shanghai!")
    print("session 1:", sess)
    sess = pipe.chat("please introduce it.", session=sess)
    print("session 2: ", sess)
