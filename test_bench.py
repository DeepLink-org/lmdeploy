import torch
import random
import numpy as np
import lmdeploy

from lmdeploy import PytorchEngineConfig

if __name__ == "__main__":
    # torch.manual_seed(10)
    # random.seed(10)
    pipe = lmdeploy.pipeline("/data/models/internlm2-chat-7b",
                            backend_config = PytorchEngineConfig(tp=1,
                                                                 block_size=16,
                                                                 device_type='muxi',
                                                                 cache_max_entry_count=0.4))
    #question = ["How are you?", "Please introduce China.", "Introduce Shanghai AI Lab"]
    #question = ["Please introduce China.", "How are you?"]
    # question = ["Please introduce Shanghai."]
    #question = ["Shanghai is"]
    # question = ["Hello, my name is"]
    question = np.random.randint(10000, size=(8, 256))
    question = question.tolist()
    response = pipe(question, do_preprocess=True)
    for idx, r in enumerate(response):
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()

