import torch
import random
import lmdeploy
from lmdeploy import PytorchEngineConfig

if __name__ == "__main__":
    torch.manual_seed(10)
    random.seed(10)
    pipe = lmdeploy.pipeline("/data/models/internlm2-chat-7b",
                            backend_config = PytorchEngineConfig(tp=1,
                                                                 block_size=16,
                                                                 device_type='maca',
                                                                 cache_max_entry_count=0.4))
    #question = ["How are you?", "Please introduce China.", "Introduce Shanghai AI Lab"]
    #question = ["Please introduce China.", "How are you?"]
    # question = ["Please introduce Shanghai."]
    #question = ["Shanghai is"]
    question = ["介绍一下上海的旅游景点。", "Hello, how are you?"][:1]
    response = pipe(question, do_preprocess=False, top_k=1, temperature=0.0)
    for idx, r in enumerate(response):
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()
