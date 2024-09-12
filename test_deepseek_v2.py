import torch
import random
import lmdeploy
from lmdeploy import PytorchEngineConfig

if __name__ == "__main__":
    torch.manual_seed(10)
    random.seed(10)
    pipe = lmdeploy.pipeline("/data/models/DeepSeek-V2-Lite-Chat",
                            backend_config = PytorchEngineConfig(tp=1,
                                                                 block_size=16,
                                                                 device_type='muxi',
                                                                 cache_max_entry_count=0.4))
    # question = ["How are you?", "Please introduce China.", "Introduce Shanghai AI Lab"]
    # question = ["Hello, my name is"]
    question = ["How are you?"]
    # question = ["Please introduce China."]
    # response = pipe(question, do_preprocess=False, top_k=1)
    response = pipe(question, do_preprocess=True, top_k=1)
    # response = pipe(question, do_preprocess=False)
    for idx, r in enumerate(response):
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()
