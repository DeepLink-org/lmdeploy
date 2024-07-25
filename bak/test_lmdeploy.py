import torch
import random
import lmdeploy
from lmdeploy import PytorchEngineConfig

from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    # torch.manual_seed(10)
    # random.seed(10)
    #pipe = lmdeploy.pipeline("/data/models/internlm2-chat-20b",
    pipe = lmdeploy.pipeline("/root/internlm2-chat-7b",
    #pipe = lmdeploy.pipeline("/root/Qwen2-7B",
                            backend_config = PytorchEngineConfig(tp=1,
                                                                 block_size=16,
                                                                 device_type='muxi',
                                                                 #session_len=256,
                                                                 cache_max_entry_count=0.4))
    #question = ["How are you?", "Please introduce China.", "Introduce Shanghai AI Lab"]
    #question = ["Please introduce China.", "How are you?"]
    #question = ["Please introduce Shanghai."]
    #question = ["Shanghai is"]
    #question = ["Shanghai is"]
    #question = ["How are you?"]
    question = ["Hi", "How are you?"]

    response = pipe(question, do_preprocess=True)
    for idx, r in enumerate(response):
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()


## profiler
#    #question = ["How are you?"]
#    question = ["Hi"]
#    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
#        with record_function("inference"):
#            response = pipe(question, do_preprocess=True)
#    prof.export_chrome_trace("internlm.muxi.json")
#    for idx, r in enumerate(response):
#        print(f"Q: {question[idx]}")
#        print(f"A: {r.text}")
#        print()
#
#    #response = pipe(question, do_preprocess=True)
#    #for idx, r in enumerate(response):
#    #    print(f"Q: {question[idx]}")
#    #    print(f"A: {r.text}")
#    #    print()

