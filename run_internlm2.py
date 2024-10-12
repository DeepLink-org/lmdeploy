# import dlinfer
import lmdeploy
import torch
from lmdeploy import PytorchEngineConfig
if __name__ == "__main__":
    seed = 1024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.set_printoptions(precision=10)
    b = PytorchEngineConfig(tp=1,block_size=16, cache_max_entry_count=0.4, device_type="camb", download_dir="/workspace/volume/shangda/share/llm_models")
    pipe = lmdeploy.pipeline("Shanghai_AI_Laboratory/internlm2_5-7b",
                            backend_config = b)
    # pipe = lmdeploy.pipeline("Shanghai_AI_Laboratory/internlm2-chat-7b",
    #                          backend_config = b)
    # question = ["Hi, pls intro yourself", "Please introduce Shanghai."]
    # question = ["Hi, pls intro yourself", "Hi, pls intro yourself"]
    question = ["Hi, pls intro yourself", "who is your father"]
    print(question)
    response = pipe(question, do_preprocess=False, top_k=1)
    print(response)
    # for idx, r in enumerate(response):
    #     print(f"Q: {question[idx]}")
    #     print(f"AAAAAA: {r.text}")
    #     print()
    # print("end")
