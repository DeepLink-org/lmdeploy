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
                                                                 device_type='maca',
                                                                 cache_max_entry_count=0.4))

    # warm up
    print("warm up:")
    response = pipe("How are you?", do_preprocess=True)
    print(response.text)
    print()

    # print("test multi batch:")
    # question = ["Please introduce Shanghai."]
    # question = ["What functions do you have?"]
    # question = ["How are you?", "Please introduce Shanghai."]
    # response = pipe(question, do_preprocess=True)
    # for idx, r in enumerate(response):
    #     print(f"Q: {question[idx]}")
    #     print(f"A: {r.text}")
    #     print()

    # # test multi session
    # print("test multi session:")
    # # question = ("Please introduce Shanghai.")
    # question = ("who are the president of USA?")
    # sess = pipe.chat(question)
    # print("session 1: ", sess)

    # question = "please introduce his family."
    # # question = "What's the key point of your description above?"
    # sess = pipe.chat(question, session=sess)
    # print("session 2: ", sess)
    # print()
