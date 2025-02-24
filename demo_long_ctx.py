import random
import torch
import lmdeploy
from lmdeploy import PytorchEngineConfig, GenerationConfig
from lmdeploy.serve.async_engine import AsyncEngine

from contextlib import nullcontext

# torch._logging.set_logs(dynamo=logging.DEBUG, output_code=True)
# torch._dynamo.config.assume_static_by_default = True
# torch._dynamo.config.automatic_dynamic_shapes = True
# torch._dynamo.assume_static_by_default = False
# torch._dynamo.config.suppress_errors = False
# torch._dynamo.config.cache_size_limit = 3000
# torch._dynamo.config.report_guard_failures = True
# torch._dynamo.config.enforce_cond_guards_match = False
# torch._dynamo.config.enforce_cond_guards_match = False



if __name__ == "__main__":
    torch.manual_seed(10)
    random.seed(10)
    path = "/pj_data/share/internlm_model/internlm2-chat-7b"
    # path = "/deeplink_afs/model_data/Qwen2-7B-Instruct/"
    # from viztracer import VizTracer
    # tracer = VizTracer(output_file="my.json", log_async=True)
    # tracer = nullcontext()
    # with tracer:
    pipe: AsyncEngine = lmdeploy.pipeline(
        path,
        backend_config=PytorchEngineConfig(tp=1,
                                            eager_mode=True,
                                            cache_max_entry_count=0.3,
                                            device_type="ascend"))
    question1 = [
        "How are you?", "Please introduce China.", "Introduce Shanghai AI Lab"
    ]
    question1 = [
        "请用200字续写一下西游记：", "能否用python写一个计算斐波那契数列的代码。", "介绍一下中国："
    ]

    with open("./zhihu1.txt", "r") as f:
        zhihu1 = f.read()
    with open("./zhihu2.txt", "r") as f:
        zhihu2 = f.read()

    question1 = [f'''请总结以下内容：{zhihu1} 请给出简明扼要的总结。''',
                 f'''请总结以下内容：{zhihu2} 请给出简明扼要的总结。'''][0:1]
    # question2 = ["Hi! Have you been to any interesting places recently?"]
    # question = ["Shanghai is"]
    gen_config = GenerationConfig()
    gen_config.max_new_tokens = 128
    gen_config.do_sample = False # this force top_k = 1


    # import torch_npu
    # import torch
    # prof = torch_npu.profiler.profiler.profile(
    #     activities=[torch_npu.profiler.ProfilerActivity.CPU,
    #                 torch_npu.profiler.ProfilerActivity.NPU],
    #     schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=0, skip_first=0),
    #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./pd_profile"),
    #     record_shapes=False,
    #     profile_memory=False,
    #     with_stack=False,
    #     with_flops=False,
    # )
    # prof.__enter__()

    response1 = pipe(question1,
                    gen_config=gen_config,
                    do_preprocess=False)
    # response2 = pipe(question2,
    #                 # gen_config=gen_config,
    #                 do_preprocess=False)
    for idx, r in enumerate(response1):
        print(f"Q: {question1[idx]}")
        print(f"A: {r.text}")
        print()
    # for idx, r in enumerate(response2):
    #     print(f"Q: {question2[idx]}")
    #     print(f"A: {r.text}")
    #     print()

    # pd profile
    for _ in range(5):
        response1 = pipe(['你好，' + question for question in question1],
                            gen_config=gen_config,
                            do_preprocess=False)
    # prof.__exit__(None, None, None)
    # prof.export_chrome_trace('./pd_profile/pd_engine_trace.json')

    print("=" * 100)
    for idx, r in enumerate(response1):
        # print(f"Q: {question1[idx]}")
        print(f"A: {r.text}")
        print()

    print("llm infer end")
    del pipe.engine.model_agent
