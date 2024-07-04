#import torch_dipu
from vllm import LLM, SamplingParams
import torch
import json
import time
from torch.profiler import record_function

# Sample prompts.
prompts = [
    "Hello, my name is",
    #"The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0)
sampling_params = SamplingParams()

# Create an LLM.
#llm = LLM(model="/data/models/internlm2-chat-20b",
llm = LLM(model="/root/internlm2-chat-7b",
          tensor_parallel_size=1, #   worker_use_ray=False,
          trust_remote_code=True,
          enforce_eager=False)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

#def save_trace(profiler, trace_file):
#    with open(trace_file, 'w') as f:
#        json.dump(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1), f)
#
#with torch.profiler.profile(
#        activities=[
#            torch.profiler.ProfilerActivity.CPU,
#            torch.profiler.ProfilerActivity.CUDA,
#        ],
#        #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#        #record_shapes=True,
#        #profile_memory=True,
#        with_stack=True
#) as prof:
##with torch.autograd.profiler.profile(with_stack=True, with_modules=True) as prof:
#    for output in outputs:
#        with record_function("one_prompt"):
#            prompt = output.prompt
#            generated_text = output.outputs[0].text
#            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#        prof.step()
#prof.export_chrome_trace('./vllm_interlm2_timeline.json')
