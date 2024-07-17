#import torch_dipu
from vllm import LLM, SamplingParams
import torch
import json
import time
from torch.profiler import record_function

# Sample prompts.
prompts = [
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'describe this image'},
            {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
        ]
    }
]
# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0)
sampling_params = SamplingParams()

# Create an LLM.
llm = LLM(model="/data/models/Mini-InternVL-Chat-2B-V1-5",
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
