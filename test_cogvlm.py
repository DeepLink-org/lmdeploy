import torch
import random

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image
from torch.profiler import profile, record_function, ProfilerActivity

#def main():
if __name__ == '__main__': 
    torch.manual_seed(10)
    random.seed(10)
    pipe = pipeline('/data/models/cogvlm-chat', backend_config = PytorchEngineConfig(tp=1, device_type='muxi', block_size=16, cache_max_entry_count=0.1))

    #image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    #response = pipe(('describe this image', image))
    #print(response)
    prompts = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe this image'},
                {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
            ]
        }
    ]

    # image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    image = load_image('https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWibMzAJHnvVSP7C5ealtYOQwSwPqO9k33PSF4zCfv9Cusxlm51o0ZxwKPHTbianBworFfkPCK2I2DBw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1')
    image = load_image("/home/costest/zhousl/tiger.jpeg")
    # response = pipe(('describe this image', image), do_preprocess=True)
    response = pipe(('describe this image'), do_preprocess=True)
    # response = pipe(prompts, do_preprocess=True)
    # response = pipe(prompts)

    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #    with record_function("inference"):
    #        response = pipe(prompts)
    #prof.export_chrome_trace("internvl.v100.json")
    print(response)
# if __name__ == '__main__':
#   main()
