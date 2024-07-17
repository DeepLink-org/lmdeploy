from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    pipe = pipeline('/data/models/Mini-InternVL-Chat-2B-V1-5', backend_config = PytorchEngineConfig(tp=1, device_type='muxi', block_size=16,))

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
    #response = pipe(prompts, do_preprocess=True)
    response = pipe(prompts)

    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #    with record_function("inference"):
    #        response = pipe(prompts)
    #prof.export_chrome_trace("internvl.v100.json")
    print(response)
if __name__ == '__main__':
    main()
