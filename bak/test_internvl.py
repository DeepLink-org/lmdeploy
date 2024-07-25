import lmdeploy

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image
from torch.profiler import profile, record_function, ProfilerActivity

#def main():
#    #pipe = pipeline('/root/Mini-InternVL-Chat-2B-V1-5', backend_config = PytorchEngineConfig(tp=1, device_type='muxi', block_size=16, cache_max_entry_count=0.5))
#    pipe = pipeline('/root/InternVL-Chat-V1-5', backend_config = PytorchEngineConfig(tp=4, device_type='muxi', block_size=16, cache_max_entry_count=0.2))
#
#    #image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
#    #response = pipe(('describe this image', image))
#    #print(response)
#    pic = []
#    #tiger_pic = {
#    #        'role': 'user',
#    #        'content': [
#    #            {'type': 'text', 'text': 'describe this image'},
#    #            {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
#    #        ]
#    #    }
#    pic.append({
#            'role': 'user',
#            'content': [
#                {'type': 'text', 'text': 'describe this image'},
#                {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
#            ]
#    })
#    prompts = []
#    for i in range(2):
#        prompts.append(pic[i % len(pic)])
#    response = pipe(prompts, do_preprocess=True)
#
#    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#    #    with record_function("inference"):
#    #        response = pipe(prompts)
#    #prof.export_chrome_trace("internvl.v100.json")
#    #for idx, r in enumerate(response):
#    #    print(response)
#    #    print()
#    print(response)
#if __name__ == '__main__':
#    main()



def main():

    pipe = pipeline('/root/InternVL-Chat-V1-5', backend_config = PytorchEngineConfig(tp=4, device_type='muxi', block_size=16, cache_max_entry_count=0.2))
    
    #pipe = pipeline('/root/Mini-InternVL-Chat-2B-V1-5', 
    #                backend_config = PytorchEngineConfig(tp=4, device_type='muxi', block_size=16, cache_max_entry_count=0.8))
    
    
    images = [
        '/root/human-pose.jpg',
        '/root/det.jpg',
        '/root/text_recog.jpg',
        '/root/text_det.jpg',
        '/root/chengdu.jpg',
    ]
    
    image_urls = []
    
    for i in range(128):
        image_urls.append(images[i % len(images)])
    
    prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
    response = pipe(prompts)
    print(response)
    #for idx, r in enumerate(response):
    #    print(r)
    #    #pass

if __name__ == '__main__':
    main()

