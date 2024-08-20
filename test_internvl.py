from infer_ext.framework import lmdeploy_ext

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image

def main():
    pipe = pipeline('/data/models/InternVL-Chat-V1-5', backend_config = PytorchEngineConfig(tp=4, device_type='muxi', block_size=16,))

    # image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    # response = pipe(('describe this image'))

    # print(response)

    prompts = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe this image'},
                {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
            ]
        }
    ]

    # image = load_image("/home/pujiang/zhousl/data/tiger.jpeg")
    # image1 = load_image("/home/pujiang/zhousl/data/cat.jpg")

    # response = pipe(('describe', image), do_preprocess=True)
    # response = pipe(('describe', image1), do_preprocess=True)
    # response = pipe(prompts, do_preprocess=True)

    response = pipe(('What functions do you have?'), do_preprocess=True)
    print(response)
    # raise ValueError("test")

    response = pipe(('How are you?'), do_preprocess=True)
    print(response)

    response = pipe(prompts, do_preprocess=True)
    print(response)

    raise ValueError("test")

if __name__ == '__main__':
    main()
