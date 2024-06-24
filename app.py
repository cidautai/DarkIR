import gradio as gr 
from PIL import Image
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
import yaml
from huggingface_hub import hf_hub_download

from archs import Network_v3
from options.options import parse

path_opt = './options/test/LOLBlur.yml'

opt = parse(path_opt)

#define some auxiliary functions
pil_to_tensor = transforms.ToTensor()
    
# define some parameters based on the run we want to make
#selected network
network = opt['network']['name']

PATH_MODEL = opt['save']['path']

model = Network_v3(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'],
                    dilations=opt['network']['dilations'])

checkpoints = torch.load(opt['save']['best'])
# print(checkpoints)
model.load_state_dict(checkpoints['model_state_dict'])


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

def load_img (filename):
    img = Image.open(filename).convert("RGB")
    img_tensor = pil_to_tensor(img)
    return img_tensor

def process_img(image):
    img = np.array(image)
    img = img / 255.
    img = img.astype(np.float32)
    y = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        x_hat = model(y)

    restored_img = x_hat.squeeze().permute(1,2,0).clamp_(0, 1).cpu().detach().numpy()
    restored_img = np.clip(restored_img, 0. , 1.)

    restored_img = (restored_img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return Image.fromarray(restored_img) #(image, Image.fromarray(restored_img))

title = "Low-Light-Deblurring ✏️🖼️ 🤗"
description = ''' ## [Low Light Image deblurring enhancement](https://github.com/cidautai/Net-Low-light-Deblurring)

[Daniel Feijoo](https://github.com/danifei)

Fundación Cidaut


> **Disclaimer:** please remember this is not a product, thus, you will notice some limitations.
**This demo expects an image with some degradations.**
Due to the GPU memory limitations, the app might crash if you feed a high-resolution image (2K, 4K). <br>
The model was trained using mostly synthetic data, thus it might not work great on real-world complex images. 

<br>
'''

examples = [['examples/inputs/0010.png'],
            ['examples/inputs/0060.png'], 
            ['examples/inputs/0075.png'], 
            ["examples/inputs/0087.png"], 
            ["examples/inputs/0088.png"]]

css = """
    .image-frame img, .image-container img {
        width: auto;
        height: auto;
        max-width: none;
    }
"""

demo = gr.Interface(
    fn = process_img,
    inputs = [
            gr.Image(type = 'pil', label = 'input')
    ],
    outputs = [gr.Image(type='pil', label = 'output')],
    title = title,
    description = description,
    examples = examples,
    css = css
)

if __name__ == '__main__':
    demo.launch()