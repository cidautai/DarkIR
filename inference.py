import os
from PIL import Image
import cv2 as cv
from options.options import parse
import argparse
from archs.retinexformer import RetinexFormer
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description="Script for prediction")
parser.add_argument('-p', '--config', type=str, default='./options/inference/LOLBlur.yml', help = 'Config file of prediction')
parser.add_argument('-i', '--inp_path', type=str, default='./images/inputs', 
                help="Folder path")
args = parser.parse_args()


path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# PyTorch library
import torch
import torch.optim
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision.transforms import Resize

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.test_utils import *
from ptflops import get_model_complexity_info

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#define some auxiliary functions
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def path_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)
    
    return img
def normalize_tensor(tensor):
    
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    
    tensor = tensor.squeeze(0)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def load_model(model, path_weights):
    map_location = 'cpu'
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
   
    weights = checkpoints['params']
    weights = {'module.' + key: value for key, value in weights.items()}

    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print(macs, params)
    model.load_state_dict(weights)
    print('Loaded weights correctly')
    
    return model

#parameters for saving model
PATH_MODEL = opt['save']['path']
resize = opt['Resize']

def predict_folder(rank, world_size):
    
    setup(rank, world_size=world_size, Master_port='12354')
    
    # DEFINE NETWORK, SCHEDULER AND OPTIMIZER
    model, _, _ = create_model(opt['network'], rank=rank)

    model = load_model(model, path_weights = opt['save']['path'])
    # create data
    PATH_IMAGES= args.inp_path
    PATH_RESULTS = './images/results'

    #create folder if it doen't exist
    not os.path.isdir(PATH_RESULTS) and os.mkdir(PATH_RESULTS)

    path_images = [os.path.join(PATH_IMAGES, path) for path in os.listdir(PATH_IMAGES) if path.endswith(('.png', '.PNG', '.jpg', '.JPEG'))]
    path_images = [file for file in path_images if not file.endswith('.csv') and not file.endswith('.txt')]
   
    model.eval()
    if rank==0:
        pbar = tqdm(total = len(path_images))
        
    for path_img in path_images:
        tensor = path_to_tensor(path_img).to(device)
        _, _, H, W = tensor.shape
        
        if resize and (H >=1500 or W>=1500):
            new_size = [int(dim//2) for dim in (H, W)]
            downsample = Resize(new_size)
        else:
            downsample = torch.nn.Identity()
        tensor = downsample(tensor)
        
        tensor = pad_tensor(tensor)

        with torch.no_grad():
            output = model(tensor, side_loss=False)
        if resize:
            upsample = Resize((H, W))
        else: upsample = torch.nn.Identity()
        output = upsample(output)
        output = torch.clamp(output, 0., 1.)
        output = output[:,:, :H, :W]
        save_tensor(output, os.path.join(PATH_RESULTS, os.path.basename(path_img)))


        pbar.update(1)
        pass

    print('Finished inference!')
    if rank == 0:
        pbar.close()   
    cleanup()

def main():
    world_size = 1
    print('Used GPUS:', world_size)
    mp.spawn(predict_folder, args =(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()










