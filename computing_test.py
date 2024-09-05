import torch
import time
from ptflops import get_model_complexity_info

from archs import Network
from archs import NAFNet

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

network = 'Network'

if network == 'Network':
    model = Network(img_channel=3, 
                    width=32, 
                    middle_blk_num_enc=2,
                    middle_blk_num_dec=2, 
                    enc_blk_nums=[1,2,3],
                    dec_blk_nums=[3, 1, 1], 
                    dilations=[1, 4, 9],
                    extra_depth_wise=True,
                    ksize=None)
elif network == 'NAFNet':
    model = NAFNet(img_channel=3, 
                    width=32, 
                    middle_blk_num=1, 
                    enc_blk_nums=[2, 2, 4, 8],
                    dec_blk_nums=[2, 2, 2, 2])

else:
    raise NotImplementedError('This network isnt implemented')
model.to(device)
macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False)
print(f'Macs:{macs}')
print(f'Params:{params}')
total_time = 0
size = 1000
for i in range(size):
    
    img = torch.randn((1, 3, 512, 512)).to(device)
    start_time = time.time()

    out = model(img)
    total_time += (time.time() - start_time)
    # print(i)

print('Estimated computing time was:' , total_time/size)


