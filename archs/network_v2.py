import torch
import torch.nn as nn
import torch.nn.functional as F
from nafnet_utils.arch_model import NAFBlock, SimpleGate, NAFNet
from fourllie_archs.SFBlock import AmplitudeNet_skip, FreBlock
from fourllie_archs.arch_util import make_layer, ResidualBlock_noBN
import kornia
import functools

class Light_Map(nn.Module):
    
    def __init__(self, img_channels = 3, width = 16):
        super(Light_Map, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels = img_channels, out_channels = width//2, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                nn.Conv2d(in_channels = width//2, out_channels = width, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                nn.Conv2d(in_channels = width, out_channels = width, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                nn.Sigmoid()
                    )
    def forward(self, input):
        return self.block(input)
    
class IBlock(nn.Module):
    def __init__(self, in_nc, spatial = True):
        super(IBlock,self).__init__()
        self.spatial = spatial
        self.spatial_process = NAFBlock( c = in_nc) if spatial else nn.Identity()
        self.frequency_process = FreBlock(nc = in_nc)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0) if spatial else nn.Conv2d(in_nc,in_nc,1,1,0)

    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        xcat = torch.cat([x_spatial,x_freq],1)
        x_out = self.cat(xcat) if self.spatial else self.cat(x_freq)

        return x_out+xori  
    
class Network(nn.Module):
    
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], 
                 enc_blk_nums_map = [], middle_blk_num_map = 3, residual_layers = 3, spatial = False):
        super(Network, self).__init__()
        
        assert len(enc_blk_nums) == len(enc_blk_nums_map), 'Not the same len for encoders in illumination map and image'
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.illumination_map = Light_Map(img_channel, width)

        self.encoders = nn.ModuleList()
        self.encoders_map = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # self.middle_blks = nn.ModuleList()
        # self.middle_blks_map = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.downs_map = nn.ModuleList()

        chan = width
        for num, num_map in zip(enc_blk_nums, enc_blk_nums_map):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            self.encoders_map.append(
                nn.Sequential(
                    *[IBlock(in_nc=chan, spatial = spatial) for _ in range(num_map)]
                )
            )
            self.downs_map.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
            
        self.middle_blks_map = nn.Sequential(*[IBlock(in_nc = chan, spatial = spatial) for _ in range(middle_blk_num_map)])
        self.reconstruct_light = nn.Sequential(*[NAFBlock(c = chan) for _ in range(residual_layers)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)        
        
        
    def forward(self, input):

        B, C, H, W = input.shape
        
        #calculate the illumination map and the starting point of the image in the encoder
        ill_map = self.illumination_map(input)
        x = self.intro(input)
        print('Ill map shape', ill_map.shape)
        print('Image shape', x.shape)
        encs = []
        i = 0
        for encoder, down, encoder_map, down_map in zip(self.encoders, self.downs, self.encoders_map, self.downs_map):
            ill_map = encoder_map(ill_map)
            x = encoder(x) * ill_map
            print(i, x.shape)
            encs.append(x)
            x = down(x)
            ill_map = down_map(ill_map)
            # i += 1

        ill_map = self.middle_blks_map(ill_map)
        x = self.middle_blks(x) * ill_map
        # print('3', x.shape)
        # apply the mask
        # x = x * mask
        
        x = self.reconstruct_light(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input
        
        return x[:, :, :H, :W]
    
    
    
if __name__ == '__main__':
    
    img_channel = 3
    width = 32

    enc_blks = [1, 1, 3]
    enc_blk_nums_map = [1, 1, 3]
    middle_blk_num = 3
    middle_blk_num_map = 3
    dec_blks = [3, 1, 1]

    
    net = Network(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, middle_blk_num_map=middle_blk_num_map,
                      enc_blk_nums_map=enc_blk_nums_map, spatial= False)

    # NAF = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    # print('MACs and params of the network:')
    print(macs, params)    
    
    tensor = torch.rand((1, 3, 256, 256))
    # print('Size of input tensor: ',tensor.shape)
    
    # output = net(tensor)
    # print('Size of output tensor: ',output.shape)