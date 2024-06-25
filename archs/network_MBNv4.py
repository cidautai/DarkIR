import torch
import torch.nn as nn
import torch.nn.functional as F
from .nafnet_utils.arch_model import NAFBlock_dilated, SimpleGate, NAFNet
from .fourllie_archs.SFBlock import AmplitudeNet_skip
from .fourllie_archs.arch_util import make_layer, ResidualBlock_noBN
import kornia
import functools


class Attention_Light(nn.Module):
    
    def __init__(self, img_channels = 3, width = 16):
        super(Attention_Light, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels = img_channels, out_channels = width//2, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                nn.Conv2d(in_channels = width//2, out_channels = width, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                nn.Conv2d(in_channels = width, out_channels = width, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                nn.Sigmoid()
                    )
    def forward(self, input):
        return self.block(input)

class UIB_Extra_DW(nn.Module):
    
    def __init__(self, img_channels = 3, expand_ratio = 4):
        super(UIB_Extra_DW, self).__init__()
        expanded_channels = int(img_channels * expand_ratio)
        self.block = nn.Sequential(
                    nn.Conv2d(in_channels = img_channels, out_channels = img_channels, kernel_size = 3, padding = 1, stride = 1, groups = img_channels, bias = True),
                    nn.Conv2d(in_channels = img_channels, out_channels = expanded_channels, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True)  
        )


    
class Network(nn.Module):
    
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], residual_layers = 3, dilations = [1]):
        super(Network, self).__init__()
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated(chan, dilations = dilations) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock_dilated(chan, dilations = dilations) for _ in range(middle_blk_num)]
            )

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
                    *[NAFBlock_dilated(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)        
        
        #define the attention layers 
        
        self.attention1 = Attention_Light(img_channel, width)
        self.attention2 = Attention_Light(img_channel, width * 2)
        self.attention3 = Attention_Light(img_channel, width * 4)
        self.attention4 = Attention_Light(img_channel, width * 8)
        
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf= chan * self.padder_size)
        self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, residual_layers)
        
   
        
    def forward(self, input):

        B, C, H, W = input.shape
        
        # we calculate the three sizes of our input image
        h1 =  input#F.interpolate(input, size=(H, W), mode = 'area')
        h2 = F.interpolate(h1, size=(H//2, W//2), mode = 'area')
        h3 = F.interpolate(h2, size=(H//4, W//4), mode = 'area')
        h4 = F.interpolate(h3, size=(H//8, W//8), mode = 'area')
        
        # print(h1.shape)
        attention1 = self.attention1(h1)
        attention2 = self.attention2(h2)
        attention3 = self.attention3(h3)
        attentions = [attention1, attention2, attention3]
        # print('Attention1', attention1.shape)
        # print('Attention2', attention2.shape)
        # print('Attention3', attention3.shape)
        
        x = self.intro(input)
        
        encs = []
        # i = 0
        for encoder, down, attention in zip(self.encoders, self.downs, attentions):
            x = encoder(x) * attention
            # print(i, x.shape)
            encs.append(x)
            x = down(x)
            # i += 1

        x = self.middle_blks(x) * self.attention4(h4)
        # print('3', x.shape)
        # apply the mask
        # x = x * mask
        
        x = self.recon_trunk_light(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input
        
        return x[:, :, :H, :W]


if __name__ == '__main__':
    
    img_channel = 3
    width = 16

    enc_blks = [2, 2, 4]
    middle_blk_num = 1
    dec_blks = [2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = Network(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

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