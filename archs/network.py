import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
try:
    from .arch_util import EBlock, Attention_Light
    from .arch_util_freq import EBlock_freq
except:
    from arch_util import EBlock, Attention_Light
    from arch_util_freq import EBlock_freq


class Network(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=16, 
                 middle_blk_num=1, 
                 enc_blk_nums=[], 
                 dec_blk_nums=[],  
                 dilations = [1], 
                 extra_depth_wise = False):
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
                    *[EBlock(chan, dilations = dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[EBlock(chan, dilations = dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num)]
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
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)        
        
        #define the attention layers 
        
        # self.recon_trunk_light = nn.Sequential(*[FBlock(c = chan * self.padder_size,
        #                                         DW_Expand=2, FFN_Expand=2, dilations = dilations, 
        #                                         extra_depth_wise = False) for i in range(residual_layers)])

        # ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf = width * self.padder_size)
        # self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, residual_layers)
        
   
        
    def forward(self, input):

        _, _, H, W = input.shape

        x = self.intro(input)
        
        encs = []
        # i = 0
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            # print(i, x.shape)
            encs.append(x)
            x = down(x)
            # i += 1

        x = self.middle_blks(x)
        # print('3', x.shape)
        # apply the mask
        # x = x * mask
        
        # x = self.recon_trunk_light(x)
        
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

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    residual_layers = 2
    dilations = [1, 4]
    
    net = Network(img_channel=img_channel, 
                  width=width, 
                  middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, 
                  dec_blk_nums=dec_blks,
                  dilations = dilations)

    # NAF = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)    
    inp = torch.randn(1, 3, 256, 256)
    out = net(inp)
    
    
