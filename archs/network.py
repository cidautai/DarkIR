import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
try:
    from .arch_util import EBlock, KernelConv2D, FAC
    from .arch_util_freq import EBlock_freq
except:
    from arch_util import EBlock, KernelConv2D, FAC
    from arch_util_freq import EBlock_freq


class Network(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=16, 
                 middle_blk_num_enc=1,
                 middle_blk_num_dec=1, 
                 enc_blk_nums=[], 
                 dec_blk_nums=[],  
                 dilations = [1], 
                 extra_depth_wise = False,
                 ksize = 5,
                 side_out = True):
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
                    *[EBlock_freq(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_enc = \
            nn.Sequential(
                *[EBlock_freq(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            nn.Sequential(
                *[EBlock(chan, dilations = dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
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
                    *[EBlock(chan,dilations = dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)        

        self.side_out = side_out
        # this layer is needed for the computing of the middle loss. It isn't necessary for anything else
        if side_out:
            self.side_out = nn.Conv2d(in_channels = width * 2**len(self.encoders), out_channels = img_channel, 
                                  kernel_size = 3, stride=1, padding=1)
        # self.facs = nn.ModuleList([nn.Identity(), nn.Identity(),
        #                           nn.Identity(),
        #                           nn.Identity())
        # self.kconv_deblur = KernelConv2D(ksize=ksize, act = True)
   
        
    def forward(self, input, side_loss = False):

        _, _, H, W = input.shape

        input = self.check_image_size(input)
        x = self.intro(input)
        
        # encs = []
        facs = []
        # i = 0
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            # x_fac = fac(x)
            facs.append(x)
            # print(i, x.shape)
            # encs.append(x)
            x = down(x)
            # i += 1

        # we apply the encoder transforms
        x_light = self.middle_blks_enc(x)
        
        if side_loss and self.side_out:
            out_side = self.side_out(x_light)
        # calculate the fac at this level
        # x_fac = self.facs[-1](x)
        # facs.append(x_fac)
        # apply the decoder transforms
        x = self.middle_blks_dec(x_light)
        # apply the fac transform over this step
        x = x + x_light

        # print('3', x.shape)
        # apply the mask
        # x = x * mask
        
        # x = self.recon_trunk_light(x)
        i = 0
        for decoder, up, fac_skip in zip(self.decoders, self.ups, facs[::-1]):
            x = up(x)
            if i == 2: # in the toppest decoder step
                x = x + fac_skip
                x = decoder(x)
            else:
                x = x + fac_skip
                x = decoder(x)
            i+=1

        x = self.ending(x)
        x = x + input
        out = x[:, :, :H, :W] # we recover the original size of the image
        if side_loss:
            return out_side, out
        else:        
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x

if __name__ == '__main__':
    
    img_channel = 3
    width = 32

    # enc_blks = [1, 1, 1, 3]
    # middle_blk_num = 3
    # dec_blks = [2, 1, 1, 1]
    
    enc_blks = [1, 2, 3]
    middle_blk_num_enc = 2
    middle_blk_num_dec = 2
    dec_blks = [3, 1, 1]
    residual_layers = None
    dilations = [1, 4, 9]
    extra_depth_wise = True
    ksize = 5
    
    net = Network(img_channel=img_channel, 
                  width=width, 
                  middle_blk_num_enc=middle_blk_num_enc,
                  middle_blk_num_dec= middle_blk_num_dec,
                  enc_blk_nums=enc_blks, 
                  dec_blk_nums=dec_blks,
                  dilations = dilations,
                  extra_depth_wise = extra_depth_wise,
                  ksize = ksize)
    
    # NAF = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)    
    inp = torch.randn(1, 3, 256, 256)
    out_side, out = net(inp, side_loss= True)
    print(out_side.shape, out.shape)
    # print(len(out))
    
    
