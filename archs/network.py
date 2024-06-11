import torch
import torch.nn as nn
import torch.nn.functional as F
from .nafnet_utils.arch_model import NAFBlock, SimpleGate, NAFNet
from .fourllie_archs.SFBlock import AmplitudeNet_skip
from .fourllie_archs.arch_util import make_layer, ResidualBlock_noBN
import kornia
import functools

class Network(nn.Module):
    
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
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
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
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
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)        
        
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf= chan * self.padder_size)
        self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, 3)
        
        self.AmpNet = nn.Sequential(
            AmplitudeNet_skip(8),
            nn.Sigmoid()
        )

    def get_mask(self, dark):

        light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1.5, 1.5))
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        return mask.float()      
        
    def forward(self, input):
        
        # Mask estimation
        
        _, _, H, W = input.shape
        image_fft = torch.fft.fft2(input, norm='backward')
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)
        curve_amps = self.AmpNet(input)
        mag_image = mag_image / (curve_amps + 0.00000001)  # * d4

        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='backward').real
        
        x_center = img_amp_enhanced

        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), "reflect")
            input = F.pad(input, (0, pad_w, 0, pad_h), "reflect")
            
        mask = self.get_mask(x_center)
        
        
        # Now we the NAFNet begins
        
        x = self.intro(input)
        
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        channel = x.shape[1]
        h_feature = x.shape[2]
        w_feature = x.shape[3]       
        mask = F.interpolate(mask, size = [h_feature, w_feature], mode = 'nearest') 
        mask = mask.repeat(1, channel, 1, 1)
        
        # print('--Info of the mask: ')
        # print('\t',mask.shape, mask.dtype, torch.max(mask), torch.min(mask))
        
        # here we need to apply the mask and the middle blocks
        # print('--Info of the middle level: ')
        # print('\t',x.shape, x.dtype, torch.max(x), torch.min(x))
        x = self.middle_blks(x)
        
        # apply the mask
        x = x * mask
        
        x = self.recon_trunk_light(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input
        
        return x[:, :, :H, :W], x_center
    

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

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    # print('MACs and params of the network:')
    print(macs, params)    
    
    tensor = torch.rand((1, 3, 256, 256))
    # print('Size of input tensor: ',tensor.shape)
    
    output = net(tensor)
    # print('Size of output tensor: ',output.shape)
    
        
