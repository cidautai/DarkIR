import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
try:
    from .arch_util import EBlock, ProcessBlock, make_layer, ResidualBlock_noBN
except:
    from arch_util import EBlock, ProcessBlock, make_layer, ResidualBlock_noBN

class Attention_Light(nn.Module):
    
    def __init__(self, img_channels = 3, width = 16, spatial = False):
        super(Attention_Light, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels = img_channels, out_channels = width//2, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                ProcessBlock(in_nc = width //2, spatial = spatial),
                nn.Conv2d(in_channels = width//2, out_channels = width, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                ProcessBlock(in_nc = width, spatial = spatial),
                nn.Conv2d(in_channels = width, out_channels = width, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
                ProcessBlock(in_nc=width, spatial = spatial),
                nn.Sigmoid()
                    )
    def forward(self, input):
        return self.block(input)


class Network(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=16, 
                 middle_blk_num=1, 
                 enc_blk_nums=[], 
                 dec_blk_nums=[], 
                 residual_layers = 3, 
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
        
        self.attention1 = Attention_Light(img_channel, width)
        # self.upconv1 = nn.Conv2d(width, width * 2, 1, 1)
        # self.upconv2 = nn.Conv2d(width * 2, width * 4, 1, 1)
        # self.upconv3 = nn.Conv2d(width * 4, width * 8, 1, 1)
        
        self.upconv1 = nn.Sequential(nn.Conv2d(width, width*2, 1, 1),
                                     nn.Conv2d(width*2, width*2, kernel_size=3, stride=2, padding=1, groups=width*2, bias = True))
        self.upconv2 = nn.Sequential(nn.Conv2d(width*2, width*4, 1, 1),
                                     nn.Conv2d(width*4, width*4, kernel_size=3, stride=2, padding=1, groups = width*4, bias = True))
        self.upconv3 = nn.Sequential(nn.Conv2d(width*4, width*8, 1, 1),
                                     nn.Conv2d(width*8, width*8, kernel_size=3, stride=2, padding=1, groups=width*8, bias = True))
        
        # self.recon_trunk_light = nn.Sequential(*[FBlock(c = chan * self.padder_size,
        #                                         DW_Expand=2, FFN_Expand=2, dilations = dilations, 
        #                                         extra_depth_wise = False) for i in range(residual_layers)])

        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf = width * self.padder_size)
        self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, residual_layers)
        
   
        
    def forward(self, input):

        B, C, H, W = input.shape

        h1 =  input
        # generate the different attention layers
        attention1 = self.attention1(input)
        # attention2 = F.interpolate(self.upconv1(attention1), size = (H//2, W//2), mode = 'bilinear')
        # attention3 = F.interpolate(self.upconv2(attention2), size = (H//4, W//4), mode = 'bilinear')
        # attention4 = F.interpolate(self.upconv3(attention3), size= (H//8, W//8), mode ='bilinear')
        attention2 = self.upconv1(attention1)
        attention3 = self.upconv2(attention2)
        attention4 = self.upconv3(attention3)
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

        x = self.middle_blks(x) * attention4
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

# class NetworkLocal(Local_Base, Network):

#     def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         Network.__init__(self, *args, **kwargs)

#         N, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))

#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
  


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
                  residual_layers=residual_layers,
                  dilations = dilations)

    # NAF = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)    
    inp = torch.randn(1, 3, 256, 256)
    out = net(inp)
    
    
