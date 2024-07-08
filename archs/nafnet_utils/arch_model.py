import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .arch_util import LayerNorm2d
    from .local_arch import Local_Base
except:
    from arch_util import LayerNorm2d
    from local_arch import Local_Base


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True) # the dconv
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp           # size [B, C, H, W]

        x = self.norm1(x) # size [B, C, H, W]

        x = self.conv1(x) # size [B, 2*C, H, W]
        x = self.conv2(x) # size [B, 2*C, H, W]
        x = self.sg(x)    # size [B, C, H, W]
        x = x * self.sca(x) # size [B, C, H, W]
        x = self.conv3(x) # size [B, C, H, W]

        x = self.dropout1(x)

        y = inp + x * self.beta # size [B, C, H, W]

        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

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

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

class Branch(nn.Module):
    def __init__(self, c, DW_Expand, dilation = 1, extra_depth_wise = False):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        self.step1 = nn.Sequential(
                       nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity(), #optional extra dw
                       nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1),
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation), # the dconv
                       SimpleGate() 
        )
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.conv = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1) 
    def forward(self, input):
        
        x = self.step1(input)
        z = self.sca(x)
        x = x* z
        x = self.conv(x)
        
        return x

class Branch_v2(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''
    def __init__(self, c, DW_Expand, dilation = 1, extra_depth_wise = False):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        self.branch = nn.Sequential(
                       nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity(), #optional extra dw
                       nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1),
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation) # the dconv
        )
    def forward(self, input):
        return self.branch(input)

class EBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, FFN_Expand = FFN_Expand, dilation = dilation, drop_out_rate = 0., extra_depth_wise=extra_depth_wise))
            
        assert len(dilations) == len(self.branches)
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        number_branches = len(self.branches)
        y = inp
        x = self.norm1(inp)
        for branch in self.branches:
            y += branch(x)/number_branches
        
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        return y + x * self.gamma

# class EBlock_v2(nn.Module):
#     '''
#     Change this block using Branch_v2. It adds concatenation of the branches instead of element addition. After the concat,
#     we apply a 1x1conv to fuse the information of the different branches before passing them through the SimpleGate and SCA.
#     '''
    
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
#         super().__init__()
#         #we define the 2 branches
        
#         self.branches = nn.ModuleList()
#         for dilation in dilations:
#             self.branches.append(Branch_v2(c, DW_Expand, dilation = dilation, extra_depth_wise=extra_depth_wise))
            
#         assert len(dilations) == len(self.branches)
#         self.dw_channel = DW_Expand * c 
#         self.sca = nn.Sequential(
#                        nn.AdaptiveAvgPool2d(1),
#                        nn.Conv2d(in_channels=self.dw_channel // 2 * len(dilations), out_channels=self.dw_channel // 2 * len(dilations), 
#                                  kernel_size=1, padding=0, stride=1,
#                                  groups=1, bias=True, dilation = 1),  
#         )
#         self.sg1 = SimpleGate()
#         self.conv_fuse= nn.Conv2d(in_channels=self.dw_channel * len(dilations), out_channels=self.dw_channel * len(dilations), 
#                                   kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        
#         self.sg2 = SimpleGate()
#         self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2 * len(dilations), out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)

#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

#     def forward(self, inp):

#         y = inp
#         x = self.norm1(inp)
#         z = self.branches[0](x)
#         for branch in self.branches[1:]:
#             z = torch.cat([z, branch(x)], dim = 1)
        
#         z = self.conv_fuse(z) #[d * 2 * C, H, W]
#         z = self.sg1(z) #[d * C, H, W]
#         x = self.sca(z) * z
#         x = self.conv3(x)
#         y = inp + self.beta * x
#         #second step
#         x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
#         x = self.sg2(x)  # size [B, C, H, W]
#         x = self.conv5(x) # size [B, C, H, W]

#         return y + x * self.gamma
    
class EBlock_v3(nn.Module):
    '''
    Change this block using Branch_v2
    '''
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch_v2(c, DW_Expand, dilation = dilation, extra_depth_wise=extra_depth_wise))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        y = inp
        x = self.norm1(inp)
        z = 0
        for branch in self.branches:
            z += branch(x)
        
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        #second step
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        return y + x * self.gamma

class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out+x

class FPA(nn.Module):
    
    def __init__(self,nc):
        super(FPA, self).__init__()
        self.process_mag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
    def forward(self, input):
        _, _, H, W = input.shape
        x_freq = torch.fft.rfft2(input, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = mag + self.process_mag(mag)
        pha = pha + self.process_pha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out
        

class FBlock(nn.Module):
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super(FBlock, self).__init__()
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch_v2(c, DW_Expand, dilation = dilation, extra_depth_wise=extra_depth_wise))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)



        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        ffn_channel = FFN_Expand * c
        self.conv_fpr_intro = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.fpa = FPA(nc = ffn_channel)
        self.conv_fpr_out = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        y = inp
        x = self.norm1(inp)
        z=0
        for branch in self.branches:
            z += branch(x)
        
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        #Frequency pixel residue
        x = self.conv_fpr_intro(self.norm2(y)) # size [B, C, H, W]
        x = self.fpa(x)  # size [B, C, H, W]
        x = self.conv_fpr_out(x)

        return y + x * self.gamma

if __name__ == '__main__':
    
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = False
    
    # net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    net  = EBlock_v2(c = img_channel, 
                            dilations = dilations,
                            extra_depth_wise=extra_depth_wise)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)


    print(macs, params)