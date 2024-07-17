import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

try:
    from .nafnet_utils.arch_util import LayerNorm2d
    from .nafnet_utils.arch_model import SimpleGate
except:
    from nafnet_utils.arch_util import LayerNorm2d
    from nafnet_utils.arch_model import SimpleGate

'''
https://github.com/wangchx67/FourLLIE.git
'''

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class FreNAFBlock(nn.Module):
    
    def __init__(self, nc, expand = 2):
        super(FreNAFBlock, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

# ------------------------------------------------------------------------------------------------

class Branch(nn.Module):
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
    
class EBlock_freq(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation, extra_depth_wise=extra_depth_wise))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreNAFBlock(nc = c, expand=2)
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
        x_step2 = self.norm2(y) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2) # size [B, C, H, W]
        x = y * x_freq 
        
        return y + x * self.gamma

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    img_channel = 128
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    # net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    net  = EBlock(c = img_channel, 
                            dilations = dilations,
                            extra_depth_wise=extra_depth_wise)

    inp_shape = (128, 32, 32)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)


    print(macs, params)