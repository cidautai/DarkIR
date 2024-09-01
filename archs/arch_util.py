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


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


# class ResidualBlock_noBN(nn.Module):
#     '''Residual block w/o BN
#     ---Conv-ReLU-Conv-+-
#      |________________|
#     '''

#     def __init__(self, nf=64):
#         super(ResidualBlock_noBN, self).__init__()
#         self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

#         # initialization
#         initialize_weights([self.conv1, self.conv2], 0.1)

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.conv1(x), inplace=True)
#         out = self.conv2(out)
#         return identity + out

# class SpaBlock(nn.Module):
#     def __init__(self, nc):
#         super(SpaBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(nc,nc,3,1,1),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(nc, nc, 3, 1, 1),
#             nn.LeakyReLU(0.1, inplace=True))

#     def forward(self, x):
#         return x+self.block(x)

# class FreBlock(nn.Module):
#     def __init__(self, nc):
#         super(FreBlock, self).__init__()
#         self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
#         self.process1 = nn.Sequential(
#             nn.Conv2d(nc, nc, 1, 1, 0),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(nc, nc, 1, 1, 0))
#         self.process2 = nn.Sequential(
#             nn.Conv2d(nc, nc, 1, 1, 0),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(nc, nc, 1, 1, 0))

#     def forward(self, x):
#         _, _, H, W = x.shape
#         x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
#         mag = torch.abs(x_freq)
#         pha = torch.angle(x_freq)
#         mag = self.process1(mag)
#         pha = self.process2(pha)
#         real = mag * torch.cos(pha)
#         imag = mag * torch.sin(pha)
#         x_out = torch.complex(real, imag)
#         x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

#         return x_out+x

# class ProcessBlock(nn.Module):
#     def __init__(self, in_nc, spatial = True):
#         super(ProcessBlock,self).__init__()
#         self.spatial = spatial
#         self.spatial_process = SpaBlock(in_nc) if spatial else nn.Identity()
#         self.frequency_process = FreBlock(in_nc)
#         self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0) if spatial else nn.Conv2d(in_nc,in_nc,1,1,0)

#     def forward(self, x):
#         xori = x
#         x_freq = self.frequency_process(x)
#         x_spatial = self.spatial_process(x)
#         xcat = torch.cat([x_spatial,x_freq],1)
#         x_out = self.cat(xcat) if self.spatial else self.cat(x_freq)

#         return x_out+xori

# class Attention_Light(nn.Module):
    
#     def __init__(self, img_channels = 3, width = 16, spatial = False):
#         super(Attention_Light, self).__init__()
#         self.block = nn.Sequential(
#                 nn.Conv2d(in_channels = img_channels, out_channels = width//2, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
#                 ProcessBlock(in_nc = width //2, spatial = spatial),
#                 nn.Conv2d(in_channels = width//2, out_channels = width, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
#                 ProcessBlock(in_nc = width, spatial = spatial),
#                 nn.Conv2d(in_channels = width, out_channels = width, kernel_size = 1, padding = 0, stride = 1, groups = 1, bias = True),
#                 ProcessBlock(in_nc=width, spatial = spatial),
#                 nn.Sigmoid()
#                     )
#     def forward(self, input):
#         return self.block(input)

class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''
    def __init__(self, c, DW_Expand, dilation = 1):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        
        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation) # the dconv
        )
    def forward(self, input):
        return self.branch(input)
    
class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
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
        x = self.conv1(self.extra_conv(x))
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

class KernelConv2D(nn.Module):
    def __init__(self, ksize=5, act=True):
        super(KernelConv2D, self).__init__()
        self.ksize = ksize
        self.act = act

    def forward(self, feat_in, kernel):
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (self.ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 4, 5).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, -1)
        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        if self.act:
            feat_out = F.leaky_relu(feat_out, negative_slope=0.2, inplace=True)
        return feat_out

class FAC_Block(nn.Module):
    def __init__(self, channels, ksize, dilation = 1, final_block = False):
        super(FAC_Block, self).__init__()
        if final_block:
            self.block_ = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=2 * channels, kernel_size=1, stride=1, padding = 0),
                nn.Conv2d(in_channels=2 * channels, out_channels= channels * ksize**2, kernel_size=3, stride=1, padding = dilation, dilation = dilation, groups = channels),
                # nn.Conv2d(in_channels= 2 * channels, out_channels=channels * ksize**2, kernel_size=1, stride=1, padding = 0),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        
        else:
            self.block_ = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=2 * channels, kernel_size=1, stride=1, padding = 0),
                nn.Conv2d(in_channels=2 * channels, out_channels= channels, kernel_size=3, stride=1, padding = dilation, dilation=dilation, groups = channels),
                # nn.Conv2d(in_channels= 2 * channels, out_channels=channels, kernel_size=1, stride=1, padding = 0),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        
    def forward(self, input):
        return self.block_(input)

class FAC(nn.Module):
    def __init__(self, channels, ksize = 5):
        super(FAC, self).__init__()
        self.block= nn.Sequential(
            FAC_Block(channels=channels, ksize=ksize, dilation = 1),
            FAC_Block(channels=channels, ksize=ksize, dilation = 1),
            FAC_Block(channels=channels, ksize=ksize, dilation = 1),
            FAC_Block(channels=channels, ksize=ksize, dilation = 1,final_block=True)
        )

    def forward(self, input):
        return self.block(input)


#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    img_channel = 3
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

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    output = net(torch.randn((4, 3, 256, 256)))
    # print('Values of EBlock:')
    # print(macs, params)

    channels = 128
    resol = 32
    ksize = 5

    # net = FAC(channels=channels, ksize=ksize)
    # inp_shape = (channels, resol, resol)
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    # print('Values of FAC:')
    # print(macs, params)
