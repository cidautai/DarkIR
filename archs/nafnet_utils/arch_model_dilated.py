import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch_util import LayerNorm2d
from archs.local_arch import Local_Base
from archs.arch_model import NAFBlock

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Branch(nn.Module):
    def __init__(self, c, DW_Expand, FFN_Expand = 2, dilation = 1, drop_out_rate = 0.):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        self.step1 = nn.Sequential(
                       LayerNorm2d(c),
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
        # print(input.shape)
        x = self.step1(input)
        z = self.sca(x)
        x = x* z
        x = self.conv(x)
        # print(x.shape)
        return x
        

class NAFBlock_dilated_124(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., dilations = [1, 2, 4]):
        super().__init__()
        dw_channel = c * DW_Expand
        #we define the 3 branches
        self.branch_1 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[0], drop_out_rate = 0.)
        self.branch_2 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[1], drop_out_rate = 0.)
        self.branch_3 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[2], drop_out_rate = 0.)
        
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.delta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.epsilon = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        x_1 = self.branch_1(inp)
        x_2 = self.branch_2(inp)
        x_3 = self.branch_3(inp)


        # y = inp + x_1 * self.beta + x_2 * self.delta + x_3 * self.epsilon # size [B, C, H, W]
        y = inp + x_1 / 3 +  x_2 / 3 + x_3 / 3 # size [B, C, H, W]
        
        
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        # x = self.dropout2(x)

        return y + x * self.gamma
    

class NAFBlock_dilated_124_concat(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., dilations = [1, 2, 4]):
        super().__init__()
        dw_channel = c * DW_Expand
        #we define the 3 branches
        self.branch_1 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[0], drop_out_rate = 0.)
        self.branch_2 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[1], drop_out_rate = 0.)
        self.branch_3 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[2], drop_out_rate = 0.)

        # # SimpleGate
        # self.sg = SimpleGate()
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=4 * c, out_channels= 2 * c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(4 * c)

        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.delta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.epsilon = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, inp):

        x_1 = self.branch_1(inp)
        x_2 = self.branch_2(inp)
        x_3 = self.branch_3(inp)

        y = inp + x_1 / 3 +  x_2 / 3 + x_3 / 3 # size [B, C, H, W]
        # print(inp.shape, x_1.shape, x_2.shape, x_3.shape)
        x = torch.cat([inp, x_1, x_2, x_3], dim = 1)
        
        x = self.conv4(self.norm2(x)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        # x = self.dropout2(x)

        return y + x * self.gamma

class NAFBlock_dilated_14(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., dilations = [1, 4]):
        super().__init__()
        dw_channel = c * DW_Expand
        # print('Instantiation')
        #we define the 3 branches
        self.branch_1 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[0], drop_out_rate = 0.)
        self.branch_2 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[1], drop_out_rate = 0.)

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.delta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.epsilon = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, inp):

        x_1 = self.branch_1(inp)
        x_2 = self.branch_2(inp)

        y = inp + x_1 / 2 +  x_2 / 2  # size [B, C, H, W]
        
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        # x = self.dropout2(x)

        return y + x * self.gamma
    
    
class NAFBlock_dilated_14_concat(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., dilations = [1,4]):
        super().__init__()
        dw_channel = c * DW_Expand
        #we define the 3 branches
        self.branch_1 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[0], drop_out_rate = 0.)
        self.branch_2 = Branch(c, DW_Expand, FFN_Expand = 2, dilation = dilations[1], drop_out_rate = 0.)

        # # SimpleGate
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=3 * c, out_channels= 2 * c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(3 * c)

        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.delta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.epsilon = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, inp):

        x_1 = self.branch_1(inp)
        x_2 = self.branch_2(inp)

        y = inp + x_1 / 3 +  x_2 / 3  # size [B, C, H, W]
        # print(inp.shape, x_1.shape, x_2.shape, x_3.shape)
        x = torch.cat([inp, x_1, x_2], dim = 1)
        
        x = self.conv4(self.norm2(x)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        # x = self.dropout2(x)

        return y + x * self.gamma   

# NOW THE MODELS
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
    
class NAFNet_dilated_124(nn.Module):
    '''
    With different NAFBlocks in each encoder level
    '''
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dilations = [1, 2, 4]):
        super().__init__()
        # print('Instantiation')
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
        # ------------------------ Encoder blocks
        for num in enc_blk_nums[:-1]: # only the three first levels
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated_124(chan, dilations = dilations) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # in the last encoder we only apply the normal NAFNet 
        self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(enc_blk_nums[-1])]
                )
        )
        
        self.downs.append(
            nn.Conv2d(chan, 2*chan, 2, 2)
        )
        
        chan = chan *2

        #---------------------- Middle blocks

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        # -------------------- Decoder blocks
        
        self.ups.append(
            nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
            )
        )
        chan = chan//2
        self.decoders.append(
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(dec_blk_nums[0])]
            )
        )
        
        
        for num in dec_blk_nums[1:]: # again, we only apply the NAFBlock standard in the lowest value
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated_124(chan, dilations = dilations) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        # print('End of Instantiation')

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        # print('Checked image size')
        # print(inp.shape)
        x = self.intro(inp)
        # print('After intro')
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # print('encoder')

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        # print(x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # print(mod_pad_w, mod_pad_h)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNet_dilated_124_concat(nn.Module):
    '''
    With concat
    '''
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dilations = [1, 2, 4]):
        super().__init__()
        # print('Instantiation')
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
        # ------------------------ Encoder blocks
        for num in enc_blk_nums[:-1]: # only the three first levels
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated_124_concat(chan, dilations = dilations) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # in the last encoder we only apply the normal NAFNet 
        self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(enc_blk_nums[-1])]
                )
        )
        
        self.downs.append(
            nn.Conv2d(chan, 2*chan, 2, 2)
        )
        
        chan = chan *2

        #---------------------- Middle blocks

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        # -------------------- Decoder blocks
        
        self.ups.append(
            nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
            )
        )
        chan = chan//2
        self.decoders.append(
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(dec_blk_nums[0])]
            )
        )
        
        
        for num in dec_blk_nums[1:]: # again, we only apply the NAFBlock standard in the lowest value
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated_124_concat(chan, dilations = dilations) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        # print('End of Instantiation')

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        # print('Checked image size')
        # print(inp.shape)
        x = self.intro(inp)
        # print('After intro')
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # print('encoder')

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        # print(x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # print(mod_pad_w, mod_pad_h)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNet_dilated_14(nn.Module):
    '''
    With 1-4 dilations
    '''
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dilations = [1, 4]):
        super().__init__()
        # print('Instantiation')
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
        # ------------------------ Encoder blocks
        for num in enc_blk_nums[:-1]: # only the three first levels
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated_14(chan, dilations = dilations) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # in the last encoder we only apply the normal NAFNet 
        self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(enc_blk_nums[-1])]
                )
        )
        
        self.downs.append(
            nn.Conv2d(chan, 2*chan, 2, 2)
        )
        
        chan = chan *2

        #---------------------- Middle blocks

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        # -------------------- Decoder blocks
        
        self.ups.append(
            nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
            )
        )
        chan = chan//2
        self.decoders.append(
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(dec_blk_nums[0])]
            )
        )
        
        
        for num in dec_blk_nums[1:]: # again, we only apply the NAFBlock standard in the lowest value
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated_14(chan, dilations = dilations) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        # print('End of Instantiation')

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        # print('Checked image size')
        # print(inp.shape)
        x = self.intro(inp)
        # print('After intro')
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # print('encoder')

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        # print(x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # print(mod_pad_w, mod_pad_h)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
class NAFNet_dilated_14_concat(nn.Module):
    '''
    With concat
    '''
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dilations = [1, 4]):
        super().__init__()
        # print('Instantiation')
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
        # ------------------------ Encoder blocks
        for num in enc_blk_nums[:-1]: # only the three first levels
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated_14_concat(chan, dilations = dilations) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # in the last encoder we only apply the normal NAFNet 
        self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(enc_blk_nums[-1])]
                )
        )
        
        self.downs.append(
            nn.Conv2d(chan, 2*chan, 2, 2)
        )
        
        chan = chan *2

        #---------------------- Middle blocks

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        # -------------------- Decoder blocks
        
        self.ups.append(
            nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
            )
        )
        chan = chan//2
        self.decoders.append(
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(dec_blk_nums[0])]
            )
        )
        
        
        for num in dec_blk_nums[1:]: # again, we only apply the NAFBlock standard in the lowest value
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock_dilated_14_concat(chan, dilations = dilations) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        # print('End of Instantiation')

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        # print('Checked image size')
        # print(inp.shape)
        x = self.intro(inp)
        # print('After intro')
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # print('encoder')

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        # print(x.shape)
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # print(mod_pad_w, mod_pad_h)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
