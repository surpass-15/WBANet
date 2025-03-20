from torch import nn
import torch
from .cga import SpatialAttention, ChannelAttention, PixelAttention
from .crc_module import DynamicFuse
from ..wtconv2d import WTConv2d
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bias=False, activate=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.activate = activate

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.activate:
            out = self.relu(out)
        return out


class HAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(HAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True).to('cuda')
        self.sigmoid = nn.Sigmoid()
        self.la = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        ).to('cuda')
        # self.conv_mining = BasicBlock(dim, dim, kernel_size=3, stride=1, padding=1, activate=True)
        # # self.msca = DynamicFuse(dim, reduction)
        # self.decoder_temp = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        # self.conv_dw = WTConv2d(dim, dim)


    def forward(self, x, y):
        # x = x + x * torch.sigmoid(self.decoder_temp * self.conv_dw(x).mean(dim=1, keepdim=True))
        initial = x.to('cuda') + y.to('cuda')
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn+self.la(initial)
        pattn2 = self.sigmoid(self.pa(initial.to('cuda'), pattn1.to('cuda')))
        result = initial.to('cuda') + pattn2.to('cuda') * x.to('cuda') + (1 - pattn2.to('cuda')) * y.to('cuda')
        # f_mining = self.conv_mining(x - y)
        # result=result+f_mining
        result = self.conv(result)
        return result