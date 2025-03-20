import torch
from torch import nn


from .deconv import DEConv
from .cga import SpatialAttention, ChannelAttention, PixelAttention
from .crc_module import DynamicFuse


class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0).to('cuda')
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False).to('cuda')
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False).to('cuda')
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False).to('cuda')
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False).to('cuda')
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        # x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        # x=torch.cat([x1, x2, x3,x4], dim=1)
        return  x1+x2+x3+x4

class CDCM2(nn.Module):
    """
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels, out_channels):
        super(CDCM2, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0).to('cuda')
        self.conv2_1 = nn.Conv2d(out_channels, out_channels//4, kernel_size=3, dilation=5, padding=5, bias=False).to('cuda')
        self.conv2_2 = nn.Conv2d(out_channels, out_channels//4, kernel_size=3, dilation=7, padding=7, bias=False).to('cuda')
        self.conv2_3 = nn.Conv2d(out_channels, out_channels//4, kernel_size=3, dilation=9, padding=9, bias=False).to('cuda')
        self.conv2_4 = nn.Conv2d(out_channels, out_channels//4, kernel_size=3, dilation=11, padding=11, bias=False).to('cuda')
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        # x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        x=torch.cat([x1, x2, x3,x4], dim=1)
        return  x




class PDCBlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(PDCBlockTrain, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim//2, kernel_size, bias=True).to('cuda')
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim//2, reduction)
        self.pa = PixelAttention(dim//2)
        self.sigmoid = nn.Sigmoid()
        self.cdcm=CDCM(dim//2,dim//2)
        self.local_att = nn.Sequential(
            nn.Conv2d(dim//2, dim//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim//2),
        ).to('cuda')

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim//2, dim//2, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim//2, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        ).to('cuda')

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)


        # cattn = self.ca(res)
        # sattn = self.sa(res)
        # pattn1 = sattn + cattn
        # pattn2 = self.pa(res, pattn1)

        res=self.cdcm(res)
        xl = self.local_att(res)
        xg = self.global_att(res)
        pattn2 = self.sigmoid(xl + xg)
        res = res * pattn2
        res = res + self.conv2(x)
        return res

class PDCBlockTrain2(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(PDCBlockTrain2, self).__init__()
        self.conv1 = DEConv(dim).to('cuda')
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True).to('cuda')
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.sigmoid = nn.Sigmoid()
        self.cdcm = CDCM2(dim, dim)
        self.local_att = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
        ).to('cuda')

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        ).to('cuda')

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res.to('cuda') + x.to('cuda')
        res = self.conv2(res)



        res = self.cdcm(res)
        xl = self.local_att(res)
        xg = self.global_att(res)
        pattn2 = self.sigmoid(xl + xg)
        res = res * pattn2
        res = res.to('cuda') + x.to('cuda')
        return res


