import torch.nn as nn
import torch
from models.archs import common

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #通过全局池化函数，实际上实现了对像素的平均
        self.conv1 = nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True)
        self.sigmod1 = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.sigmod1(y)
        return x*y


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction=reduction))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res


if __name__ == '__main__':
    model = RCAB(conv=common.default_conv, n_feat=16, kernel_size=1, reduction=8)
    a = torch.randn(1, 16, 32, 32)
    a = model(a)
    print(a.shape)