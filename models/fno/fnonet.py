import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

# fourier_2d_time.py 70-141行
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        # 保留了几行源代码所带的注释，可以根据这个注释去看下面的代码
        # input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        # input shape: (batchsize, x=64, y=64, c=12)
        # output: the solution of the next timestep
        # output shape: (batchsize, x=64, y=64, c=1)
        self.modes1 = modes1 # Fourier Layer所需要的参数
        self.modes2 = modes2 # Fourier Layer所需要的参数
        self.width = width # Fourier Layer所需要的参数
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width) #将输入的12个channel映射到想要的channel，这里设置为width个channel
        # 对应着上图(a)里的4个Fourier Layer，具体结构后面会讲
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        # 对应着上图(b)里的W，类似ResNet的shortcut结构
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        # 设置了bn层，但是该代码并没有使用
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)
        # 全连接层，用于将channel数从width映射到128
        self.fc1 = nn.Linear(self.width, 128)
        # 全连接层，用于将channel数从128映射到1，即得到最终输出的维度
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 这部分就是源代码注释里的 10 timesteps + 2 locations，增加两个维度存储位置信息
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # x = self.fc0(x)
        # x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        # 经过Fourier Layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # 经过Fourier Layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # 经过Fourier Layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # 经过Fourier Layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # 经过两个全连接层，将channel维度映射到目标输出的维度
        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        # x = x.permute(0, 2, 3, 1)
        # x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        # width = 20, modes1=12, modes2=12
        # in_channels = out_channels = width =20
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2 #傅里叶模态相乘的数量，最多floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels)) # 保证scale不变所设置的量
        # 先随机初始化两个parameter，维度为[20,20,12,12]，并乘上缩放因子scale
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # 定义向量乘的规则，即定义input和weights如何做乘，理解起来略微抽象
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # 可以简单理解成前两个维度做向量乘，即(batch, in_channel),(in_channel, out_channel) -> (batch, out_channel）
        # [20,20,12,12] * [20,20,12,12] = [20,20,12,12]
        # 在这里主要是in_channel和out_channel与batch维度都是20，所以理解起来容易混淆
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # 最初输入的x的维度为[20,64,64,20]->[bathsize，resolution，resolution，channel by fc0]
        # 经过permute之后，x的维度变成了[20, 20, 64, 64]，即[bathsize，channel by fc0，resolution，resolution]
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # 将x做傅里叶变换，维度变为x_ft[20, 20, 64, 33]，可自行查阅fft算法的原理，网上资料充足在此不赘述
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        # 定义out_ft变量用于存储结果，维度为[20, 12, 64, 33]，因为modes设置的是12，因此只有12个channel用于相乘
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        # 根据前面的向量乘法定义规则，即(batch, in_channel),(in_channel, out_channel) -> (batch, out_channel）
        # out_ft[0:20,0:12,0:12,0:12] = x_ft[0:20, 0:20, 0:12, 0:12] * weights1[0:20, 0:20, 0:12, 0:12]
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # 同理，只不过倒着处理12个模态
        # out_ft[0:20,0:12,-12:end,-12:end] = x_ft[0:20,0:12,-12:end,-12:end] * weights2[0:20, 0:20, 0:12, 0:12]
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        # 经过一波相乘后，out_ft在[0,12]和[-12:end]处有信息，而之外的区间内是0
        # Return to physical space
        # 傅里叶逆变换，将傅里叶空间转换到物理空间
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class Fnounet(nn.Module):
    def __init__(self, in_channels, nf, n=1):
        super(Fnounet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, nf, 1, 1, 0),
            FNO2d(modes1=floor(nf / 2) + 1, modes2=floor(nf / 2) + 1, width=nf),
        )
        self.conv1 = FNO2d(modes1=floor(nf / 2) + 1, modes2=floor(nf / 2) + 1, width=nf)
        self.conv2 = FNO2d(modes1=floor(nf / 2) + 1, modes2=floor(nf / 2) + 1, width=nf)
        self.conv3 = FNO2d(modes1=floor(nf / 2) + 1, modes2=floor(nf / 2) + 1, width=nf)
        self.conv4 = nn.Sequential(
            FNO2d(modes1=floor(2*nf / 2) + 1, modes2=floor(2*nf / 2) + 1, width=2*nf),
            nn.Conv2d(nf * 2, nf, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            FNO2d(modes1=floor(2*nf / 2) + 1, modes2=floor(2*nf / 2) + 1, width=2*nf),
            nn.Conv2d(nf * 2, nf, 1, 1, 0),
        )
        self.convout = nn.Sequential(
            FNO2d(modes1=floor(2*nf / 2) + 1, modes2=floor(2*nf / 2) + 1, width=2*nf),
            nn.Conv2d(nf * 2, in_channels, 1, 1, 0),
        )

    def forward(self, x):

        x = self.conv0(x)
        #x = self.calayer1(x)#
        x1 = self.conv1(x)
        #x1 = self.calayer1(x1)
        x2 = self.conv2(x1)
        #x2 = self.calayer1(x2)#
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat((x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x1, x4), dim=1))
        xout = self.convout(torch.cat((x, x5), dim=1))

        return xout

if __name__ == '__main__':
    a = torch.randn(1, 3, 400, 600)
    model = Fnounet(3, 29)
    result = model(a)
    print(result.shape)