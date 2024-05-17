import torch
import torch.nn as nn
from einops import rearrange


class FreBlock(nn.Module):
    '''
    The Fourier Processing (FP) Block in paper
    '''
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

class SpaBlock(nn.Module):
    '''
    The Spatial Processing (SP) Block in paper
    '''
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return x+self.block(x)


class Attention(nn.Module):
    '''
    Attention module: A part in the frequency-spatial interaction block
    '''
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FuseBlock(nn.Module):
    '''
    The FuseBlock: The module to make frequency-spatial interaction block
    '''
    def __init__(self, nc):
        super(FuseBlock, self).__init__()
        self.fre = nn.Conv2d(nc, nc, 3, 1, 1)
        self.spa = nn.Conv2d(nc, nc, 3, 1, 1)
        self.fre_att = Attention(dim=nc)
        self.spa_att = Attention(dim=nc)
        self.fuse = nn.Sequential(nn.Conv2d(2*nc, nc, 3, 1, 1), nn.Conv2d(nc, 2*nc, 3, 1, 1), nn.Sigmoid())

    def forward(self, fre, spa):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.fre_att(fre, spa)+fre
        spa = self.fre_att(spa, fre)+spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class FSAIO(nn.Module):
    '''
    Frequency Spatial All in One Module
    '''
    def __init__(self, nc):
        super(FSAIO, self).__init__()
        self.freblock = FreBlock(nc=nc)
        self.spablock = SpaBlock(nc=nc)
        self.fuseblock = FuseBlock(nc=nc)

    def forward(self, fre, spa):
        fre_out = self.freblock(fre)
        spa_out = self.spablock(spa)
        spa_out = self.fuseblock(fre=fre_out, spa=spa_out)
        return fre_out, spa_out


if __name__ == '__main__':
    spa = torch.randn(1, 64, 64, 64)
    fre = torch.randn(1, 64, 64, 64)
    model = FSAIO(nc=64)
    fre_out, spa_out = model(spa=spa, fre=fre)
    print(fre_out.shape)
    print(spa_out.shape)

