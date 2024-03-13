import functools
import models.archs.arch_util as arch_util
from models.archs.SFBlock import *
import torch.nn.functional as F
from models.archs.FASIO import FSAIO
from models.archs.torch_rgbto import rgb_to_ycbcr, ycbcr_to_rgb
from models.fno.fnonet import FNO2d
from math import floor


class FNONet(nn.Module):
    def __init__(self, nf=64):
        super(FNONet, self).__init__()
        self.nf = nf
        self.fnonet_1 = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1, 0),
            FNO2d(modes1=floor(nf / 2) + 1, modes2=floor(nf / 2) + 1, width=nf),
            nn.Conv2d(nf, 3, 1, 1, 0),
            nn.Sigmoid()
        )
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.conv_first_1 = nn.Conv2d(3 * 2, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.myupconv = nn.Conv2d(nf*2, nf*2, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fsaio1 = FSAIO(nc=nf)
        self.fsaio2 = FSAIO(nc=nf)
        self.fsaio3 = FSAIO(nc=nf)
        self.fsaio4 = FSAIO(nc=nf)
        self.fsaio5 = FSAIO(nc=nf)
        self.fsaio6 = FSAIO(nc=nf)
        self.fsaio7 = FSAIO(nc=nf)

    def forward(self, x):

        _, _, H, W = x.shape
        # transform here
        Y, Cb, Cr = rgb_to_ycbcr(x)
        tensor_cbcr = torch.cat([Y, Cb, Cr], dim=1)
        image_fft = torch.fft.fft2(tensor_cbcr, norm='backward')
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)
        curve_amps = self.fnonet_1(tensor_cbcr)
        mag_image = mag_image * (curve_amps + 0.00000001)  # * d4
        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='backward').real
        R, G, B = torch.split(img_amp_enhanced, 1, dim=1)
        x_center = ycbcr_to_rgb(R, G, B)

        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), "reflect")
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        L1_fea_1 = self.lrelu(self.conv_first_1(torch.cat((x_center, x), dim=1)))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)

        fea_fre, fea_spa = self.fsaio1(fea, fea)
        fea_fre, fea_spa = self.fsaio2(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio3(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio4(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio5(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio6(fea_fre, fea_spa)
        fea = fea_spa

        out_image = self.recon_trunk(fea) # nf
        out_image = torch.cat([out_image, L1_fea_3], dim=1) # 2*nf
        out_image = self.lrelu(self.pixel_shuffle(self.upconv1(out_image))) # upconv1->4*nf,pixel_shuffle->nf
        out_image = torch.cat([out_image, L1_fea_2], dim=1) # 2*nf
        out_image = self.lrelu(self.pixel_shuffle(self.upconv2(out_image)))  #upconv1->4*nf,pixel_shuffle->nf
        out_image = torch.cat([out_image, L1_fea_1], dim=1) # 2*nf
        out_image = self.lrelu(self.HRconv(out_image)) # nf
        out_image = self.conv_last(out_image) # 3
        out_image = out_image + x
        out_image = out_image[:, :, :H, :W]

        return out_image, mag_image, x_center

if __name__ == '__main__':
    a = torch.randn(1, 3, 400, 600)
    model = FNONet()
    out_noise, mag_image, x_center= model(a)
    print(mag_image.shape)
