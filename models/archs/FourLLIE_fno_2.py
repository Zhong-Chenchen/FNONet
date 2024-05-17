import functools
import models.archs.arch_util as arch_util
from models.archs.SFBlock import *
from models.archs.FSIB import FuseBlock
import kornia
import torch.nn.functional as F
from models.archs.myblock import FSAIO
from models.archs.torch_rgbto import rgb_to_ycbcr, ycbcr_to_rgb
from models.fno.fnonet import FNO2d, Fnounet
from math import floor
###############################
# 参数增强版

class FourLLIE_fno(nn.Module):
    def __init__(self, nf=64):
        super(FourLLIE_fno, self).__init__()

        # AMPLITUDE ENHANCEMENT
        # self.AmpNet = nn.Sequential(
        #     AmplitudeNet_skip(8),
        #     nn.Sigmoid()
        # )

        self.nf = nf
        self.fnonet_1 = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1, 0),
            FNO2d(modes1=floor(nf / 2) + 1, modes2=floor(nf / 2) + 1, width=nf),
            nn.Conv2d(nf, 3, 1, 1, 0),
            nn.Sigmoid()
        )
        self.fnonet = nn.Sequential(
            Fnounet(3, 16),
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
        self.transformer = SFNet(nf)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)
        self.fuseblock = FuseBlock(64)
        self.fourget = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.fsaio1 = FSAIO(nc=nf)
        self.fsaio2 = FSAIO(nc=nf)
        self.fsaio3 = FSAIO(nc=nf)
        self.fsaio4 = FSAIO(nc=nf)
        self.fsaio5 = FSAIO(nc=nf)
        self.fsaio6 = FSAIO(nc=nf)
        self.fsaio7 = FSAIO(nc=nf)

    def get_mask(self, dark):

        light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1.5, 1.5))
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        return mask.float()

    def forward(self, x):

        _, _, H, W = x.shape
        # transform here
        Y, Cb, Cr = rgb_to_ycbcr(x)
        tensor_cbcr = torch.cat([Y, Cb, Cr], dim=1)
        # enhence Y
        image_fft = torch.fft.fft2(tensor_cbcr, norm='backward')
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)
        curve_amps = self.fnonet_1(tensor_cbcr)
        mag_image = mag_image / (curve_amps + 0.00000001)  # * d4
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
        # # 这里得到了Fspatail
        # fea_light = self.recon_trunk_light(fea)
        # # 这里得到F_fourier
        # fea_unfold = self.transformer(fea)
        fea_fre, fea_spa = self.fsaio1(fea, fea)
        fea_fre, fea_spa = self.fsaio2(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio3(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio4(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio5(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio6(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio7(fea_fre, fea_spa)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = self.get_mask(x_center)
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)

        # fea = fea_fre * (1 - mask) + fea_spa * mask
        fea = fea_spa

        out_noise = self.recon_trunk(fea) # nf
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1) # 2*nf
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise))) # upconv1->4*nf,pixel_shuffle->nf
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1) # 2*nf
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))  #upconv1->4*nf,pixel_shuffle->nf
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1) # 2*nf
        out_noise = self.lrelu(self.HRconv(out_noise)) # nf
        out_noise = self.conv_last(out_noise) # 3
        out_noise = out_noise + x
        out_noise = out_noise[:, :, :H, :W]

        return out_noise, mag_image, x_center, mask

if __name__ == '__main__':
    a = torch.randn(1, 3, 400, 600)
    model = FourLLIE_fno()
    out_noise, mag_image, x_center, mask = model(a)
    print(mag_image.shape)
