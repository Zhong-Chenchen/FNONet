import torch


def rgb_to_ycbcr(input):
    R, G, B = torch.split(input, 1, dim=1)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.172 * R - 0.339 * G + 0.511 * B + 0.5
    Cr = 0.511 * R - 0.428 * G - 0.083 * B + 0.5

    return Y, Cb, Cr


def ycbcr_to_rgb(Y, Cb, Cr):

    R = Y + 1.371 * (Cr - 0.5)
    G = Y - 0.698 * (Cr - 0.5) - 0.336 * (Cb - 0.5)
    B = Y + 1.732 * (Cb - 0.5)

    rgb = torch.cat((R, G, B), dim=1)

    return rgb


if __name__ == '__main__':
    input = torch.randn(1, 3, 32, 32)
    Y, Cb, Cr = rgb_to_ycbcr(input)
    print(Y.shape)
    print(Cb.shape)
    print(Cr.shape)
    rgb = ycbcr_to_rgb(Y, Cb, Cr)
    print(rgb.shape)

