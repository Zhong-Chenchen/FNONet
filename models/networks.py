import torch
import models.archs.FNONet as FNONet

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'FNONet':
        netG = FNONet.FNONet(nf=opt_net['nf'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

