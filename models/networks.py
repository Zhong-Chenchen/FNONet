import torch
import models.archs.FourLLIE as FourLLIE
import models.archs.FourLLIE_orj as FourLLIE_orj
import models.archs.FourLLIE_my as FourLLIE_my
import models.archs.FourLLIE_my_withycbcr as FourLLIE_Ycbcr
import models.archs.FourLLIE_my_withycbcr_change as FourLLIE_change
import models.archs.FourLLIE_my_withycbcr_changessh as FourLLIE_changessh
import models.archs.FourLLIE_fno_2 as FourLLIE_fno
from models.archs.EnhanceN_arch_LOL import InteractNet as UHD_Net
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'FourLLIE':
        netG = FourLLIE.FourLLIE(nf=opt_net['nf'])
    elif which_model == 'FourLLIE_orj':
        netG = FourLLIE_orj.FourLLIE(nf=opt_net['nf'])
    elif which_model == 'FourLLIE_my':
        netG = FourLLIE_my.FourLLIE(nf=opt_net['nf'])
    elif which_model == 'UHD_Net':
        netG = UHD_Net()
    elif which_model == 'FourLLIE_Ycbcr':
        netG = FourLLIE_Ycbcr.FourLLIE_my(nf=opt_net['nf'])
    elif which_model == 'FourLLIE_Ycbcr_change':
        netG = FourLLIE_change.FourLLIE_my(nf=opt_net['nf'])
    elif which_model == 'FourLLIE_Ycbcr_changessh':
        netG = FourLLIE_changessh.FourLLIE_my(nf=opt_net['nf'])
    elif which_model == 'FourLLIE_fno':
        netG = FourLLIE_fno.FourLLIE_fno(nf=opt_net['nf'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

