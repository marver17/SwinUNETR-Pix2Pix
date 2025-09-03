import torch.nn as nn
import functools
from models.utils import get_norm_layer, init_weights, init_net


def define_D(opt):
    """Create a discriminator

    Parameters:
        opt (argparse.Namespace) -- Configuration object containing all parameters:
            opt.input_nc -- the number of channels in input images
            opt.ndf -- the number of filters in the first conv layer
            opt.netD -- the architecture's name: basic | n_layers | pixel
            opt.n_layers_D -- the number of conv layers in the discriminator
            opt.norm -- the type of normalization layers used in the network
            opt.init_type -- the name of the initialization method
            opt.init_gain -- scaling factor for normal, xavier and orthogonal
            opt.gpu_ids -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier (default n_layers=3)
        [n_layers]: User-specified number of conv layers in the discriminator
        [pixel]: 1x1 PixelGAN discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm)

    if opt.netD == 'basic':
        net = NLayerDiscriminator3D(opt.input_nc, opt.ndf, n_layers=3, norm_layer=norm_layer)
    elif opt.netD == 'n_layers':
        net = NLayerDiscriminator3D(opt.input_nc, opt.ndf, opt.n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % opt.netD)

    return init_net(net, opt.init_type, opt.init_gain)



class NLayerDiscriminator3D(nn.Module):
    """Definisce un discriminatore PatchGAN per dati 3D"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """
        Costruisce un discriminatore PatchGAN per dati volumetrici 3D.
        
        Parametri:
            input_nc (int)  -- numero di canali negli input volumetrici.
            ndf (int)       -- numero di filtri nel livello convoluzionale iniziale.
            n_layers (int)  -- numero di layer convoluzionali nel discriminatore.
            norm_layer      -- layer di normalizzazione (default: nn.BatchNorm3d).
        """
        super(NLayerDiscriminator3D, self).__init__()


        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d    # no need to use bias as BatchNorm2d has affine parameters
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence  = []


        if norm_layer == "spectral_norm":
            sequence += [
                norm_layer(nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),  ### applico la spectral normalization direttamente alla matrice dei pesi
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence += [
                nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]


        nf_mult = 1
        nf_mult_prev = 1


        # Graduale incremento del numero di filtri in modo da ottenere il receptive field desisderato, per patchGAN ho n=3 e un RF pari a 70x70x70


        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        if norm_layer == "spectral_norm":
            sequence += [
                norm_layer(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Layer finale per produrre una mappa di predizioni a canale singolo

        if norm_layer == "spectral_norm":
            sequence += [norm_layer(nn.Conv3d(ndf * nf_mult, 1,
                                           kernel_size=kw, stride=1, padding=padw))]
        else:
            sequence += [nn.Conv3d(ndf * nf_mult, 1,
                                 kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Passaggio forward standard."""
        return self.model(input)

