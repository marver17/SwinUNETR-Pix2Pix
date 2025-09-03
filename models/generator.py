import monai.networks
import torch
import torch.nn as nn
from torch.nn import init
import functools
import monai
from models.utils import get_norm_layer,init_net


def define_G(opt):
    """Create a generator

    Parameters:
        opt  -- Configuration object containing all parameters:
            opt.input_nc -- the number of channels in input images
            opt.output_nc -- the number of channels in output images
            opt.ngf -- the number of filters in the last conv layer
            opt.netG -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
            opt.norm -- the name of normalization layers used in the network: batch | instance | none
            opt.use_dropout -- if use dropout layers
            opt.init_type -- the name of initialization method
            opt.init_gain -- scaling factor for normal, xavier and orthogonal initialization

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm)
    print(opt.netG)
    if opt.netG == 'resnet_9blocks':
        net = ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, 
                            norm_layer=norm_layer, use_dropout=opt.use_dropout, n_blocks=9)
    elif opt.netG == 'resnet_6blocks':
        net = ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, 
                            norm_layer=norm_layer, use_dropout=opt.use_dropout, n_blocks=6)
    elif opt.netG == "unet_128":
        net = UnetGenerator3D(opt.input_nc, opt.output_nc, 7, opt.ngf, 
                            norm_layer=norm_layer, use_dropout=opt.use_dropout)
    elif opt.netG == "swinunetr_128":
        net = SwinUNETR(opt)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % opt.netG)

    return init_net(net, opt.init_type, opt.init_gain)


class SwinUNETR(nn.Module) :
    """
        Swin unetr wrap adding an activaion function at the end 
    """
    def init(self, opt):
        super(SwinUNETR, self).__init__()
        net = monai.networks.nets.SwinUNETR(
            in_channels=opt.input_nc,
            out_channels=opt.output_nc,
            dropout_path_rate= 0.4 if opt.use_dropout else 0 
        )
        activation_function = nn.Sigmoid()
        self.model = nn.Sequential(net, activation_function)

    def forward(self, x):
        return self.model(x)


class UnetGenerator3D(nn.Module):
    """Crea un generatore basato su U-Net per dati volumetrici 3D."""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """
        Costruisce un generatore U-Net 3D.
        
        Parametri:
            input_nc (int)  -- numero di canali in ingresso (volumi)
            output_nc (int) -- numero di canali in uscita (volumi)
            num_downs (int) -- numero di downsampling nel U-Net. 
                               Ad esempio, se num_downs == 7, un volume di dimensione 128x128x128
                               verrà ridotto a 1x1x1 al collo di bottiglia.
            ngf (int)       -- numero di filtri nel layer convoluzionale iniziale.
            norm_layer      -- layer di normalizzazione (default: nn.BatchNorm3d).
            use_dropout (bool) -- se utilizzare i layer di dropout.
        """
        super(UnetGenerator3D, self).__init__()

        # Innermost block: input_nc defaults to outer_nc (ngf * 8)
        unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                               norm_layer=norm_layer, innermost=True)  

        # Add intermediate layers with ngf * 8 filters, building outwards
        # For num_downs = 7, num_downs - 5 = 2. This loop runs twice.
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer, use_dropout=use_dropout)

        # Gradually reduce the number of filters while building outwards.
        # The current unet_block (output of the loop above or innermost if loop didn't run)
        # expects ngf*8 input channels.

        # This block expects ngf*4 input channels (input_nc defaults to outer_nc=ngf*4).
        # Its downsampling outputs ngf*8 channels, which matches the submodule's expectation.
        unet_block = UnetSkipConnectionBlock3D(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                               norm_layer=norm_layer)

        # This block expects ngf*2 input channels.
        # Its downsampling outputs ngf*4 channels for the submodule.
        unet_block = UnetSkipConnectionBlock3D(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                               norm_layer=norm_layer)

        # This block expects ngf input channels.
        # Its downsampling outputs ngf*2 channels for the submodule.
        unet_block = UnetSkipConnectionBlock3D(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                               norm_layer=norm_layer)

        # Outermost block: takes input_nc (image channels) and its downsampling outputs ngf channels
        # for the submodule (which is the unet_block constructed just above, expecting ngf channels).
        self.model = UnetSkipConnectionBlock3D(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                               outermost=True, norm_layer=norm_layer)  

    def forward(self, x):
        return self.model(x)

class UnetSkipConnectionBlock3D(nn.Module) :

    """Defines te Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        """
        Construct a Unet submodule with skip connections for 3D inputs.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer.
            inner_nc (int) -- the number of filters in the inner conv layer.
            input_nc (int) -- the number of channels in input volumes/features.
            submodule (UnetSkipConnectionBlock3D) -- previously defined submodules.
            outermost (bool)    -- if this module is the outermost module.
            innermost (bool)    -- if this module is the innermost module.
            norm_layer          -- normalization layer (default: nn.BatchNorm3d).
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock3D,self).__init__()
        self.outermost = outermost

    
        if type(norm_layer) == functools.partial : 
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        if input_nc is None : 
            input_nc = outer_nc
            
        # Downsampling layers: using 3D convolutions
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        # Upsampling layers: using 3D transposed convolutions
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x)
        if not self.outermost:  # Controllo dimensioni per evitare errori in torch.cat
            if x.shape != output.shape:
                print(f"Dimensioni diverse: x={x.shape}, output={output.shape}")
            return torch.cat([x, output], 1)
        else:
            return output
