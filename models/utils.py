import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

#### Utily function to customize the network 

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. 
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.


    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters.
    For InstanceNorm, we do not use learnable affine parameters. 
    
    """

    if norm_type == 'batch' :
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance' :
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'spectral' :
        norm_layer = lambda x: spectral_norm(x)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        
    return norm_layer


def save_image(real_ct, real_mr, fake_mr, subject_information, output_dir,metrics):
    """Save images and metrics in NIFTI format.
    
    Args:
        real_ct (torch.Tensor): Input CT image
        real_mr (torch.Tensor): Reference MR image
        fake_mr (torch.Tensor): Generated MR image
        subject_information (dict): Contains dataset and subject information
        output_dir (str): Base output directory path
        metrics (dict, optional): Dictionary containing metrics to save
    """

    import os
    import json
    from monai.transforms import SaveImage

    subject_path = os.path.join(output_dir, subject_information["dataset"], subject_information["subject"])
    os.makedirs(subject_path, exist_ok=True)

    ct_saver = SaveImage(
        output_dir=subject_path,
        output_postfix="input_ct",
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False,
        print_log=False
    )
    #ct_saver(real_ct)
    
    ref_mr_saver = SaveImage(
        output_dir=subject_path,
        output_postfix="reference_mr",
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False,
        print_log=False
    )
    #ref_mr_saver(real_mr)
    
    gen_mr_saver = SaveImage(
        output_dir=subject_path,
        output_postfix="generated_mr",
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False,
        print_log=False
    )
    gen_mr_saver(fake_mr)
    
    if metrics is not None:
        metrics_file = os.path.join(subject_path, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)


def save_image_2(real_A,real_B,synth, subject_information,opt,metrics):
    """Save images and metrics in NIFTI format.
    
    Args:
        real_A (torch.Tensor): Input CT image
        real_B (torch.Tensor): Reference MR image
        fake_A (torch.Tensor): Generated MR image
        subject_information (dict): Contains dataset and subject information
        output_dir (str): Base output directory path
        metrics (dict, optional): Dictionary containing metrics to save
    """

    import os
    import json
    from monai.transforms import SaveImage

    subject_path = os.path.join(output_dir, subject_information["dataset"], subject_information["subject"])
    os.makedirs(subject_path, exist_ok=True)

    
    if "A" in opt.save_image.modalities : 
        real_A_saver = SaveImage(
        output_dir=subject_path,
        output_postfix="input",
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False,
        print_log=False
        )
        real_A_saver(real_A)
    elif  "B" in opt.save_image.modalities : 
        real_B_saver = SaveImage(
            output_dir=subject_path,
            output_postfix="target",
            output_ext=".nii.gz",
            resample=False,
            separate_folder=False,
            print_log=False
        )
        real_B_saver(real_B)
    
    elif "synth" in opt.save_image.modalities : 
        synth_saver = SaveImage(
            output_dir=subject_path,
            output_postfix="synth",
            output_ext=".nii.gz",
            resample=False,
            separate_folder=False,
            print_log=False
        )
        synth_saver(synth)
    else :
        raise ValueError("No modalities selected")

    if metrics is not None:
        metrics_file = os.path.join(subject_path, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
    else : 
        print("Metrics were not printed")

class GAN_loss_definition(nn.Module) :
    
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Define the GAN loss function
            https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

        Parameters:
            gan_mode (str) -- The name of the GAN loss function: 'lsgan' | 'gan' | 'wgan' | 'hinge'.
                                 'gan' refers to the vanilla GAN loss with BCEWithLogits.
            target_real_label (float) -- Label for real images (default: 1.0).
            target_fake_label (float) -- Label for fake images (default: 0.0).
        """
        super().__init__()
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_mode == 'gan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode in ['wgan', 'wgan-gp']: 
            self.loss = None
        elif self.gan_mode == 'hinge':
            self.loss = None
        else:
            raise NotImplementedError(
                f"GAN mode {self.gan_mode} not implemented. Supported modes are: 'lsgan', 'gan', 'wgan', 'hinge'."
            )

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) -- Typically the prediction from a discriminator.
            target_is_real (bool) -- If the ground truth label is for real images or fake images.

        Returns:
            A label tensor filled with the ground truth label, and with the same size and device as the input.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real, is_discriminator=True):
        """Calculate the GAN loss"""
        
        if self.gan_mode in ['lsgan', 'gan']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
            
            # Controllo stabilità numerica
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Invalid loss detected in {self.gan_mode}")
                print(f"  prediction: {prediction.min().item():.4f} to {prediction.max().item():.4f}")
                print(f"  target: {target_tensor.min().item():.4f} to {target_tensor.max().item():.4f}")
                # Fallback per evitare crash
                loss = torch.tensor(1.0, device=prediction.device, requires_grad=True)
                
        elif self.gan_mode in ['wgan', "wgan-gp"]:
            # WGAN: non usa target_is_real nello stesso modo
            loss = -prediction.mean() if target_is_real else prediction.mean()
            
        elif self.gan_mode == 'hinge':
            if is_discriminator:
                # Discriminatore: max(0, 1 - D(real)) per real, max(0, 1 + D(fake)) per fake
                loss = torch.relu(1.0 - prediction).mean() if target_is_real else torch.relu(1.0 + prediction).mean()
            else:
                # Generatore: -D(fake) (assume target_is_real=True quando chiamato dal generatore)
                loss = -prediction.mean()
        else:
            raise NotImplementedError(f"GAN mode {self.gan_mode} forward pass not implemented.")
        
        return loss

def compute_gradient_penalty(critic, real_B_samples, fake_B_samples, real_A_samples, constant=1, device='cpu', lambda_gp=10.0):

    """Calculates the gradient penalty loss for WGAN GP.
       Penalizes the gradient norm of the critic's output w.r.t. interpolated samples.

    Parameters:
        critic (nn.Module): The critic (discriminator) network.
        real_B_samples (torch.Tensor): Real samples from the target domain (e.g., real_B).
        fake_B_samples (torch.Tensor): Fake samples generated for the target domain (e.g., fake_B).
        real_A_samples (torch.Tensor): Real samples from the source domain (e.g., real_A, the condition).
        # type (str)                 : If we mix real and fake data or not [real | fake | mixed], # Removed as GP for WGAN-GP is typically 'mixed'
        constatnt(float) : L in the formula.In numerical formula we have (||gradient||_2 - constant)
        lambda_gp (float): Weight for the gradient penalty. Not used in this function for calculation,
                           but often passed around with GP logic. The penalty itself is returned.

    Returns:
        torch.Tensor: The gradient penalty.
    """
    # Random weight term for interpolation between real and fake samples
    # Shape of alpha: (batch_size, 1, 1, 1, 1) for 3D data to broadcast correctly
    # Get random interpolation between real and fake samples
    # Use .data for samples if they might have been involved in other ops and to avoid graph issues for interpolation
    
    alpha_shape = [real_B_samples.size(0)] + [1] * (real_B_samples.dim() - 1)
    alpha = torch.rand(alpha_shape, device=device, dtype=real_B_samples.dtype)
    # Interpolate samples from domain B (target domain)
    interpolated_B = (alpha * real_B_samples.data + ((1 - alpha) * fake_B_samples.data)).requires_grad_(True)

    # Use discriminator 
    # print(interpolated_samples.shape) # Debug print, can be removed or kept
    # For cGAN, concatenate interpolated target samples with source domain condition
    # real_A_samples.data is used as the condition 'A' should be fixed during GP calculation w.r.t. 'B'
    critic_input = torch.cat((real_A_samples.data, interpolated_B), 1)
    
    d_interpolates = critic(critic_input)
    
    # Gradients are taken with respect to the interpolated_B part
    gradients  = torch.autograd.grad(outputs = d_interpolates, inputs = interpolated_B,
                                    grad_outputs = torch.ones(d_interpolates.size()).to(device),
                                    create_graph = True, retain_graph = True, only_inputs = True)[0]
    # Reshape gradients to [batch_size, -1]
    gradients = gradients.view(gradients.size(0), -1)


    #### I could use gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #### adding little number like for example 1e-16 ensure numerical stability
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp    ### optimize calculation formula

    
    
    return gradient_penalty,gradients   



