import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision.transforms.functional import pad, scale


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


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
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_mult=False, att=False, multsc=False, use_bias_anyway=0):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_64':
        net = UnetGenerator(input_nc, output_nc, 6, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
                            use_dropout=use_dropout, use_mult=use_mult, att=att, multsc=multsc)
    elif netG == 'unet_2':
        net = UnetGenerator2(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             use_mult=use_mult, att=att, multsc=multsc, use_bias_anyway=use_bias_anyway)
    elif netG == 'unet_3':
        net = UnetGenerator3(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             use_mult=use_mult, att=att, multsc=multsc, use_bias_anyway=use_bias_anyway)
    elif netG == 'unet_4':
        net = UnetGenerator4(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             use_mult=use_mult, att=att, multsc=multsc, use_bias_anyway=use_bias_anyway)
    elif netG == 'unet_5':
        net = UnetGenerator5(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             use_mult=use_mult, att=att, multsc=multsc, use_bias_anyway=use_bias_anyway)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70x70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        # either use real images, fake images, or a linear interpolation of two.
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement(
            ) // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(
                                            disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) -
                            constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7,
                           padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, mult=(i == 4))]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)+input


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, mult=False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, mult=False):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=p, bias=use_bias), norm_layer(dim)]
        if mult:
            conv_block += [Mult(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Mult(nn.Module):
    def __init__(self, nc):
        super(Mult, self).__init__()

        self.register_parameter(name='exp',
                                param=torch.nn.Parameter(torch.diag(torch.ones(nc)).unsqueeze(-1).unsqueeze(-1)))

        # self.exp=torch.diag(torch.ones(nc)).unsqueeze(-1).unsqueeze(-1).to('cuda:1')
        '''self.register_parameter(name='weight',
                                param=torch.nn.Parameter(torch.ones(nc).unsqueeze(-1).unsqueeze(-1)))
                                '''
        self.register_parameter(name='bias',
                                param=torch.nn.Parameter(torch.zeros(nc).unsqueeze(-1).unsqueeze(-1)))
        self.relu = nn.ReLU()

    def forward(self, x):
        # return self.leaky_relu(x.unsqueeze(-3).pow(self.exp).prod(1)*self.weight+self.bias)
        x = self.relu(x)+0.1
        return x.unsqueeze(-3).pow(self.exp).prod(1)+self.bias


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_mult=False, att=False, multsc=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.output_nc = output_nc
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, multsc=multsc)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, mult=use_mult, att=att, multsc=multsc)  # True)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, multsc=multsc)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, multsc=multsc)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, tanh=(output_nc == 3), multsc=multsc)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        if(self.output_nc == 3):
            return self.model(input)
        else:
            return self.model(input) + input[:, :1, :, :] # Heightmaps 


class UnetGenerator2(nn.Module):
    """視野較小的unet"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_mult=False, att=False, multsc=False, use_bias_anyway=0):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator2, self).__init__()
        self.output_nc = output_nc
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, use_bias_anyway=use_bias_anyway)  # add the innermost layer
        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout, multsc=multsc, stride2=(num_downs-i) % 2, use_bias_anyway=use_bias_anyway)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             mult=use_mult, att=att, multsc=multsc, use_bias_anyway=use_bias_anyway)  # True)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, multsc=multsc, stride2=0, use_bias_anyway=use_bias_anyway)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, multsc=multsc, use_bias_anyway=use_bias_anyway)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, tanh=(
            output_nc == 3), multsc=multsc, stride2=0, use_bias_anyway=use_bias_anyway)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        if(self.output_nc == 3):
            return self.model(input)
        else:
            return self.model(input) + input[:, :1, :, :]


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, mult=False, tanh=False, att=False, multsc=False, stride2=1, use_bias_anyway=0):

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.multsc = multsc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if use_bias_anyway:
            use_bias = 1
        if input_nc is None:
            input_nc = outer_nc
        kernel_size = 4 if stride2 else 3
        stride = 2 if stride2 else 1
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size,
                             stride=stride, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc + inner_nc if submodule.multsc else inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1)
            down = [downconv]
            if tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc + inner_nc if submodule.multsc else inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        if mult:
            model = model + [Mult(outer_nc)]
        if att:
            model += [Self_Attn(outer_nc, 'relu')]

        self.model = nn.Sequential(*model)
        if self.multsc:
            self.a = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                               stride=1, padding=1, bias=True)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        elif self.multsc and not self.innermost:
            a = self.model(x)
            return torch.cat([self.a(x)*a, x], 1)
        else:
            return torch.cat([x, self.model(x)], 1)


latent_dim = 4


class UnetGenerator3(nn.Module):
    """把style code當成gate"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_mult=False, att=False, multsc=False, use_bias_anyway=0):

        super(UnetGenerator3, self).__init__()
        self.output_nc = output_nc
        # construct unet structure
        unet_block = UnetSkipConnectionBlock3(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                              innermost=True, use_bias_anyway=use_bias_anyway, depth=3)  # add the innermost layer
        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                                  use_dropout=use_dropout, multsc=multsc, stride2=(num_downs-i) % 2, use_bias_anyway=use_bias_anyway, depth=(num_downs - 2-i)//2)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock3(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                              mult=use_mult, att=att, multsc=multsc, use_bias_anyway=use_bias_anyway, depth=1)  # True)
        unet_block = UnetSkipConnectionBlock3(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer, multsc=multsc, stride2=0, use_bias_anyway=use_bias_anyway, depth=1)
        unet_block = UnetSkipConnectionBlock3(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer, multsc=multsc, use_bias_anyway=use_bias_anyway, depth=0)
        self.model = UnetSkipConnectionBlock3(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, tanh=(
            output_nc == 3), multsc=multsc, stride2=0, use_bias_anyway=use_bias_anyway, depth=0)

    def forward(self, input):
        self.model.latent = input[:, 1:latent_dim+1]
        return self.model(input)


class StyleGate(nn.Module):
    def __init__(self, out_dim, depth):
        super(StyleGate, self).__init__()
        self.fc_sigmoid = nn.Sequential(torch.nn.AvgPool2d(2**depth, stride=2**depth),
                                        nn.Conv2d(latent_dim, out_dim //
                                                  2, kernel_size=1),
                                        nn.LeakyReLU(0.01),
                                        nn.Conv2d(out_dim//2, out_dim,
                                                  kernel_size=1),
                                        nn.Sigmoid())

    def forward(self, x):
        return self.fc_sigmoid(self.latent)*x


class UnetSkipConnectionBlock3(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, mult=False, tanh=False, att=False, multsc=False, stride2=1, use_bias_anyway=0, depth=0):

        super(UnetSkipConnectionBlock3, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.multsc = multsc

        use_bias = 1
        if input_nc is None:
            input_nc = outer_nc
        kernel_size = 4 if stride2 else 3
        stride = 2 if stride2 else 1
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size,
                             stride=stride, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        self.style_gates = []
        self.submodule = submodule
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1)
            down = [downconv, downrelu]
            if tanh:
                up = [upconv, nn.Tanh()]
            else:
                up = [upconv]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=use_bias)
            down = [downconv, downrelu]
            up = [upconv, uprelu, upnorm]
            model = down + up

        else:
            up_gate = StyleGate(outer_nc, depth)
            self.style_gates += [up_gate]

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=use_bias)
            down = [downconv, downrelu, downnorm]
            up = [upconv, uprelu, upnorm, up_gate]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        if mult:
            model = model + [Mult(outer_nc)]
        if att:
            model += [Self_Attn(outer_nc, 'relu')]

        self.model = nn.Sequential(*model)
        if self.multsc:
            self.a = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                               stride=1, padding=1, bias=True)

    def forward(self, x):
        if not self.innermost:
            self.submodule.latent = self.latent
        for g in self.style_gates:
            g.latent = self.latent
        if self.outermost:
            return self.model(x)
        elif self.multsc and not self.innermost:
            a = self.model(xt)
            return torch.cat([self.a(x)*a, x], 1)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator4(nn.Module):
    """conv relu bn"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_mult=False, att=False, multsc=False, use_bias_anyway=0):

        super(UnetGenerator4, self).__init__()
        self.output_nc = output_nc
        # construct unet structure
        unet_block = UnetSkipConnectionBlock4(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                              innermost=True, use_bias_anyway=use_bias_anyway, depth=3)  # add the innermost layer
        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock4(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                                  use_dropout=use_dropout, multsc=multsc, stride2=(num_downs-i) % 2, use_bias_anyway=use_bias_anyway, depth=(num_downs - 2-i)//2)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock4(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                              mult=use_mult, att=att, multsc=multsc, use_bias_anyway=use_bias_anyway, depth=1)  # True)
        unet_block = UnetSkipConnectionBlock4(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer, multsc=multsc, stride2=0, use_bias_anyway=use_bias_anyway, depth=1)
        unet_block = UnetSkipConnectionBlock4(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer, multsc=multsc, use_bias_anyway=use_bias_anyway, depth=0)
        self.model = UnetSkipConnectionBlock4(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, tanh=(
            output_nc == 3), multsc=multsc, stride2=0, use_bias_anyway=use_bias_anyway, depth=0)

    def forward(self, input):
        self.model.latent = input[:, 1:latent_dim+1]
        return self.model(input)


class UnetSkipConnectionBlock4(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, mult=False, tanh=False, att=False, multsc=False, stride2=1, use_bias_anyway=0, depth=0):

        super(UnetSkipConnectionBlock4, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.multsc = multsc

        use_bias = 1
        if input_nc is None:
            input_nc = outer_nc
        kernel_size = 4 if stride2 else 3
        stride = 2 if stride2 else 1
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size,
                             stride=stride, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        self.submodule = submodule
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1)
            down = [downconv, downrelu]
            if tanh:
                up = [upconv, nn.Tanh()]
            else:
                up = [upconv]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=use_bias)
            down = [downconv, downrelu]
            up = [upconv, uprelu, upnorm]
            model = down + up

        else:

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=use_bias)
            down = [downconv, downrelu, downnorm]
            up = [upconv, uprelu, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        if mult:
            model = model + [Mult(outer_nc)]
        if att:
            model += [Self_Attn(outer_nc, 'relu')]

        self.model = nn.Sequential(*model)
        if self.multsc:
            self.a = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                               stride=1, padding=1, bias=True)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        elif self.multsc and not self.innermost:
            a = self.model(x)
            return torch.cat([self.a(x)*a, x], 1)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator5(nn.Module):
    """like unet 4 but every down conv is stride 2 (except the outermost layer)"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_mult=False, att=False, multsc=False, use_bias_anyway=0):

        super(UnetGenerator5, self).__init__()
        self.output_nc = output_nc
        # construct unet structure
        unet_block = UnetSkipConnectionBlock4(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                              innermost=True, use_bias_anyway=use_bias_anyway)  # add the innermost layer
        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock4(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                  norm_layer=norm_layer, use_dropout=use_dropout, multsc=multsc, use_bias_anyway=use_bias_anyway)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock4(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer, mult=use_mult, att=att, multsc=multsc, use_bias_anyway=use_bias_anyway)  # True)
        unet_block = UnetSkipConnectionBlock4(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer, multsc=multsc, use_bias_anyway=use_bias_anyway)
        unet_block = UnetSkipConnectionBlock4(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, multsc=multsc, use_bias_anyway=use_bias_anyway)
        self.model = UnetSkipConnectionBlock4(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, tanh=(
            output_nc == 3), multsc=multsc, use_bias_anyway=use_bias_anyway, stride2=0)

    def forward(self, input):
        self.model.latent = input[:, 1:latent_dim+1]
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            # if n==2:
            #sequence+=[Mult(ndf* nf_mult_prev)]
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1,
                      stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return super().__getattr__('module')
        else:
            return getattr(self.module, name)


class VAEEncoder2(nn.Module):
    def block(self, innc, outnc, conv_type='3', bn=0, dropout_rate=0, leakyrelu=0.1, relu=True, wn=0):
        use_bias = bn == 0
        # use_bias=1
        layers = []
        if conv_type == '1':
            layers += [nn.Conv2d(innc, outnc, kernel_size=1,
                                 stride=1, padding=0, bias=use_bias)]
        elif conv_type == '3':
            layers += [nn.Conv2d(innc, outnc, kernel_size=3,
                                 stride=1, padding=1, bias=use_bias)]
        elif conv_type == 'down':
            layers += [nn.Conv2d(innc, outnc, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)]
            #layers+=[nn.Conv2d(innc, outnc, kernel_size=3,stride=1, padding=1, bias=use_bias)]
            #layers+=[torch.nn.MaxPool2d(2, stride=2)]
        if wn:
            layers[0] = nn.utils.weight_norm(layers[0])

        if bn > 0:
            layers += [nn.BatchNorm2d(outnc, momentum=bn)]
        if relu:
            layers += [nn.LeakyReLU(leakyrelu)]
        if dropout_rate > 0:
            layers += [nn.Dropout(dropout_rate)]
        return layers

    def __init__(self, innc, outnc, nc, bn=0.1, dropout_rate=0, wn=0):
        super(VAEEncoder2, self).__init__()
        depth = len(nc)
        cnn = []
        self.bn = bn
        self.dropout_rate = dropout_rate
        for i in range(depth):
            if i == 0:
                block =\
                    self.block(innc, nc[i], '3', bn, dropout_rate, wn=wn) +\
                    self.block(nc[i], nc[i], '3', bn, dropout_rate, wn=wn) +\
                    self.block(nc[i], nc[i], 'down', bn, dropout_rate, wn=wn)
            else:
                block =\
                    self.block(nc[i-1], nc[i], 'down', bn, dropout_rate, wn=wn)
            cnn += block
        self.cnn = nn.Sequential(*cnn)
        self.bottleneckMiu = nn.Sequential(*(self.block(nc[-1], nc[-1], '1', bn, dropout_rate, wn=wn)
                                             + self.block(nc[-1], outnc, '1', 0, dropout_rate, relu=False, wn=wn)))
        self.bottleneckVar = nn.Sequential(*(self.block(nc[-1], nc[-1], '1', bn, dropout_rate, wn=wn)
                                             + self.block(nc[-1], outnc, '1', 0, dropout_rate, relu=False, wn=wn)))
        self.upsampler = torch.nn.Upsample(
            scale_factor=2 ** depth, mode="bicubic")

    def forward(self, t, sample=False):
        t = self.cnn(t)
        self.miu = self.bottleneckMiu(t)
        if self.training or sample:
            self.log_var = self.bottleneckVar(t)
            self.var = torch.clamp(torch.exp(self.log_var), 0, 10000)
            return self.upsampler(self.miu + torch.normal(torch.zeros_like(self.miu), torch.ones_like(self.var)) * torch.sqrt(self.var))
        else:
            return self.upsampler(self.miu)

    def loss(self, lossWeight=1, lower_var=0):
        return lossWeight * torch.sum(self.miu ** 2 + self.var - self.log_var - 1.0) / 2 + self.var.sum() * lower_var

    def rmsMiu(self):
        return (self.miu**2).mean().sqrt().item()

    def meanVar(self):
        return self.var.mean().item()


class VAEEncoder(nn.Module):
    def block(self, innc, outnc, conv_type='3', bn=0, dropout_rate=0, leakyrelu=0.1, relu=True, wn=0, ins_norm=0):
        # use_bias=bn==0
        use_bias = 1
        layers = []
        if conv_type == '1':
            layers += [nn.Conv2d(innc, outnc, kernel_size=1,
                                 stride=1, padding=0, bias=use_bias)]
        elif conv_type == '3':
            layers += [nn.Conv2d(innc, outnc, kernel_size=3,
                                 stride=1, padding=1, bias=use_bias)]
        elif conv_type == 'down':
            layers += [nn.Conv2d(innc, outnc, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)]
            #layers+=[nn.Conv2d(innc, outnc, kernel_size=3,stride=1, padding=1, bias=use_bias)]
            #layers+=[torch.nn.MaxPool2d(2, stride=2)]
        if wn:
            layers[0] = nn.utils.weight_norm(layers[0])
        if relu:
            layers += [nn.LeakyReLU(leakyrelu)]
        if bn > 0:
            layers += [nn.BatchNorm2d(outnc, momentum=bn)]
        if ins_norm > 0:
            layers += [nn.InstanceNorm2d(outnc, momentum=ins_norm)]
        if dropout_rate > 0:
            layers += [nn.Dropout(dropout_rate)]
        return layers

    def __init__(self, innc, outnc, nc, bn=0.1, dropout_rate=0, wn=0, ins_norm=0):
        super(VAEEncoder, self).__init__()
        depth = len(nc)
        cnn = []
        self.bn = bn
        self.dropout_rate = dropout_rate
        for i in range(depth):
            if i == 0:
                block =\
                    self.block(innc, nc[i], '3', bn, dropout_rate, wn=wn, ins_norm=ins_norm) +\
                    self.block(nc[i], nc[i], '3', bn, dropout_rate, wn=wn, ins_norm=ins_norm) +\
                    self.block(nc[i], nc[i], 'down', bn,
                               dropout_rate, wn=wn, ins_norm=ins_norm)
            elif i < 4:
                block =\
                    self.block(nc[i-1], nc[i], 'down', bn,
                               dropout_rate, wn=wn, ins_norm=ins_norm)
            else:
                block = self.block(nc[i-1], nc[i], 'down',
                                   dropout_rate=dropout_rate)
            cnn += block
        self.cnn = nn.Sequential(*cnn)
        self.bottleneckMiu = nn.Sequential(*(self.block(nc[-1], nc[-1], '1', 0, 0, leakyrelu=0.02, wn=wn)
                                             + self.block(nc[-1], outnc, '1', 0, 0, relu=False, wn=wn)))
        self.bottleneckVar = nn.Sequential(*(self.block(nc[-1], nc[-1], '1', 0, 0, leakyrelu=0.02, wn=wn)
                                             + self.block(nc[-1], outnc, '1', 0, 0, relu=False, wn=wn)))
        self.upsampler = torch.nn.Upsample(
            scale_factor=2 ** depth, mode="bicubic")
        print(self)

    def forward(self, t, sample=False):
        t = self.cnn(t)
        self.miu = self.bottleneckMiu(t)
        if self.training or sample:
            self.log_var = self.bottleneckVar(t)-8  # small initial var
            self.var = torch.clamp(torch.exp(self.log_var), 0, 1000)
            return self.upsampler(self.miu + torch.normal(torch.zeros_like(self.miu), torch.ones_like(self.var)) * torch.sqrt(self.var))
            #return self.upsampler(self.miu)####################################################################################################
        else:
            return self.upsampler(self.miu)

    def loss(self, lossWeight=1, lower_var=0):
        return lossWeight * torch.sum(self.miu ** 2 + self.var - self.log_var - 1.0) / 2 + self.var.sum() * lower_var

    def rmsMiu(self):
        return (self.miu**2).mean().sqrt().item()

    def meanVar(self):
        return self.var.mean().item()


class VAEEncoder3(nn.Module):
    '''skip connection'''

    def block(self, innc, outnc, conv_type='3', bn=0, dropout_rate=0, leakyrelu=0.1, relu=True, wn=0, ins_norm=0):
        # use_bias=bn==0
        use_bias = 1
        layers = []
        if conv_type == '1':
            layers += [nn.Conv2d(innc, outnc, kernel_size=1,
                                 stride=1, padding=0, bias=use_bias)]
        elif conv_type == '3':
            layers += [nn.Conv2d(innc, outnc, kernel_size=3,
                                 stride=1, padding=1, bias=use_bias)]
        elif conv_type == 'down':
            layers += [nn.Conv2d(innc, outnc, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)]
            #layers+=[nn.Conv2d(innc, outnc, kernel_size=3,stride=1, padding=1, bias=use_bias)]
            #layers+=[torch.nn.MaxPool2d(2, stride=2)]
        if wn:
            layers[0] = nn.utils.weight_norm(layers[0])
        if relu:
            layers += [nn.LeakyReLU(leakyrelu)]
        if bn > 0:
            layers += [nn.BatchNorm2d(outnc, momentum=bn)]
        if ins_norm > 0:
            layers += [nn.InstanceNorm2d(outnc, momentum=ins_norm)]
        if dropout_rate > 0:
            layers += [nn.Dropout(dropout_rate)]
        return layers

    def __init__(self, innc, outnc, nc, bn=0.1, dropout_rate=0, wn=0, ins_norm=0):
        super(VAEEncoder3, self).__init__()
        depth = len(nc)
        cnn = []
        self.bn = bn
        self.dropout_rate = dropout_rate
        for i in range(depth):
            if i == 0:
                block =\
                    self.block(innc, nc[i], '3', bn, dropout_rate, wn=wn, ins_norm=ins_norm) +\
                    self.block(nc[i], nc[i], '3', bn, dropout_rate, wn=wn, ins_norm=ins_norm) +\
                    self.block(nc[i], nc[i], 'down', bn,
                               dropout_rate, wn=wn, ins_norm=ins_norm)
            else:
                block = self.block(nc[i-1], nc[i], 'down',
                                   bn, dropout_rate=dropout_rate)
            cnn.append(nn.Sequential(*block))
        self.cnn = nn.ModuleList(cnn)
        self.downsampler = torch.nn.AvgPool2d(2)
        self.bottleneckMiu = nn.Sequential(*(self.block(nc[-1], nc[-1], '1', 0, 0, leakyrelu=0.02, wn=wn)
                                             + self.block(nc[-1], outnc, '1', 0, 0, relu=False, wn=wn)))
        self.bottleneckVar = nn.Sequential(*(self.block(nc[-1], nc[-1], '1', 0, 0, leakyrelu=0.02, wn=wn)
                                             + self.block(nc[-1], outnc, '1', 0, 0, relu=False, wn=wn)))
        self.upsampler = torch.nn.Upsample(
            scale_factor=2 ** depth, mode="bicubic")

        print(self)

    def forward(self, t, sample=None):
        for layer in self.cnn:
            t_ = self.downsampler(t)
            t = layer(t)
            t[:, :t_.shape[1]] += t_

        self.miu = self.bottleneckMiu(t)
        if ((sample == None) and self.training) or (sample == True):
            self.log_var = self.bottleneckVar(t)-8  # small initial var
            self.var = torch.clamp(torch.exp(self.log_var), 0, 10000)
            return self.upsampler(self.miu + torch.normal(torch.zeros_like(self.miu), torch.ones_like(self.var)) * torch.sqrt(self.var))
        else:
            self.var = None
            return self.upsampler(self.miu)

    def loss(self, lossWeight=1, lower_var=0):
        if self.var == None:
            return 0
        return lossWeight * torch.sum(self.miu ** 2 + self.var - self.log_var - 1.0) / 2 + self.var.sum() * lower_var

    def rmsMiu(self):
        return (self.miu**2).mean().sqrt().item()

    def meanVar(self):
        if self.var == None:
            return 0
        return self.var.mean().item()


class CNNEncoder(nn.Module):
    def __init__(self, in_w=256, in_h=256, out_w=256, out_h=256, in_dim=4, out_dim=32, num_layer=4, num_filter=[64, 64, 64, 64, 64], kernel_size=10):
        super(CNNEncoder, self).__init__()
        assert(num_layer == len(num_filter) + 2)
        self.in_conv = nn.Conv2d(in_channels=in_dim, out_channels=num_filter[0], kernel_size=5, stride=1, padding=2)
        self.out_conv = nn.Conv2d(in_channels=num_filter[-1], out_channels=out_dim, kernel_size=5, stride=1, padding=2)
        layer = [
            self.in_conv,
        ]
        print(num_filter)
        for i in range(len(num_filter)-1):
            layer.append(nn.LeakyReLU())
            layer.append(nn.Conv2d(in_channels=num_filter[i], out_channels=num_filter[i+1], kernel_size=4, stride=2, padding=1)) # cut in half
            # layer.append(nn.Conv2d(in_channels=num_filter[i], out_channels=num_filter[i+1], kernel_size=5, stride=1, padding=2))


        layer.append(nn.LeakyReLU())
        layer.append(self.out_conv)
        layer.append(nn.Upsample(scale_factor=2 ** (len(num_filter) - 1), mode='bicubic', align_corners=False))
        layer.append(nn.Tanh())
        self.seq = nn.Sequential(*layer)
        print(self.seq)
    def forward(self, x):
        
        return self.seq(x)


# https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        # return out,attention
        return out
