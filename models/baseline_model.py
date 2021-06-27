"""
空照圖和高度圖一起train
輸入是模糊圖，輸出是空照和清晰圖
SatHeiDataset
"""
import torch
from .base_model import BaseModel
from . import networks
import random
import numpy as np
from collections import OrderedDict

def blur(x,k):
    k=1/(k+0.0001)
    kernel_size = min(120,int(21/k))
    if kernel_size%2 ==0:kernel_size+=1
    arr = [
        [[x - kernel_size / 2 + 0.5, y - kernel_size / 2 + 0.5] for x in range(kernel_size)]
        for y in range(kernel_size)
    ]
    arr = torch.tensor(arr,device=x.device)
    kernel = (
        torch.exp(-0.015*k*k * (arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    kernel/=kernel.sum()
    pad=torch.nn.ReplicationPad2d([(kernel_size-1)//2]*4)
    return torch.nn.functional.conv2d(pad(x), kernel)


class BaselineModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', pool_size=0, gan_mode='vanilla',dataset_mode='sathei')
        if is_train:
            
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        self.align_minmax= opt.align_minmax
        
        """Initialize the pix2pix class.
        
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'real_C', 'fake_B', 'fake_C']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        latent_dim = 8
        networks.latent_dim=latent_dim
        self.beta = 0.005
        self.start_var=50
        self.sat_weight = 1
        
        self.netG = networks.define_G(1, 4, opt.ngf, opt.netG, opt.norm,not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,att=opt.attention,multsc=opt.mult_skip_conn,use_bias_anyway=opt.use_bias_anyway)
        
        self.tanh=torch.nn.Tanh()

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(5, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.epoch = 0
        self.add_real=opt.add_real
    def set_input(self, input,epoch=0):
        self.epoch=epoch
        self.real_A = input['A'].to(self.device) #blurry
        self.real_B = input['B'].to(self.device) #sat
        self.real_C = input['C'].to(self.device) #hei
        self.image_paths = input['A_paths']
        if self.opt.isTrain:    
            if random.randint(0,1):
                self.real_A=torch.flip(self.real_A,[2])
                self.real_B=torch.flip(self.real_B,[2])
                self.real_C=torch.flip(self.real_C,[2])
            if random.randint(0,1):
                self.real_A=torch.flip(self.real_A,[3])
                self.real_B=torch.flip(self.real_B,[3])
                self.real_C=torch.flip(self.real_C,[3])
            if random.randint(0,1):
                self.real_A=self.real_A.transpose(2,3)
                self.real_B=self.real_B.transpose(2,3)    
                self.real_C=self.real_C.transpose(2,3) 
        if self.align_minmax:
            for i in range(self.real_A.shape[1]):
                min=self.real_A[i].min()
                max=self.real_A[i].max()
                self.real_A[i]=((self.real_A[i]-min)/(max-min+0.000001))*2-1
                self.real_C[i]=((self.real_C[i]-min)/(max-min+0.000001))*2-1

    def test(self,blur_=3):
        self.visual_names = ['real_A', 'real_B', 'real_C', 'fake_B', 'fake_C','rand_lat_B','rand_lat_C']
        self.netG.eval
        fake_BC = self.netG(self.real_A)
        self.fake_B=self.tanh(fake_BC[:,0:3])
        self.fake_C=torch.clamp((fake_BC[:,3:4])+(self.real_A if self.add_real else 0), -1, 1)
        

        fake_BC = self.netG(self.real_A)
        self.rand_lat_B=self.tanh(fake_BC[:,0:3])
        self.rand_lat_C = torch.clamp(fake_BC[:,3:4]+(self.real_A if self.add_real else 0), -1, 1)
        

            
    def forward(self, latent=None,test=0):
        rand_bias=np.random.normal(0,2)
        rand_scale=2.72**np.random.normal(0,0.15)
        rand_blur=np.random.uniform(1,2)
        self.real_A=blur(self.real_A,rand_blur)
        if self.isTrain:
            self.real_A=self.real_A*rand_scale+rand_bias
            self.real_C=self.real_C*rand_scale+rand_bias
        
        fake_BC = self.netG(self.real_A)
        self.fake_B=self.tanh(fake_BC[:,0:3])
        self.fake_C=(fake_BC[:,3:4])+(self.real_A if self.add_real else 0)
        
        if test:
            "make visuals fit visdom"
            self.real_A=(self.real_A-rand_bias)/rand_scale
            self.real_C=(self.real_C-rand_bias)/rand_scale
            self.fake_C=(self.fake_C-rand_bias)/rand_scale
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat([self.real_A, self.fake_B,self.fake_C], dim=1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat([self.real_A, self.real_B,self.real_C], dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat([self.real_A, self.fake_B,self.fake_C], dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = (self.criterionL1(self.fake_B, self.real_B)*self.sat_weight+self.criterionL1(self.fake_C, self.real_C)) * self.opt.lambda_L1


        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 
        self.loss_G.backward()
        

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()            # udpate G's weights
    def l2(self,c):
        self.forward() 
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.loss_G_L2=self.criterionL2(self.fake_B, self.real_A)*c
        self.loss_G_L2.backward()
        self.optimizer_G.step()