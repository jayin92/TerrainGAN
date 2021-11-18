"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import torch.nn as nn
from statistics import *
from PIL import Image
from torchvision import transforms

loader = transforms.ToTensor()

def image_loader(path, device="cuda:1"):
    img = Image.open(path).convert("RGB")
    img = loader(img).unsqueeze(0)
    img *= 255 # (0, 1) -> (0, 255)

    return img.to(device, torch.float)

def cal_single_loss(path1, path2):
    t1 = image_loader(path1)
    t2 = image_loader(path2)

    L1 = nn.L1Loss()
    L2 = nn.MSELoss()

    return (L1(t1, t2).item(), L2(t1, t2).item())

def cal_folder_loss(path):
    imgs = sorted(os.listdir(path))
    hei_l1 = []
    hei_l2 = []
    sat_l1 = []
    sat_l2 = []
    cnt = 0
    # for i in range(0, len(imgs), 7):
    #     l1, l2 = cal_single_loss(os.path.join(path, imgs[i+1]), os.path.join(path, imgs[i+6]))
    #     hei_l1.append(l1)
    #     hei_l2.append(l2)
    #     l1, l2 = cal_single_loss(os.path.join(path, imgs[i]), os.path.join(path, imgs[i+5]))
    #     sat_l1.append(l1)
    #     sat_l2.append(l2)

    for i in range(0, len(imgs), 5):
        l1, l2 = cal_single_loss(os.path.join(path, imgs[i+1]), os.path.join(path, imgs[i+4]))
        hei_l1.append(l1)
        hei_l2.append(l2)
        l1, l2 = cal_single_loss(os.path.join(path, imgs[i]), os.path.join(path, imgs[i+3]))
        sat_l1.append(l1)
        sat_l2.append(l2)

    print(f"Avg. Loss for heightmaps: (L1, L2, L1 std, L2 std) = ({round(mean(hei_l1), 3)}, {round(mean(hei_l2), 3)}, {round(pstdev(hei_l1), 3)}, {round(pstdev(hei_l2), 3)})")
    print(f"Avg. Loss for satellite : (L1, L2, L1 std, L2 std) = ({round(mean(sat_l1), 3)}, {round(mean(sat_l2), 3)}, {round(pstdev(sat_l1), 3)}, {round(pstdev(sat_l2), 3)})")



if __name__ == '__main__':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    res = 0
    print(len(dataset))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        start.record()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        end.record()
        torch.cuda.synchronize()
        res += start.elapsed_time(end)
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML

    print("Time per image: {} ms".format(res / min(opt.num_test, len(dataset))))
    path = f"results/{opt.name}/{opt.phase}_{opt.epoch}/images"
    cal_folder_loss(path)
