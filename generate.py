import torch
import PIL
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import torchvision.transforms as tr
import torchvision.transforms.functional as tf
from tqdm import tqdm 
'''
large=Image.open('./datasets/23.800,121.048.png')
large=large.crop((1500,0,2500,2000))
'''
large=Image.open('../scifair/datasets/heightmap/26.900,100.000.png')
large=tr.Compose([tr.ToTensor(),
                       tr.Lambda(lambda x: x[0]+x[1]/256.0+x[2]/65536.0),
                       tr.ToPILImage(mode='F')
                      ])(large)
large=large.resize((large.size[0],int(large.size[1]*0.9)),resample=1) #cos(25deg)=0.9
trainIm=large.crop((800,0,2500,2250))
valIm=large.crop((400,0,800,2250))
testIm=large.crop((0,0,400,2250))

sample=tr.Compose([tr.RandomCrop((363,363)),
                      tr.RandomAffine(180,resample=PIL.Image.BICUBIC),
                      tr.CenterCrop((256,256))
                     ])

import random
img_num=-1
def generate_batch(size,source,same=False):
    A=[]
    B=[]
    global img_num
    img_num+=1
    for i in range(size):
        
        #filter_size=random.randrange(1,31,2)
        
        filter_size=(img_num//25)*2+1
        im=sample(source) 
        t=tf.to_tensor(im)
        bias=(t.max()+t.min())/2
        scale=2.4
        if (t.max()-bias)*scale>1:
            print((t.max()-bias)*scale)
            print(str(filter_size)+'_1_'+str(img_num))
          
        A.append((tf.to_tensor(im.filter(PIL.ImageFilter.MedianFilter(filter_size)))-bias)*scale)
        B.append((t-bias)*scale)
        
        
    
    return {'A':torch.stack(A,0),'B':torch.stack(B,0),'A_paths':[str(filter_size)+'_1_'+str(img_num)],'B_paths':[str(filter_size)+'_1_'+str(img_num)]}

n_train=1024
n_test=375
n_val=256

name='test_loss'
import os
os.mkdir('datasets/'+name+'/')
os.mkdir('datasets/'+name+'/test')
os.mkdir('datasets/'+name+'/train')
os.mkdir('datasets/'+name+'/val')
dataset=[]
'''
for i in tqdm(range(n_train),'Generating dataset'):
    b=generate_batch(1,trainIm)
    tr.ToPILImage()( torch.cat([b['A'],b['B']],3)[0]/2+.5).save('datasets/'+name+'/train/'+b['A_paths'][0]+'.png')
'''
for i in tqdm(range(n_test),'Generating dataset'):
    b=generate_batch(1,testIm)
    tr.ToPILImage()( torch.cat([b['A'],b['B']],3)[0]/2+.5).save('datasets/'+name+'/test/'+b['A_paths'][0]+'.png') 
'''
for i in tqdm(range(n_val),'Generating dataset'):
    b=generate_batch(1,valIm)
    tr.ToPILImage()( torch.cat([b['A'],b['B']],3)[0]/2+.5).save('datasets/'+name+'/val/'+b['A_paths'][0]+'.png')
'''