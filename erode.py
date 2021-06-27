import torch
from torch import nn,tensor
import util
import matplotlib.pyplot as plt
import numpy as np 
device='cuda:0'
tx=tensor([[[[-1.,1.]]]]).to(device)
ty=tensor([[[[-1.],[1.]]]]).to(device)
ox=tensor([[[[.5,.5]]]]).to(device)
oy=tensor([[[[.5],[.5]]]]).to(device)
def dx(z):
    return nn.functional.conv2d(z,tx)
def dy(z):
    return nn.functional.conv2d(z,ty)
def mr(z):
    return nn.functional.conv2d(z,-tx,padding=(0,1))
def ml(z):
    return nn.functional.conv2d(z,tx,padding=(0,1))
def mu(z):
    return nn.functional.conv2d(z,-ty,padding=(1,0))
def md(z):
    return nn.functional.conv2d(z,ty,padding=(1,0))
def ax(z):
    return nn.functional.conv2d(z,ox,padding=(0,1))
def ay(z):
    return nn.functional.conv2d(z,oy,padding=(1,0))
def r(z):
    return z[:,:,:,:-1]
def l(z):
    return z[:,:,:,1:]
def u(z):
    return z[:,:,:-1,:]
def d(z):
    return z[:,:,1:,:]
    
width=height=256
a=50

drag=0.0005
dt=.04
v_fac=(1-drag)**dt
evaporation=0.001
rain=0.0003
rain_period=1
ks=0.02
kd=0.001

from tqdm import tqdm
def erode(t,n=1000):
    w=torch.zeros_like(t)
    wr=torch.zeros_like(t)[:,:,:,:-1]
    wl=torch.zeros_like(t)[:,:,:,:-1]
    wu=torch.zeros_like(t)[:,:,:-1,:]
    wd=torch.zeros_like(t)[:,:,:-1,:]
    s=torch.zeros_like(t)
    for i in tqdm(range(n)):
        w*=(1-evaporation)**dt
        deposit=s*(1-(1-evaporation)**dt)
        s-=deposit
        t+=deposit
        #w+=rain*dt*(-np.cos(1+6.28*i*dt/rain_period)*.5+.5)
        w+=rain*dt

        g=dx(t+w+s)
        wr=(wr-g*a*dt)*v_fac
        wr=torch.clamp(torch.min(wr,w[:,:,:,:-1]/dt/4),min=0)
        wl=(wl+g*a*dt)*v_fac
        wl=torch.clamp(torch.min(wl,w[:,:,:,1:]/dt/4),min=0)

        g=dy(t+w+s)
        wu=(wu-g*a*dt)*v_fac
        wu=torch.clamp(torch.min(wu,w[:,:,:-1,:]/dt/4),min=0)
        wd=(wd+g*a*dt)*v_fac
        wd=torch.clamp(torch.min(wd,w[:,:,1:,:]/dt/4),min=0)
        dw=(mr(wr)+ml(wl)+mu(wu)+md(wd))*dt

        c=s/(w+0.001)
        ds=(mr(wr*r(c))+ml(wl*l(c))+mu(wu*u(c))+md(wd*d(c)))*dt

        w+=dw
        s+=ds

        erode=(ax(wr**2+wl**2)+ay(wu**2+wd**2))**0.5*ks*dt
        s+=erode
        t-=erode

        deposit=s*(1-(1-kd)**dt)
        s-=deposit
        t+=deposit
    return t+s