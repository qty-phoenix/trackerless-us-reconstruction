import torch
from torch import nn

class DeformationNet(nn.Module):
    """VoxelMorph-style regularized registration branch used after rigid alignment."""
    def __init__(self, channels=2):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(channels,32,3,padding=1),nn.InstanceNorm2d(32),nn.LeakyReLU(.2),nn.Conv2d(32,32,3,padding=1),nn.LeakyReLU(.2),nn.Conv2d(32,2,3,padding=1))
    def forward(self, fixed, moving): return self.net(torch.cat([fixed,moving],1))

def warp(moving, flow):
    b,_,h,w=moving.shape; yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=moving.device),torch.linspace(-1,1,w,device=moving.device),indexing='ij')
    grid=torch.stack([xx,yy],-1).expand(b,-1,-1,-1).clone(); grid[...,0]+=flow[:,0]/max(w-1,1)*2; grid[...,1]+=flow[:,1]/max(h-1,1)*2
    return torch.nn.functional.grid_sample(moving,grid,align_corners=True,padding_mode='border')
