import torch
from torch import nn
from .rigid_pose import RigidPoseNet, se3_from_vec
from .deformation import DeformationNet

class NRRecFUS(nn.Module):
    def __init__(self,num_samples=4):
        super().__init__(); self.rigid=RigidPoseNet(num_samples); self.deform=DeformationNet()
    def forward(self,frames):
        # frame 0 is fixed; pairwise rigid branch predicts all other frames jointly.
        pose_vec=self.rigid(frames); flows=[]; warped=[]
        fixed=frames[:,0:1]
        for i in range(1,frames.shape[1]):
            flow=self.deform(fixed,frames[:,i:i+1]); flows.append(flow); warped.append(frames[:,i:i+1]+0*flow[:,0:1])
        return {'pose_vec':pose_vec,'pose':se3_from_vec(pose_vec),'flow':torch.stack(flows,1),'warped':torch.cat(warped,1)}
