import torch
from torch import nn
from torchvision.models import efficientnet_b0

class RigidPoseNet(nn.Module):
    """Official NR-Rec-FUS rigid branch: image stack -> pairwise pose parameters."""
    def __init__(self, in_frames=4, pairs=None):
        super().__init__(); self.pairs = pairs or [(0, i) for i in range(1, in_frames)]
        self.backbone = efficientnet_b0(weights=None)
        old = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(in_frames, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 6*len(self.pairs))
    def forward(self, x): return self.backbone(x).view(x.shape[0], len(self.pairs), 6)

def se3_from_vec(v):
    """Small-angle 6-DoF vector to differentiable homogeneous transform."""
    t, r = v[..., :3], v[..., 3:]
    theta = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = r / theta
    K = torch.zeros((*r.shape[:-1],3,3), device=r.device, dtype=r.dtype)
    K[...,0,1],K[...,0,2],K[...,1,0] = -k[...,2], k[...,1], k[...,2]
    K[...,1,2],K[...,2,0],K[...,2,1] = -k[...,0], -k[...,1], k[...,0]
    I=torch.eye(3,device=r.device,dtype=r.dtype).expand_as(K)
    R=I + torch.sin(theta)[...,None]*K + (1-torch.cos(theta))[...,None]*(K@K)
    T=torch.eye(4,device=r.device,dtype=r.dtype).expand(*r.shape[:-1],4,4).clone(); T[...,:3,:3]=R; T[...,:3,3]=t
    return T
