from lib.networks.encoding.triplane import TriPlane,Plane
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
eps=1e-6
class FourPlane(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        config_for_planes = cfg.network['Fourplane_Encoder']
        self.out_activation=config_for_planes["aggregation"]
        if config_for_planes["aggregation"] == "concat":
            self.out_dims=6*32
        plane_config={}
        for key in config_for_planes:
            if key!="aggregation":
                plane_config[key]=config_for_planes[key]
        self.xy_plane = Plane(**plane_config)
        self.yz_plane = Plane(**plane_config)
        self.xz_plane = Plane(**plane_config)
        self.tx_plane = Plane(**plane_config)
        self.ty_plane = Plane(**plane_config)
        self.tz_plane = Plane(**plane_config)
    def forward(self, xyz,timesteps,wbounds):
        wbounds=wbounds.reshape(-1)
        xyz=xyz.reshape(-1,3)
        timesteps_re=timesteps.reshape(-1,1).repeat(xyz.shape[0],1)
        xyzt=torch.cat([xyz,timesteps_re],dim=-1)
        wbounds=wbounds.reshape(-1)
        xyzt_bounded = torch.clamp(xyzt, min=wbounds[:4], max=wbounds[4:8])
        xyzt_bounded=xyzt_bounded.reshape(-1,xyzt_bounded.shape[-1])
        inputs=xyzt
        inputs = inputs - wbounds[None][:, :4]
        inputs = inputs / ((wbounds[4:8] - wbounds[:4]).max().item() + eps)
        
        print(torch.max(inputs),torch.min(inputs))
        xy_feat = self.xy_plane(inputs[..., [0, 1]])
        yz_feat = self.yz_plane(inputs[..., [1, 2]])
        xz_feat = self.xz_plane(inputs[..., [0, 2]])
        tx_feat = self.tx_plane(inputs[..., [0, 3]])
        ty_feat = self.ty_plane(inputs[..., [1, 3]])
        tz_feat = self.tz_plane(inputs[..., [2, 3]])
        if self.out_activation=="concat":
            return torch.cat([xy_feat, yz_feat, xz_feat,tx_feat,ty_feat,tz_feat], dim=-1)
        if self.out_activation=='sum':
            feature=torch.stack([xy_feat, yz_feat, xz_feat,tx_feat,ty_feat,tz_feat], dim=0)
            feature=torch.sum(feature,dim=0)
            return feature
            # return torch.sum([xy_feat, yz_feat, xz_feat,tx_feat,ty_feat,tz_feat], dim=-1)