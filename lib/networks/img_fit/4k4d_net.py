import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.networks.encoding.fourplanes import FourPlane
from lib.networks.blocks.mlp import MLP
from lib.networks.blocks.resunet import ResUNet
from lib.networks.blocks.ibrnet import IBRnet
from lib.config import cfg


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        geometry_config=cfg.network['Geometry_Encoder'] 
        fourplane_config=cfg.network['Fourplane_Encoder']
        ibrnet_config=cfg.network['IBRnet']
        self.use_sigmoid=cfg.network["use_sigmoid"]
        self.feature_encoder=FourPlane(**fourplane_config)
        self.geometry_mlp=MLP(**geometry_config)
        self.ibrnet=IBRnet(ibrnet_config)
    def get_normalized_direction(self,xyz,ray_o):
        ray_o=ray_o.permute(0,2,1)
        rays_d=xyz-ray_o.repeat(1,xyz.shape[1],1)
        rays_d=rays_d/torch.norm(rays_d,dim=-1,keepdim=True)
        return rays_d
    def forward(self,batch):
        xyz=batch['pcd']
        # xyz=xyz.reshape(-1,3)
        rgb=batch['rgb']
        time_step=batch['time_step']
        wbounds=batch['wbounds']
        # camera=batch['cam']
        rays_o=batch['rays_o']
        H,W=batch['meta']['H'],batch['meta']['W']
        projections=batch['projections']
        rgb_reference_images=batch['rgb_reference_images']
        
        xyz_feature=self.feature_encoder(xyz,time_step,wbounds)
        xyz_feature=xyz_feature.float()
        sigmas_radius=self.geometry_mlp(xyz_feature)
        
        sigmas=sigmas_radius[...,0]
        if self.use_sigmoid:
            sigmas=F.sigmoid(sigmas)
        radius=sigmas_radius[...,1]
        direction_each=self.get_normalized_direction(xyz,rays_o)
        rgb_references=rgb_reference_images.squeeze(0).permute(0,3,1,2).float()
        rgb_compose,rgb_discrete,rgb_shs= self.ibrnet(rgb_references,xyz,projections,H,W,direction_each,xyz_feature)
        
        
        