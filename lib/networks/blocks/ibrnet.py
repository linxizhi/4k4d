import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.networks.blocks.resunet import ResUNet
from lib.networks.blocks.mlp import MLP,SH_MLP
from lib.utils.sh_utils import eval_sh,SH2RGB
def project_xyz_to_uv(projection:torch.Tensor,xyz:torch.Tensor):
    xyz=xyz.reshape(-1,3)

    projection=projection.squeeze(0)
    ones_w=torch.ones(xyz.shape[0],1,device=xyz.device,dtype=xyz.dtype)
    xyzw=torch.cat([xyz,ones_w],dim=-1)
    xyz=xyz.unsqueeze(0).repeat(projection.shape[0],1,1)
    
    projection=projection.unsqueeze(1).repeat(1,xyz.shape[1],1,1)
    uvw=torch.matmul(projection,xyzw.unsqueeze(-1)).squeeze()
    w=uvw[...,2].unsqueeze(-1)+1e-6
    uv=uvw[...,:2]/w
    return uv
def get_normalized_uv(uv,H,W):
    # uv=uv[:,:2]
    uv=torch.clamp(uv,min=0)
    uv=torch.clamp(uv,max=torch.tensor([W-1,H-1],device=uv.device,dtype=uv.dtype))
    uv=2*uv/torch.tensor([W-1,H-1],device=uv.device,dtype=uv.dtype)-1
    return uv
def set_to_ndc_corrd(uv,H,W):
    uv=torch.clamp(uv,min=0)
    uv=torch.clamp(uv,max=torch.tensor([W-1,H-1],device=uv.device,dtype=uv.dtype))
    uv=(uv+0.5)/torch.tensor([W,H],device=uv.device,dtype=uv.dtype)
    return uv

def get_bilinear_feature(feature_map,rgb_map,uv):
    rgb_map=rgb_map.float()
    uv=uv.float()
    uv=uv.unsqueeze(2)
    rgb_map_bilinear=F.grid_sample(rgb_map,uv,align_corners=True)
    feature_map_bilinear=F.grid_sample(feature_map,uv,align_corners=True)
    feature_cated=torch.cat([rgb_map_bilinear,feature_map_bilinear],dim=1)
    feature_cated=feature_cated.squeeze().permute(0,2,1)
    return feature_cated
    
def get_rgb_feature(feature_map:torch.Tensor,rgb_map:torch.Tensor,xyz:torch.TensorType,projection:torch.Tensor,H:torch.Tensor,W:torch.Tensor,uv_rgb):
    uv=project_xyz_to_uv(projection,xyz)
    # print(torch.max(uv[...,0]))
    uv_rgb=uv_rgb.unsqueeze(0).repeat(uv.shape[0],uv.shape[1],1)
    uv_rgb=uv_rgb.flip(-1)
    uv-=uv_rgb
    uv=get_normalized_uv(uv,628,344)
    rgb_feature=get_bilinear_feature(feature_map,rgb_map,uv)
    return rgb_feature

class IBRnet(nn.Module):
    def __init__(self, ibr_cfg) -> None:
        super().__init__()
        self.cfg=ibr_cfg
        feature_map_config=ibr_cfg['Feature_map_encoder']
        self.feature_map_encoder=ResUNet(**feature_map_config)
        weights_encoder_config=ibr_cfg['Weights_encoder']
        self.weights_encoder=MLP(**weights_encoder_config)
        self.softmax=nn.Softmax(dim=-1)
        sh_config=ibr_cfg['SH_consistent_encoder']
        self.sh_degree=sh_config["degree"]
        self.SH_consistent_Encoder=SH_MLP(**sh_config)
    def forward(self,rgbs,xyz,projection,H,W,direction,xyz_feature,uv_rgb):
        rgb_map=rgbs
        if self.cfg['Feature_map_encoder']['coarse_only']:
            feature_map_corse,_=self.feature_map_encoder(rgbs)
        else:
            feature_map_corse,feature_map_fine=self.feature_map_encoder(rgbs)
        rgb_feature=get_rgb_feature(feature_map_corse,rgb_map,xyz,projection,H,W,uv_rgb)
        rgb_raw=rgb_feature[...,:3].permute(1,0,2)
        xyz_feature_repeat=xyz_feature.unsqueeze(0).repeat(rgb_feature.shape[0],1,1)
        feature_compose_xyz_rgb=torch.cat([rgb_feature,xyz_feature_repeat],dim=-1)
        feature_compose_xyz_rgb=feature_compose_xyz_rgb.permute(1,0,2)
        weights_for_each=self.weights_encoder(feature_compose_xyz_rgb)
        weights_for_each=weights_for_each.squeeze()
        weights_for_each=self.softmax(weights_for_each).unsqueeze(-1)
        rgb_discrete=torch.sum(rgb_raw*weights_for_each,dim=1)
        
        shs=self.SH_consistent_Encoder(xyz_feature)
        rgb_showed_in_shs=eval_sh(self.sh_degree,shs,direction.reshape(-1,3))
        # rgb_shs=rgb_showed_in_shs.tanh()*0.25
        rgb_shs=SH2RGB(rgb_showed_in_shs)
        rgb_shs=rgb_shs.clip(min=0)
        
        rgb_compose=rgb_discrete+rgb_shs
        rgb_compose=torch.clip(rgb_compose,0,1)
        return rgb_compose,rgb_discrete,rgb_shs