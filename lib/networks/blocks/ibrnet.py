import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.networks.blocks.resunet import ResUNet
from lib.networks.blocks.mlp import MLP,SH_MLP
from lib.utils.sh_utils import eval_sh,SH2RGB
from lib.utils.ndc_utils import get_ndc_for_uv
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
    
def get_rgb_feature(feature_map:torch.Tensor,rgb_map:torch.Tensor,xyz:torch.TensorType,K:torch.Tensor,R:torch.Tensor,T:torch.Tensor,H:torch.Tensor,W:torch.Tensor):

    ndc_uv=get_ndc_for_uv(xyz,K,R,T,H,W)
    print(K,R,T)
    print(torch.max(ndc_uv[...,0]),torch.max(ndc_uv[...,1]),torch.min(ndc_uv[...,0]),torch.min(ndc_uv[...,0]))
    # uv=project_xyz_to_uv(projection,xyz)
    # if (torch.max(uv[...,0]<torch.max(uv[...,1]))):
    #     uv=uv.flip(-1)
    # uv=uv.flip(-1)
    # print(torch.max(uv[...,0]),torch.max(uv[...,1]),torch.min(uv[...,0]),torch.min(uv[...,0]))
    # uv=get_normalized_uv(uv,H,W)
    # ndc_uv=ndc_uv.flip(-1)
    rgb_feature=get_bilinear_feature(feature_map,rgb_map,ndc_uv)
    return rgb_feature



def get_rgb_feature_by_ndc(feature_map:torch.Tensor,rgb_map:torch.Tensor,xyz:torch.TensorType,projection:torch.Tensor,H:torch.Tensor,W:torch.Tensor,scale:torch.Tensor):
    
    uv=project_xyz_to_uv(projection,xyz)
    # uv[...,0]/=2
    # uv[...,0]-=80

    scale=scale.permute(1,0,2).flip(-1)
    uv=uv/scale
    uv=get_normalized_uv(uv,H,W)
    
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
    def forward(self,rgbs,xyz,projection,H,W,direction,xyz_feature,Ks,RTs,scale):
        rgb_map=rgbs
        if self.cfg['Feature_map_encoder']['coarse_only']:
            feature_map_corse,_=self.feature_map_encoder(rgbs)
        else:
            feature_map_corse,feature_map_fine=self.feature_map_encoder(rgbs)
        rgb_feature=get_rgb_feature_by_ndc(feature_map_corse,rgb_map,xyz,projection,H,W,scale)
        
        
        RTs=RTs
        Rs,Ts=RTs[...,:-1],RTs[...,-1:]
        Ks=Ks
        
        # rgb_feature=get_rgb_feature(feature_map_corse,rgb_map,xyz,Ks,Rs,Ts,H,W)
        
        # scales=torch.tensor([[1.0,1.0]],device=Ks.device,dtype=Ks.dtype).reshape(2,-1)
        # exts=RTs
        # zeros=torch.tensor([0,0,0,1.0],device=Rs.device,dtype=Rs.dtype).reshape(1,1,1,4).repeat(1,4,1,1)
        # exts=torch.cat([exts,zeros],dim=-2)
        # ixts=Ks
        
        # rgb_feature=sample_geometry_feature_image(xyz,feature_map_corse.unsqueeze(0),rgb_map.unsqueeze(0),exts,ixts,scales)
        
        rgb_raw=rgb_feature[...,:3].permute(1,0,2)

        xyz_feature_repeat=xyz_feature.unsqueeze(0).repeat(rgb_feature.shape[0],1,1)
        feature_compose_xyz_rgb=torch.cat([rgb_feature,xyz_feature_repeat],dim=-1)
        feature_compose_xyz_rgb=feature_compose_xyz_rgb.permute(1,0,2)
        weights_for_each=self.weights_encoder(feature_compose_xyz_rgb)
        weights_for_each=weights_for_each.squeeze()
        weights_for_each=self.softmax(weights_for_each).unsqueeze(-1)
        rgb_discrete=torch.sum(rgb_raw*weights_for_each,dim=1)
        
        # rgb_discrete=rgb_raw[:,0,...]
        
        shs=self.SH_consistent_Encoder(xyz_feature)
        rgb_showed_in_shs=eval_sh(self.sh_degree,shs,direction.reshape(-1,3))
        rgb_shs=rgb_showed_in_shs.tanh()*0.25
        # rgb_shs=SH2RGB(rgb_showed_in_shs)
        rgb_shs=rgb_shs.clip(min=0)
        
        rgb_compose=rgb_discrete+rgb_shs
        rgb_compose=torch.clip(rgb_compose,0,1)
        return rgb_compose,rgb_discrete,rgb_shs
def sample_geometry_feature_image(xyz: torch.Tensor,  # B, P, 3
                                  src_feat_rgb: torch.Tensor, 
                                  src_raw_rgb,# B, S, C, H, W
                                  src_exts: torch.Tensor,  # B, S, 3, 4
                                  src_ixts: torch.Tensor,  # B, S, 3, 3
                                  src_scale: torch.Tensor,
                                  padding_mode: str = 'border',

                                  #   sample_msk: bool = False,
                                  #   src_size: torch.Tensor = None,  # S, 2
                                  ):
    # xyz: B, P, 3
    # src_feat_rgb: B, S, C, Hs, Ws
    B, S, C, Hs, Ws = src_feat_rgb.shape
    _,_,_,hr,wr=src_raw_rgb.shape
    B, P, _ = xyz.shape
    xyz1 = torch.cat([xyz, torch.ones_like(xyz[..., -1:])], dim=-1)  # homogeneous coordinates

    src_ixts = src_ixts.clone()
    src_ixts[..., :2, :] *= src_scale

    # B, P, 4 -> B, 1, P, 4
    # B, S, 4, 4 -> B, S, 4, 4
    # -> B, S, P, 4
    xyz1 = (xyz1[..., None, :, :] @ src_exts.mT)
    xyzs = xyz1[..., :3] @ src_ixts.mT  # B, S, P, 3 @ B, S, 3, 3
    xy = xyzs[..., :-1] / (xyzs[..., -1:] + 1e-8)  # B, S, P, 2
    x, y = xy.chunk(2, dim=-1)  # B, S, P, 1
    xy = torch.cat([x / Ws * 2 - 1, y / Hs * 2 - 1], dim=-1)  # B, S, P, 2

    xy=xy.float()
    # Actual sampling of the image features (along with rgb colors)
    src_feat_rgb = F.grid_sample(src_feat_rgb.view(-1, C, Hs, Ws), xy.view(-1, 1, P, 2), padding_mode=padding_mode).view(B, S, C, P).permute(0, 1, 3, 2)  # BS, C, 1, P -> B, S, C, P -> B, S, P, C
    raw_rgb=F.grid_sample(src_raw_rgb.view(-1, 3, hr, wr), xy.view(-1, 1, P, 2), padding_mode=padding_mode).view(B, S, 3, P).permute(0, 1, 3, 2)  # BS, C, 1, P -> B, S, C, P -> B, S, P, C
    feat_rgb=torch.cat([raw_rgb,src_feat_rgb],dim=-1)
    return feat_rgb.squeeze(0)