import torch
from lib.utils.ndc_utils  import get_ndc
from pytorch3d.renderer.points.rasterizer import rasterize_points
from pytorch3d.renderer import PerspectiveCameras, PointsRasterizer, AlphaCompositor
from lib.datasets.preprocess import process_voxels
from pytorch3d.structures import Pointclouds
import nerfacc
def get_weights(sigmas,ray_indices):

        ones_concat=torch.ones_like(sigmas[...,0:1],device=sigmas.device,dtype=sigmas.dtype)
        sigmas_concat=torch.cat([ones_concat,sigmas],dim=-1)
        weights_cum=torch.cumprod(sigmas_concat,dim=-1)
        weights_f=weights_cum[...,:-1]*(1-sigmas_concat[...,1:])
        weights_f=weights_f.reshape(-1)
        return weights_f
def render(rgb_compose,xyz,H,W,sigmas,radius,K,ray_o,R):
        
        H_d,W_d=H.item(),W.item()
        
        
        ndc_corrd,ndc_radius=get_ndc(xyz,K,R,ray_o,H,W,radius)

        ndc_radius=ndc_radius.float()
        ndc_radius=ndc_radius.reshape(-1)

    
        ndc_corrd=ndc_corrd.unsqueeze(0).float()
        ndc_corrd=ndc_corrd.reshape(1,-1,3)
        ndc_radius_c=ndc_radius.float().clone()
        
        pcd_real=Pointclouds(ndc_corrd)

        idx, depth, dists = rasterize_points(pcd_real, (H.item(), W.item()), ndc_radius_c, self.K_points, 0, None)

        num=rgb_compose.shape[0]

        dists_c=dists.clip(min=0)

        dists_c=dists_c.reshape(-1)
        idx=idx.reshape(-1)


        rgb_zeros=torch.zeros((1,3)).to(dtype=rgb_compose.dtype,device=rgb_compose.device)
        sigma_zeros=torch.zeros((1)).to(dtype=sigmas.dtype,device=sigmas.device)
        radius_zeros=torch.ones((1)).to(dtype=radius.dtype,device=radius.device)
        
        idx_c=idx.clone()
        idx_c[idx==-1]=num
        
        rgb_compose_new=torch.cat([rgb_compose,rgb_zeros],dim=0)
        sigmas_new=torch.cat([sigmas,sigma_zeros],dim=-1)
        ndc_radius_new=torch.cat([ndc_radius,radius_zeros],dim=-1)
        
        
        ray_indices=torch.arange(H_d*W_d,device=ndc_corrd.device).reshape(-1,1).repeat(1,self.K_points).reshape(-1)
        sigmas_each_pixel=sigmas_new[idx_c]
        radius_each_pixel=ndc_radius_new[idx_c]
        pcd_real_sigma=sigmas_each_pixel*(1-dists_c/(radius_each_pixel*radius_each_pixel))
        
        pcd_real_sigma_clip=pcd_real_sigma.clip(min=0)
        # print(torch.max(pcd_real_sigma),torch.min(pcd_real_sigma))
        rgbs_each_pixel=rgb_compose_new.reshape(-1,3)[idx_c]
        
        
        weights_= get_weights(pcd_real_sigma_clip.reshape(H_d,W_d,-1),ray_indices)

        rgb_final=nerfacc.accumulate_along_rays(weights_,rgbs_each_pixel,ray_indices,H_d*W_d)
        weights_final=nerfacc.accumulate_along_rays(weights_,None,ray_indices,H_d*W_d)
        rgb_final=rgb_final.view(1,H,W,3)
        # self.save_image(rgb_final)
        weights_final=weights_final.view(1,H,W,1)
        return rgb_final,weights_final