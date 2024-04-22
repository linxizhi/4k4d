import pytorch3d.renderer
import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.networks.encoding.fourplanes import FourPlane
from lib.networks.blocks.mlp import MLP
from lib.networks.blocks.resunet import ResUNet
from lib.networks.blocks.ibrnet import IBRnet
from lib.config import cfg
from lib.networks.blocks.ibrnet import project_xyz_to_uv ,set_to_ndc_corrd
from lib.utils.camera_utils import prepare_feedback_transform,affine_inverse
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.points.rasterizer import rasterize_points
from pytorch3d.renderer import PerspectiveCameras, PointsRasterizer, AlphaCompositor
from lib.datasets.preprocess import process_voxels
import os
import imageio
import nerfacc
import pytorch3d
import numpy as np
from lib.utils.data_utils import to_cuda

# from easyvolcap.engine import SAMPLERS, EMBEDDERS, REGRESSORS
# from easyvolcap.models.networks.embedders.kplanes_embedder import KPlanesEmbedder
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
        self.near=0.1
        self.far=10
        self.K_points=15
        self.all_masks=[]
        self.get_masks()
        self.all_masks=np.stack(self.all_masks)
        self.all_timestep_pcds={}
        self.cameras_all=[]
        self.sigma_shift=-5.0
        self.radius_shift=-5.0
        self.radius_min=0.01
        self.radius_max=0.015
        
        # self.pcd_embedder= KPlanesEmbedder()
        
        for i in range(self.all_masks.shape[1]):
            mask=torch.tensor(self.all_masks)[:,i,:,:]
            voxel_now_step=process_voxels(cfg,self.cameras_all,mask,pcd_index=i)
            
            self.all_timestep_pcds.update({f"pcd_{i}":voxel_now_step})
        # to_cuda(self.allactivation_timestep_pcds)
        self.set_to_cuda()
        self.set_pcd_params()
        for key ,param in self.named_parameters():
            print(key)

    def set_to_cuda(self):
        for key in self.all_timestep_pcds.keys():
            self.all_timestep_pcds[key]=torch.tensor(self.all_timestep_pcds[key]).unsqueeze(0) .cuda()
    def set_pcd_params(self):
        self.all_timestep_pcds=nn.ParameterDict(self.all_timestep_pcds)
    def get_xyz(self):
        self.xyz=0
        
    def geo_actvn(self,sigma,radius):
        a=sigma
        r=radius
        radius_min=self.radius_min
        radius_max=self.radius_max
        radius_shift=self.radius_shift
        sigma_shift=self.sigma_shift
        r = (r + radius_shift).sigmoid() * (radius_max - radius_min) + radius_min
        a = (a + sigma_shift).sigmoid()
        return r, a    
        
    
    def get_masks(self):
        mask_path=os.path.join(cfg.train_dataset['data_root'],'masks')
        angles_all=os.listdir(mask_path)
        angles_all.sort()
        for cam_angle in angles_all:
            mask_files=os.listdir(os.path.join(mask_path,cam_angle))
            mask_files.sort()
            mask_angle=[]
            for index,mask_file in enumerate(mask_files):
                mask=imageio.imread(os.path.join(mask_path,cam_angle,mask_file))
                mask=np.array(mask).astype(np.float32)
                mask_angle.append(mask)
                if index>1:
                    break
            mask_angle=np.stack(mask_angle)
            
            self.all_masks.append(mask_angle)
                # self.all_timestep_pcds.append(mask)       
        
    def get_normalized_direction(self,xyz,ray_o):
        ray_o=ray_o.permute(0,2,1)
        rays_d=xyz-ray_o.repeat(1,xyz.shape[1],1)
        rays_d=rays_d/torch.norm(rays_d,dim=-1,keepdim=True)
        return rays_d
    def turn_pcd_world_to_cam(self,pcd,RT):
        pcd=pcd.reshape(-1,3)
        w2c_cat=torch.tensor([[[0,0,0,1.0]]],device=pcd.device,dtype=pcd.dtype)
        w2cs=torch.cat([RT,w2c_cat],dim=1)
        # c2w=affine_inverse(w2cs)
        # w2c=affine_inverse(c2w)
        pcdw=torch.cat([pcd,torch.ones(pcd.shape[0],1,device=pcd.device,dtype=pcd.dtype)],dim=-1)
        w2c_repeat=w2cs.repeat(pcd.shape[0],1,1)
        # RT_repeat=RT.repeat(pcd.shape[0],1,1)
        pcdw_cam=torch.bmm(w2c_repeat,pcdw.unsqueeze(-1)).squeeze(-1)
        pcd_cam=pcdw_cam.clone()[...,:-1]
        pcd_cam=pcd_cam.unsqueeze(0)
        return pcd_cam
    
    def forward(self,batch):
        pcd_idx=batch['pcd'].item()
        
        self.near=batch['near']
        self.far=batch['far']
        xyz=self.all_timestep_pcds[f"pcd_{pcd_idx}"]
        # xyz=xyz.reshape(-1,3)
        rgb=batch['rgb']
        time_step=batch['time_step']
        wbounds=batch['wbounds']
        # camera=batch['cam']
        rays_o=batch['rays_o']
        H,W=batch['meta']['H'],batch['meta']['W']
        K,R,P,RT=batch['K'],batch['R'],batch['P'],batch['RT']
        smallest_uv=batch['smallest_uv']
        uv_rgb=batch['uv_rgb']


        # xyz=self.turn_pcd_world_to_cam(xyz,RT)

        wbounds_space_max=torch.max(xyz,dim=1,keepdim=True)[0]
        wbounds_space_min=torch.min(xyz,dim=1,keepdim=True)[0]
        wbounds_space=torch.cat([wbounds_space_min,wbounds_space_max],dim=1)
        wbounds[...,:-1]=wbounds_space
        # wbounds[...,:-1]=wbounds_space
        
        
        fov=batch['fov']
        projections=batch['projections']
        rgb_reference_images=batch['rgb_reference_images']
        
        xyz_feature=self.feature_encoder(xyz,time_step,wbounds)
        xyz_feature=xyz_feature.float()
        sigmas_radius=self.geometry_mlp(xyz_feature)
        
        sigmas=sigmas_radius[...,0]
        radius=sigmas_radius[...,1]
        if self.use_sigmoid:
            radius,sigmas=self.geo_actvn(sigmas,radius)
            # print(torch.min(radius),torch.max(radius))
            # radius=F.sigmoid(radius)
            # radius=radius.clip(min=0)
        rays_o_cam=self.turn_pcd_world_to_cam(rays_o,RT)
        direction_each=self.get_normalized_direction(xyz,rays_o)
        rgb_references=rgb_reference_images.squeeze(0).permute(0,3,1,2).float()
        rgb_compose,rgb_discrete,rgb_shs= self.ibrnet(rgb_references,xyz,projections,H,W,direction_each,xyz_feature,uv_rgb)
        # self.project_pcd_to_ndc(xyz, K, RT, rays_o)
        # prepare_feedback_transform(H, W, K,R,rays_o,self.near,self.far,xyz)
        self.save_image(batch['rgb'],0)
        rgb_out,weight_out=self.render(rgb_compose,xyz,P.unsqueeze(0),H,W,sigmas,radius,K,RT,rays_o,R,fov)
        out={}
        out.update({"rgb":rgb_out,"weight":weight_out})
        return out
    def set_ndc_matrix(self,K,H,W):
        H,W=H.item(),W.item()
        n,f=self.near,self.far
        ndc_matrix=torch.zeros(1,4,4,device=K.device,dtype=K.dtype)
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        s = K[..., 0, 1]
        ndc_matrix[..., 0, 0] = 2 * fx / W
        ndc_matrix[..., 0, 1] = 2 * s / W
        ndc_matrix[..., 0, 2] = 1 - 2 * (cx / W)
        ndc_matrix[..., 1, 1] = 2 * fy / H
        ndc_matrix[..., 1, 2] = 1 - 2 * (cy / H)
        ndc_matrix[..., 2, 2] = (f + n) / (n - f)
        ndc_matrix[..., 2, 3] = 2 * f * n / (n - f)
        ndc_matrix[..., 3, 2] = -1

        return ndc_matrix

    def project_pcd_to_ndc(self,pcd, K, RT, T):
        pcd=pcd.reshape(-1,3)
        ons_w=torch.ones(pcd.shape[0],1,device=pcd.device,dtype=pcd.dtype)
        pcdw=torch.cat([pcd,ons_w],dim=-1)
        RT_repeat=RT.repeat(pcd.shape[0],1,1)
        pcd_cam=torch.bmm(RT_repeat,pcdw.unsqueeze(-1)).squeeze(-1)
        pcd_z=-pcd_cam[...,2].unsqueeze(-1)
        pcd_ndc=pcd_cam.clone()
        pcd_ndc[...,:2]=pcd_cam[...,:2]/pcd_z
        K_repeat=K.repeat(pcd.shape[0],1,1)
        tt=torch.bmm(K_repeat[:,:2,:2],pcd_ndc[...,:2].unsqueeze(-1)).squeeze(-1)
        bias=K_repeat[...,:2,2]
        pcd_ndc[...,:2]=tt+bias
        pcd_ndc[..., 0] = 2 * (pcd_ndc[..., 0] / K[0,0, 2]) - 1
        pcd_ndc[..., 1] = 1 - 2 * (pcd_ndc[..., 1] / K[0,1, 2])
        pcd_ndc=torch.clip(pcd_ndc,min=-1,max=1)
        return pcd_ndc

    def project_xyz_to_ndc(self,ndc_matrix,pcd):
        pcd=pcd.reshape(-1,3)
        ons_w=torch.ones(pcd.shape[0],1,device=pcd.device,dtype=pcd.dtype)
        pcdw=torch.cat([pcd,ons_w],dim=-1)
        ndc_matrix_repeat=ndc_matrix.repeat(pcd.shape[0],1,1)
        ndc_pcd=torch.bmm(ndc_matrix_repeat,pcdw.unsqueeze(-1)).squeeze(-1) 
        ndc_w=ndc_pcd[...,-1].unsqueeze(-1)
        ndc_xyz=ndc_pcd[...,:-1]/ndc_w
        return ndc_xyz
    
    def get_ndc_xyz(self,K,H,W):
        ndc_matrix=self.set_ndc_matrix(K,H,W)
        
    def translate(self,R,T,P):
        return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)
    def save_image(self,rgb,ii=None):
        rgb=rgb.clone().detach()
        rgb=rgb.squeeze(0).cpu().numpy()
        rgb=rgb*255
        rgb=rgb.astype(np.uint8)
        import cv2
        if ii is None:
            cv2.imwrite("test.png",rgb)
        else:
            cv2.imwrite(f"test_{ii}.png",rgb)
    def save_npy(self,coord):
        coord=coord.clone().detach()
        coord=coord.squeeze(0).cpu().numpy()
        np.save("test.npy",coord)
        
    def project_to_ndc(self,points, R, T, K,device="cuda:0",image_size=(1024,1024)):


        K_zeros=torch.zeros((1,4,4),device=device,dtype=K.dtype)
        K_zeros[0,:3,:3]=K
        K_zeros[0,3,3]=1
        T=T.reshape(-1,3)
        cameras = PerspectiveCameras(device=device, R=R.float(), T=T.float(),K=K_zeros.float(),image_size=image_size)


        points=points.reshape(-1,3).float()
        points_ndc = cameras.transform_points_ndc(points)

        return points_ndc    
    def get_weights(self,sigmas,ray_indices):

        ones_concat=torch.ones_like(sigmas[...,0:1],device=sigmas.device,dtype=sigmas.dtype)
        sigmas_concat=torch.cat([ones_concat,sigmas],dim=-1)
        weights_cum=torch.cumprod(sigmas_concat,dim=-1)
        weights_f=weights_cum[...,:-1]*(1-sigmas_concat[...,1:])
        weights_f=weights_f.reshape(-1)
        return weights_f
        
    def get_ndc_radius(self,fov,z,radius):
        fov=fov.float()
        z=z.float()
        radius=radius.float()
        max_radius=2*torch.tan(fov/2)*z
        max_radius=max_radius.reshape(-1)
        ndc_radius=radius/max_radius
        ndc_radius=torch.abs(ndc_radius)
        return ndc_radius
    
    
    def render(self,rgb_compose,xyz,projection,H,W,sigmas,radius,K,RT,ray_o,R,fov):
        
        H_d,W_d=H.item(),W.item()
        ndc_corrd,ndc_w=prepare_feedback_transform(H,W,K,R,ray_o,self.near,self.far,xyz)
        # print(torch.max(ndc_corrd[...,0]),torch.min(ndc_corrd[...,0]))
        # print(torch.max(ndc_corrd[...,1]),torch.min(ndc_corrd[...,1]))
        # print(torch.max(ndc_corrd[...,2]),torch.min(ndc_corrd[...,2]))
        
        self.save_npy(ndc_corrd)
        # ndc_radius = torch.abs(K[..., 1, 1][..., None] * radius[..., 0] / (ndc_w+ 1e-10)).reshape(-1)/1000
        ndc_radius=self.get_ndc_radius(fov,ndc_w,radius)
    
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
        print(torch.max(pcd_real_sigma),torch.min(pcd_real_sigma))
        rgbs_each_pixel=rgb_compose_new.reshape(-1,3)[idx_c]
        
        
        weights_= self.get_weights(pcd_real_sigma_clip.reshape(H_d,W_d,-1),ray_indices)

        rgb_final=nerfacc.accumulate_along_rays(weights_,rgbs_each_pixel,ray_indices,H_d*W_d)
        weights_final=nerfacc.accumulate_along_rays(weights_,None,ray_indices,H_d*W_d)
        rgb_final=rgb_final.view(1,H,W,3)
        self.save_image(rgb_final)
        weights_final=weights_final.view(1,H,W,1)
        return rgb_final,weights_final

        