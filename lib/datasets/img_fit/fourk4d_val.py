import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
import json
import yaml
import torch
from lib.datasets.preprocess import process_voxels
# import pyyaml
from typing import NamedTuple
from lib.datasets.img_fit.caminfo import CameraInfo4K4D

from scipy.ndimage import zoom
from lib.utils.camera_utils import resize_array
import math
def pad_image(image, target_size):

    pad_dims = [(0, max_size - img_size) for img_size, max_size in zip(image.shape, target_size)]
    

    padded_image = np.pad(image, pad_dims, mode='constant')
    
    return padded_image

class Camera:
    def __init__(self,cam_path_intri,cam_path_extri) -> None:
        # data/my_387/optimized
        self.intrix_file=cv2.FileStorage(cam_path_intri,cv2.FILE_STORAGE_READ)
        self.extrix_file=cv2.FileStorage(cam_path_extri,cv2.FILE_STORAGE_READ)
        
        # pcd_path=os.path.join(cfg.train_dataset['data_root'],"pcd")
        # self.all_timestep_pcds=[]
        self.cameras_all=[]
        self.set_cameras()
        self.all_masks=[]
        self.get_masks()
        self.all_masks=np.stack(self.all_masks)
        self.all_timestep_pcds=[]
        self.mask_max_len_x=0
        self.mask_max_len_y=0
        for i in range(self.all_masks.shape[1]):
            mask=torch.tensor(self.all_masks)[:,i,:,:]
        # bouding_box=self.extrix_file.getNode("bounds_00").mat()
        
            voxel_now_step=process_voxels(cfg,self.cameras_all,mask,pcd_index=i)
            self.all_timestep_pcds.append(voxel_now_step)
        # self.delet_bg()

    def delet_bg(self):
        masks=self.all_masks
        rgb_reference_images_list=[]
        
        max_len_x=0
        max_len_y=0
        masks=masks.reshape(-1,masks.shape[-2],masks.shape[-1])
        for index,mask in enumerate(masks):
            mask_min_x,mask_max_x,mask_min_y,mask_max_y=np.where(mask>0)[0].min(),np.where(mask>0)[0].max(),np.where(mask>0)[1].min(),np.where(mask>0)[1].max()
            if (mask_max_x-mask_min_x>max_len_x):
                max_len_x=mask_max_x-mask_min_x
            if (mask_max_y-mask_min_y>max_len_y):
                max_len_y=mask_max_y-mask_min_y
        self.mask_max_len_x=max_len_x
        self.mask_max_len_y=max_len_y
    
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
      
    def set_cameras(self):
        cam_names = self.read('names',True ,dt='list')
        
        for cam in cam_names:
            # Intrinsics
            cam_dict={}
            
            cam_dict['K'] = self.read('K_{}'.format(cam),True)
            cam_dict['H'] = int(self.read('H_{}'.format(cam), True,dt='real')) or -1
            cam_dict['W']= int(self.read('W_{}'.format(cam), True,dt='real')) or -1
            cam_dict['H']= 256
            cam_dict['W']= 256
            cam_dict['invK'] = np.linalg.inv(cam_dict['K'])

            # Extrinsics
            Tvec = self.read('T_{}'.format(cam),False)
            Rvec = self.read('R_{}'.format(cam),False)
            if Rvec is not None: R = cv2.Rodrigues(Rvec)[0]
            else:
                R = self.read('Rot_{}'.format(cam))
                Rvec = cv2.Rodrigues(R)[0]
            RT = np.hstack((R, Tvec))

            cam_dict['R'] = R
            cam_dict["T"] = Tvec
            cam_dict["C"] = - Rvec.T @ Tvec
            cam_dict["RT"] = RT
            cam_dict["Rvec"] = Rvec
            cam_dict["P"]= cam_dict['K'] @ cam_dict['RT']

            if (cam_dict['P'][0,0]<0):
                a=0
            # Distortion
            D = self.read('D_{}'.format(cam),True)
            if D is None: D = self.read('dist_{}'.format(cam),True)
            cam_dict["D"]= D

            # Time input
            cam_dict['t'] = self.read('t_{}'.format(cam), False,dt='real') or 0  # temporal index, might all be 0
            cam_dict['v'] = self.read('v_{}'.format(cam), False,dt='real') or 0  # temporal index, might all be 0

            # Bounds, could be overwritten
            cam_dict['n'] = self.read('n_{}'.format(cam), False,dt='real') or 0.0001  # temporal index, might all be 0
            cam_dict['f'] = self.read('f_{}'.format(cam), False,dt='real') or 1e6  # temporal index, might all be 0
            cam_dict['bounds'] = self.read('bounds_{}'.format(cam),False)
            cam_dict['bounds'] = np.array([[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]]) if cam_dict['bounds'] is None else cam_dict['bounds']

            cam_dict['fov']=2*np.arctan(cam_dict['W']/(2*cam_dict['K'][0,0]))

            # CCM
            cam_dict['ccm'] = self.read('ccm_{}'.format(cam),True)
            cam_dict['ccm'] = np.eye(3) if cam_dict['ccm'] is None else cam_dict['ccm']     
            mm=-cam_dict['R'].T@cam_dict['T']
            c2w=   np.concatenate([cam_dict['R'].T,mm],axis=-1)
            cam_dict['c2w']=  c2w
            cam_dict['center']=cam_dict['c2w'][:,-1].reshape(3,1)
            caminfo=CameraInfo4K4D(**cam_dict)
            self.cameras_all.append(caminfo)
        

    def read(self,node,use_intrix=True,dt="mat"):
        if use_intrix:
            fs=self.intrix_file 
        else:
            fs=self.extrix_file
        if dt == 'mat':
            output = fs.getNode(node).mat()
        elif dt == 'list':
            results = []
            n =fs.getNode(node)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'real':
            output = fs.getNode(node).real()
        else:
            raise NotImplementedError
        return output
    def get_pcds(self,time_step:int):
        return self.all_timestep_pcds[time_step]
    def get_camera(self,camera_index:int):
        return self.cameras_all[camera_index]
    @property
    def get_camera_length(self):
        return len(self.cameras_all)
    @property
    def get_timestep_length(self):
        return len(self.all_timestep_pcds)
class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split = kwargs['data_root'], kwargs['split']
        self.nearset_num=cfg.train_dataset['nearset_num']
        camera_matrix_intrix_optimized_path=os.path.join(cfg.train_dataset['data_root'],'optimized')
        camera_intri_path=os.path.join(camera_matrix_intrix_optimized_path,"intri.yml")
        camera_extri_path=os.path.join(camera_matrix_intrix_optimized_path,"extri.yml")
        self.camera=Camera(camera_intri_path,camera_extri_path)

        self.camera_len=self.camera.get_camera_length
        self.time_step_len=self.camera.get_timestep_length

        self.len=self.time_step_len

        self.input_ratio = kwargs['input_ratio']
        self.data_root = data_root
        self.split = split
        self.batch_size = cfg.task_arg.N_pixels

        image_path=os.path.join(self.data_root,'images')
        angles_all=os.listdir(image_path)
        angles_all.sort()
        self.img=[]
        for angle in angles_all:
            image_files=os.listdir(os.path.join(image_path,angle))
            image_files.sort()
            now_angle_images=[]
            for index,image_file in enumerate(image_files):
                image=imageio.imread(os.path.join(image_path,angle,image_file))/255
                # image=resize_array(image,(256,256,3))
                now_angle_images.append(image)
                if index>1:
                    break
            self.img.append(np.stack(now_angle_images))
        self.img=np.stack(self.img)
        # set uv
        H, W = self.img.shape[-3:-1]
        self.H,self.W=H,W
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        u, v = X.astype(np.float32) / (W-1), Y.astype(np.float32) / (H-1)
        self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)

        

    def __getitem__(self, index):
        # index=0
        index=23*index+22
        cam_index=index%self.camera_len
        time_step_index=index//self.camera_len
        cam=self.camera.get_camera(cam_index)
        # pcd=self.camera.get_pcds(time_step_index)
        # pcd=pcd.reshape(-1,3)
        t_bounds=np.array([0.0,self.time_step_len-1]).reshape(2,-1)
        wbounds=cam.bounds
        wbounds=np.concatenate([wbounds,t_bounds],axis=1)
        # wbounds=wbounds.reshape(-1)
        rgb=self.img[cam_index,time_step_index]
        mask=self.camera.all_masks[cam_index,time_step_index,:,:]/255

        mask_min_x,mask_max_x,mask_min_y,mask_max_y=np.where(mask>0)[0].min(),np.where(mask>0)[0].max(),np.where(mask>0)[1].min(),np.where(mask>0)[1].max()

        uv_rgb=np.array([mask_min_x,mask_min_y])

        rgb=rgb*mask[...,np.newaxis]
        mask=mask[mask_min_x:mask_max_x,mask_min_y:mask_max_y]
        rgb=rgb[mask_min_x:mask_max_x,mask_min_y:mask_max_y,:]
        H=rgb.shape[0]
        W=rgb.shape[1]
        
        cam_k=cam.K.copy()
        # print(cam_k)
        # cam_k[0,...]=cam_k[0,...]*(mask_max_y-mask_min_y)/self.W
        # cam_k[1,...]=cam_k[1,...]*(mask_max_x-mask_min_x)/self.H
        cam_k[0,2]=cam_k[0,2]-mask_min_y
        cam_k[1,2]=cam_k[1,2]-mask_min_x
        cam_p=cam_k @ cam.RT
        ret = {'rgb': rgb} # input and output. they will be sent to cuda
        ret.update({'mask':mask})
        ret.update({'pcd': time_step_index,'cam':cam,"time_step":time_step_index,"cam_index":cam_index,"wbounds":wbounds})
        ret.update({'rays_o':cam.T })
        ret.update({"R":cam.R,"K":cam_k,"P":cam_p,"RT":cam.RT,"near":cam.n,"far":cam.f,"fov":cam.fov})
        ret.update({'meta': {'H': H, 'W': W}}) # meta means no need to send to cuda
        N_reference_images_index,RTS,KS= self.get_nearest_pose_cameras(cam_index)
        rgb_reference_images=self.img[N_reference_images_index,time_step_index]
        
        ret.update({"N_reference_images_index":N_reference_images_index})
        masks=self.camera.all_masks[N_reference_images_index]
        rgb_reference_images_list=[]

        
        smallest_uv=[]
        max_len_x,max_len_y=0,0
        cam_k_all=[]
        mask_allindex_lists=[]
        for index,mask in enumerate(masks):
            mask_real_index=N_reference_images_index[index]
            cam_k_temp=self.camera.get_camera(mask_real_index).K.copy()
            mask_min_x,mask_max_x,mask_min_y,mask_max_y=np.where(mask>0.9)[1].min(),np.where(mask>0.9)[1].max(),np.where(mask>0.5)[2].min(),np.where(mask>0.5)[2].max()
            rgb_reference_images_list.append(rgb_reference_images[index,mask_min_x:mask_max_x,mask_min_y:mask_max_y,:])
            uv_temp=np.array([mask_min_x,mask_min_y])
            smallest_uv.append(uv_temp)
            cam_k_temp[0,2]-=mask_min_y
            cam_k_temp[1,2]-=mask_min_x
            
            mask_allindex_lists.append([mask_min_x,mask_max_x,mask_min_y,mask_max_y])

            cam_k_all.append(cam_k_temp)
            if (mask_max_x-mask_min_x>max_len_x):
                max_len_x=mask_max_x-mask_min_x
            if (mask_max_y-mask_min_y>max_len_y):
                max_len_y=mask_max_y-mask_min_y
        max_len_x=math.ceil(max_len_x/8)*8
        max_len_y=math.ceil(max_len_y/8)*8
        for i,mask_for in enumerate(mask_allindex_lists):
            mask_min_x,mask_max_x,mask_min_y,mask_max_y=mask_for
            len_pad_x=max(max_len_x-(mask_max_x-mask_min_x),0)
            len_pad_y=max(max_len_y-(mask_max_y-mask_min_y),0)
            # cam_k_all[i][0,2]-=100
            # cam_k_all[i][1,2]-=100
        K_for_reference=np.stack(cam_k_all)
        #TODO 这里可能是错误的
        # K_for_reference=cam.K.copy()

        # K_for_reference[:,0,:]=K_for_reference[:,0,:]*max_len_x/self.H
        # K_for_reference[:,1,:]=K_for_reference[:,1,:]*max_len_y/self.W
        smallest_uv=np.stack(smallest_uv)   
        rgb_reference_images_list_final=[]
        for rgb_reference_image in rgb_reference_images_list:
            rgb_reference_image_temp=pad_image(rgb_reference_image,(max_len_x,max_len_y,3))
            rgb_reference_images_list_final.append(rgb_reference_image_temp)
        rgb_reference_images_list_final=np.stack(rgb_reference_images_list_final)
        # K_for_reference=K_for_reference[np.newaxis,...]
        projections=K_for_reference @ RTS
        ret.update({"rgb_reference_images":rgb_reference_images_list_final})
        ret.update({"projections":projections})
        ret.update({"uv_rgb":uv_rgb})
        ret.update({"refernce_k":K_for_reference,"refernce_RTs":RTS})
        # ret.update({"smallest_uv":smallest_uv})
        
        return ret
    def get_nearest_pose_cameras(self,now_index):
        num_cams=self.camera_len
        target_cam=self.camera.get_camera(now_index)
        all_cams=self.camera.cameras_all
        target_cam_center=target_cam.center
        all_cam_centers=[cam.center for cam in all_cams]
        all_RTs=[cam.RT for cam in all_cams]
        all_Ks=[cam.K for cam in all_cams]

        all_cam_centers=np.stack(all_cam_centers)
        all_RTs=np.stack(all_RTs)
        all_Ks=np.stack(all_Ks)

        all_cam_centers=all_cam_centers.reshape(-1,3)
        target_cam_center=target_cam_center.reshape(-1,3)
        all_cam_distance=all_cam_centers-target_cam_center
        all_cam_distance=np.linalg.norm(all_cam_distance,axis=-1)
        distance_index=np.argsort(all_cam_distance,axis=-1)
        distance_index=distance_index[1:self.nearset_num+1]
        RTs_choose=all_RTs[distance_index].copy()
        KS_choose=all_Ks[distance_index].copy()

        return distance_index,RTs_choose,KS_choose
        
        
        
    def __len__(self):
        # we only fit 1 images, so we return 1
        return self.len