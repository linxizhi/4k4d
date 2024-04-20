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
        for i in range(self.all_masks.shape[1]):
            mask=torch.tensor(self.all_masks)[:,i,:,:]
        # bouding_box=self.extrix_file.getNode("bounds_00").mat()
        
            voxel_now_step=process_voxels(cfg,self.cameras_all,mask,pcd_index=i)
            self.all_timestep_pcds.append(voxel_now_step)
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
            cam_dict['H']=1024
            cam_dict['W']=1024
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

            # CCM
            cam_dict['ccm'] = self.read('ccm_{}'.format(cam),True)
            cam_dict['ccm'] = np.eye(3) if cam_dict['ccm'] is None else cam_dict['ccm']          
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

        self.len=self.camera_len*self.time_step_len

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

                now_angle_images.append(image)
                if index>1:
                    break
            self.img.append(np.stack(now_angle_images))
        self.img=np.stack(self.img)
        # set uv
        H, W = self.img.shape[-3:-1]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        u, v = X.astype(np.float32) / (W-1), Y.astype(np.float32) / (H-1)
        self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)

        

    def __getitem__(self, index):
        cam_index=index%self.camera_len
        time_step_index=index//self.camera_len
        cam=self.camera.get_camera(cam_index)
        pcd=self.camera.get_pcds(time_step_index)
        # pcd=pcd.reshape(-1,3)
        t_bounds=np.array([0.0,self.time_step_len-1]).reshape(2,-1)
        wbounds=cam.bounds
        wbounds=np.concatenate([wbounds,t_bounds],axis=1)
        wbounds=wbounds.reshape(-1)
        rgb=self.img[cam_index,time_step_index]
        ret = {'rgb': rgb} # input and output. they will be sent to cuda
        
        ret.update({'pcd': pcd,'cam':cam,"time_step":time_step_index,"cam_index":cam_index,"wbounds":wbounds})
        ret.update({'rays_o':cam.T })
        ret.update({'meta': {'H': self.img.shape[2], 'W': self.img.shape[3]}}) # meta means no need to send to cuda
        N_reference_images_index,projections= self.get_nearest_pose_cameras(cam_index)
        rgb_reference_images=self.img[N_reference_images_index,time_step_index]
        
        ret.update({"N_reference_images_index":N_reference_images_index})
        ret.update({"rgb_reference_images":rgb_reference_images})
        ret.update({"projections":projections})
        
        return ret
    def get_nearest_pose_cameras(self,now_index):
        num_cams=self.camera_len
        target_cam=self.camera.get_camera(now_index)
        all_cams=self.camera.cameras_all
        target_cam_center=target_cam.T
        all_cam_centers=[cam.T for cam in all_cams]
        all_projections=[cam.P for cam in all_cams]
        all_cam_centers=np.stack(all_cam_centers)
        all_projections=np.stack(all_projections)
        
        all_cam_centers=all_cam_centers.reshape(-1,3)
        target_cam_center=target_cam_center.reshape(-1,3)
        all_cam_distance=all_cam_centers-target_cam_center
        all_cam_distance=np.linalg.norm(all_cam_distance,axis=-1)
        distance_index=np.argsort(all_cam_distance,axis=-1)
        distance_index=distance_index[:self.nearset_num]
        projections=all_projections[distance_index,:,:]
        # all_cam_centers=all_cam_centers.argsort()
        return distance_index,projections
        
        
        
    def __len__(self):
        # we only fit 1 images, so we return 1
        return self.len