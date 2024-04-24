import torch
from pytorch3d.renderer import PerspectiveCameras, PointsRasterizer, AlphaCompositor
from pytorch3d.structures import Pointclouds
def get_pytorch3d_ndc_K(K: torch.Tensor, H: int, W: int):
    M = min(H, W)
    K = torch.cat([K, torch.zeros_like(K[..., -1:, :])], dim=-2)
    K = torch.cat([K, torch.zeros_like(K[..., :, -1:])], dim=-1)
    K[..., 3, 2] = 1  # ...? # HACK: pytorch3d magic
    K[..., 2, 2] = 0  # ...? # HACK: pytorch3d magic
    K[..., 2, 3] = 1  # ...? # HACK: pytorch3d magic

    K[..., 0, 1] = 0
    K[..., 1, 0] = 0
    K[..., 2, 0] = 0
    K[..., 2, 1] = 0
    # return K

    K[..., 0, 0] = K[..., 0, 0] * 2.0 / M  # fx
    K[..., 1, 1] = K[..., 1, 1] * 2.0 / M  # fy
    K[..., 0, 2] = -(K[..., 0, 2] - W / 2.0) * 2.0 / M  # px
    K[..., 1, 2] = -(K[..., 1, 2] - H / 2.0) * 2.0 / M  # py
    return K

def get_pytorch3d_camera_params(R,T,K,H,W):
    # Extract pytorc3d camera parameters from batch input
    # R and T are applied on the right (requires a transposed R from OpenCV camera format)
    # Coordinate system is different from that of OpenCV (cv: right down front, 3d: left up front)
    # However, the correction has to be down on both T and R... (instead of just R)
    C = -R.mT @ T  # B, 3, 1
    R = R.clone()
    R[..., 0, :] *= -1  # flip x row
    R[..., 1, :] *= -1  # flip y row
    T = (-R @ C)[..., 0]  # c2w back to w2c
    R = R.mT  # applied left (left multiply to right multiply, god knows why...)

    H = H.item()  # !: BATCH
    W = W.item()  # !: BATCH
    K = get_pytorch3d_ndc_K(K, H, W)

    return H, W, K, R, T, C

def get_ndc_for_uv(pcd,K,R,T,H,W):
    pcd=pcd.float()
    # H[0]=609
    # W[0]=251
    H, W, K, R, T, C=get_pytorch3d_camera_params(R,T,K,H,W)
    K=K.float()
    R=R.float()
    T=T.float()
    rasterizer=PointsRasterizer()
    # for i in range(K.shape[0]):
        # K_temp=K[i,...].unsqueeze(0)
        # R_temp=R[i,...].unsqueeze(0)
        # T_temp=T[i,...].unsqueeze(0)
        # pcd_temp=pcd.reshape(1,-1,3)
    pcd=pcd.repeat(K.shape[0],1,1)
    ndc_pcd = rasterizer.transform(Pointclouds(pcd), cameras=PerspectiveCameras(K=K, R=R, T=T, device=pcd.device)).points_padded()
    ndc_uv=ndc_pcd[...,:-1]
    ndc_uv=ndc_uv.clip(min=-1,max=1)
    return ndc_uv
def get_ndc(pcd,K,R,T,H,W,rad):
    pcd=pcd.float()
    # H[0]=609
    # W[0]=251
    H, W, K, R, T, C=get_pytorch3d_camera_params(R,T,K,H,W)
    K=K.float()
    R=R.float()
    T=T.float()
    rasterizer=PointsRasterizer()
    ndc_pcd = rasterizer.transform(Pointclouds(pcd), cameras=PerspectiveCameras(K=K, R=R, T=T, device=pcd.device)).points_padded()
    ndc_rad = abs(K[..., 1, 1][..., None] * rad[..., 0] / (ndc_pcd[..., -1] + 1e-10))
    return ndc_pcd,ndc_rad