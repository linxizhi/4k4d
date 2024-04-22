import torch
from scipy.ndimage import zoom
def set_ndc_matrix(K,H,W,near,far):
    H,W=H.item(),W.item()
    n,f=near,far
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


def get_ndc_matrix(K,H,W,near,far):
    ndc_matrix=torch.zeros((4,4),device=K.device,dtype=K.dtype)
    ndc_matrix[3,2]=-1
    ndc_matrix[2,2]=(far+near)/(near-far)
    ndc_matrix[2,3]=2*far*near/(near-far)
    ndc_matrix[0,0]=2*K[...,0,0]/W
    ndc_matrix[1,1]=2*K[...,1,1]/H
    ndc_matrix=ndc_matrix.unsqueeze(0)
    return ndc_matrix

def affine_padding(c2w: torch.Tensor):
    # Already padded
    if c2w.shape[-2] == 4:
        return c2w
    # Batch agnostic padding
    sh = c2w.shape
    pad0 = c2w.new_zeros(sh[:-2] + (1, 3))  # B, 1, 3
    pad1 = c2w.new_ones(sh[:-2] + (1, 1))  # B, 1, 1
    pad = torch.cat([pad0, pad1], dim=-1)  # B, 1, 4
    c2w = torch.cat([c2w, pad], dim=-2)  # B, 4, 4
    return c2w
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)
def prepare_feedback_transform(H: int, W: int, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor,
                               n: torch.Tensor,
                               f: torch.Tensor,
                               xyz: torch.Tensor):
    # ixt = set_ndc_matrix(K, H, W, n, f).to(xyz.dtype)  # to opengl, remove last dim of n and f
    H,W=H.to(device=xyz.device),W.to(device=xyz.device)
    ixt=get_ndc_matrix(K,H,W,n,f)
    
    w2c = affine_padding(torch.cat([R, T], dim=-1)).to(xyz.dtype)
    c2w = affine_inverse(w2c)
    # c2w[..., 0] *= 1  # flip x
    # c2w[..., 1] *= -1  # flip y
    # c2w[..., 2] *= -1  # flip z
    ext = affine_inverse(c2w)
    pix_xyz = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1) @ ext.mT @ ixt.mT
    ndc_xyz=pix_xyz[..., :-1]
    ndc_xyz_final=ndc_xyz.clone()
    
    ndc_xyz_final[...,:]=ndc_xyz[...,:]/pix_xyz[...,-1:]
    # print(torch.max(ndc_xyz),torch.min(ndc_xyz))
    return ndc_xyz_final,pix_xyz[...,-1:]

def resize_array(arr, new_shape):
    
    factors = [n / o for n, o in zip(new_shape, arr.shape)]

    resized_arr = zoom(arr, factors)

    return resized_arr