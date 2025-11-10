import torch
from torch import Tensor
import numpy as np

def skew(v):
    zero = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
    ], dim=-2)

def se3_log_map(T, eps=1e-6):
    R = T[:3, :3]
    t = T[:3, 3]
    trace = torch.trace(R)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps))
    
    if theta < eps:
        w = torch.zeros(3, device=T.device)
        v = t.clone()
    else:
        w_hat = (R - R.T) / (2 * torch.sin(theta))
        w = torch.tensor([w_hat[2,1], w_hat[0,2], w_hat[1,0]], device=T.device)
        V_inv = (torch.eye(3, device=T.device) / theta -
                 0.5 * w_hat +
                 (1 - theta * torch.cos(theta/2) / (2 * torch.sin(theta/2))) / (theta**2) * (w_hat @ w_hat))
        v = V_inv @ t
    return torch.cat([v, w])

def se3_exp_map(twist, s):
    device = twist.device
    v = twist[:3]
    w = twist[3:]
    theta = torch.norm(w)
    
    T = torch.eye(4, device=device)
    
    if theta < 1e-6:
        T[:3, 3] = s * v
    else:
        w_norm = w / theta
        w_skew = skew(w_norm.unsqueeze(0)).squeeze(0)
        cos_t = torch.cos(s * theta)
        sin_t = torch.sin(s * theta)
        
        R = cos_t * torch.eye(3, device=device) + \
            (1 - cos_t) * (w_norm[:, None] @ w_norm[None, :]) + \
            sin_t * w_skew
        T[:3, :3] = R
        
        V = (torch.eye(3, device=device) * (sin_t / theta) +
             (1 - cos_t) / (theta**2) * skew(w) +
             (1 - sin_t / theta) / (theta**2) * (w[:, None] @ w[None, :]))
        T[:3, 3] = V @ (s * v)
    return T

def pose_to_se3(pos, quat):
    T = torch.eye(4, device=pos.device)
    q = quat[[3, 0, 1, 2]]  # xyzw -> wxyz
    # 手动计算旋转矩阵（避免 pytorch3d 依赖）
    w, x, y, z = q
    T[:3, :3] = torch.tensor([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]
    ], device=pos.device)
    T[:3, 3] = pos
    return T

def se3_to_pose(T):
    R = T[:3, :3]
    # 从旋转矩阵提取四元数（简化版，仅用于设置）
    trace = torch.trace(R)
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    quat_xyzw = torch.tensor([x, y, z, w], device=T.device)
    return T[:3, 3], quat_xyzw