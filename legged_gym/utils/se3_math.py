import torch
from torch import Tensor
import numpy as np
# from pytorch3d.transforms import matrix_to_quaternion

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

def quat_to_rot_matrix(quat):
    """Convert quaternion [x, y, z, w] to rotation matrix (3x3).
    Args:
        quat: (..., 4)
    Returns:
        rot: (..., 3, 3)
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.stack([
        torch.stack([1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)], dim=-1)
    ], dim=-2)
    return R

# ==================== Batched SE(3) Utilities ====================

def skew_batch(v):
    """Batched skew-symmetric matrix.
    Args:
        v: (..., 3)
    Returns:
        (..., 3, 3)
    """
    zero = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
    ], dim=-2)

def quat_to_rot_matrix_batch(quat):
    """Convert batch of quaternions [x, y, z, w] to rotation matrices.
    Args:
        quat: (..., 4)
    Returns:
        rot: (..., 3, 3)
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.stack([
        torch.stack([1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)], dim=-1)
    ], dim=-2)
    return R

def pose_to_se3_batch(pos, quat):
    """Batched pose to SE(3) matrix.
    Args:
        pos: (..., 3)
        quat: (..., 4) in [x,y,z,w]
    Returns:
        T: (..., 4, 4)
    """
    N = pos.shape[:-1]
    T = torch.eye(4, device=pos.device).repeat(*N, 1, 1)
    R = quat_to_rot_matrix_batch(quat)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    return T

# def se3_to_pose_batch(T):
#     R = T[..., :3, :3]
#     trans = T[..., :3, 3]
#     quat_wxyz = matrix_to_quaternion(R)  # returns [w, x, y, z]
#     quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]  # reorder to [x, y, z, w]
#     return trans, quat_xyzw

def se3_exp_map_batch(twist, s):
    """Batched exponential map: exp(s * twist^) -> SE(3)
    Args:
        twist: (N, 6)  # [v; w]
        s: (N,)        # scalar path parameter
    Returns:
        T: (N, 4, 4)
    """
    N = twist.shape[0]
    device = twist.device
    v = twist[:, :3]      # (N, 3)
    w = twist[:, 3:]      # (N, 3)
    theta = torch.norm(w, dim=1)  # (N,)

    T = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)  # (N,4,4)

    eps = 1e-6
    small_angle = theta < eps
    big_angle = ~small_angle

    if small_angle.any():
        T[small_angle, :3, 3] = (s[small_angle].unsqueeze(1) * v[small_angle])

    if big_angle.any():
        w_big = w[big_angle]
        theta_big = theta[big_angle]
        s_big = s[big_angle]
        v_big = v[big_angle]
        N_big = w_big.shape[0]

        w_norm = w_big / theta_big.unsqueeze(1)  # (N_big, 3)
        w_skew = skew_batch(w_norm)  # (N_big, 3, 3)

        cos_t = torch.cos(s_big * theta_big)  # (N_big,)
        sin_t = torch.sin(s_big * theta_big)

        I = torch.eye(3, device=device).unsqueeze(0)  # (1,3,3)
        R = (cos_t[:, None, None] * I +
             (1 - cos_t)[:, None, None] * (w_norm.unsqueeze(-1) @ w_norm.unsqueeze(-2)) +
             sin_t[:, None, None] * w_skew)
        T[big_angle, :3, :3] = R

        # Compute V matrix
        theta2 = theta_big ** 2
        st = s_big * theta_big
        sin_st = torch.sin(st)
        one_minus_cos = 1.0 - torch.cos(st)

        V = (I * (sin_st / theta_big)[:, None, None] +
             (one_minus_cos / theta2)[:, None, None] * skew_batch(w_big) +
             ((1.0 - sin_st / (st + eps)) / theta2)[:, None, None] * (w_big.unsqueeze(-1) @ w_big.unsqueeze(-2)))
        
        trans = torch.bmm(V, (s_big.unsqueeze(1) * v_big).unsqueeze(-1)).squeeze(-1)  # (N_big, 3)
        T[big_angle, :3, 3] = trans

    return T

def se3_log_map_batch(T, eps=1e-6):
    """Batched logarithmic map: SE(3) -> se(3)
    Args:
        T: (N, 4, 4)
    Returns:
        twist: (N, 6) = [v; w]
    """
    N = T.shape[0]
    device = T.device
    R = T[:, :3, :3]  # (N,3,3)
    t = T[:, :3, 3]   # (N,3)

    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)  # (N,)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)  # (N,)

    w = torch.zeros(N, 3, device=device)
    v = torch.zeros(N, 3, device=device)

    small_angle = theta < eps
    big_angle = ~small_angle

    if small_angle.any():
        w[small_angle] = 0.0
        v[small_angle] = t[small_angle]

    if big_angle.any():
        R_big = R[big_angle]
        t_big = t[big_angle]
        theta_big = theta[big_angle]
        N_big = R_big.shape[0]

        # Compute w from R
        w_hat = (R_big - R_big.transpose(-2, -1)) / (2 * torch.sin(theta_big).unsqueeze(-1).unsqueeze(-1))
        w_vec = torch.stack([
            w_hat[:, 2, 1],
            w_hat[:, 0, 2],
            w_hat[:, 1, 0]
        ], dim=-1)  # (N_big, 3)
        w[big_angle] = w_vec

        # Compute V^{-1}
        I = torch.eye(3, device=device).unsqueeze(0)  # (1,3,3)
        theta_big_sq = theta_big ** 2
        A = theta_big
        B = (1 - theta_big * torch.cos(theta_big / 2) / (2 * torch.sin(theta_big / 2) + eps)) / (theta_big_sq + eps)
        V_inv = (I / A.unsqueeze(-1).unsqueeze(-1) -
                 0.5 * w_hat +
                 B.unsqueeze(-1).unsqueeze(-1) * torch.bmm(w_hat, w_hat))
        v[big_angle] = torch.bmm(V_inv, t_big.unsqueeze(-1)).squeeze(-1)

    return torch.cat([v, w], dim=-1)  # (N, 6)