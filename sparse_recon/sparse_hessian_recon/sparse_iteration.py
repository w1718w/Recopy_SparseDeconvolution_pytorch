import warnings

import torch

if not torch.cuda.is_available():
    warnings.warn("Could not import CUDA... falling back to CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_diff(data, step, dim):
    # data --- input image (torch.Tensor)
    # step
    # dim --- determine which is the dimension to calculate derivative
    # dim = 0 --> z axis
    # dim = 1 --> y axis
    # dim = 2 --> x axis

    assert dim <= 2
    r, n, m = data.shape
    size = [r, n, m]
    position = [0, 0, 0]
    temp1 = torch.zeros([x + 1 for x in size], dtype=torch.float32, device=device)
    temp2 = torch.zeros([x + 1 for x in size], dtype=torch.float32, device=device)

    size[dim] += 1
    position[dim] += 1

    temp1[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data
    temp2[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data

    size[dim] -= 1
    temp2[: size[0], : size[1], : size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] += 1

    out = temp1[position[0] : size[0], position[1] : size[1], position[2] : size[2]]
    return -out


def back_diff(data, step, dim):
    # data --- input image (torch.Tensor)
    # step
    # dim --- determine which is the dimension to calculate derivative
    # dim = 0 --> z axis
    # dim = 1 --> y axis
    # dim = 2 --> x axis
    assert dim <= 2
    r, n, m = data.shape
    size = [r, n, m]
    position = [0, 0, 0]
    temp1 = torch.zeros([x + 1 for x in size], dtype=torch.float32, device=device)
    temp2 = torch.zeros([x + 1 for x in size], dtype=torch.float32, device=device)

    temp1[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data
    temp2[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data

    size[dim] += 1
    position[dim] += 1

    temp2[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] -= 1
    out = temp1[: size[0], : size[1], : size[2]]
    return out


def shrink(x, L):  # shrink函数
    s = torch.abs(x)
    xs = torch.sign(x) * torch.maximum(s - 1 / L, torch.tensor(0.0, device=device))
    return xs


def iter_xx(g, bxx, para, mu):  # v和u还有L迭代 xx方向
    gxx = back_diff(forward_diff(g, 1, 1), 1, 1)
    dxx = shrink(gxx + bxx, mu)  # u
    bxx = bxx + (gxx - dxx)
    Lxx = para * back_diff(forward_diff(dxx - bxx, 1, 1), 1, 1)  # L
    return Lxx, bxx


def iter_xy(g, bxy, para, mu):
    gxy = forward_diff(forward_diff(g, 1, 1), 1, 2)
    dxy = shrink(gxy + bxy, mu)
    bxy = bxy + (gxy - dxy)
    Lxy = para * back_diff(back_diff(dxy - bxy, 1, 2), 1, 1)
    return Lxy, bxy


def iter_xz(g, bxz, para, mu):
    gxz = forward_diff(forward_diff(g, 1, 1), 1, 0)
    dxz = shrink(gxz + bxz, mu)
    bxz = bxz + (gxz - dxz)
    Lxz = para * back_diff(back_diff(dxz - bxz, 1, 0), 1, 1)
    return Lxz, bxz


def iter_yy(g, byy, para, mu):
    gyy = back_diff(forward_diff(g, 1, 2), 1, 2)
    dyy = shrink(gyy + byy, mu)
    byy = byy + (gyy - dyy)
    Lyy = para * back_diff(forward_diff(dyy - byy, 1, 2), 1, 2)
    return Lyy, byy


def iter_yz(g, byz, para, mu):
    gyz = forward_diff(forward_diff(g, 1, 2), 1, 0)
    dyz = shrink(gyz + byz, mu)
    byz = byz + (gyz - dyz)
    Lyz = para * back_diff(back_diff(dyz - byz, 1, 0), 1, 2)
    return Lyz, byz


def iter_zz(g, bzz, para, mu):
    gzz = back_diff(forward_diff(g, 1, 0), 1, 0)
    dzz = shrink(gzz + bzz, mu)
    bzz = bzz + (gzz - dzz)
    Lzz = para * back_diff(forward_diff(dzz - bzz, 1, 0), 1, 0)
    return Lzz, bzz


def iter_sparse(gsparse, bsparse, para, mu):
    dsparse = shrink(gsparse + bsparse, mu)
    bsparse = bsparse + (gsparse - dsparse)
    Lsparse = para * (dsparse - bsparse)
    return Lsparse, bsparse
