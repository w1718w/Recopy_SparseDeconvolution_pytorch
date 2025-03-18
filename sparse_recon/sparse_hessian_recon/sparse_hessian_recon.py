import gc  # 垃圾回收模块

import torch
import torch.nn.functional as F

from .operation import *
from .sparse_iteration import *


def sparse_hessian(f, iteration_num=100, fidelity=150, sparsity=10, contiz=0.5, mu=1):
    """function g = SparseHessian_core(f,iteration_num,fidelity,sparsity,iteration,contiz,mu)
    -----------------------------------------------
    Source code for argmin_g { ||f-g ||_2^2 +||gxx||_1+||gxx||_1+||gyy||_1+lamdbaz*||gzz||_1+2*||gxy||_1
     +2*sqrt(lamdbaz)||gxz||_1+ 2*sqrt(lamdbaz)|||gyz||_1+2*sqrt(lamdbal1)|||g||_1}

    Args:
        f (Torch.tensor): 输入图像，可以是 N 维张量。
        iteration_num (int, optional): 迭代次数. Defaults to 100.
        fidelity (int, optional): 保真度权重. Defaults to 150.
        sparsity (int, optional): 稀疏性权重. Defaults to 10.
        contiz (float, optional): 沿 z 轴的连续性权重. Defaults to 0.5.
        mu (int, optional): 正则化参数. Defaults to 1.

    Returns:
        - g: 优化后的图像张量。
    """
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = f.to(device)
    contiz = torch.tensor(contiz, dtype=torch.float32).to(device)

    # 记录f的维度
    f_flag = f.dim()
    flage = 0

    if f_flag == 2:
        contiz = 0
        flage = 1
        f = f.unsqueeze(0).repeat(3, 1, 1)  # 扩展为 3 层
    elif f_flag > 2:
        if f.size(0) < 3:
            contiz = 0
            f = F.pad(
                f, (0, 0, 0, 0, 0, 3 - f.size(0)), mode="replicate"
            )  # 补充到 3 层
        else:
            pass
    imgsize = f.shape

    print("Start the Sparse deconvolution...")

    xxfft = operation_xx(imgsize, device=device)
    yyfft = operation_yy(imgsize, device=device)
    zzfft = operation_zz(imgsize, device=device)
    xyfft = operation_xy(imgsize, device=device)
    xzfft = operation_xz(imgsize, device=device)
    yzfft = operation_yz(imgsize, device=device)

    operationfft = (
        xxfft
        + yyfft
        + (contiz**2) * zzfft
        + 2 * xyfft
        + 2 * contiz * xzfft
        + 2 * contiz * yzfft
    )
    normalize = (fidelity / mu) + (sparsity**2) + operationfft

    # 和原本逻辑类似的清理工作
    # del xxfft, yyfft, zzfft, xyfft, xzfft, yzfft, operationfft
    # torch.cuda.empty_cache()

    # b是原式中的v吧
    bxx = torch.zeros(imgsize, dtype=torch.float32, device=device)
    byy = torch.zeros_like(bxx)
    bzz = torch.zeros_like(bxx)
    bxy = torch.zeros_like(bxx)
    bxz = torch.zeros_like(bxx)
    byz = torch.zeros_like(bxx)
    bl1 = torch.zeros_like(bxx)
    g_update = (fidelity / mu) * f

    # 迭代过程
    for iter in range(iteration_num):
        g_update_fft = torch.fft.fftn(g_update, dim=(-2, -1))

        if iter == 0:
            g = torch.fft.ifftn(g_update_fft / (fidelity / mu), dim=(-2, -1)).real
        else:
            g = torch.fft.ifftn(g_update_fft / normalize, dim=(-2, -1)).real

        g_update = (fidelity / mu) * f

        Lxx, bxx = iter_xx(g, bxx, 1, mu)
        g_update += Lxx
        del Lxx
        # gc.collect()  # 回收

        Lyy, byy = iter_yy(g, byy, 1, mu)
        g_update += Lyy
        del Lyy
        # gc.collect()

        Lzz, bzz = iter_zz(g, bzz, contiz**2, mu)
        g_update += Lzz
        del Lzz
        # gc.collect()

        Lxy, bxy = iter_xy(g, bxy, 2, mu)
        g_update += Lxy
        del Lxy
        # gc.collect()

        Lxz, bxz = iter_xz(g, bxz, 2 * contiz, mu)
        g_update += Lxz
        del Lxz
        # gc.collect()

        Lyz, byz = iter_yz(g, byz, 2 * contiz, mu)
        g_update += Lyz
        del Lyz
        # gc.collect()

        Lsparse, bl1 = iter_sparse(g, bl1, sparsity, mu)
        g_update += Lsparse
        del Lsparse
        # gc.collect()
        # 100次回收一次————显存不够用的话在每个del下面加回收
        if iter % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"{iter + 1} iterations done\r", end="")

        g = torch.clamp(g, min=0)

    # 清理内存
    del bxx, byy, bzz, bxy, byz, bl1, f, normalize, g_update
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return g[1, :, :] if flage else g
