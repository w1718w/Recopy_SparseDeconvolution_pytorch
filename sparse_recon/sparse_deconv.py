import torch
import numpy as np
import warnings
import time


from .sparse_hessian_recon.sparse_hessian_recon import sparse_hessian
from .iterative_deconv.iterative_deconv import iterative_deconv
from .iterative_deconv.kernel import Gauss
from .utils.background_estimation import background_estimation
from .utils.upsample import spatial_upsample, fourier_upsample

def sparse_deconv(img, sigma, sparse_iter=100, fidelity=150, sparsity=10, tcontinuity=0.5,
                  background=1, deconv_iter=7, deconv_type=1, up_sample=0):
    """Sparse deconvolution
    ----------
    通用荧光显微图像后处理框架，支持：
    - 2D (XY) / 3D (XY-T/Z) 图像处理
    - 基于自然先验知识：空间稀疏性与时序连续性
    
    Parameters
    ----------
    img : ndarray or Tensor
        输入图像 (支持 T×X×Y 或 Z×X×Y 格式)
        支持numpy数组或PyTorch张量输入
    sigma : list
        点扩散函数尺寸 [x, y] 或 [x, y, z] (单位：像素)
        当前版本暂不支持3D反卷积
    sparse_iter: int, optional
        稀疏Hessian重建迭代次数 (默认: 100)
    fidelity : int, optional
        数据保真项权重 (默认: 150)
    tcontinuity : float, optional
        时序连续性权重 (默认: 0.5)
    sparsity : int, optional
        空间稀疏性权重 (默认: 10)
    background : int, optional
        背景估计模式 (默认:1)：
        0: 无背景扣除
        1: 高信噪比-弱背景
        2: 高信噪比-强背景
        3: 低信噪比-弱背景
        4: 低信噪比-中背景
        5: 低信噪比-强背景
    deconv_iter : int, optional
        反卷积迭代次数 (示例:7)
    deconv_type : int, optional
        反卷积算法选择：
        0: 无反卷积       
        1: Richardson-Lucy
        2: LandWeber
    up_sample : int, optional
        上采样方式 (2倍)：
        0: 无上采样       
        1: 傅里叶上采样
        2: 空间插值上采样（需降低保真/稀疏权重）

    Returns
    -------
    Tensor
        重建后的图像张量，保持与输入相同的维度
        数值范围自动还原到原始尺度

    Examples
    --------

    >>> img_tensor = torch.from_numpy(img).cuda()
    >>> recon = sparse_deconv(img_tensor, [5,5])

    References
    ----------
      [1] Weisong Zhao et al. Sparse deconvolution improves
      the resolution of live-cell super-resolution 
      fluorescence microscopy, Nature Biotechnology (2022),
      https://doi.org/10.1038/s41587-021-01092-2
    """
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).to(device)
    elif not isinstance(img, torch.Tensor):
        raise TypeError("输入类型应为ndarray或Tensor")
    else:
        img = img.to(device)
    

    scaler = img.max()
    img = img.float() / scaler


    if background > 0:
        bg_img = img.clone()
        if background in [1,2]:
            bg_img /= (2.5 if background==1 else 2.0)
            bg = background_estimation(bg_img)
        else:
            med_val = torch.mean(img)
            cutoff = med_val / (2.5 if background==3 else 
                               2.0 if background==4 else 1.0)
            bg_img = torch.where(img > cutoff, cutoff, img)
            bg = background_estimation(bg_img)
        
        img = torch.clamp(img - bg, min=0) 


    img = img / img.amax(dim=(-2,-1), keepdim=True)
    

    if up_sample == 1:
        img = fourier_upsample(img)
    elif up_sample == 2:
        img = spatial_upsample(img)
    

    start_time = time.process_time()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    img_sparse = sparse_hessian(
        img,
        iterations=sparse_iter,
        fidelity_weight=fidelity,
        sparsity_weight=sparsity,
        continuity_weight=tcontinuity
    )
    
    print(f'sparse-hessian time {time.process_time()-start_time:.2f}s')
    

    if deconv_type > 0:
        if not sigma:
            warnings.warn("The PSF's sigma is not given, turning off the iterative deconv...")
        else:
            kernel = Gauss(sigma).to(device)
            start_time = time.process_time()
            img_sparse = iterative_deconv(
                img_sparse, 
                kernel,
                iterations=deconv_iter,
                method=deconv_type
            )
            print(f'deconv time{time.process_time()-start_time:.2f}s')


    return img_sparse * scaler
