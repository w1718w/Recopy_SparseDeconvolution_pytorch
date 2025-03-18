# utils/upsample.py
from typing import Tuple

import torch


def spatial_upsample(SIMmovie, n=2):
    """通过零值插入 空间域上采样

    Args:
        SIMmovie (torch.Tensor): 输入张量，支持2D或3D格式
            - 2D形状: (height, width)
            - 3D形状: (t, height, width)
        n (int, optional): 上采样倍数，默认值为2

    Returns:
    torch.Tensor: 上采样后的张量，数据类型为float32，设备与输入一致
        - 2D输入输出形状: (height*n, width*n)
        - 3D输入输出形状: (t, height*n, width*n)
    """
    if SIMmovie.ndim == 3:
        sz, sx, sy = SIMmovie.shape
        y = torch.zeros(
            (sz, sx * n, sy * n), dtype=torch.float32, device=SIMmovie.device
        )
        y[:, ::n, ::n] = SIMmovie  # 替换 步长为n，从开始到结尾
    else:
        sx, sy = SIMmovie.shape
        y = torch.zeros((sx * n, sy * n), dtype=torch.float32, device=SIMmovie.device)
        y[::n, ::n] = SIMmovie
    return y


def fourier_upsample(imgstack, n=2):
    """基于傅里叶变换的频域插值上采样

    Args:
        imgstack (torch.Tensor): 输入张量，支持2D或3D格式
            - 2D形状: (height, width)
            - 3D形状: (frames, height, width)
        n (int, optional): 上采样倍数，默认值为2

    Returns:
        torch.Tensor: 上采样后的张量，数据类型为float32，设备与输入一致
            - 2D输入输出形状: (height*n, width*n)
            - 3D输入输出形状: (frames, height*n, width*n)

    Raises:
        ValueError: 当输入维度不在2-3维时抛出异常
    """
    if imgstack.ndim not in [2, 3]:
        raise ValueError(f"输入维度应为2D或3D，当前维度：{imgstack.ndim}")

    original_ndim = imgstack.ndim
    if original_ndim == 2:
        imgstack = imgstack.unsqueeze(0)

    z, sx, sy = imgstack.shape
    device = imgstack.device
    n_tensor = torch.tensor([n, n], device=device)
    new_shape = (n_tensor * torch.tensor([sx, sy], device=device)).long()
    imgfl = torch.zeros(
        (z, new_shape[0], new_shape[1]), dtype=torch.float32, device=device
    )

    for i in range(z):
        img = imgstack[i]
        imgsz = torch.tensor(img.shape)

        sz = imgsz if imgsz[0] % 2 == 1 else imgsz - 1
        padsize = sz.float() / 2
        k = torch.ceil(padsize).int()
        f = torch.floor(padsize).int()

        padded = symm_pad(img, (k[1].item(), f[1].item(), k[0].item(), f[0].item()))

        newsz = n_tensor * sz
        imgl = _fInterp_2D(padded, newsz)

        idx = torch.ceil(sz / 2) + 1 + (n_tensor - 1) * torch.floor(sz / 2)
        start_h = idx[0].long()
        start_w = idx[1].long()
        end_h = start_h + n * sz[0].int()
        end_w = start_w + n * sz[1].int()

        imgfl[i] = imgl[start_h:end_h, start_w:end_w]

    return imgfl.squeeze() if original_ndim == 2 else imgfl


def _fInterp_2D(img: torch.Tensor, newsz: torch.Tensor) -> torch.Tensor:
    """傅里叶插值核心实现"""
    imgsz = torch.tensor(img.shape)
    if torch.any(newsz == 0):
        return torch.tensor([], dtype=img.dtype, device=img.device)

    B = (newsz[0] / imgsz[0] * newsz[1] / imgsz[1]).to(img.dtype)
    img_fft = torch.fft.fft2(img) * B

    # 初始化输出频谱
    a, b = newsz.int().tolist()
    img_ip = torch.zeros((a, b), dtype=torch.complex64, device=img.device)

    # 计算Nyquist索引
    nyqst = torch.ceil((imgsz + 1) / 2).int()
    h, w = nyqst.tolist()

    # 四象限复制
    img_ip[:h, :w] = img_fft[:h, :w]  # 左上
    img_ip[:h, b - (imgsz[1] - w) :] = img_fft[:h, w:]  # 右上
    img_ip[a - (imgsz[0] - h) :, :w] = img_fft[h:, :w]  # 左下
    img_ip[a - (imgsz[0] - h) :, b - (imgsz[1] - w) :] = img_fft[h:, w:]  # 右下

    # 处理偶数尺寸
    if imgsz[0] % 2 == 0 and a != imgsz[0]:
        img_ip[h - 1, :] *= 0.5
        img_ip[a - (imgsz[0] - h), :] = img_ip[h - 1, :]
    if imgsz[1] % 2 == 0 and b != imgsz[1]:
        img_ip[:, w - 1] *= 0.5
        img_ip[:, b - (imgsz[1] - w)] = img_ip[:, w - 1]

    # 逆变换并返回实数部分
    return torch.fft.ifft2(img_ip).real


def symm_pad(im: torch.Tensor, padding: Tuple[int, int, int, int]):
    """优化的对称填充实现"""
    pl, pr, pt, pb = padding

    # 宽度填充
    if pl > 0:
        left = im[..., :, :pl].flip(-1)
        im = torch.cat([left, im], dim=-1)
    if pr > 0:
        right = im[..., :, -pr:].flip(-1)
        im = torch.cat([im, right], dim=-1)

    # 高度填充
    if pt > 0:
        top = im[..., :pt, :].flip(-2)
        im = torch.cat([top, im], dim=-2)
    if pb > 0:
        bottom = im[..., -pb:, :].flip(-2)
        im = torch.cat([im, bottom], dim=-2)

    return im
