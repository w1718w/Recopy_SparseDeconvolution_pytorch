import math

# import warnings
import torch
import torch.fft
import torch.nn.functional as F


def iterative_deconv(data, kernel, iteration, rule):
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.as_tensor(kernel, dtype=torch.float32)

    device = data.device

    if data.ndim > 2:
        data_de = torch.zeros(
            (data.shape[0], data.shape[1], data.shape[2]),
            dtype=torch.float32,
            device=device,
        )
        for i in range(data.shape[0]):
            data_de[i] = deblur_core(data[i], kernel, iteration, rule).real
    else:
        data_de = deblur_core(data, kernel, iteration, rule).real

    return data_de


def deblur_core(data, kernel, iteration, rule):
    device = data.device
    kernel = kernel.to(device)
    kernel = kernel / kernel.sum()
    kernel_initial = kernel.clone()

    dx, dy = data.shape[-2], data.shape[-1]
    B = math.floor(min(dx, dy) / 6)

    # 边缘填充
    # TODO:似乎是pytorch的pad函数的bug，需要添加[None,None]才能运行
    # pytorch 2.6.0+cu124
    # https://github.com/pytorch/pytorch/issues/147506
    pad_layer = torch.nn.ReplicationPad2d((B, B, B, B))
    data = pad_layer(data[None, None])

    yk = data.clone()
    xk = torch.zeros_like(data)
    vk = torch.zeros_like(data)

    otf = psf2otf(kernel_initial, data.shape)

    if rule == 2:
        t = 1
        gamma1 = torch.tensor(1.0, device=device)
        xk_prev = torch.zeros_like(data)
        for i in range(iteration):
            if i == 0:
                xk = (
                    data
                    + t
                    * torch.fft.ifftn(
                        torch.conj(otf)
                        * (torch.fft.fftn(data) - otf * torch.fft.fftn(data))
                    ).real
                )
                xk_prev = xk.clone()  # 保存当前值
            else:
                gamma2 = 0.5 * (4 * gamma1**2 + gamma1**4).sqrt() - gamma1**2
                beta = -gamma2 * (1 - 1 / gamma1)
                yk_update = xk + beta * (xk - xk_prev)
                yk_term = torch.fft.ifftn(
                    torch.conj(otf)
                    * (torch.fft.fftn(data) - otf * torch.fft.fftn(yk_update))
                )
                yk = yk_update + t * yk_term.real
                yk = torch.clamp(yk, min=1e-6)
                gamma1 = gamma2
                xk_prev = xk.clone()
                xk = yk.clone()

    elif rule == 1:  # Richardson-Lucy方法
        for iter in range(iteration):
            xk_prev = xk.clone()
            rliter1 = rliter(yk, data, otf)

            # 计算反卷积更新项
            numerator = torch.fft.ifftn(torch.conj(otf) * rliter1).real
            denominator = torch.fft.ifftn(
                otf * torch.fft.fftn(torch.ones_like(data))
            ).real
            xk = yk * (numerator / denominator.clamp(min=1e-6))

            xk = torch.clamp(xk, min=1e-6)
            vk_prev = vk.clone()
            vk = torch.clamp(xk - yk, min=1e-6)

            if iter == 0:
                alpha = torch.tensor(0.0, device=device)
                yk = xk.clone()
            else:
                alpha = (vk_prev * vk).sum() / (vk_prev**2).sum().clamp(min=1e-10)
                alpha = torch.clamp(alpha, min=1e-6, max=1.0)
                yk = xk + alpha * (xk - xk_prev)
                yk = torch.nan_to_num(yk, nan=1e-6)

    yk = torch.clamp(yk, min=0)
    return yk[..., B:-B, B:-B]  # 去除填充部分


def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return (x, y)


def psf2otf(psf, outSize):
    psf_size = torch.tensor(psf.shape[-2:])
    out_size = (
        torch.tensor(outSize[-2:])
        if isinstance(outSize, torch.Size)
        else torch.tensor(outSize)
    )

    pad_size = (out_size - psf_size).tolist()
    psf = F.pad(psf, (0, pad_size[1], 0, pad_size[0]))

    for i in range(2):
        shift = -int(psf_size[i] // 2)
        psf = torch.roll(psf, shifts=shift, dims=i)

    otf = torch.fft.fftn(psf, dim=(-2, -1))
    return otf


def rliter(yk, data, otf):
    denominator = torch.fft.ifftn(otf * torch.fft.fftn(yk)).real.clamp(min=1e-6)
    return torch.fft.fftn(data / denominator)
