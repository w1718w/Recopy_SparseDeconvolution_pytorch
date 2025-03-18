import torch
import math
import warnings
import numpy as np

def iterative_deconv(data, kernel, iteration, rule):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32, device=device)
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.as_tensor(kernel, dtype=torch.float32, device=device)
    
    if data.ndim > 2:
        data_de = torch.zeros((data.shape[0], data.shape[1], data.shape[2]), 
                            dtype=torch.float32, device=device)
        for i in range(data.shape[0]):
            data_de[i] = deblur_core(data[i], kernel, iteration, rule).real
    else:
        data_de = deblur_core(data, kernel, iteration, rule).real
    
    return data_de

def deblur_core(data, kernel, iteration, rule):
    device = data.device
    kernel = kernel.to(device)
    
    kernel = kernel / torch.sum(kernel)
    kernel_initial = kernel.clone()
    
    dx, dy = data.shape[-2], data.shape[-1]
    B = math.floor(min(dx, dy) / 6)
    data = torch.nn.functional.pad(data, [B, B, B, B], mode='replicate')
    

    yk = data.clone()
    xk = torch.zeros_like(data)
    vk = torch.zeros_like(data)
    otf = psf2otf(kernel_initial, data.shape)
    
    # LandWeber反卷积
    if rule == 2:
        t = 1.0
        gamma1 = 1.0
        for i in range(iteration):
            if i == 0:
                xk = data + t * torch.fft.ifftn(
                    torch.conj(otf) * (torch.fft.fftn(data) - otf * torch.fft.fftn(data))
                ).real
            else:
                gamma2 = 0.5 * (4 * gamma1**2 + gamma1**4)**0.5 - gamma1**2
                beta = -gamma2 * (1 - 1 / gamma1)
                yk_update = xk + beta * (xk - xk_update)
                yk = yk_update + t * torch.fft.ifftn(
                    torch.conj(otf) * (torch.fft.fftn(data) - otf * torch.fft.fftn(yk_update))
                ).real
                yk = torch.clamp(yk, min=1e-6)
                gamma1 = gamma2
                xk_update = xk.clone()
                xk = yk.clone()
    
    # Richardson-Lucy反卷积
    elif rule == 1:
        for iter in range(iteration):
            xk_update = xk.clone()
            rliter_val = rliter(yk, data, otf)
            
            numerator = torch.fft.ifftn(torch.conj(otf) * rliter_val).real
            denominator = torch.fft.ifftn(torch.fft.fftn(torch.ones_like(data)) * otf).real
            xk = yk * (numerator / torch.clamp(denominator, min=1e-6))
            
            xk = torch.clamp(xk, min=1e-6)
            vk_update = vk.clone()
            vk = torch.clamp(xk - yk, min=1e-6)
            
            if iter == 0:
                alpha = 0.0
                yk = xk.clone()
            else:
                alpha = torch.sum(vk_update * vk) / torch.clamp(torch.sum(vk_update**2), min=1e-10)
                alpha = torch.clamp(alpha, min=1e-6, max=1.0)
                yk = xk + alpha * (xk - xk_update)
                yk = torch.where(torch.isnan(yk), torch.tensor(1e-6, device=device), yk)
    
    yk = torch.clamp(yk, min=0)
    data_decon = yk[..., B:-B, B:-B]
    return data_decon

def psf2otf(psf, outSize):
    psf = psf.to(torch.float32)
    pad_size = (outSize[-2] - psf.shape[-2], outSize[-1] - psf.shape[-1])
    psf = torch.nn.functional.pad(psf, [0, pad_size[1], 0, pad_size[0]])
    
    for i in range(2):
        psf = torch.roll(psf, shifts=-psf.shape[-2+i]//2, dims=-2+i)
    
    otf = torch.fft.fftn(psf, dim=(-2, -1), norm='ortho')
    return otf

def rliter(yk, data, otf):
    denominator = torch.fft.ifftn(otf * torch.fft.fftn(yk)).real
    return torch.fft.fftn(data / torch.clamp(denominator, min=1e-6))
