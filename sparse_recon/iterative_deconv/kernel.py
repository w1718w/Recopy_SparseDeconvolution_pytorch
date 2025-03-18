import torch
import math
from math import log

def Gauss(sigma):
    # 和旧的一致，旧的用numpy，这里先不指定gpu
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=torch.float32)
    
    s = sigma.numel()
    if s == 1:
        sigma = torch.tensor([sigma, sigma], dtype=torch.float32)
    
    psfN = torch.ceil(sigma / math.sqrt(8 * math.log(2)) * math.sqrt(-2 * math.log(0.0002))) + 1
    N = psfN * 2 + 1
    sigma = sigma / (2 * math.sqrt(2 * math.log(2)))
    
    dim = len(N)
    if dim > 1:
        N[1] = torch.maximum(N[0], N[1])
        N[0] = N[1]
    
    if dim == 1:
        x = torch.arange(-torch.fix(N / 2), torch.ceil(N / 2), dtype=torch.float32)
        PSF = torch.exp(-0.5 * (x * x) / (sigma * sigma))
        PSF = PSF / PSF.sum()
        return PSF
    
    if dim == 2:
        m, n = N[0], N[1]
        x = torch.arange(-torch.fix(n/2), torch.ceil(n/2), dtype=torch.float32)
        y = torch.arange(-torch.fix(m/2), torch.ceil(m/2), dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='xy')  # 保持与numpy一致的坐标顺序
        s1, s2 = sigma[0], sigma[1]
        PSF = torch.exp(-(X**2)/(2*s1**2) - (Y**2)/(2*s2**2))
        PSF = PSF / PSF.sum()
        return PSF
    
    if dim == 3:
        m, n, k = N[0], N[1], N[2]
        x = torch.arange(-torch.fix(n/2), torch.ceil(n/2), dtype=torch.float32)
        y = torch.arange(-torch.fix(m/2), torch.ceil(m/2), dtype=torch.float32)
        z = torch.arange(-torch.fix(k/2), torch.ceil(k/2), dtype=torch.float32)
        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')  # 3D坐标处理
        s1, s2, s3 = sigma[0], sigma[1], sigma[2]
        PSF = torch.exp(-(X**2)/(2*s1**2) - (Y**2)/(2*s2**2) - (Z**2)/(2*s3**2))
        PSF = PSF / PSF.sum()
        return PSF


'''
def Generate_PSF(pixel, lamda, n, NA, z):
    sin2 = (1 - (1 - NA**2)) / 2
    u = 8 * math.pi * z * sin2 / lamda
    
    # 创建坐标网格
    x = torch.arange(-n*pixel, n*pixel + pixel, step=pixel, dtype=torch.float32)
    X, Y = torch.meshgrid(x, x, indexing='xy')
    
    s1 = torch.sqrt(X**2 + Y**2)
    idx = s1 <= 1
    
    p = torch.linspace(0, 1, 100, dtype=torch.float32)
    dp = p[1] - p[0]
    
    r = s1[idx]
    pr = p[None, :] * r[:, None]  # 创建二维积分网格
    arg = 2 * math.pi * (NA / lamda) * pr  # 整合所有常数项
    
    bes = torch.special.bessel_j0(arg)    # 直接计算零阶Bessel函数
    integrand = 2 * torch.exp(1j * u * (pr**2)/2) * bes
    
    integrals = torch.sum(integrand * dp, dim=1)
    
    IP = torch.zeros_like(s1)
    IP[idx] = torch.abs(integrals)**2
    Ipsf = IP / IP.sum()
    
    return Ipsfv
'''