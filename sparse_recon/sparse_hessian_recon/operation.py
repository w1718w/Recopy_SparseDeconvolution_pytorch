import warnings

import torch

if not torch.cuda.is_available():
    warnings.warn("CUDA is not available... falling back to CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def operation_xx(gsize, device):
    delta_xx = torch.tensor([[[1, -2, 1]]], dtype=torch.float32, device=device)
    xxfft = torch.fft.fftn(delta_xx, s=gsize, norm="backward") * torch.conj(
        torch.fft.fftn(delta_xx, s=gsize, norm="backward")
    )
    return xxfft


def operation_xy(gsize, device):
    delta_xy = torch.tensor([[[1, -1], [-1, 1]]], dtype=torch.float32, device=device)
    xyfft = torch.fft.fftn(delta_xy, s=gsize, norm="backward") * torch.conj(
        torch.fft.fftn(delta_xy, s=gsize, norm="backward")
    )
    return xyfft


def operation_xz(gsize, device):
    delta_xz = torch.tensor([[[1, -1]], [[-1, 1]]], dtype=torch.float32, device=device)
    xzfft = torch.fft.fftn(delta_xz, s=gsize, norm="backward") * torch.conj(
        torch.fft.fftn(delta_xz, s=gsize, norm="backward")
    )
    return xzfft


def operation_yy(gsize, device):
    delta_yy = torch.tensor([[[1], [-2], [1]]], dtype=torch.float32, device=device)
    yyfft = torch.fft.fftn(delta_yy, s=gsize, norm="backward") * torch.conj(
        torch.fft.fftn(delta_yy, s=gsize, norm="backward")
    )
    return yyfft


def operation_yz(gsize, device):
    delta_yz = torch.tensor(
        [[[1], [-1]], [[-1], [1]]], dtype=torch.float32, device=device
    )
    yzfft = torch.fft.fftn(delta_yz, s=gsize, norm="backward") * torch.conj(
        torch.fft.fftn(delta_yz, s=gsize, norm="backward")
    )
    return yzfft


def operation_zz(gsize, device):
    delta_zz = torch.tensor([[[1]], [[-2]], [[1]]], dtype=torch.float32, device=device)
    zzfft = torch.fft.fftn(delta_zz, s=gsize, norm="backward") * torch.conj(
        torch.fft.fftn(delta_zz, s=gsize, norm="backward")
    )
    return zzfft
