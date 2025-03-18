import ptwt
import torch


def Low_frequency_resolve(coeffs, dlevel):
    """
    高频置零 pytorch版本
        coeffs: list 小波分解后的系数列表
        dlevel: int  小波分解层数
    """
    device = coeffs[0].device
    cAn = coeffs[0]  # 低频分量
    vec = [cAn]
    for i in range(1, dlevel + 1):
        # 获取高频分量的形状并创建零张量
        # cH_shape = coeffs[i][0].shape
        zero_tensor = torch.zeros_like(coeffs[i][0], device=device)
        vec.append((zero_tensor, zero_tensor, zero_tensor))
    return vec


def rm_1(Biter, x, y):
    """调整逆变换后的张量尺寸"""
    Biter_new = torch.zeros((x, y), dtype=torch.uint8)
    if x % 2 and y % 2 == 0:
        Biter_new[:, :] = Biter[:x, :]
    elif x % 2 == 0 and y % 2:
        Biter_new[:, :] = Biter[:, :y]
    elif x % 2 and y % 2:
        Biter_new[:, :] = Biter[:x, :y]
    else:
        Biter_new = Biter
    return Biter_new


def background_estimation(imgs, th=1, dlevel=7, wavename="db6", iter=3):
    """基于小波变换的图像背景估计
    Args:
        imgs (torch.Tensor): 输入图像张量，支持形状：
            - 单张图像：(Height, Width)
            - 时序图像：(Time, Height, Width)
        th (int, optional): 1时启用迭代优化，0时仅单次分解。默认1
        dlevel (int, optional): 小波分解层数。默认7
        wavename (str, optional): 小波基名称，参考`pywt.wavelist(kind='discrete')`。默认'db6'
        iter (int, optional): 优化迭代次数。默认3

    Returns:
        torch.Tensor: 背景图像，形状与输入imgs相同

    Raises:
        AssertionError: 当输入维度不符合要求时抛出
    """
    assert imgs.dim() in [2, 3], "输入维度需为2D(H,W)或3D(T,H,W)"
    device = imgs.device  # 设备和img一致
    if imgs.dim() == 3:
        t, x, y = imgs.shape
        Background = torch.zeros_like(imgs)
        for taxial in range(t):
            img = imgs[taxial, ...]
            for _ in range(iter):
                res = img.clone()
                coeffs = ptwt.wavedec2(
                    res, wavelet=wavename, level=dlevel, mode="symmetric"
                )
                vec = Low_frequency_resolve(coeffs, dlevel)
                Biter = ptwt.waverec2(vec, wavelet=wavename)
                Biter_new = rm_1(Biter, x, y)

                if th > 0:
                    eps = torch.sqrt(torch.abs(res)) / 2
                    ind = img > (Biter_new + eps)
                    res[ind] = Biter_new[ind] + eps[ind]
                    coeffs1 = ptwt.wavedec2(
                        res, wavelet=wavename, level=dlevel, mode="symmetric"
                    )
                    vec = Low_frequency_resolve(coeffs1, dlevel)
                    Biter = ptwt.waverec2(vec, wavelet=wavename)
                    Biter_new = rm_1(Biter, x, y)
                Background[taxial, ...] = Biter_new
    else:
        x, y = imgs.shape
        Background = torch.zeros_like(imgs)
        res = imgs.clone()
        for i in range(iter):
            coeffs = ptwt.wavedec2(
                res, wavelet=wavename, level=dlevel, mode="symmetric"
            )
            vec = Low_frequency_resolve(coeffs, dlevel)
            Biter = ptwt.waverec2(vec, wavelet=wavename)

            Biter = Biter.squeeze(0)
            Biter_new = rm_1(Biter, x, y)

            if th > 0:
                eps = torch.sqrt(
                    torch.abs(res)
                )  # 假设噪声服从泊松分布，其方差与信号强度成正比
                ind = imgs > (Biter_new + eps)
                res[ind] = Biter_new[ind] + eps[ind]
                coeffs1 = ptwt.wavedec2(
                    res, wavelet=wavename, level=dlevel, mode="symmetric"
                )
                vec = Low_frequency_resolve(coeffs1, dlevel)
                Biter = ptwt.waverec2(vec, wavelet=wavename)
                Biter_new = rm_1(Biter, x, y)
            Background = Biter_new

    return Background
