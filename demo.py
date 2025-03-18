import torch
from matplotlib import pyplot as plt
from skimage import io
from sparse_recon.sparse_deconv import sparse_deconv

if __name__ == "__main__":
    im = io.imread("2222.tif")
    plt.imshow(im, cmap="gray")
    plt.show()

    im = torch.tensor(im, device="cuda")
    pixelsize = 65  # (nm)
    resolution = 280  # (nm)
    img_recon = sparse_deconv(im, resolution / pixelsize)
    img_recon = img_recon.detach().cpu().numpy()

    plt.imshow(img_recon / img_recon.max() * 255, cmap="gray")
    plt.show()
    io.imsave("test_processed.tif", img_recon.astype(im.dtype))
