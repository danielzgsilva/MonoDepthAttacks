import torch
import torch.nn.functional as F
from torch import nn
import scipy.stats as st
import numpy as np


class GaussianSmooth(nn.Module):
    """
        Gaussian Smoothing function, also used in generating translatin invariant adversarial examples
        Arguments:  
            kernlen: (2k + 1) Kernel size: k
            nsig: upper and lower limits of gaussian kernel

    """

    def __init__(self, kernlen=21, nsig=3):
        super(GaussianSmooth, self).__init__()
        self.kernlen = kernlen
        self.nsig = nsig

        kernel = self.gkern().astype(np.float32)
        self.stack_kernel = np.stack([kernel, kernel, kernel])
        self.stack_kernel = np.expand_dims(self.stack_kernel, 0)
        self.stack_kernel = torch.from_numpy(self.stack_kernel).cuda()
        print('using TI with kernel size ', self.stack_kernel.size(-1))
        self.stride = 1

    def gkern(self):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-self.nsig, self.nsig, self.kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def forward(self, x):
        padding = int((((self.stride - 1) * x.size(-1) -
                        self.stride + self.stack_kernel.size(-1)) / 2) + 0.5)
        noise = F.conv2d(x, self.stack_kernel,
                         stride=self.stride, padding=padding)
        noise = noise / torch.mean(torch.abs(noise), [1, 2, 3], keepdim=True)
        x = x + noise
        return x
