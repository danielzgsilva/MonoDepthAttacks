import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks.translation_invariance import GaussianSmooth
import numpy as np
from utils import criteria


class PGD(nn.Module):
    """
    PGD from the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    https://arxiv.org/pdf/1706.06083.pdf

    # Norms Linf or L2
    # L1, L2 and Huber loss implemented

    Arguments:
        model: model to attack.
        eps: maximum perturbation
        decay: momentum factor
        steps: number of iterations

    """

    def __init__(self, model, device, loss, norm, eps, alpha, iters, mean=0.5, std=0.5, TI=False, k_=0, test=None):
        super(PGD, self).__init__()
        assert(2 <= eps <= 20)
        assert(norm in [2, 'inf', np.inf])
        self.eps = (eps / 255.0) / std
        self.alpha = (alpha / 255.0) / std
        self.norm = norm
        self.iterations = iters
        if loss == 'l1':
            self.loss = criteria.MaskedL1Loss()
        elif loss == 'l2':
            self.loss = criteria.MaskedMSELoss()
        elif loss == 'berhu':
            self.loss = criteria.berHuLoss()
        else:
            assert (False, '{} loss not supported'.format(loss))
        self.model = model
        self.device = device
        self.lower_lim = (0.0 - mean) / std
        self.upper_lim = (1.0 - mean) / std
        self.TI = TI
        self.test = test
        if self.TI:
            k = k_
            w = 2*k + 1
            sig = k / np.sqrt(3)
            self.smoothing = GaussianSmooth(w, sig)

    def forward(self, images, labels):
        adv = images.clone().detach().requires_grad_(True).to(self.device)

        for i in range(self.iterations):
            _adv = adv.clone().detach().requires_grad_(True)
            # outputs = self.model(_adv)
            if self.test == 'dpt':
                outputs = self.model(_adv)
                outputs = torch.unsqueeze(outputs, 1)
            elif self.test == 'adabins':
                _, outputs = self.model(_adv)
                labels = F.interpolate(
                    labels, size=(114, 456), mode='bilinear')
            else:
                outputs = self.model(_adv)

            self.model.zero_grad()
            cost = self.loss(outputs, labels)
            cost.backward()
            grad = _adv.grad

            if self.TI:
                grad = self.smoothing(grad)

            if self.norm in ["inf", np.inf]:
                grad = grad.sign()

            elif self.norm == 2:
                ind = tuple(range(1, len(images.shape)))
                grad = grad / \
                    (torch.sqrt(torch.sum(grad * grad, dim=ind, keepdim=True)) + 10e-8)

            assert(images.shape == grad.shape)

            adv = adv + grad * self.alpha

            # project back onto Lp ball
            if self.norm in ["inf", np.inf]:
                adv = torch.max(
                    torch.min(adv, images + self.eps), images - self.eps)

            elif self.norm == 2:
                delta = adv - images

                mask = delta.view(
                    delta.shape[0], -1).norm(self.norm, dim=1) <= self.eps

                scaling_factor = delta.view(
                    delta.shape[0], -1).norm(self.norm, dim=1)
                scaling_factor[mask] = self.eps

                delta *= self.eps / scaling_factor.view(-1, 1, 1, 1)

                adv = images + delta

            adv = adv.clamp(self.lower_lim, self.upper_lim)

        return adv.detach()
