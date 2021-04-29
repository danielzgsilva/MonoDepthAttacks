__author__ = 'Kesaroid'

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from attacks.translation_invariance import GaussianSmooth
from utils import criteria


class MIFGSM(nn.Module):
    """
    MI-FGSM from the paper 'Boosting Adversarial Attacks with Momentum'
    https://arxiv.org/abs/1710.06081

    # Only Linf implemented
    # L1, L2 and Huber loss implemented
    # Targeted attack implemented

    Arguments:
        model: model to attack.
        eps: maximum perturbation
        decay: momentum factor
        steps: number of iterations

    """

    def __init__(self, model, device, loss, eps=6.0, steps=5, decay=1.0, mean=0.5, std=0.5, alpha=1, TI=False, k_=0, targeted=False, test=None):
        super(MIFGSM, self).__init__()
        self.model = model
        self.eps = (eps / 255.0) / std
        self.steps = steps
        self.decay = decay
        self.device = device
        self.alpha = alpha
        self.TI = TI
        self.test = test

        if loss == 'l1':
            self.loss = criteria.MaskedL1Loss()
        elif loss == 'l2':
            self.loss = criteria.MaskedMSELoss()
        elif loss == 'berhu':
            self.loss = criteria.berHuLoss()
        else:
            assert (False, '{} loss not supported'.format(loss))

        if targeted:
            self._targeted = 1
        else:
            self._targeted = -1

        if self.TI:
            k = k_
            w = 2*k + 1
            sig = k / np.sqrt(3)
            self.smoothing = GaussianSmooth(w, sig)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # labels = self._transform_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            if self.test == 'dpt':
                outputs = self.model(adv_images)
                outputs = torch.unsqueeze(outputs, 1)
            elif self.test == 'adabins':
                _, outputs = self.model(adv_images)
                labels = F.interpolate(
                    labels, size=(114, 456), mode='bilinear')
            else:
                outputs = self.model(adv_images)
            # print(outputs.shape, labels.shape)
            cost = self._targeted * self.loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            if self.TI:
                grad = self.smoothing(grad)

            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)

            adv_images = torch.max(
                torch.min(adv_images, images + self.eps), images - self.eps)

        return adv_images
