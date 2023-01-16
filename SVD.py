"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from attack_methods import DI, gkern
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from dct import *
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss

def cut(model, n, x,topk):
    for name, module in model._modules.items():
        if name in ['avgpool_1a','avgpool']:
            break
        if name == n:
            B, C, H, W = x.size()
            feat = x.view(B, C, H * W)
            u, s, v = torch.linalg.svd(feat, full_matrices=False)
            top_sum = 0
            for ii in range(topk):
                top_sum = top_sum + s[:, ii:ii + 1].unsqueeze(2) * u[:, :, ii:ii + 1].bmm(v[:, ii:ii + 1, :])
            x = top_sum.view(B, C, H, W)
        if name == 'AuxLogits':
            continue
        else:
            x = module(x)
    return feat , x


def inv4_forward(model,layer,x,topk):
    for (name, module) in model.features._modules.items():
        x = module(x)
        if name == layer:
            B, C, H, W = x.size()
            feat = x.view(B, C, H * W)
            u, s, v = torch.linalg.svd(feat, full_matrices=False)
            top_sum = 0
            for ii in range(topk):
                top_sum = top_sum + s[:, ii:ii + 1].unsqueeze(2) * u[:, :, ii:ii + 1].bmm(v[:, ii:ii + 1, :])
                x = top_sum.view(B, C, H, W)
            x = x.view(B, C, H, W)
    return feat , x


def SVD_model(smodel,img,layer,beta=0.5,topk=1):
    img = smodel[0](img)
    model = smodel[1]
    if model.__class__.__name__=='InceptionV4':
        feat, rank_1 = inv4_forward(model, layer, img,topk)
    else:
        feat, rank_1 = cut(model, layer, img,topk)
    Rlogit = model.logits(rank_1)
    logits = beta * Rlogit + (1 - beta) * model(img)
    return feat, logits

def SVD(smodel,img,beta=0.5,topk=1):
    img = smodel[0](img)
    model = smodel[1]
    x = model.features(img)
    B, C, H, W = x.size()
    feat = x.view(B, C, H * W)
    u, s, v = torch.linalg.svd(feat, full_matrices=False)
    x = s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
    x = x.view(B, C, H, W)
    Rlogit = model.logits(x)
    logits = beta * Rlogit + (1 - beta) * model(img)
    return feat, logits










