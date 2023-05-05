"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse
import sys
import pretrainedmodels

from utils.loader import ImageNet
from utils.Normalize import Normalize
from utils.attack_methods import DI,gkern
from utils.SVD import svd_inv3

T_kernel = gkern(7, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./adv_img', help='Source Models.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")

opt = parser.parse_args()
torch.backends.cudnn.benchmark = True

transforms = T.Compose([T.CenterCrop(opt.image_width), T.ToTensor()])

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def graph(model ,x, gt):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    decay = 1.0
    m = 5
    grad = 0

    momentum = torch.zeros_like(x).detach().cuda()
    adv_images = x.clone().detach()

    for i in range(num_iter):
        adv_images.requires_grad = True
        nes_image = adv_images + decay * alpha * momentum
        # Calculate sum the gradients over the scale copies of the input image
        adv_grad = torch.zeros_like(x).detach().cuda()
        for i in torch.arange(m):
            nes_images = nes_image / torch.pow(2, i)
            output = svd_inv3(model[1], model[0](nes_images))
            ## DI-FGSM
            # output = svd_inv3(model[1], model[0](DI(nes_images)))

            cost = F.cross_entropy(output, gt)

            adv_grad += torch.autograd.grad(cost, adv_images,
                                            retain_graph=False, create_graph=False)[0]
        adv_grad = adv_grad / m

        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        # adv_grad = F.conv2d(adv_grad, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # Update adversarial images
        grad = decay * momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True)
        momentum = grad
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - x, min=-eps, max=eps)
        adv_images = torch.clamp(x + delta, min=0, max=1).detach()

    return adv_images


def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir +'/'+ name)


def main():
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    model = torch.nn.Sequential(Normalize(mean, std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        adv_img = graph(model , images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)


if __name__ == '__main__':
    main()

