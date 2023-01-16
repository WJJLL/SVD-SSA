"""Implementation of evaluate attack result."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--adv_dir', type=str, default='./outputs', help='Output directory with adversarial images.')
opt = parser.parse_args()

batch_size = 10

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pretrainedmodels
def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),)
    return model


def verify(model_name, path):

    model = get_model(model_name, path)
    X = ImageNet(opt.adv_dir, opt.input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images)[0].argmax(1) != (gt + 1)).detach().sum().cpu()
    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))

import numpy as np
import torchvision.models as models
def main():

    # model_names = ['tf_inception_v3','tf_inception_v4','tf_inc_res_v2','tf_resnet_v2_50','tf_resnet_v2_101','tf_resnet_v2_152','tf_ens3_adv_inc_v3','tf_ens4_adv_inc_v3','tf_ens_adv_inc_res_v2']

    # model_names = ['resnet101','vgg19','inceptionv3','resnet152','resnet50','vgg16','densenet121','googlenet','inceptionv4','inceptionresnetv2']
    model_names = ['inceptionv3',  'inceptionv4', 'inceptionresnetv2','resnet152', 'resnet50', 'resnet101',]
    for model_name in model_names:
        if model_name in ['inceptionv3','inceptionv4', 'inceptionresnetv2']:
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

        if model_name =='inceptionv3':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
                                        # models.inception_v3(num_classes=1000,
                                        #                            pretrained='imagenet').eval().cuda())
        elif model_name =='resnet152':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.resnet152(num_classes=1000,
                                                                     pretrained='imagenet').eval().cuda())
                                        # models.resnet152(num_classes=1000,
                                        #                            pretrained='imagenet').eval().cuda())
        elif model_name =='resnet50':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.resnet50(num_classes=1000,
                                                                   pretrained='imagenet').eval().cuda())
        elif model_name =='resnet101':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.resnet101(num_classes=1000,
                                                                  pretrained='imagenet').eval().cuda())
        elif model_name =='vgg16':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.vgg16(num_classes=1000,
                                                                   pretrained='imagenet').eval().cuda())
        elif model_name =='vgg19':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.vgg19(num_classes=1000,
                                                               pretrained='imagenet').eval().cuda())
        elif model_name =='densenet121':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.densenet121(num_classes=1000,
                                                               pretrained='imagenet').eval().cuda())
        elif model_name =='googlenet':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.googlenet(num_classes=1000,
                                                                     pretrained='imagenet').eval().cuda())
        elif model_name =='inceptionv4':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.inceptionv4(num_classes=1000,
                                                                     pretrained='imagenet').eval().cuda())
        elif model_name =='inceptionresnetv2':
            model = torch.nn.Sequential(Normalize(mean, std),
                                        pretrainedmodels.inceptionresnetv2(num_classes=1000,
                                                                     pretrained='imagenet').eval().cuda())

        X = ImageNet(opt.adv_dir, opt.input_csv, T.Compose([T.ToTensor()]))
        data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        sum = 0
        for images, _, gt_cpu in data_loader:
            gt = gt_cpu.cuda()
            images = images.cuda()
            with torch.no_grad():
                sum += (model(images).argmax(1) != (gt)).detach().sum().cpu()
        print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))
        print("===================================================")

    # model_names = ['tf_inception_v4','tf_ens_adv_inc_res_v2','tf_inc_res_v2','tf_adv_inception_v3']
    # models_path = './models/'
    # for model_name in model_names:
    #     verify(model_name, models_path)
    #     print("===================================================")

if __name__ == '__main__':
    main()