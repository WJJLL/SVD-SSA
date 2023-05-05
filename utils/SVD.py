"""Implementation of SVD."""

import torch
import torch.nn.functional as F
def _l2normalize(v, eps=1e-10):
    return v / (torch.norm(v,dim=2,keepdim=True) + eps)

#Power Iteration as SVD substitute for accleration
def power_iteration(A, iter=10):
    u = torch.FloatTensor(1, A.size(1)).normal_(0, 1).view(1,1,A.size(1)).repeat(A.size(0),1,1).to(A)
    v = torch.FloatTensor(A.size(2),1).normal_(0, 1).view(1,A.size(2),1).repeat(A.size(0),1,1).to(A)
    for _ in range(iter):
      v = _l2normalize(u.bmm(A)).transpose(1,2)
      u = _l2normalize(A.bmm(v).transpose(1,2))
    sigma = u.bmm(A).bmm(v)
    sub = sigma * u.transpose(1,2).bmm(v.transpose(1,2))
    return sub

def inv3_logit(model,x):
    x = model.Mixed_7a(x)  # 8 x 8 x 1280
    x = model.Mixed_7b(x)  # 8 x 8 x 2048
    x = model.Mixed_7c(x)  # 8 x 8 x 2048

    x = F.avg_pool2d(x, kernel_size=8)  # 1 x 1 x 2048
    x = F.dropout(x, training=False)  # 1 x 1 x 2048
    x = x.view(x.size(0), -1)  # 2048
    x = model.last_linear(x)  # 1000 (num_classes)
    return x

def svd_inv3(model,input):
    x = model.Conv2d_1a_3x3(input)  # 149 x 149 x 32
    x = model.Conv2d_2a_3x3(x)  # 147 x 147 x 32
    x = model.Conv2d_2b_3x3(x)  # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)  # 73 x 73 x 64
    x = model.Conv2d_3b_1x1(x)  # 73 x 73 x 80
    x = model.Conv2d_4a_3x3(x)  # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)  # 35 x 35 x 192
    x = model.Mixed_5b(x)  # 35 x 35 x 256
    x = model.Mixed_5c(x)  # 35 x 35 x 288
    x = model.Mixed_5d(x)  # 35 x 35 x 288
    x = model.Mixed_6a(x)  # 17 x 17 x 768
    x = model.Mixed_6b(x)  # 17 x 17 x 768
    x = model.Mixed_6c(x)  # 17 x 17 x 768
    x = model.Mixed_6d(x)  # 17 x 17 x 768
    x = model.Mixed_6e(x)  # 17 x 17 x 768

    ### using the svd to decompose feature for obtain the Top-1  singular value-related feature ###
    B, C, H, W = x.size()
    feat = x.view(B, C, H * W)
    # u, s, v = torch.linalg.svd(feat, full_matrices=False)
    # x = s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
    ##Power Iteration as SVD substitute for accleration
    x = power_iteration(feat, iter=20)
    x = x.view(B, C, H, W)

    svd_logit = inv3_logit(model,x)

    ori_logit = inv3_logit(model,feat.view(B, C, H, W))

    return (svd_logit+ori_logit)/2












