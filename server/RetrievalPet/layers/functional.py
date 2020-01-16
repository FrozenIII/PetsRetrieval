import math
import pdb

import torch
import torch.nn.functional as F
import numpy as np

# --------------------------------------
# pooling
# --------------------------------------

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative


def spoc(x):

    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative

def hnip(x0, x90, x180, x270):
    eps = 1e-6
    nt = 2
    ns = 1
    # nr=max
    Scale = [0.75, 0.5, 0.25, 1]
    channle_size = x0.shape[1]
    fina_vec = torch.empty(4, len(Scale), channle_size)
    for idx, x in enumerate([x0, x90, x180, x270]):
        W = x.shape[-2]
        H = x.shape[-1]
        for i in range(len(Scale)):
            x_s = x[:, :, :int(W * Scale[i]), :int(H * Scale[i])]
            x_s = F.avg_pool2d(x_s.clamp(min=eps).pow(nt), (x_s.size(-2), x_s.size(-1))).pow(1. / nt)
            fina_vec[idx, i:] = l2n(x_s).squeeze(-1).squeeze(-1)
    # shape(2048,R,S)
    fina_vec = fina_vec.transpose(2, 0).transpose(2, 1)

    #s变换
    fina_vec = F.avg_pool1d(fina_vec.clamp(min=eps).pow(ns), fina_vec.size(-1)).pow(1. / ns)
    fina_vec = fina_vec.squeeze(-1)

    #r 变换
    fina_vec = fina_vec.max(1)[0]

    return fina_vec.unsqueeze(0)

def hnip_rmac(x0, x90, x180, x270):
    eps = 1e-6
    nt = 2
    ns = 1
    # nr=max
    Scale = [0.75, 0.5, 0.25, 1]
    channle_size = x0.shape[1]
    fina_vec = torch.empty(4, channle_size)
    for idx, x in enumerate([x0, x90, x180, x270]):
        fina_vec[idx,:] = l2n(rmac(x, nt=2)).squeeze(-1).squeeze(-1)
    # shape(2048,R,S)
    fina_vec = fina_vec.transpose(1,0)
    #r 变换
    fina_vec = fina_vec.max(1)[0]

    return fina_vec.unsqueeze(0)
def hnip_lyh(x0, x90, x180, x270):
    eps = 1e-6
    nt = 2
    ns = 1
    # nr=max
    Scale = [2,3,4,5,6,7,8]
    channle_size = x0.shape[1]
    fina_vec = torch.empty(4, channle_size)
    for idx, x in enumerate([x0, x90, x180, x270]):
        fina_vec[idx,:] = rmac_lyh(x, nt=2).squeeze(-1).squeeze(-1)
    # shape(2048,R,S)
    fina_vec = fina_vec.transpose(1,0)
    #r 变换
    fina_vec = fina_vec.max(1)[0]

    return fina_vec.unsqueeze(0)

def rmac(x, L=3, eps=1e-6, nt=-1):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1
    if nt == -1:
        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    else:
        v = F.lp_pool2d(x, nt, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                if nt == -1:
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                else:
                    vt = F.lp_pool2d(R, nt, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def rmac(x, L=3, eps=1e-6, nt=-1):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1
    if nt == -1:
        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    else:
        v = F.lp_pool2d(x, nt, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                if nt == -1:
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                else:
                    vt = F.lp_pool2d(R, nt, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


def roipool(x, rpool, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    _, idx = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    vecs.append(rpool(x).unsqueeze(1))

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b).int() - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b).int() - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                vecs.append(rpool(x.narrow(2, i_, wl).narrow(3, j_, wl)).unsqueeze(1))

    return torch.cat(vecs, dim=1)


# --------------------------------------
# normalization
# --------------------------------------

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


def powerlaw(x, eps=1e-6):
    x = x + self.eps
    return x.abs().sqrt().mul(x.sign())


# --------------------------------------
# loss
# --------------------------------------

def contrastive_loss(x, label, margin=0.7, eps=1e-6):
    # x is D x N
    dim = x.size(0)  # D
    nq = torch.sum(label.data == -1)  # number of tuples
    S = x.size(1) // nq  # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1, 0).repeat(1, S - 1).view((S - 1) * nq, dim).permute(1, 0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label != -1]

    dif = x1 - x2
    D = torch.pow(dif + eps, 2).sum(dim=0).sqrt()

    y = 0.5 * lbl * torch.pow(D, 2) + 0.5 * (1 - lbl) * torch.pow(torch.clamp(margin - D, min=0), 2)
    y = torch.sum(y)
    return y


def beida_loss_couple(x, label, margin=0.7, eps=1e-6, margin2=0.8, beta=0.75):
    # x is D x N
    dim = x.size(0)  # D
    nq = 1  # number of tuples
    # S = x.size(1) // nq # number of images per tuple including query: 1+1+n
    idx0 = [i for i in range(len(label)) if label.data[i] == 0]
    x_0 = x[:, idx0]
    center_0 = torch.sum(x_0, dim=1).repeat(1, len(idx0)).view(len(idx0) * nq, dim).permute(1, 0)/x_0.shape[1]
    idx1 = [i for i in range(len(label)) if label.data[i] == 1]
    x_1 = x[:, idx1]
    ##寻找x_1中离center_0最近的点 xx1是1类别离0类别中心最近的点再拓展形成
    dif10 = x_1 - center_0
    min_10, index10 = torch.min(torch.sum(dif10, dim=0), dim=0)
    xx1 = x_1.permute(1, 0)[index10].repeat(1, len(idx0)).view(len(idx0) * nq, dim).permute(1, 0)

    # 开始计算
    D00 = torch.pow((x_0 - center_0) + eps, 2).sum(dim=0).sqrt()
    D01 = torch.pow((xx1 - center_0) + eps, 2).sum(dim=0).sqrt()

    L1 = 0.5 * (torch.pow(torch.clamp(margin + D00 - D01, min=0), 2))
                # + beta * torch.pow(torch.clamp(margin2 + Dxx - Dcenter, min=0), 2))
    # print("_________________________________________")
    # print(torch.sum(torch.pow(torch.clamp(margin + D00 - D02 - D01, min=0), 2)))
    # print(torch.sum(torch.pow(torch.clamp(margin + D10 - D12 - D11, min=0), 2)))
    # print(torch.sum(Dxx - Dcenter))
    # print(torch.sum(beta * torch.pow(torch.clamp(margin2 + Dxx - Dcenter, min=0), 2)))
    L1 = torch.sum(L1)

    return L1

def contrastive_loss_couple(x, label, margin=0.7, eps=1e-6, margin2=0.8, beta=0.75):
    # x is D x N
    dim = x.size(0)  # D
    nq = 1  # number of tuples
    # S = x.size(1) // nq # number of images per tuple including query: 1+1+n
    idx0 = [i for i in range(len(label)) if label.data[i] == 0]
    x_0 = x[:, idx0]
    center_0 = torch.sum(x_0, dim=1).repeat(1, len(idx0)).view(len(idx0) * nq, dim).permute(1, 0)/x_0.shape[1]
    idx1 = [i for i in range(len(label)) if label.data[i] == 1]
    x_1 = x[:, idx1]
    center_1 = torch.sum(x_1, dim=1).repeat(1, len(idx1)).view(len(idx1) * nq, dim).permute(1, 0)/x_1.shape[1]
    ##寻找x_1中离center_0最近的点 xx1是1类别离0类别中心最近的点再拓展形成
    dif10 = x_1 - center_0
    min_10, index10 = torch.min(torch.sum(dif10, dim=0), dim=0)
    xx1 = x_1.permute(1, 0)[index10].repeat(1, len(idx0)).view(len(idx0) * nq, dim).permute(1, 0)

    dif01 = x_0 - center_1
    min_01, index01 = torch.min(torch.sum(dif01, dim=0), dim=0)
    xx0 = x_0.permute(1, 0)[index01].repeat(1, len(idx1)).view(len(idx1) * nq, dim).permute(1, 0)

    # 开始计算
    D00 = torch.pow((x_0 - center_0) + eps, 2).sum(dim=0).sqrt()
    D01 = torch.pow((xx1 - center_0) + eps, 2).sum(dim=0).sqrt()
    D02  = torch.pow((x_0 - center_1) + eps, 2).sum(dim=0).sqrt()

    D10 = torch.pow((x_1 - center_1) + eps, 2).sum(dim=0).sqrt()
    D11 = torch.pow((xx0 - center_1) + eps, 2).sum(dim=0).sqrt()
    D12 = torch.pow((x_1 - center_0) + eps, 2).sum(dim=0).sqrt()

    Dxx = torch.pow((xx1 - xx0) + eps, 2).sum(dim=0).sqrt()
    Dcenter = torch.pow((center_0 - center_1) + eps, 2).permute(1, 0)[0].sum(dim=0).sqrt()

    L1 = 0.5 * (torch.pow(torch.clamp(margin + D00 - D01 - D02, min=0), 2)
                + torch.pow(torch.clamp(margin + D10 - D11 - D12, min=0), 2))
                # + beta * torch.pow(torch.clamp(margin2 + Dxx - Dcenter, min=0), 2))
    # print("_________________________________________")
    # print(torch.pow(torch.clamp(margin + D00 - D02 - D01, min=0), 2))
    # print(torch.pow(torch.clamp(margin + D10 - D12 - D11, min=0), 2))
    # print(Dxx - Dcenter)
    # print(beta * torch.pow(torch.clamp(margin2 + Dxx - Dcenter, min=0), 2))
    L1 = torch.sum(L1)

    return L1


def baidu_loss(x, label, margin=0.7, eps=1e-6):
    idxa = [i for i in range(len(label)) if label.data[i] == -1]
    x_a = x[:, idxa]
    idxp = [i for i in range(len(label)) if label.data[i] == 1]
    x_p = x[:, idxp]
    idxn = [i for i in range(len(label)) if label.data[i] == 0]
    x_n = x[:, idxn]
    x_c=(x_a+x_p)/2
    Dap = torch.pow((x_a - x_p) + eps, 2).sum(dim=0).sqrt()
    Dnc = torch.pow((x_c - x_n) + eps, 2).sum(dim=0).sqrt()

    L1 = 0.5 * (torch.clamp(torch.pow(Dap, 2)-margin*torch.pow(Dnc, 2), min=0))
    L1 = torch.sum(L1)
    return L1

def triangle_loss(x, label, margin=0.7, eps=1e-6):
    idxa = [i for i in range(len(label)) if label.data[i] == -1]
    x_a = x[:, idxa]
    idxp = [i for i in range(len(label)) if label.data[i] == 1]
    x_p = x[:, idxp]
    idxn = [i for i in range(len(label)) if label.data[i] == 0]
    x_n = x[:, idxn]
    Dap = torch.pow((x_a - x_p) + eps, 2).sum(dim=0).sqrt()
    Dan = torch.pow((x_a - x_n) + eps, 2).sum(dim=0).sqrt()
    Dpn = torch.pow((x_p - x_n) + eps, 2).sum(dim=0).sqrt()

    L1 = 0.5 * (torch.clamp(margin+torch.pow(Dap, 2)-torch.pow(Dan, 2)-torch.pow(Dpn, 2), min=0))
    L1 = torch.sum(L1)
    return L1

def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))

def triplet_loss_npairs(anchor, positive, target, l2_reg=0.02):
    batch_size = anchor.size(0)
    target = target.view(target.size(0), 1)

    target = (target == torch.transpose(target, 0, 1)).float()
    target = target / torch.sum(target, dim=1, keepdim=True).float()

    logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
    loss_ce = cross_entropy(logit, target)


    l2_loss = torch.sum(anchor ** 2) / batch_size + torch.sum(positive ** 2) / batch_size

    loss = loss_ce + l2_reg * l2_loss * 0.25
    return loss
def triplet_loss_npairs_new(embeddings, target, l2_reg=0.02):
    #get the labels of target
    #target shape([2*minibatch]) npairs shape [minibatch,2] n_negatives shape [minibatch,minibatch-1]
    #n is the minibatch
    n_pairs, n_negatives = get_n_pairs(np.array(target))
    if embeddings.is_cuda:
        n_pairs = n_pairs.cuda()
        n_negatives = n_negatives.cuda()

    anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
    positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
    negatives = torch.stack([embeddings[i] for i in n_negatives])  # (n, n-1, embedding_size)
    losses = n_pair_loss(anchors, positives, negatives) + l2_reg * l2_loss(anchors, positives)
    return losses
def angular_loss_npairs_new(embeddings, target, angle_bound=1., l2_reg=0.02):
    n_pairs, n_negatives = get_n_pairs(np.array(target))
    if embeddings.is_cuda:
        n_pairs = n_pairs.cuda()
        n_negatives = n_negatives.cuda()

    anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
    positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
    negatives = torch.stack([embeddings[i] for i in n_negatives])  # (n, n-1, embedding_size)
    losses = angular_loss(anchors, positives, negatives, angle_bound) + l2_reg * l2_loss(anchors, positives)
    return losses
def triangle_loss_npairs_new(embeddings, target, margin=1.7, l2_reg=0.02):
    n_pairs, n_negatives = get_n_pairs(np.array(target))
    if embeddings.is_cuda:
        n_pairs = n_pairs.cuda()
        n_negatives = n_negatives.cuda()

    anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
    positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
    negatives = torch.stack([embeddings[i] for i in n_negatives])  # (n, n-1, embedding_size)
    losses = n_pair_triangle_loss(anchors, positives, negatives, margin) + l2_reg * l2_loss(anchors, positives)
    return losses
def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1 + x))
        return loss

def n_pair_triangle_loss(anchors, positives, negatives, margin=1.7):
    anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
    positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

    x = margin - 2.0 + 2. * (torch.matmul(anchors, negatives.transpose(1, 2)) \
                            + torch.matmul(positives, negatives.transpose(1, 2)) \
                            - torch.matmul(anchors, positives.transpose(1, 2)))
    # Preventing overflow
    with torch.no_grad():
        t = torch.max(x, dim=2)[0]

    x = torch.exp(x - t.unsqueeze(dim=1))
    x = torch.log(torch.exp(-t) + torch.sum(x, 2))
    loss = torch.mean(t + x)

    return loss

def angular_loss(anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0]

        x = torch.exp(x - t.unsqueeze(dim=1))
        x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        loss = torch.mean(t + x)

        return loss
def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        n_pairs = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = label_indices[0], label_indices[1]
            n_pairs.append([anchor, positive])
        n_pairs = np.array(n_pairs)
        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i + 1:, 1]])
            n_negatives.append(negative)
        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)


def l2_loss(anchors, positives):
    """
    Calculates L2 norm regularization loss
    :param anchors: A torch.Tensor, (n, embedding_size)
    :param positives: A torch.Tensor, (n, embedding_size)
    :return: A scalar
    """
    return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]

