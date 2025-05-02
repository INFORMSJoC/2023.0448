import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def evidence_p(out):
    alpha = out_alpha(out)
    s = alpha.sum(-1, keepdim=True)
    p = alpha / s
    # p = F.softmax(out, dim=-1)
    return p

def out_alpha(out):
    evidence = F.softplus(out)
    alpha = evidence + 1
    return alpha

def uncertain_raw(out):
    alpha = out_alpha(out)
    k = alpha.size(-1)
    s = alpha.sum(-1, keepdim=True)
    u = k / s
    return u

def ratio_uncertain(out):
    p = evidence_p(out)
    pk = p.topk(2)[0]
    u = pk[:, :, 1] / pk[:, :, 0]
    return u.unsqueeze(-1)

def energy_ratio_uncertain(out, t=10):
    p = torch.exp(out / t)
    pk = p.topk(2)[0]
    u = pk[:, :, 1] / pk[:, :, 0]
    return u.unsqueeze(-1)

def energy_margin_uncertain(out, t=10):
    p = torch.exp(out / t)
    pk = p.topk(2)[0]
    u = 1 - (pk[:, :, 0] - pk[:, :, 1])
    return u.unsqueeze(-1)

def predict_entropy_uncertain(out):
    k = out.size(-1)
    p = evidence_p(out)
    u = - (p * torch.log2(p) / np.log2(k)).sum(-1, keepdim=True)
    return u

def generalized_entropy_score(out, gamma=0.1):
    p = evidence_p(out)
    u = (p ** gamma * (1 - p) ** gamma ).sum(-1, keepdim=True)
    return u


def least_cofidence_uncertain(out):
    k = out.size(-1)
    p = evidence_p(out)
    u = (1 - p.max(-1, keepdim=True)[0]) * k / (k - 1)
    return u

def margin_uncertain(out):
    p = evidence_p(out)
    pk = p.topk(2)[0]
    u = 1 - (pk[:, :, 0] - pk[:, :, 1])
    return u.unsqueeze(-1)

def energy_uncertain(out):
    u = torch.log(torch.sum(torch.exp(out), dim=-1, keepdim=True)) / 10
    return u

def uenergy_uncertain(out):
    # u = 1 / torch.log(torch.sum(torch.exp(out), dim=-1, keepdim=True)) # / 10
    u = - torch.log(1 / torch.sum(torch.exp(out), dim=-1, keepdim=True)) # / 10
    return u


def uncertain_weighting(evidence, u):
    c = 1 - u
    e = evidence * c
    e = e.sum(0)
    return e

def relative_uncertain_weighting(evidence, u):
    c = 1 - u / u.sum(0,  keepdim=True)
    e = evidence * c
    e = e.sum(0)
    return e

def relative_uncertain_weighting2(out, u):
    w = 1 - u
    return relative_weighting(out, w)

def weighting(out, w):
    o = out * w
    o = o.sum(0)
    return o

def relative_weighting(out, w):
    w = w / w.sum(0,  keepdim=True)
    return weighting(out, w)


def ratio_fusion(outs):
    u = ratio_uncertain(outs)
    out = relative_uncertain_weighting2(outs, u)
    return out


def accuracy(correct):
    acc = correct.sum(-1) / correct.size(-1)
    # acc = torch.tensor(float(f"{acc:.4f}"))
    return acc


def metrics(outs:torch.Tensor, tgts:torch.Tensor, out:torch.Tensor):
    preds = outs.argmax(-1)
    corrects = preds == tgts

    pred = out.argmax(-1)
    correct = pred == tgts

    results = {}
    MIN, MAX = None, None
    for i in range(corrects.size(0)):
        MIN = corrects[i] if MIN is None else corrects[i] & MIN
        MAX = corrects[i] if MAX is None else corrects[i] | MAX
        results[f"M{i + 1}"] = accuracy(corrects[i])

    results["Min"] = accuracy(MIN)
    results["Max"] = accuracy(MAX)
    results["M"] = accuracy(correct)

    return results


def metric_format(results:dict):
    str_ = " ".join(map(lambda x: f"{x[0]}: {x[1] * 100:.2f}", results.items()))
    return str_


