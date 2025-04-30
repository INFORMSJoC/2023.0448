#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from ..models.bert import BertEncoder,BertClf, PruneBertClf
from ..models.image import ImageEncoder,ImageClf
from ..models.ImageEncoder import ViTClf, torchViTClf
from ..models.TextEncoder import BertTextClf
import torch.nn.functional as F

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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

def ratio_uncertain(out):
    p = evidence_p(out)
    pk = p.topk(2)[0]
    u = pk[:, :, 1] / pk[:, :, 0]
    return u.unsqueeze(-1)

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


def lq2_loss(out, tgt):
    b, k = out.size()
    p = F.softmax(out, dim=-1)
    pt = p.argmax(-1)
    q = torch.zeros_like(pt) + 0.1
    q[pt == tgt] = 1
    # q = p[torch.arange(b), tgt].detach()
    loss = (1 - p[torch.arange(b), tgt] ** q) / q
    return loss.unsqueeze(-1)


def emc_loss(txt_img, txt_img_a, tgt):
    loss = 0
    for i in range(0, len(txt_img)):
        loss += lq2_loss(txt_img[i], tgt)
    loss += lq2_loss(txt_img_a, tgt)
    loss = torch.mean(loss)
    return loss


class WModel():
    def __init__(self, args, logger=None):
        super().__init__()
        # self.model = RandomForestRegressor(n_estimators=args.n_estimators, verbose=1, random_state=42, n_jobs=12)
        self.model = LinearRegression()
        self.trained = False
        self.logger = logger

    def summary(self, data, log=True):
        tgts = data["tgts"]
        outs = data["outs"]
        m, x, y = [], [], []
        min_ = np.ones(len(tgts), dtype=np.bool_)
        max_ = np.zeros(len(tgts), dtype=np.bool_)
        for i in outs.keys():
            acc, data = self.accuracy(outs[i], tgts)
            m.append(acc)
            x.append(outs[i])
            y.append(data)
            min_ = min_ & data
            max_ = max_ | data
        min_acc = min_.sum() / len(tgts)
        max_acc = max_.sum() / len(tgts)

        logger_ = "Acc: "
        for i in range(len(m)):
            logger_ += f"M{i+1}: {m[i]:.4f} "
        logger_ += f"Min: {min_acc:.4f} "
        logger_ += f"Max: {max_acc:.4f} "
        # print(logger_)
        if log:
            self.logger.info(logger_)

        x = np.row_stack(x)
        y = np.hstack(y).astype(np.int_)
        return x, y

    def accuracy(self, x, y):
        data = x.argmax(-1) == y
        acc = data.sum() / len(y)
        return acc, data

    def fit(self, x, y):
        logger_ = "training model."
        self.logger.info(logger_)
        self.model.fit(x, y)
        # self.trained = True

    def wpredict(self, x, y):
        predicted = self.model.predict(x)
        error = abs(predicted - y).sum()/len(x)
        # print(f"W error: {error:.4f}")
        logger_ = f"W error: {error:.4f} "
        return predicted, logger_

    def weights(self, outs, w):
        d = len(outs)
        w = w.reshape(d, -1, 1)
        out = []
        for i in range(d):
            out.append(outs[i])
        out = np.stack(out)
        wout = out*w
        out_ = wout.sum(0)
        return out_

    def predict(self, test, log=True):
        outs, tgts = test["outs"], test["tgts"]
        x, y = self.summary(test, log)
        x.sort(-1)
        w, logger_ = self.wpredict(x, y)
        predicted = self.weights(outs, w)
        acc, data = self.accuracy(predicted, tgts)
        # print(f"Acc: {acc:.4f}")
        logger_ += f"Acc: {acc:.4f}"
        return predicted, logger_

    def train_predict(self, val, test):
        val_x, val_y = self.summary(val)
        val_x.sort(-1)
        self.fit(val_x, val_y)

        predicted, logger_ = self.predict(val)
        logger_ = "Val  " + logger_
        self.logger.info(logger_)

        predicted, logger_ = self.predict(test)
        logger_ = "Test " + logger_
        self.logger.info(logger_)
        return predicted



class EMC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.text_model == "transforms_bert":
            self.txtclf = BertTextClf(args)
        else:
            self.txtclf = BertClf(args)

        if self.args.vision_model == "vit":
            self.imgclf = torchViTClf(args)
        else:
            self.imgclf= ImageClf(args)
        self.wmodel = WModel(args)

    def forward(self, txt, mask, segment, img):#
        txt_out = self.txtclf(txt, mask, segment)
        img_out = self.imgclf(img)
        txt_img_out = torch.cat((txt_out.unsqueeze(0), img_out.unsqueeze(0)), dim=0)

        u = ratio_uncertain(txt_img_out)
        txt_img_out_a = relative_uncertain_weighting2(txt_img_out, u.detach())

        return txt_img_out, txt_img_out_a
