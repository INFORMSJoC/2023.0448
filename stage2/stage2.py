import torch
from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

from util import set_seed, load_checkpoint, save_checkpoint, collate, copy_fn, AverageMeter, count_parameters, check_dirs, create_logger
from u import ratio_fusion, metrics, metric_format, relative_weighting, ratio_uncertain, weighting, evidence_p
from u import *

from tqdm import tqdm
import argparse

import numpy as np
import copy

from collections import defaultdict

import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# from kan import KAN
# from deepkan import DeepKAN

import matplotlib.pyplot as plt
from plots import reliability_plot, bin_strength_plot, reliability_plot2

from calibration import Calibration

class WModel():
    def __init__(self, model, use_train=True, is_numpy=True):
        self.model = model
        self.use_train = use_train
        self.is_numpy = is_numpy
        self.models = defaultdict(lambda :None)

    def __call__(self, data_i, data_type, noise_i, i, args):
        train_X = data_i["dev"]["outs"]
        tgts = data_i["dev"]["tgts"]
        if self.use_train:
            train_X = torch.hstack((data_i["train"]["outs"], train_X))
            tgts = torch.hstack((data_i["train"]["tgts"], tgts))

        if i == "1":
            args.logger.info(f"Trian data: {train_X.size()}")

        train_Y = pred_correct(train_X, tgts)

        # train_X = F.softplus(train_X)

        train_X = train_X.view(-1, train_X.size(-1))
        train_X = train_X.sort(-1)[0]
        train_Y = train_Y.view(-1)

        if self.models[noise_i + i] is None:
            self.model.fit(train_X, train_Y)
            self.models[noise_i + i] = copy.deepcopy(self.model)
        else:
            self.model = self.models[noise_i + i]

        train_pred_Y = self.model.predict(train_X)
        train_pred_Y = torch.from_numpy(train_pred_Y) if self.is_numpy else train_pred_Y
        train_mse = ((train_pred_Y - train_Y) ** 2).mean() ** 0.5

        test_X = data_i[data_type]["outs"]
        test_tgts = data_i[data_type]["tgts"]
        # test_X = F.softplus(test_X)
        test_Y = pred_correct(test_X, test_tgts).unsqueeze(-1)
        b, n, _ = test_X.size()
        pred_Y = self.model.predict(test_X.view(-1, _).sort(-1)[0])
        pred_Y = torch.from_numpy(pred_Y) if self.is_numpy else pred_Y
        pred_Y = pred_Y.view(b, n, -1)
        outs = relative_weighting(test_X, pred_Y)

        test_mse = ((pred_Y - test_Y) ** 2).mean() ** 0.5

        trian_metric = {
            "Train mse": train_mse,
            "Test mse": test_mse,
        }

        return outs, trian_metric


class CModel():
    def __init__(self, model, use_train=True, is_numpy=False):
        self.model = model
        self.use_train = use_train
        self.is_numpy = is_numpy
        self.models = defaultdict(lambda :None)

    def __call__(self, data_i, data_type, noise_i, i, args):
        train_X = data_i["dev"]["outs"]
        tgts = data_i["dev"]["tgts"]
        if self.use_train:# and noise_i != "0": 
            train_X = torch.hstack((data_i["train"]["outs"], train_X))
            tgts = torch.hstack((data_i["train"]["tgts"], tgts))

        if i == "1":
            args.logger.info(f"Trian data: {train_X.size()}")

        if self.models[noise_i + i] is None:#
            self.model.fit(train_X, tgts)
            self.models[noise_i + i] = copy.deepcopy(self.model)
        else:
            self.model = self.models[noise_i + i]

        train_pred_Y = self.model.predict(train_X)

        train_mse = ((train_pred_Y - pred_correct(train_X, tgts).unsqueeze(-1)) ** 2).mean() ** 0.5

        test_X = data_i[data_type]["outs"]
        test_tgts = data_i[data_type]["tgts"]
        pred_Y = self.model.predict(test_X)
        outs = relative_weighting(test_X, pred_Y)

        test_mse = ((pred_Y - pred_correct(test_X, test_tgts).unsqueeze(-1)) ** 2).mean() ** 0.5

        trian_metric = {
            "Train mse": train_mse,
            "Test mse": test_mse,
        }
        ece = ECE()
        trian_metric.update({f"Train M{m + 1} ECE": ece(train_X[m], tgts) for m in range(len(train_X))})
        trian_metric.update({f"Test M{m + 1} ECE": ece(test_X[m], test_tgts) for m in range(len(test_X))})

        if isinstance(self.model, CalParamW):
            cal_train_X = self.model.cal_predict(train_X)
            cal_test_X = self.model.cal_predict(test_X)
            trian_metric.update({f"cal Train M{m + 1} ECE": ece(cal_train_X[m], tgts) for m in range(len(cal_train_X))})
            trian_metric.update({f"cal Test M{m + 1} ECE": ece(cal_test_X[m], test_tgts) for m in range(len(cal_test_X))})


    # if i == "1":
        #     for m in range(len(train_X)):
        #         reliability_plot2(train_X[m].softmax(-1).max(-1)[0], train_X[m].argmax(-1), tgts, save=f"{args.path}/train{m}_{noise_i}_{i}.png")
        #         reliability_plot2(test_X[m].softmax(-1).max(-1)[0], test_X[m].argmax(-1), test_tgts, save=f"{args.path}/test{m}_{noise_i}_{i}.png")
        #         bin_strength_plot(train_X[m].softmax(-1).max(-1)[0], train_X[m].argmax(-1), tgts, save=f"{args.path}/train{m}_{noise_i}_{i}_strength.png")
        #         bin_strength_plot(test_X[m].softmax(-1).max(-1)[0], test_X[m].argmax(-1), test_tgts, save=f"{args.path}/test{m}_{noise_i}_{i}_strength.png")
        #     plt.close('all')
        return outs, trian_metric


class ECE(nn.Module):
    def __init__(self, bins=15):
        super().__init__()
        self.bins = torch.linspace(0, 1, bins + 1).unsqueeze(0)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def forward(self, logits, labels):
        p = max_class_probability(logits)
        bin_index = self.get_bin_index(logits)
        correct = logits.argmax(-1) == labels
        ece = 0
        for i in range(self.bins_num):
            i_index = bin_index == i
            a_i = correct[i_index].float().mean()
            c_i = p[i_index].mean()
            # c_i = 1 / self.bins_num * (i + 1/2)
            e_i = torch.nan_to_num(c_i - a_i, nan=0.0)
            ece += torch.abs(e_i) * i_index.float().mean()
        return ece




class Calibration_net():
    def __init__(self, args, bins=15):
        self.args = args
        self.bins = torch.linspace(0, 1, bins + 1).unsqueeze(0)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def fit(self, train_X, train_Y):
        M, _, K = train_X.size()
        bin_index = self.get_bin_index(train_X)
        correct = train_X.argmax(-1) == train_Y
        self.bins_acc = (self.bins - 1 / self.bins.size(-1) / 2)[:, 1:].expand(M, -1)
        for m in range(M):
            for i in range(self.bins_num):
                i_index = bin_index[m] == i
                a_i = correct[m, i_index].float().mean()
                self.bins_acc[m, i] = a_i
        self.w = (self.bins_acc / (1 - self.bins_acc)).log() + np.log(K - 1)

    def predict(self, test_X):
        bin_index = self.get_bin_index(test_X)
        w = torch.gather(self.w, 1, bin_index).unsqueeze(-1)
        return w


class ParamW(nn.Module):
    def __init__(self, args, bins=15):
        super().__init__()
        self.args = args
        self.bins = torch.linspace(0, 1, bins + 1).unsqueeze(0)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def fit(self, train_X, train_Y):
        M, _, K = train_X.size()
        self.w = nn.Parameter(torch.ones(M, self.bins_num))
        self.to(self.args.device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.w], lr=0.001, max_iter=100)
        train_X, train_Y = train_X.to(self.args.device), train_Y.to(self.args.device)

        def eval():
            optimizer.zero_grad()
            w = self.predict(train_X)
            outs = relative_weighting(train_X, w)
            loss = nll_criterion(outs, train_Y)
            loss.backward()
            return loss
        optimizer.step(eval)
        self.cpu()


    def predict(self, test_X):
        bin_index = self.get_bin_index(test_X)
        w = torch.gather(self.w, 1, bin_index).unsqueeze(-1)
        return w



class ParamCW(nn.Module):
    def __init__(self, args, bins=15):
        super().__init__()
        self.args = args
        self.bins = torch.linspace(0, 1, bins + 1).unsqueeze(0)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def fit(self, train_X, train_Y):
        M, _, K = train_X.size()
        self.w = nn.Parameter(torch.ones(M, K, self.bins_num))
        self.to(self.args.device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.w], lr=0.001, max_iter=100)
        train_X, train_Y = train_X.to(self.args.device), train_Y.to(self.args.device)

        def eval():
            optimizer.zero_grad()
            w = self.predict(train_X)
            outs = relative_weighting(train_X, w)
            loss = nll_criterion(outs, train_Y)
            loss.backward()
            return loss
        optimizer.step(eval)
        self.cpu()


    def predict(self, test_X):
        bin_index = self.get_bin_index(test_X)
        c = test_X.argmax(-1).unsqueeze(-1).expand(-1, -1, self.w.size(-1))
        w = torch.gather(torch.gather(self.w, 1, c), -1, bin_index.unsqueeze(-1))
        return w


class CalParamW():
    def __init__(self, args, bins=15):
        self.param_model = ParamW(args, bins)
        # self.calibration_model = Calibration(times=6, bins=19)

    def cal_fit(self, train_X, train_Y):
        calibration_models = []
        for m in range(train_X.size(0)):
            calibration_model = Calibration(times=6, bins=19)
            calibration_model.fit(train_X[m], train_Y)
            calibration_models.append(calibration_model)
        self.calibration_models = calibration_models

    def cal_predict(self, test_X):
        cab_test_X = []
        for m in range(test_X.size(0)):
            cab_test_X_m = self.calibration_models[m].predict(test_X[m])
            cab_test_X.append(cab_test_X_m)
        return torch.stack(cab_test_X)

    def fit(self, train_X, train_Y):
        self.cal_fit(train_X, train_Y)
        cal_train_X = self.cal_predict(train_X)
        self.param_model.fit(cal_train_X, train_Y)

    def predict(self, test_X):
        cal_test_X = self.cal_predict(test_X)
        return self.param_model.predict(cal_test_X)






class Cnet():
    def __init__(self, args):
        self.args = args

    def fit(self, train_X, train_Y):
        epochs = self.args.epochs
        batch_size = self.args.batch_size
        lr = self.args.lr

        self.net = Net(
            nlayers=self.args.nlayers,
            n_nodes=self.args.n_nodes,
            length_logits=train_X.size(-1),
            top_k_logits=self.args.top_k_logits,
        )
        # self.net = MNet(
        #     M=train_X.size(0),
        #     nlayers=self.args.nlayers,
        #     n_nodes=self.args.n_nodes,
        #     length_logits=train_X.size(-1),
        #     top_k_logits=self.args.top_k_logits,
        # )
        self.net.to(self.args.device)
        #

        train_datasets = XYDatasets(train_X.transpose(0, 1), train_Y)
        train_loader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr, weight_decay=0)
        for epoch in (range(epochs)):
            optimizer.zero_grad()
            for step, data in enumerate(train_loader):#
                batch_data = []
                for da in data:
                    batch_data.append(da.to(self.args.device))
                labels = batch_data[-2]
                x = batch_data[0].transpose(0, 1)#.sort(-1)[0]
                preds = self.net(x)
                out = relative_weighting(x, preds)
                # loss = F.cross_entropy(out, labels)
                loss = F.mse_loss(preds, pred_correct(x, labels).unsqueeze(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        self.net.eval().cpu()

    def predict(self, test_X):
        test_X = test_X#.sort(-1)[0]
        with torch.no_grad():
            w = self.net(test_X)
        return w


def CNet(args, use_train=True):
    model = Cnet(args)
    return CModel(model, use_train, is_numpy=False)



class Kannet():
    def __init__(self, args):
        self.args = args

    def fit(self, train_X, train_Y):
        epochs = self.args.epochs
        batch_size = self.args.batch_size
        lr = self.args.lr

        # self.net = Net(
        #     nlayers=self.args.nlayers,
        #     n_nodes=self.args.n_nodes,
        #     length_logits=train_X.size(-1),
        #     top_k_logits=self.args.top_k_logits,
        # )
        #= KAN([train_X.size(-1), self.args.n_nodes, 1])

        # self.net = Mlp(
        #     in_features=train_X.size(-1),
        #     hidden_features=self.args.n_nodes,
        #     out_features=1, drop=0.3, act_layer=nn.PReLU,
        #     )
        self.net = MlpN(
            features=[train_X.size(-1), *[self.args.n_nodes] * self.args.nlayers,1], drop=0.3, act_layer=nn.PReLU,
        )
        # import robust_loss_pytorch.general
        # adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        # num_dims = 1, float_dtype=np.float32, device=self.args.device)
        
        self.net.to(self.args.device)
        

        train_datasets = XYDatasets(train_X.transpose(0, 1), train_Y)
        train_loader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,)

        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # best_loss = torch.inf
        # best_model = self.net.state_dict()
        total_losss = []
        for epoch in (range(epochs)):
            optimizer.zero_grad()
            total_loss = 0
            for step, data in enumerate(train_loader):#
                batch_data = []
                for da in data:
                    batch_data.append(da.to(self.args.device))
                labels = batch_data[-2]
                x = batch_data[0].transpose(0, 1)#.sort(-1)[0]
                input_x = x#[:, :, 1:self.args.top_k_logits]#.softmax(-1)
                preds = self.net(input_x)
                # half_correct = pred_correct(x, labels).sum(0) == 1
                # no_correct = pred_correct(x, labels).sum(0) < 2
                # out = relative_weighting(x, preds)
                # out = weighting(x, preds)
                # out_preds = self.net(out)
                # print(out_preds.size())
                # not_correct = out.argmax(-1) != labels
                # loss = F.cross_entropy(out, labels, reduction="none")[half_correct].mean()

                # w1 = preds.squeeze()
                # y1 = pred_correct(x, labels)
                # w2 = w1.roll(-1, -1)
                # y2 = y1.roll(-1, -1)
                # loss += F.margin_ranking_loss(w1, w2, y1 - y2, margin=0)
                # loss += F.margin_ranking_loss(w1[0], w2[1], y1[0] - y2[1])

                # loss = lq2_loss(out, labels)[half_correct].mean()
                loss = F.mse_loss(preds, pred_correct(x, labels).unsqueeze(-1)) #, reduction="none").mean()
                # loss += F.mse_loss(out_preds, pred_correct(out, labels).unsqueeze(-1))
                # m, b, k = x.size()
                # loss = F.mse_loss(preds, true_class_probability(x, labels))
                # loss = ((preds - pred_correct(x, labels).unsqueeze(-1)).abs() ** 2).mean()
                # loss = F.l1_loss(preds, pred_correct(x, labels).unsqueeze(-1))#, reduction="none").squeeze().sum(0).mean()
                # print(loss.size())
                # [half_correct].mean()
                # loss -= torch.cosine_similarity(preds.squeeze(), pred_correct(x, labels), dim=-1).mean()
                # loss = torch.mean(adaptive.lossfun((preds - pred_correct(x, labels).unsqueeze(-1)).view(-1, 1)))
                # loss = (out.max(-1)[0] - out.gather(-1, labels.unsqueeze(-1)).squeeze())[half_correct].mean()#out.topk(2, dim=-1)[0].min(-1)[0] + 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            total_loss /= len(train_loader)
            scheduler.step(total_loss)
            total_losss.append(total_loss)
            # if total_loss < best_loss:
                # best_model = copy.deepcopy(self.net.state_dict())
        self.net.eval().cpu()
        # self.net.load_state_dict(best_model)
        plt.plot(total_losss)
        plt.savefig("loss.png")

    def predict(self, test_X):
        test_X = test_X#.sort(-1)[0]
        with torch.no_grad():
            w = self.net(test_X)
        return w


def KNet(args, use_train=True):
    model = Kannet(args)
    return CModel(model, use_train, is_numpy=False)


def CalNet(args, use_train=False):
    model = Calibration_net(args)
    return CModel(model, use_train, is_numpy=False)


def PWCNet(args, use_train=True):
    model = ParamCW(args)
    return CModel(model, use_train, is_numpy=False)


def CalWNet(args, use_train=True):
    model = ParamW(args)
    return CModel(model, use_train, is_numpy=False)

def CalPWNet(args, use_train=False):
    model = CalParamW(args)
    return CModel(model, use_train, is_numpy=False)



def Ratio(args, use_train=False):
    class ratio():
        def __init__(self, args):
            self.args = args
        def fit(self, train_X, train_Y):
            pass
        def predict(self, test_X):
            w = 1 - ratio_uncertain(test_X)
            # w = 1 - energy_ratio_uncertain(test_X, t=1)
            # w = 1 - energy_margin_uncertain(test_X)
            return w

    model = ratio(args)
    return CModel(model, use_train, is_numpy=False)


def Linear(args, use_train=True):
    model = LinearRegression()
    return WModel(model, use_train)


def svr(args, use_train=True):
    # from joblib import Parallel, delayed
    # class SVR_pall

    model = SVR(kernel="rbf", degree=3, gamma="auto", coef0=0.0, 
          tol=0.001, C=1.0, epsilon=0.1, shrinking=True, 
          cache_size=200, verbose=False, max_iter=-1)
    return WModel(model, use_train)




def rf(args, use_train=True):
    model = RandomForestRegressor(n_jobs=-1, max_depth=50, max_leaf_nodes=50)
    return WModel(model, use_train)



class Net(nn.Module):
    def __init__(self,
                 nlayers=2,
                 n_nodes=5,
                 length_logits=2,
                 top_k_logits=200):
        super().__init__()
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        self.net = nn.Sequential(
            nn.Linear(min(length_logits, top_k_logits), n_nodes),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(n_nodes, n_nodes),
                            nn.ReLU()) for i in range(nlayers)],
            nn.Linear(n_nodes, 1),
            # nn.Sigmoid(),
        )

    def forward(self, logits):
        if len(logits.size()) == 2:
            logits = logits[:, -self.top_k_logits:]
        elif len(logits.size()) == 3:
            logits = logits[:, :, -self.top_k_logits:]
        w = self.net(logits)
        return w

class Mlp(nn.Module):
    """
    MLP as used in networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.PReLU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpN(nn.Module):
    """
    MLP as used in networks
    """
    def __init__(self, features: list, act_layer=nn.PReLU, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(features[i], features[i + 1]),
                act_layer(),
                nn.Dropout(drop),
            )
            for i in range(len(features) - 2)
        ])
        self.fc = nn.Linear(features[-2], features[-1])
        self.act = nn.Softplus()


    def forward(self, x):
        # x = F.tanh(x)
        x = self.act(x)
        # x = evidence_p(x)
        # k = 10
        # topk_values, topk_indices = torch.topk(x, k, dim=-1)
        # x = torch.zeros_like(x)
        # x.scatter_(-1, topk_indices, topk_values)
        x = self.net(x)
        x = self.fc(x)
        # x = F.sigmoid(x)
        # x = F.tanh(x)
        # x = F.relu(x)
        return x



class MNet(nn.Module):
    def __init__(self,
                 M=2,
                 nlayers=2,
                 n_nodes=5,
                 length_logits=2,
                 top_k_logits=200):
        super().__init__()
        self.M = M
        self.models = nn.ModuleList([Net(nlayers=nlayers, n_nodes=n_nodes, length_logits=length_logits, top_k_logits=top_k_logits)
                                     for i in range(M)])

    def forward(self, logits):
        outs = []
        for i in range(self.M):
            out = self.models[i](logits[i])
            outs.append(out)
        outs = torch.stack(outs)
        return outs


class XYDatasets(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        # random_mask = torch.bernoulli(torch.full(x.size(), 0.7))
        # random_mask[torch.arange(x.size(0)), x.argmax(-1)] = 1
        # x = x * random_mask
        # print(x.size())
        # p = torch.rand(1).item()
        # if p > 0.1:
        #     y_i = torch.where(self.y == y)[0]
        #     mix_i = y_i[torch.randint(0, len(y_i), (1,))].item()
        #     # mix_i = torch.randint(0, len(self.x), (1,)).item()
        #     # if y == self.y[mix_i]:
        #     x = (x + self.x[mix_i]) / 2
        return x, y, item


class Wnet():
    def __init__(self, args):
        self.args = args

    def fit(self, train_X, train_Y):
        epochs = self.args.epochs
        batch_size = self.args.batch_size
        lr = self.args.lr

        self.net = Net(
            nlayers=self.args.nlayers,
            n_nodes=self.args.n_nodes,
            length_logits=train_X.size(-1),
            top_k_logits=self.args.top_k_logits,
        )
        self.net.to(self.args.device)
        #

        train_datasets = XYDatasets(train_X, train_Y)
        train_loader = DataLoader(train_datasets,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True,)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr, weight_decay=0)
        for epoch in (range(epochs)):
            optimizer.zero_grad()
            for step, data in enumerate(train_loader):#
                batch_data = []
                for da in data:
                    batch_data.append(da.to(self.args.device))
                labels = batch_data[-2].unsqueeze(-1)
                preds = self.net(*batch_data[:-2])
                loss = F.mse_loss(preds, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        self.net.eval().cpu()

    def predict(self, test_X):
        w = self.net(test_X)
        return w


def WNet(args, use_train=True):
    model = Wnet(args)
    return WModel(model, use_train, is_numpy=False)


def true_class_probability(out, tgt):
    p = F.softmax(out, dim=-1)
    b = tgt.view(1, -1, 1).expand(2, -1, 1) if len(out.size()) == 3 else tgt.view(-1, 1)
    tcp = p.gather(-1, b)
    return tcp

def max_class_probability(out):
    p = F.softmax(out, dim=-1)
    return p.max(-1)[0]

def pred_correct(preds, labels):
    return (preds.argmax(-1) == labels).float()


def ratio_fusion_model(data_i, data_type, *args, **kargs):
    return ratio_fusion(data_i[data_type]["outs"])


def read_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def result_report(fusion, all_data, args, data_type="test", noise_type=None):
    all_results = {}
    format_results = {}
    args.logger.info(f"{args.path} {data_type}")

    for noise in args.noise.split(",") if noise_type is None else noise_type.split(","):
        results = []
        for i in tqdm(args.data_nums.split(",")):
            data_i = all_data[f"noise_{noise}"][int(i) - 1]
            data = data_i[data_type]
            out = fusion if isinstance(fusion, torch.Tensor) else fusion(data_i, data_type, noise, i, args)
            trian_metric = {}
            if isinstance(out, tuple):
                out, trian_metric = out
            metric_result = metrics(out=out, **data)
            metric_result.update(trian_metric)
            results.append(metric_result)
        # args.logger.info(f"Trian data: ", train_size)
        results = collate(results)

        format_result = {}
        for key in results.keys():
            scale = 100 if "M" in key else 1
            format_result[key] = f"{results[key].mean() * scale:.2f}±{results[key].std(unbiased=False) * scale:.2f}"
        str_ = " ".join(map(lambda x: f"{x[0]}: {x[1]}", format_result.items()))
        format_results[f"noise_{noise}"] = str_
        # args.logger.info(f"noise_{noise} {str_}")

        all_results[f"noise_{noise}"] = results

    args.logger.info(
        f"{args.path} {data_type}\n" +
        "\n".join(
            f"{k:<8}: {v}" for k, v in format_results.items()
        )
    )

    return format_results, all_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="MVSA_Single_emc",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--output_dir", default="emc_data", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--dataset", default="MVSA", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--data_nums', type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help="seed data for calculate")
    parser.add_argument('--noise', type=str, default="0,5,10",
                        help="noise strength")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--gpu", default="0", type=str,
                        help="The gpu used.")
    parser.add_argument("--model", default="Linear", type=str,
                        help="The stage2 model.")

    parser.add_argument('--nlayers', type=int, default=0, help="")
    parser.add_argument('--n_nodes', type=int, default=100, help="")
    parser.add_argument('--top_k_logits', type=int, default=200, help="")
    parser.add_argument('--epochs', type=int, default=100, help="")
    parser.add_argument('--batch_size', type=int, default=512, help="")
    parser.add_argument('--lr', type=float, default=5e-5, help="")


    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device


    args.path = f"./{args.output_dir}/{args.dataset}"
    check_dirs(args.path)
    logger = create_logger(f"{args.path}/logfile.log", args)
    args.logger = logger

    set_seed(args.seed)

    all_data = defaultdict(list)
    transform_tensor = lambda outs: torch.stack(list(map(torch.from_numpy, outs)))
    process_data = lambda data: {"tgts": torch.from_numpy(data["tgts"]), "outs": transform_tensor(data["outs"].values())}

    for noise in args.noise.split(","):
        datas = []
        for i in args.data_nums.split(","):
            train_data = read_pkl(f"{args.path}/{args.name}_{i}_train_{noise}.0.pkl")
            dev_data = read_pkl(f"{args.path}/{args.name}_{i}_val_{noise}.0.pkl")
            test_data = read_pkl(f"{args.path}/{args.name}_{i}_test_{noise}.0.pkl")
            data = {
                "train": process_data(train_data),
                "dev": process_data(dev_data),
                "test": process_data(test_data),
            }
            datas.append(data)
        all_data[f"noise_{noise}"] = datas

    noise_type = None# "0,5,10"

    # model = ratio_fusion_model
    # result_report(model, all_data, args, data_type="train", noise_type=noise_type)
    # result_report(model, all_data, args, data_type="dev", noise_type=noise_type)
    # _, _ = result_report(model, all_data, args, noise_type=noise_type)
    # print("\n" * 3)

    # model = Linear(use_train=True)
    # model = WNet(args)
    model = eval(args.model)(args)
    result_report(model, all_data, args, data_type="train", noise_type=noise_type)
    result_report(model, all_data, args, data_type="dev", noise_type=noise_type)
    format_results, all_results = result_report(model, all_data, args, noise_type=noise_type)


    # from tsne import show_tsne
    a = all_data["noise_0"][0]["test"]["outs"]
    b = all_data["noise_0"][0]["test"]["tgts"]
    c = a.argmax(-1) == b
    # show_tsne(a, c)
    # args.logger.info(all_results)
    # args.logger.info(
    #     "\n".join(
    #         f"{k:<8}: {v}" for k, v in format_results.items()
    #     )
    # )

