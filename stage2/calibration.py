import torch
from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

# from utils import set_seed, load_checkpoint, save_checkpoint, collate, copy_fn, AverageMeter, count_parameters, check_dirs, create_logger

from plots import reliability_plot, bin_strength_plot, reliability_plot2

from tqdm import tqdm
import argparse

import numpy as np
import copy


def true_class_probability(out, tgt):
    b, k = out.size()
    p = F.softmax(out, dim=-1)
    return p[torch.arange(b), tgt]

def max_class_probability(out):
    p = F.softmax(out, dim=-1)
    return p.max(-1)[0]

def metric(test_logits, test_labels, logits, labels):
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = _ECELoss()
    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(test_logits, test_labels).item()
    before_temperature_ece = ece_criterion(test_logits, test_labels).item()
    before_temperature_brier = brier(test_logits.softmax(-1), test_labels).item()
    before_temperature_acc = accuracy(test_logits, test_labels).item()
    print('Before temperature - ACC: %.2f, ECE: %.2f, NLL: %.3f, Brier: %.3f'
          % (before_temperature_acc * 100, before_temperature_ece * 100, before_temperature_nll, before_temperature_brier))

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(logits, labels).item()
    after_temperature_ece = ece_criterion(logits, labels).item()
    after_temperature_brier = brier(logits.softmax(-1), labels).item()
    after_temperature_acc = accuracy(logits, labels).item()
    print('After  temperature - ACC: %.2f, ECE: %.2f, NLL: %.3f, Brier: %.3f'
          % (after_temperature_acc * 100, after_temperature_ece * 100, after_temperature_nll, after_temperature_brier))

def brier(probs, labels):
    return torch.mean(torch.sum((probs - F.one_hot(labels, 2).float())**2, dim=-1))

def accuracy(probs, labels):
    probs = probs.argmax(-1)
    return torch.sum(labels == probs) * 1.0 / len(labels)

def calc_mce(logits, labels, bins=15, ctype=0):
    '''
    Maximum Calibration Error
    :param n_bins: how many bins to evaluate
    :return: mce value
    '''
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0 #torch.ones(1)

    if ctype == 0:
        softmax = F.softmax(logits, dim=1)
        confidence, predictions = torch.max(softmax, 1)
    else:
        confidence, predictions = logits
    accuracy = predictions.eq(labels.long())

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) * (confidence <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracy[in_bin].float().mean()
            avg_confidence_in_bin = confidence[in_bin].mean()

            mce = max(abs(avg_confidence_in_bin - accuracy_in_bin), mce)

    return mce


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


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece




class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, ):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits = logits.cuda()
        labels = labels.cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece * 100))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece * 100))

        return self


class ModelWithTemperature2(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, bins=15):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(bins) * 1.5)
        self.bins = torch.linspace(0, 1, bins + 1).unsqueeze(0)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        logits = copy.deepcopy(logits)
        bin_index = self.get_bin_index(logits)
        for i in range(self.bins_num):
            i_index = bin_index == i
            t_i = self.temperature[i]
            logits[i_index] = logits[i_index] / t_i
        return logits

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits = logits.cuda()
        labels = labels.cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece * 100))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: ', self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece * 100))

        return self


def temperature_scale(train_datas, test_datas, val_datas=None, args=None):
    temperature_model = ModelWithTemperature()
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()
    temperature_model.set_temperature(train_logits, train_labels)

    temperature_model.cpu()
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = _ECELoss()
    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(test_logits, test_labels).item()
    before_temperature_ece = ece_criterion(test_logits, test_labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.2f' % (before_temperature_nll, before_temperature_ece * 100))

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(temperature_model.temperature_scale(test_logits), test_labels).item()
    after_temperature_ece = ece_criterion(temperature_model.temperature_scale(test_logits), test_labels).item()
    print('After temperature - NLL: %.3f, ECE: %.2f' % (after_temperature_nll, after_temperature_ece * 100))




# Ji B, Jung H, Yoon J, et al. Bin-wise temperature scaling (BTS): Improvement in confidence calibration performance through simple scaling techniques[C]//2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW). IEEE, 2019: 4190-4196.
def bin_wise_ts(train_datas, test_datas, val_datas=None, args=None):
    temperature_model = ModelWithTemperature()
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()

    ece_ = ECE()
    bin_index = ece_.get_bin_index(train_logits)
    temperatures = []
    for i in range(ece_.bins_num):
        i_index = bin_index == i
        temperature_model.set_temperature(train_logits[i_index], train_labels[i_index])
        temperatures.append(temperature_model.state_dict())

    # temperature_model.set_temperature(train_logits, train_labels)
    logits_list = []
    labels_list = []
    temperature_model.cpu()
    bin_index = ece_.get_bin_index(test_logits)
    for i in range(ece_.bins_num):
        temperature_model.load_state_dict(temperatures[i])
        i_index = bin_index == i
        logits = temperature_model.temperature_scale(test_logits[i_index])
        logits_list.append(logits)
        labels_list.append(test_labels[i_index])

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    metric(test_logits, test_labels, logits, labels)


# Tomani C, Cremers D, Buettner F. Parameterized temperature scaling for boosting the expressive power in post-hoc uncertainty calibration[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022: 555-569.
class PTS_calibrator(nn.Module):
    def __init__(self,
                 nlayers=2,
                 n_nodes=5,
                 length_logits=2,
                 top_k_logits=10):
        super().__init__()
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        self.temperature = nn.Sequential(
            nn.Linear(length_logits, n_nodes),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(n_nodes, n_nodes),
                           nn.ReLU()) for i in range(nlayers)],
            nn.Linear(n_nodes, 1),
        )

    def forward(self, logits):
        logits = logits.clamp(-1e2, 1e2)
        sort_logits = logits.sort(-1)[0]
        temperature = self.temperature(sort_logits)
        temperature = torch.abs(temperature).clamp(1e-12, 1e12)
        p = (logits / temperature).softmax(-1)
        return p

class XYDatasets(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item], item

def param_ts(train_datas, test_datas, val_datas=None, args=None):
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()

    epochs = 1000
    batch_size = 128
    lr = 0.00005

    train_datasets = XYDatasets(train_logits, train_labels)
    train_loader = DataLoader(train_datasets,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,)
    model = PTS_calibrator()
    model.to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0)
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        for step, data in enumerate(train_loader):#
            batch_data = []
            for da in data:
                batch_data.append(da.to(device))
            labels = batch_data[-2]
            preds = model(*batch_data[:-2])
            loss = F.mse_loss(preds, F.one_hot(labels, 2).float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.cpu().eval()
    train_preds = model(train_logits)
    logits = train_preds.log()
    metric(train_logits, train_labels, logits, train_labels)

    test_preds = model(test_logits)
    logits = test_preds.log()
    metric(test_logits, test_labels, logits, test_labels)


# Joy T, Pinto F, Lim S N, et al. Sample-dependent adaptive temperature scaling for improved calibration[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(12): 14919-14926.
class SDATS_calibrator(nn.Module):
    def __init__(self,
                 length_logits=2,):
        super().__init__()

        self.temperature = nn.Sequential(
            nn.Linear(length_logits, 1),
        )

    def forward(self, logits):
        p = logits.softmax(-1)
        temperature = self.temperature(p)
        temperature = torch.abs(temperature).clamp(1e-12, 1e12)
        logits = logits / temperature
        return logits


def sample_adaptive_ts(train_datas, test_datas, val_datas=None, args=None):
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()

    epochs = 1000
    batch_size = 128
    lr = 0.00005

    train_datasets = XYDatasets(train_logits, train_labels)
    train_loader = DataLoader(train_datasets,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,)
    model = SDATS_calibrator()
    model.to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0)
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        for step, data in enumerate(train_loader):#
            batch_data = []
            for da in data:
                batch_data.append(da.to(device))
            labels = batch_data[-2]
            preds = model(*batch_data[:-2])
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.cpu().eval()
    train_preds = model(train_logits)
    logits = train_preds
    metric(train_logits, train_labels, logits, train_labels)

    test_preds = model(test_logits)
    logits = test_preds
    metric(test_logits, test_labels, logits, test_labels)


# Balanya S A, Maroñas J, Ramos D. Adaptive temperature scaling for robust calibration of deep neural networks[J]. Neural Computing and Applications, 2024: 1-23.
class HTS_calibrator(nn.Module):
    def __init__(self,
                 length_logits=1,):
        super().__init__()

        self.temperature = nn.Sequential(
            nn.Linear(length_logits, 1),
        )

    def forward(self, logits):
        p = logits.softmax(-1)
        h = (p * torch.log(p)).sum(-1, keepdim=True) / np.log(p.size(-1))
        temperature = self.temperature(h)
        temperature = F.softplus(temperature)
        logits = logits / temperature
        return logits

def adaptive_ts(train_datas, test_datas, val_datas=None, args=None):
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()

    epochs = 1000
    batch_size = 128
    lr = 1e-4

    train_datasets = XYDatasets(train_logits, train_labels)
    train_loader = DataLoader(train_datasets,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,)
    model = HTS_calibrator()
    model.to(args.device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-7)
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        for step, data in enumerate(train_loader):#
            batch_data = []
            for da in data:
                batch_data.append(da.to(device))
            labels = batch_data[-2]
            preds = model(*batch_data[:-2])
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_loss = F.cross_entropy(model(train_logits.to(device)), train_labels.to(device))
        scheduler.step(val_loss)
        if 1e9 * abs(optimizer.param_groups[0]['lr'] - 1e-7) <= 1:
            break

    model.cpu().eval()
    train_preds = model(train_logits)
    logits = train_preds
    metric(train_logits, train_labels, logits, train_labels)

    test_preds = model(test_logits)
    logits = test_preds
    metric(test_logits, test_labels, logits, test_labels)


# Frenkel L, Goldberger J. Network calibration by temperature scaling based on the predicted confidence[C]//2022 30th European Signal Processing Conference (EUSIPCO). IEEE, 2022: 1586-1590.
class CBT_calibrator(nn.Module):
    def __init__(self,
                 bins=15,):
        super().__init__()
        self.temperatures = [1] * bins
        self.bins = torch.zeros(1, bins + 1)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def forward(self, logits, labels):
        correct = logits.argmax(-1) == labels
        p = max_class_probability(logits)
        p, _ = p.sort(-1)
        correct = correct[_]

        ni = len(p) // self.bins_num
        for i in range(self.bins_num):
            start = i * ni
            end = (i + 1) * ni if i != self.bins_num - 1 else -1
            pi = p[start:end]
            correct_i = correct[start:end]
            self.bins[0, i + 1] = pi[-1]
            a_i = correct_i.float().mean()
            logits_i = logits[_]
            t_i = self.binary_search_t(logits_i, a_i)
            self.temperatures[i] = t_i

    def binary_search_t(self, logits, a, min_t=1e-5, max_t=10, epsilon=1e-5, max_iter=100):
        t = 1
        for i in range(max_iter):
            t = (min_t + max_t) / 2
            c = max_class_probability(logits / t).mean()
            if c > a:
                min_t = t
            else:
                max_t = t
            if abs(c - a) < epsilon:
                break
        return t

    def calibration(self, logits):
        logits = copy.deepcopy(logits)
        bin_index = self.get_bin_index(logits)
        for i in range(self.bins_num):
            i_index = bin_index == i
            t_i = self.temperatures[i]
            logits[i_index] = logits[i_index] / t_i
        return logits

def confidence_ts(train_datas, test_datas, val_datas=None, args=None):
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()

    model = CBT_calibrator()
    model(train_logits, train_labels)
    train_preds = model.calibration(train_logits)
    logits = train_preds
    metric(train_logits, train_labels, logits, train_labels)

    test_preds = model.calibration(test_logits)
    logits = test_preds
    metric(test_logits, test_labels, logits, test_labels)



class CT_calibrator(nn.Module):
    def __init__(self,
                 bins=15,):
        super().__init__()
        self.temperatures = [1] * bins
        self.bins = torch.linspace(0, 1, bins + 1).unsqueeze(0)
        self.bins_num = bins

    def get_bin_index(self, logits):
        p = max_class_probability(logits)
        bins = self.bins.to(p.device)
        p = p.detach().unsqueeze(-1)
        return ((p.gt(bins)) * p.le(bins.roll(-1, dims=-1))).max(-1)[1]

    def forward(self, logits, labels):
        correct = logits.argmax(-1) == labels
        bin_index = self.get_bin_index(logits)

        for i in range(self.bins_num):
            i_index = bin_index == i
            correct_i = correct[i_index]
            a_i = correct_i.float().mean()
            logits_i = logits[i_index]
            t_i = self.binary_search_t(logits_i, a_i)
            self.temperatures[i] = t_i

    def binary_search_t(self, logits, a, min_t=1e-8, max_t=5, epsilon=1e-8, max_iter=100):
        t = 1
        for i in range(max_iter):
            t = (min_t + max_t) / 2
            c = max_class_probability(logits / t).mean()
            if c > a:
                min_t = t
            else:
                max_t = t
            if abs(c - a) < epsilon:
                break
        return t

    def calibration(self, logits):
        logits = copy.deepcopy(logits)
        bin_index = self.get_bin_index(logits)
        for i in range(self.bins_num):
            i_index = bin_index == i
            t_i = self.temperatures[i]
            logits[i_index] = logits[i_index] / t_i
        return logits


def confidence_s(train_datas, test_datas, val_datas=None, args=None):
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()

    times = 6
    bins = 19

    def calibration(train_logits, train_labels, test_logits, test_labels, i=1):
        model = CT_calibrator(bins=bins)
        if i != 0:
            model(train_logits, train_labels)
        train_preds = model.calibration(train_logits)
        train_cab_logits = logits = train_preds
        print(f"Train [{i}]: ")
        metric(train_logits, train_labels, logits, train_labels)


        confs, preds = logits.softmax(-1).max(-1)
        labels = train_labels
        reliability_plot2(confs, preds, labels, save=f"{args.path}/calibration_train_reliability_{i}.png")
        bin_strength_plot(confs, preds, labels, save=f"{args.path}/calibration_train_strength_{i}.png")


        test_preds = model.calibration(test_logits)
        test_cab_logits = logits = test_preds
        print("Test : ")
        metric(test_logits, test_labels, logits, test_labels)
        print()

        confs, preds = logits.softmax(-1).max(-1)
        labels = test_labels
        reliability_plot2(confs, preds, labels, save=f"{args.path}/calibration_test_reliability_{i}.png")
        bin_strength_plot(confs, preds, labels, save=f"{args.path}/calibration_test_strength_{i}.png")

        return train_cab_logits, train_labels, test_cab_logits, test_labels



    for i in range(times):
        calibration_data = calibration(train_logits, train_labels, test_logits, test_labels, i)
        train_logits, train_labels, test_logits, test_labels = calibration_data


class Calibration():
    def __init__(self, times = 6, bins = 19):
        self.times = times
        self.bins = bins
        self.models = []

    def calibration(self, train_X, train_Y, i):
        model = CT_calibrator(bins=self.bins)
        model(train_X, train_Y)
        train_preds = model.calibration(train_X)
        mce = calc_mce(train_preds, train_Y)
        return model, train_preds, mce

    def fit(self, train_X, train_Y):
        models, mces = [], []
        for i in range(self.times):
            model, train_X, mce = self.calibration(train_X, train_Y, i)
            models.append(model)
            mces.append(mces)
        best_i = mces.index(min(mces))
        self.models = models[:best_i + 1]

    def predict(self, test_X):
        for model in self.models:
            test_X = model.calibration(test_X)
        return test_X




class TNet(nn.Module):
    def __init__(self, input=1, hidden=100, output=1):
        super().__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.softplus(out)


class CPS_calibrator(nn.Module):
    def __init__(self, n_nodes=100,):
        super().__init__()
        self.temperature = TNet(1, n_nodes, 1)

    def forward(self, logits):
        p = max_class_probability(logits)
        temperature = self.temperature(p)
        logits = logits / temperature
        return logits


def confidence_pts(train_datas, test_datas, val_datas=None, args=None):
    if val_datas is not None:
        train_datas = val_datas
    train_logits, train_labels = train_datas["logits"], train_datas["labels"].long()
    test_logits, test_labels = test_datas["logits"], test_datas["labels"].long()

    epochs = 1000
    batch_size = 128
    lr = 0.00005

    train_datasets = XYDatasets(train_logits, train_labels)
    train_loader = DataLoader(train_datasets,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,)
    model = CPS_calibrator()
    model.to(args.device)
    loss_fn = _ECELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0)
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        for step, data in enumerate(train_loader):#
            batch_data = []
            for da in data:
                batch_data.append(da.to(device))
            labels = batch_data[-2]
            preds = model(*batch_data[:-2])
            # loss = F.mse_loss(preds, F.one_hot(labels, 2).float())
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # # Next: optimize the temperature w.r.t. NLL
    # optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, max_iter=50)
    #
    # def eval():
    #     optimizer.zero_grad()
    #     loss = loss_fn(model(logits), labels)
    #     loss.backward()
    #     return loss
    # optimizer.step(eval)

    model.cpu().eval()
    train_preds = model(train_logits)
    logits = train_preds.log()
    metric(train_logits, train_labels, logits, train_labels)

    test_preds = model(test_logits)
    logits = test_preds.log()
    metric(test_logits, test_labels, logits, test_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="twitter",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--gpu", default="0", type=str,
                        help="The gpu used.")
    parser.add_argument("--calibration_model", default="temperature_scale", type=str,
                        help="The calibration model.")



    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_seed(args.seed)

    args.path = f"./{args.output_dir}/{args.name}"
    check_dirs(args.path)
    logger = create_logger(f"{args.path}/logfile.log", args)
    args.logger = logger

    set_seed(args.seed)

    train_datas = torch.load(f"{args.path}/train_save.pt")
    training_datas = torch.load(f"{args.path}/training_save.pt")
    # train_datas = {
    #     "logits": torch.vstack((train_datas["logits"], training_datas["logits"])),
    #     "feat": torch.vstack((train_datas["feat"], training_datas["feat"])),
    #     "labels": torch.hstack((train_datas["labels"], training_datas["labels"].squeeze())),
    # }

    dev_datas = torch.load(f"{args.path}/dev_save.pt")
    test_datas = torch.load(f"{args.path}/test_save.pt")

    eval(args.calibration_model)(train_datas, test_datas, args=args)










