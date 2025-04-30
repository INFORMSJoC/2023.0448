
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Times",
#     "font.sans-serif": ["Times"]})

import numpy as np

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict


def reliability_plot(confs, preds, labels, num_bins=15, save="save.png"):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align='edge', width=0.05, color='pink', label='Expected')
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    plt.savefig(save)
    plt.show()


def bin_strength_plot(confs, preds, labels, num_bins=15, save="save.png"):
    '''
    Method to draw a plot for the number of samples in each confidence bin.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples)) * 100
        y.append(n)
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Percentage samples')
    plt.ylabel('Percentage of samples')
    plt.xlabel('Confidence')
    plt.savefig(save)
    plt.show()


def rel_diagram_sub(accs, confs, ax, M = 10, name = "Reliability Diagram", xname = "", yname="", ece=None):

    acc_conf = np.column_stack([accs,confs])

    # acc_conf.sort(axis=1)
    outputs = acc_conf[:, 0]
    gap = acc_conf[:, 1]


    bin_size = 1/M
    positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)


    # Next add error lines
    #for i in range(M):
    #plt.plot([i/M,1], [0, (M-i)/M], color = "red", alpha=0.5, zorder=1)

    #Bars with outputs
    output_plt = ax.bar(positions, outputs, width = bin_size, edgecolor = "black", color = "royalblue", label="Outputs", zorder = 3, alpha=1)

    # ax.bar(positions, gap, width = bin_size, edgecolor = "black", color = "blue", label="Outputs", zorder = 5, alpha=0.7)

    # Plot gap first, so its below everything
    # gap_plt = ax.bar(positions, gap, width = bin_size, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=2)
    gap_plt = ax.bar(positions, np.abs(outputs - gap), bottom=acc_conf.min(-1), width = bin_size, edgecolor = "red", color = "lightcoral", alpha = 0.5, hatch="/", label="Gap", linewidth=2, zorder=4)

    # ax.bar(positions, outputs, width = bin_size, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=4)


    # Line plot with center line.
    # ax.set_aspect('equal')
    ax.plot([0,1], [0,1], linestyle = "--", color = "black")
    ax.legend(handles = [gap_plt, output_plt])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(name, fontsize=24)
    ax.set_xlabel(xname, fontsize=22, color = "black")
    ax.set_ylabel(yname, fontsize=22, color = "black")
    bbox_props = dict(ec='lightgray', lw=1, fc='white', alpha=1)
    ax.text(0.19, 0.06, f'ECE={100 * ece:.2f}', ha='center', fontsize=20, color='black', bbox=bbox_props)



def reliability_plot2(confs, preds, labels, num_bins=15, save="save.png"):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    accs = np.array(list(map(lambda x:x["bin_acc"], bin_dict.values())))
    confs =  np.array(list(map(lambda x:x["bin_conf"], bin_dict.values())))
    ece = sum(map(lambda x:abs(x["acc"] - x["conf"]), bin_dict.values())) / sum(map(lambda x:x["count"], bin_dict.values()))
    # print(ece.item())

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    rel_diagram_sub(accs, confs, ax, M=num_bins, name="", ece=ece)
    plt.savefig(save, dpi=256, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    import torch
    test = torch.load("test_save_cab.pt")
    reliability_plot2(test["logits"].softmax(-1).max(-1)[0], test["preds"], test["labels"])
    # reliability_plot(test["logits"].softmax(-1).max(-1)[0], test["preds"], test["labels"])
