#%%
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import logging, pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.style.use('seaborn')
import seaborn as sns

def print_text(*txt):
    if logging:
        logging.info(*txt)
    else:
        print(*txt)
    
def load_dataset(dataset):
    dataset = dataset.lower()
    # ----------------------------------------------- #
    # Dataset parameter setting
    # ----------------------------------------------- #
    if dataset == "mhealth":
        filename = "../datasets/mHealth/mHealth_subject1.pkl"
    elif dataset == "pamap2":
        filename = "../datasets/PAMAP2/subject108.pkl"
    elif dataset == "skoda":
        filename = "../datasets/skoda/skoda.pkl"
    elif dataset == "skodaold":
        filename = "../datasets/skoda/skodaold.pkl"
    elif dataset == "skodanew":
        filename = "../datasets/skoda/skodanew.pkl"
    elif dataset == "opportunity":
        filename = "../datasets/opportunity/opportunity.pkl"
    elif dataset == "opportunity_loco":
        filename = "../datasets/opportunity/opportunity_loco.pkl"
    elif dataset == "waveglove":
        filename = "../datasets/waveglove/waveglove_multi.pkl"
    else:
        print("File Not Found")
        import sys; sys.exit()
    # ----------------------------------------------- #
    # Dataset load
    # ----------------------------------------------- #

    data_X, data_y = pickle.load(open(filename,"rb"))
    data_y = relabel(data_y)

    if "skoda" in dataset:
        data_X, data_y = data_X[::5], data_y[::5]
        
    n_ts, n_dim = data_X.shape
    n_clusters = len(np.unique(data_y))

    print_text("{}: Data size: {} {}".format(dataset, data_X.shape, data_y.shape))
    return data_X, data_y

def create_assigning_dataset(inputs, labels, window = 1024, shift = 256, stride = 1):
    '''
    TODO stride
    '''
    n_ts, d = inputs.shape

    X_train = []
    y_train = []
    if labels is not None:
        for t in range(0, n_ts, shift):
            if t + window >= n_ts:
                X_window = inputs[-window::stride,:]
                y_window = labels[-window::stride]
                X_train.append(X_window)
                y_train.append(y_window)
                break
            X_window = inputs[t:(t+window):stride,:]
            X_train.append(X_window)

            y_window = labels[t:(t+window):stride]
            y_train.append(y_window)
            
        return np.array(X_train), np.array(y_train)
    else:
        for t in range(0, n_ts, shift):
            if t + window >= n_ts:
                X_window = inputs[-window::stride,:]
                X_train.append(X_window)
                break
            X_window = inputs[t:(t+window):stride,:]
            X_train.append(X_window)

        return np.array(X_train)
               

def create_batch(inputs, labels, batch_size = 50, remaining = True):
    n_ts, w, d = inputs.shape
    if labels is None:
        labels = inputs

    X = []
    y = []
    ns_batch = n_ts // batch_size
    remaining = n_ts % batch_size
    for i in range(ns_batch):
        st = i * batch_size
        ed = (i+1) * batch_size
        X.append(inputs[st:ed])
        y.append(labels[st:ed])

    if (remaining != 0) and remaining:
        X.append(inputs[-remaining:])
        y.append(labels[-remaining:])
    return zip(X, y)

def metrics(true, pred):
    l = len(pred) if len(true) > len(pred) else len(true)
    nmi = normalized_mutual_info_score(true[:l], pred[:l])#,average_method="geometric")
    ami = adjusted_mutual_info_score(true[:l], pred[:l])
    ari = adjusted_rand_score(true[:l], pred[:l])
    f1 = metric_f1(true[:l], pred[:l])
    acc = metric_accuracy(true[:l], pred[:l])
    return nmi, ami, ari, f1, acc

def print_metrics(true, pred):
    nmi, ami, ari, f1, acc = metrics(true, pred)
    return "# Scoring: nmi %.3f ami %.3f ari %.3f f1 %.3f acc %.3f" % (nmi, ami, ari, f1, acc)

def metric_f1(true, pred, average = 'macro'):
    counts = []
    # hungarian algorithm
    for i in np.unique(true):
        sub_counts = [0 for _ in range(len(np.unique(true)))]
        label, count = np.unique(pred[true == i], return_counts = True)
        for l, c in zip(label, count):
            sub_counts[int(l)] = -c
        counts.append(sub_counts)
    row_idx, col_idx = linear_sum_assignment(counts)
    matching = {v:int(k) for k,v in zip(row_idx, col_idx)}
    pred = np.array([matching[k] for k in pred])
    # return matching
    return f1_score(true, pred, average = average)

def metric_accuracy(true, pred):
    counts = [] 
    # hungarian algorithm
    for i in np.unique(true):
        sub_counts = [0 for _ in range(len(np.unique(true)))]
        label, count = np.unique(pred[true == i], return_counts = True)
        for l, c in zip(label, count):
            sub_counts[int(l)] = -c
        counts.append(sub_counts)
    row_idx, col_idx = linear_sum_assignment(counts)
    matching = {v:int(k) for k,v in zip(row_idx, col_idx)}
    pred = np.array([matching[k] for k in pred])
    # return matching
    return accuracy_score(true, pred)

def relabel(y):
    c, idx = np.unique(y, return_index = True)
    origin = {k:v for k,v in zip(c, idx)}
    sorting = dict(sorted(origin.items(), key = lambda x : x[1]))

    k = np.array(list(sorting.keys()))
    v = np.array(list(origin.keys()))
    
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks, y)]

def get_segment_boundary(data):
    labels = np.unique(data)
    boundary = {k:[] for k in labels}
    for label in labels:
        a = np.pad(np.array(list(map(int, data == label))), 1)
        sts = np.where((a[1:] - a[:-1]) == 1)[0]
        if sts.size == 0:
            sts = np.array([0])
        eds = np.where((a[1:] - a[:-1]) == -1)[0]
        if eds.size == 0:
            eds = np.array([len(data)])
        for st, ed in zip(sts, eds):
            boundary[label].append([st,ed])
    boundary = dict(sorted(boundary.items(), key = lambda x : x[1][0]))
    boundary = {i:v for i,v in enumerate(boundary.values())}
    return boundary

def plotResults(x, y, pred):
    # Get color maps
    colormaps = []
    cmap = cm.get_cmap('Set3', len(np.unique(y)))  # matplotlib color palette name, n colors
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        colormaps.append(mpl.colors.rgb2hex(rgb))

    fig = plt.figure(figsize = (10, 3))

    # Plot y
    ax = fig.add_subplot(211)
    boundary_y = get_segment_boundary(y)
    for k, v in boundary_y.items():
        for st, ed in v:
            ax.axvspan(st, ed, facecolor = colormaps[int(k)], alpha = 0.8)

    # Plot pred
    ax = fig.add_subplot(212)
    boundary_pred = get_segment_boundary(pred)
    for k, v in boundary_pred.items():
        for st, ed in v:
            ax.axvspan(st, ed, facecolor = colormaps[int(k)], alpha = 0.8)
    fig.tight_layout()
    return fig

#------------------------------------------------#

def create_initial_label(n_ts, n_clusters):
    num = n_ts // n_clusters
    remainder = n_ts % n_clusters
    return np.concatenate([np.zeros(remainder), np.repeat(np.arange(n_clusters), repeats = num)], axis = -1)


def _metric_f1(true, pred, average = 'macro'):
    counts = []
    for i in np.unique(true):
        sub_dict = {k:0 for k,_ in enumerate(range(len(np.unique(true))))}
        label, count = np.unique(pred[true == i], return_counts = True)
        for l, c in zip(label, count):
            sub_dict[l] = c
        counts.append(sub_dict)
    label_count_dict = {str(k) :v for k,v in enumerate(counts)}
    matching = algorithm.find_matching(label_count_dict, matching_type = 'max', return_type = 'list')
    matching = {v:int(k) for (k,v),_ in matching}
    pred = np.array([matching[k] for k in pred])
    # return f1_score(true, pred, average = average)
    return matching
<<<<<<< HEAD
=======

# def tsne(data, d = 2, p = 50):
#     tsne = TSNE(n_components = d, perplexity = p)
#     data_embedded = tsne.fit_transform(data)
#     return data_embedded
# 
# def consecutive(data, stepsize = 1):
#     result = [0]
#     c = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
#     for a in c :
#         result.append(result[-1]+len(a))
#     return np.array(result)
# 
# def plotSegment(data_result_dict, length = 1):
#     colors = ["maroon", "olive", "lime", "tomato", "magenta", "cyan", "brown", "peru"]
# 
#     data_y = data_result_dict["data_y"]
#     n_ts = len(data_y)
#     w = data_result_dict["args"]["window"]
#     k = len(np.unique(data_y))
#     # truncated
#     data_y = data_y[:(n_ts//w)*w]
# 
#     x = 1
#     plt.figure(figsize = (20 * length, 3 * k))
#     for i in np.unique(data_y):
#         plt.subplot(k, 1, x)
#         plt.title("y = %d" % i)
#         plt.xlim(0, len(data_y[data_y==i])*length)
#         plt.ylim(min(data_y),max(data_y))
#         c = 0
#         plt.plot(data_result_dict["pred"][:(n_ts//w)*w][data_y==i], c = colors[c], linewidth = 3, label = "pred")
#         plt.vlines(consecutive(np.where(data_y==i)[0]), ymin=0, ymax=10, colors = "black",linestyles="--", linewidth = 5)
#         plt.legend(loc = "upper left")
#         # plt.show()
#         x += 1
#     plot_savefig(data_result_dict = data_result_dict, func = "plotSegment")
# 
# def plotSegmentWithX(data_result_dict, length = 1):
#     colors = ["maroon", "olive", "lime", "tomato", "magenta", "cyan", "brown", "peru"]
#     data_X = data_result_dict["data_X"]
#     data_y = data_result_dict["data_y"]
# 
#     n_ts = len(data_y)
#     w = data_result_dict["args"]["window"]
#     k = len(np.unique(data_y))
# 
#     # truncated
#     data_X = data_X[:(n_ts//w)*w]
#     data_y = data_y[:(n_ts//w)*w]
#     x = 1
#     plt.figure(figsize = (20 * length, 3 * k))
#     for i in np.unique(data_y):
#         ax = plt.subplot(k, 1, x)
#         ax.set_title("y = %d" % i)
#         ax.set_xlim(0, len(data_X[data_y == i])*length)
#         ax.plot(data_X[data_y==i])
#         ax2 = ax.twinx()
#         ax2.set_ylim(min(data_y),max(data_y))
#         ax2.plot(data_result_dict["pred"][:(n_ts//w)*w][data_y==i], c = colors[0], linewidth = 3, label = "pred")
#         ax2.vlines(consecutive(np.where(data_y==i)[0]), ymin=0, ymax=10, colors = "red",linestyles="--", linewidth = 3)
#         ax2.legend(loc = "upper left")
#         x += 1
#     plot_savefig(data_result_dict = data_result_dict, func = "plotSegmentWithX")
# 
# def plotY(data_result_dict):
#     keys = data_result_dict.keys()
#     colors = ["maroon", "olive", "lime", "tomato", "magenta", "cyan", "brown", "peru"]
#     plt.figure(figsize = (20, 5))
#     c = 0
#     plt.plot(data_result_dict["pred"], label = "pred", c = colors[c])
#     plt.plot(data_result_dict["data_y"], label = "True", c = "black", linewidth = 3)
#     plt.legend(loc="upper left")
#     plot_savefig(data_result_dict = data_result_dict, func = "plotY")
# 
# def plotYCounts(data_result_dict):
#     plt.figure(figsize = (10,10))
#     r = 1
#     plt.subplot(r, 2, 1)
#     my_cmap = plt.get_cmap("tab20")
#     rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
#     i = 1
#     for k in ["data_y", "pred"]:
#         plt.subplot(r, 2, i)
#         if k in data_result_dict.keys():
#             plt.title(k, fontsize = 15)
#             plt.xticks(fontsize = 15)
#             plt.yticks(fontsize = 15)
#             c = np.unique(data_result_dict[k], return_counts = True)
#             c1_with_order = [i[0] for i in sorted(enumerate(c[1]), key = lambda x:x[1], reverse = True)]
#             c_new = (c[0], sorted(c[1], reverse = True))
#             bar = plt.bar(c_new[0], c_new[1], color = my_cmap(rescale(c[0])))
#             for idx, rect in enumerate(bar):
#                 heights = rect.get_height()
#                 plt.text(rect.get_x() + rect.get_width() / 2.0, heights, f'{c1_with_order[idx]:.0f}', ha='center', va='bottom', fontsize = 15)
#         i += 1
#     plt.tight_layout()
#     plot_savefig(data_result_dict = data_result_dict, func = "plotYCounts")
# 
# # %%
# 
>>>>>>> 2f5dc7987a42190942903bb0ca173411330d4670
