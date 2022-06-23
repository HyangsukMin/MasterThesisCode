#%%
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
# %%
result = pickle.load(open("../outputs/hapt_AE_w10_oFalse.pkl","rb"))
result_bs = pickle.load(open("../baseline/outputs/cluster_result_hapt_w10_b50.pkl","rb"))
# %%
data_y = result["data_y"]
pred_y = result["pred"]
eu_y = result_bs["euclidean"]
dtw_y = result_bs["dtw"]
gmm_y = result_bs["gmm"]
ticc_y = result_bs["ticc"]
# %%
def adjustLabel(pred, label):
    pred_t = pred[label<5]
    label_t = label[label<5]
    label, order = np.unique(label_t, return_index = True)
    order_dict = {k:v for k,v in zip(label, order)}
    order_dict = dict(sorted(order_dict.items(), key = lambda x : x[1]))
    for key, _ in order_dict.items():
        print(key, np.unique(pred_t[label_t == key], return_counts = True))
# %%
adjustLabel(ticc_y, data_y)
# %%
