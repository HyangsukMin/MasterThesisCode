#!/home/hmin/anaconda3/envs/tensor-gpu/bin/python3.8

import pathlib, argparse, pickle, os, glob
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(prog="preprocess", description='Data preprocess')
parser.add_argument('--dataset', type = str, default = "hapt", required = True, choices = ["GTEA","FS","welllog","harsc", "hapt", "mhealth", "pamap2", "EMG", "DSA"])
parser.add_argument('--scaling_method', type = str, default = "minmax", required = False, choices = ['minmax','st'])
parser.add_argument('--EMGuser', type = int, default = 1, required = False)
parser.add_argument('--EMGexp', type = int, default = 0, required = False)
parser.add_argument('--DSAuser', type = int, default = 1, required = False)
args = parser.parse_args()

def adjustLabel(data_y):
    label = 0
    for i in np.unique(data_y):
        data_y[data_y == i] = label
        label += 1
    return data_y

def scaling(data, method = "minmax"):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    if method == "st":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def patchwork(feature_list, label_list, label_seg_list):
    num_file = len(feature_list)
    permuted_file_indices = np.arange(num_file)
    length = 0
    X_long = []
    y_long = []
    y_seg_long = []
    file_boundaries = []
    for i in permuted_file_indices:
        length += len(feature_list[i])
        X_long.append(feature_list[i])
        y_long.append(label_list[i])
        y_seg_long.append(label_seg_list[i])
        file_boundaries.append(length)
    return np.concatenate(X_long, axis=0), np.concatenate(y_long, axis=0), np.concatenate(y_seg_long, axis=0), np.array(file_boundaries, dtype=np.int64)

def generate_boundary_labels(label_list, mapping_dict):
    boundary_list = []
    segment_len_list = []
    label_seg_list = []

    for video_label in label_list:
        for class_label, class_name in mapping_dict.items():
            video_label[video_label == class_name] = int(class_label) # change class name into class integer

        label_seg_list.append(np.zeros(len(video_label)))
        boundaries = []
        segment_len = []
        length = 0
        for ind, (prev_label, curr_label) in enumerate(zip(video_label, video_label[1:])):
            length += 1
            if prev_label != curr_label:
                boundaries.append(ind)
                segment_len.append(length)
                length = 0
        if length != 0:
            segment_len.append(length)  # put last segment(no boundary at the last of file)
        if len(boundaries) != len(segment_len)-1:
            segment_len.append(1)
        boundary_list.append(boundaries)
        segment_len_list.append(segment_len)

    for i in range(len(boundary_list)):
        for j in range(len(boundary_list[i])):
            label_seg_list[i][boundary_list[i][j]] = 1
    return label_seg_list

def FS_preprocess(scaling_method = "minmax"):
    data_path = './datasets/FS/features'
    label_path = './datasets/FS/groundTruth'
    label_map_file_name = './datasets/FS/mapping.txt'
    feature_file_names = sorted(glob.glob(os.path.join(data_path, "*.npy")))
    label_file_names = sorted(glob.glob(os.path.join(label_path, "*.txt")))

    feature_list = [np.load(f).transpose() for f in feature_file_names]
    label_list = [np.array(pd.read_csv(f, sep=" ", index_col=None, header=None)[0].to_numpy()) for f in
                    label_file_names]
    mapping_dict = pd.read_csv(label_map_file_name, sep=" ", index_col=None, header=None)[1].to_dict()

    label_seg_list = generate_boundary_labels(label_list, mapping_dict)
    data_X, data_y, y_seg_long, file_boundaries_indice = patchwork(feature_list, label_list, label_seg_list)
    y_seg_long = np.array(generate_boundary_labels([data_y],{})).flatten()

    data_X = data_X[data_y != 17]
    data_y = data_y[data_y != 17]
    
    data_X = data_X[data_y != 18]
    data_y = data_y[data_y != 18]
    
    data_X = scaling(data_X, method = scaling_method)
    data_y = adjustLabel(data_y)
    print(data_X.shape, data_y.shape)
    pickle.dump((data_X, data_y), open("./datasets/FS/FS.pkl", "wb"))

def GTEA_preprocess(scaling_method = 'minmax'):
    data_path = './datasets/GTEA/features'
    label_path = './datasets/GTEA/groundTruth'
    label_map_file_name = './datasets/GTEA/mapping.txt'

    feature_file_names = sorted(glob.glob(os.path.join(data_path, "*.npy")))
    label_file_names = sorted(glob.glob(os.path.join(label_path, "*.txt")))
    mapping_dict = pd.read_csv(label_map_file_name, sep=" ", index_col=None, header=None)[1].to_dict()
    feature_list = [np.load(f).transpose() for f in feature_file_names]
    label_list = [np.array(pd.read_csv(f, sep=" ", index_col=None, header=None)[0].to_numpy()) for f in
                    label_file_names]

    label_seg_list = generate_boundary_labels(label_list, mapping_dict)
    data_X, data_y, y_seg_long, file_boundaries_indice = patchwork(feature_list, label_list, label_seg_list)
    y_seg_long = np.array(generate_boundary_labels([data_y], {})).flatten()

    data_X = scaling(data_X, method = scaling_method)
    data_y = adjustLabel(data_y)
    print(data_X.shape, data_y.shape)
    pickle.dump((data_X, data_y), open("./datasets/GTEA/GTEA.pkl", "wb"))

def welllog_preprocess(scaling_method = "minmax"):
    import pandas as pd
    filepaths = list(pathlib.Path("./datasets/WELLLOG").glob("*.csv"))
    for filepath in filepaths:
        data = pd.read_csv(filepath)
        y = data["Facies"]
        X = data.drop(["Facies", "Formation"], axis = 1)
        well = data["Well Name"].unique()
        i = 0
        for w in well:
            data_X = X[X["Well Name"] == w] 
            data_y = y[X["Well Name"] == w]

            data_X = data_X.drop("Well Name", axis = 1)
            data_X = scaling(data_X, method = scaling_method)
            data_y = adjustLabel(data_y)
            print(i,": ",data_X.shape, data_y.shape)
            pickle.dump((data_X, data_y), open("./datasets/WELLLOG/welllog_{:d}.pkl".format(i),"wb"))
            i += 1
            
def harsc_preprocess(scaling_method = "minmax"):
    filepaths = list(pathlib.Path("./datasets/HARSC").glob("*.csv"))
    for filepath in filepaths:
        data = pd.read_csv(filepath, header = None)
        data_X = data.iloc[:,1:4]
        data_y = data.iloc[:,-1]
        data_X = data_X[data_y!=0]
        data_y = data_y[data_y!=0]

        # 너무 길어서 앞 뒤 길이 잘라주기
        labels, idx = np.unique(data_y, return_index = True)
        label_dict = {k:v for k,v in zip(labels, idx)}
        label_dict = dict(sorted(label_dict.items(), key = lambda x: x[1]))
        start_point = label_dict[list(label_dict.keys())[1]] //2 
        end_point = ((len(data_y) - label_dict[list(label_dict.keys())[-1]]) // 4)*3

        data_X = np.array(data_X)[start_point:-end_point]
        data_y = np.array(data_y)[start_point:-end_point]
        
        data_X = scaling(data_X, method = scaling_method)
        data_y = adjustLabel(data_y)
        print(data_X.shape, data_y.shape)
        pickle.dump((data_X, data_y), open("{}.pkl".format(str(filepath).split(".")[0]),"wb"))

def hapt_preprocess(exp = 1, user = 1, scaling_method = "minmax"):
    # # (1) preprocess
    # if exp < 10:
    #     exp = "0" + str(exp)
    # if user < 10:
    #     user = "0"+str(user)
    # filepaths = sorted(pathlib.Path("HAPT","RawData").glob("*_exp{}_user{}.txt".format(exp,user)))
    # data = []
    # for filepath in filepaths:
    #     data.append(np.loadtxt(filepath))
    # X = np.concatenate(data, axis = 1)

    # # (2) get labels
    # labels = np.loadtxt("HAPT/RawData/labels.txt")
    # y = []
    # idx = np.where(labels[:,0] == float(exp))
    # for i in idx[0]:
    #     label = labels[i,2]
    #     y += [int(label)] * int(labels[i,4] - labels[i,3])
    
    import numpy as np
    data_X = np.load("./datasets/HAPT/labeled_X.npy")
    data_y = np.load("./datasets/HAPT/labeled_y.npy")
    data_X = data_X[data_y != 0][:30000]
    data_y = data_y[data_y != 0][:30000]
    
    data_X = scaling(data_X, method = scaling_method)
    data_y = adjustLabel(data_y)
    print(data_X.shape, data_y.shape)
    pickle.dump((data_X, data_y), open("./datasets/HAPT/hapt1_1.pkl","wb"))

def mhealth_preprocess(scaling_method = "minmax"):
    filepaths = list(pathlib.Path("./datasets/mHealth").glob("*.log"))
    i = 0
    for filepath in filepaths:
        data_X = []
        data_y = []
        with open(filepath,"r") as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                line = list(map(float, line.split()))
                data_X.append(line[:-1])
                data_y.append(line[-1])
        
        data_X = np.array(data_X)
        data_y = np.array(data_y)

        data_X = data_X[data_y!=0]
        data_y = data_y[data_y!=0]
        
        data_X = scaling(data_X, method = scaling_method)
        data_y = adjustLabel(data_y)
        print(i,": ",data_X.shape, data_y.shape)
        pickle.dump((data_X, data_y), open("{}.pkl".format(str(filepath).split(".")[0]),"wb"))
        i += 1

def pamap2_preprocess(scaling_method = "minmax"):
    filepaths = list(pathlib.Path("./datasets/PAMAP2").glob("*.dat"))
    i = 0
    for filepath in filepaths:
        data = np.loadtxt(filepath)
        del_row_ind = np.where(data[:,1] == 0)[0]
        data = np.delete(data, del_row_ind, axis = 0)

        nan_row_ind = np.isnan(data[:,3:]).any(axis=1)
        data = np.delete(data, nan_row_ind, axis = 0)
        data_X = data[:,3:]
        data_y = data[:,1]
        
        data_X = data_X[data_y != 0]
        data_y = data_y[data_y != 0]

        # data is too long
        data_X = data_X[::5]
        data_y = data_y[::5]
        data_X = scaling(data_X, method = scaling_method)
        data_y = adjustLabel(data_y)
        
        print(i,": ",data_X.shape, data_y.shape)
        pickle.dump((data_X, data_y),open("{}.pkl".format(str(filepath).split(".")[0]),"wb"))
        i += 1
        
def EMG_preprocess(user = 1, exp = 0, scaling_method = "minmax"):
    assert exp <= 1 and exp >= 0
    dirpath = "datasets/EMG/{}".format(str(user).zfill(2))
    fileLists = os.listdir(dirpath)

    data = np.loadtxt(os.path.join(dirpath,fileLists[exp]), skiprows = 1)
    
    data_X = data[:,1:-1]
    data_y = data[:,-1]

    data_X = data_X[data_y != 0]
    data_y = data_y[data_y != 0]
    
    data_X = scaling(data_X, method = scaling_method)
    data_y = adjustLabel(data_y)
    
    print(data_X.shape, data_y.shape)
    pickle.dump((data_X, data_y), open("./datasets/EMG/subject{}_exp{}.pkl".format(str(user).zfill(2),str(exp)), "wb"))

def DSA_preprocess(user = 1, scaling_method = "minmax"):
    user = 1
    dirpath = "datasets/DSA/"
    labels = os.listdir(dirpath)
    labels_dir = [[os.path.join(dirpath, label, "p{:d}".format(user), "s{:s}.txt".format(str(d).zfill(2))) for d in range(1,61)] for label in labels ]

    X = []
    y = []
    for l in range(len(labels)):
        A = []
        for d in range(60):
            A.append(np.loadtxt(labels_dir[0][2], delimiter = ","))
        X.append(np.concatenate(A, axis = 0))
        y.append([l]*len(X[l]))
    X = np.concatenate(X, axis = 0)
    y = np.concatenate(y, axis = 0)

    data_X = scaling(X, method = scaling_method)
    data_y = adjustLabel(y)
    
    print(data_X.shape, data_y.shape)
    pickle.dump((data_X, data_y), open("./datasets/DSA/subject{}.pkl".format(str(user).zfill(2)), "wb"))

def skoda_preprocess(scaling_method = "minmax"):
    import scipy.io as sio
data_dict = sio.loadmat(file_name="skodaminicp_2015_08 (1)/SkodaMiniCP_2015_08/right_classall_clean.mat", squeeze_me=True)
all_data = data_dict[list(data_dict.keys())[3]]
# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
def label_count_from_zero(all_data):
    """ start all labels from 0 to total number of activities"""

    labels = {32: 'null class', 48: 'write on notepad', 49: 'open hood', 50: 'close hood',
              51: 'check gaps on the front door', 52: 'open left front door',
              53: 'close left front door', 54: 'close both left door', 55: 'check trunk gaps',
              56: 'open and close trunk', 57: 'check steering wheel'}

    a = np.unique(all_data[:, 0])

    for i in range(len(a)):
        all_data[:, 0][all_data[:, 0] == a[i]] = i
    #         print(i, labels[a[i]])

    return all_data

def normalize(data):
    """ l2 normalization can be used"""

    y = data[:, 0].reshape(-1, 1)
    X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(X)
    X = transformer.transform(X)

    return np.concatenate((y, X), 1)
def split(data):
    """ get 80% train, 10% test and 10% validation data from each activity """

    y = data[:, 0]  # .reshape(-1, 1)
    X = np.delete(data, 0, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

def get_train_val_test(data):

    # removing sensor ids
    for i in range(1, 60, 6):
        data = np.delete(data, i, 1)

    # data = data[data[:, 0] != 32]  # remove null class activity

    data = label_count_from_zero(data)
    data = normalize(data)

    activity_id = np.unique(data[:, 0])
    number_of_activity = len(activity_id)

    for i in range(number_of_activity):

        data_for_a_single_activity = data[np.where(data[:, 0] == activity_id[i])]
        trainx, trainy, testx, testy, valx, valy = split(data_for_a_single_activity)

        if i == 0:
            x_train, y_train, x_test, y_test, x_val, y_val = trainx, trainy, testx, testy, valx, valy

        else:
            x_train = np.concatenate((x_train, trainx))
            y_train = np.concatenate((y_train, trainy))

            x_test = np.concatenate((x_test, testx))
            y_test = np.concatenate((y_test, testy))

            x_val = np.concatenate((x_val, valx))
            y_val = np.concatenate((y_val, valy))

    return x_train, y_train, x_test, y_test, x_val, y_val
# %%
x_train, y_train, x_test, y_test, x_validation, y_validation = get_train_val_test(all_data)
if __name__ == "__main__":
    if args.dataset == "all":
        FS_preprocess(scaling_method = args.scaling_method)
        GTEA_preprocess(scaling_method = args.scaling_method)
        welllog_preprocess(scaling_method = args.scaling_method)
        harsc_preprocess(scaling_method = args.scaling_method)
        hapt_preprocess(scaling_method = args.scaling_method)
        mhealth_preprocess(scaling_method = args.scaling_method)
        pamap2_preprocess(scaling_method = args.scaling_method)
        EMG_preprocess(user = args.EMGuser, exp = args.EMGexp, scaling_method = args.scaling_method)
        DSA_preprocess(user = args.DSAuser, scaling_method = args.scaling_method)
        
    elif args.dataset == "FS":
        FS_preprocess(scaling_method = args.scaling_method)

    elif args.dataset == "GTEA":
        GTEA_preprocess(scaling_method = args.scaling_method)

    elif args.dataset == "welllog":
        welllog_preprocess(scaling_method = args.scaling_method)

    elif args.dataset == "harsc":
        harsc_preprocess(scaling_method = args.scaling_method)

    elif args.dataset == "hapt":
        hapt_preprocess(scaling_method = args.scaling_method)

    elif args.dataset == "mhealth":
        mhealth_preprocess(scaling_method = args.scaling_method)

    elif args.dataset == "pamap2":
        pamap2_preprocess(scaling_method = args.scaling_method)

    elif args.dataset == "EMG":
        EMG_preprocess(user = args.EMGuser, exp = args.EMGexp, scaling_method = args.scaling_method)
    elif args.dataset == "DSA":
        DSA_preprocess(user = args.DSAuser, scaling_method = args.scaling_method)
        
    print("Finish preprocessing %s dataset" % (args.dataset.upper()))