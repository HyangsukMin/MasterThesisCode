import numpy as np
import pandas as pd
from utils import *
from tslearn.clustering import TimeSeriesKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from ticc.TICC_solver import TICC
import time

def bs_ticc(data, y_hat, w = 3, k = 6, l = 11e-2, b = 0, maxIters = 1, random_state = 102):
    logging.info("Data shape: {}".format(data.shape))
    st = time.time()
    ticc = TICC(window_size = w, number_of_clusters = k, lambda_parameter = l, beta = b, maxIters = maxIters, random_state = random_state, threshold = 2e-5,
                write_out_file = False, num_proc = 1)
    (cluster_assignment, cluster_MRFs) = ticc.fit(data, y_hat)
    ed = time.time()
    print("{:.3f}".format(ed-st))
    return cluster_assignment, cluster_MRFs, ticc

def bs_kmeans(data, k = 6, m = "Euclidean", random_state = 1234):
    kmeans = KMeans(n_clusters = k, random_state = random_state).fit(data)
    return kmeans.labels_, kmeans

def bs_gmm(data, k = 6, covariance_type = "full", random_state = 1234):
    gmm = GaussianMixture(n_components = k, covariance_type = covariance_type, random_state = random_state).fit(data)
    return gmm.predict(data), gmm

def bs_dtw(data, k, n_jobs = 10, random_state = 1234):
    logging.info("Data shape: {}".format(data.shape))
    dtw = TimeSeriesKMeans(n_clusters = k, metric = "dtw",
                            verbose = False, max_iter_barycenter = 10,
                            n_jobs = n_jobs, random_state = random_state).fit(data)
    return dtw.labels_, dtw

def bs_ghmm(data, k, random_state = 1234):
    from hmmlearn.hmm import GaussianHMM
    hmm = GaussianHMM(n_components = k, n_iter = 10000)#, params = "st", init_params = "st")
    hmm.fit(data, int(len(data)))
    _, pred = hmm.decode(data, algorithm = "viterbi")
    return pred, hmm


def stack_training_data(data, w):
    n, dim = data.shape
    complete_D_train = np.zeros([n-w+1, w * dim])
    for i in range(w, n):
        complete_D_train[(i-w+1),:] = data[(i-w):i,:].reshape(-1)
    return complete_D_train

def baseline_data(data_X, data_y, w):
    n_ts, n_dim = data_X.shape
    k = len(np.unique(data_y))
    
    remainder = n_ts % k
    data_y_hat = np.repeat(np.arange(k), n_ts // k)
    data_y_hat = np.concatenate([np.repeat(data_y_hat[0], remainder), data_y_hat])

    data = stack_training_data(data_X, w)
    y_hat = data_y_hat[(w-1):]
    return data, y_hat

#-------------------------------------------------------------------------------------
# DNN Baseline
#-------------------------------------------------------------------------------------
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, ReLU, Dropout, Flatten, Activation, AveragePooling1D, UpSampling1D, Conv1DTranspose, MaxPooling1D, LSTM, GRU, TimeDistributed
from tensorflow.keras import Model

class AE(Model):
    def __init__(self, module, window, n_latent, n_hidden):
        super(AE, self).__init__()
        
        if module == "LSTM":
            base = LSTM
        elif module == "GRU":
            base = GRU
        
        self.encoder = []
        self.encoder.append(base(n_latent, activation = "relu", return_sequences = False, return_state = True))

        self.repeatvector = RepeatVector(window)
        self.decoder = []
        self.decoder.append(base(n_hidden, activation = "relu", return_sequences = True))
        
        self.outputs = TimeDistributed(Dense(n_hidden))
        
    def encode(self, inputs, training = True):
        x = inputs
        for i in range(len(self.encoder)):
            encoder_outputs = self.encoder[i](x, training = training)
            z = self.repeatvector(encoder_outputs[0])
        return z, encoder_outputs[1:]
    
    def call(self, inputs, training = True):
        x = inputs
        encoder_outputs = self.encode(x, training = training)
        
        for j in range(len(self.decoder)):
            z = self.decoder[j](z, initial_state = encoder_outputs[1:], training = training)
        
        x_hat = self.outputs(z, training = training)
        return x_hat  


def run(dataset, data_X, data_y,window, shift, module, model_epochs = 100, n_latent = 10, batch_size = 32, exp = 5, l = 0.11, b = 300, maxIters = 20):
    n_ts, n_dim = data_X.shape
    print(n_ts, n_dim)
    train_X, train_y = create_assigning_dataset(data_X, data_y, window = window, shift = shift)

    print("#----------------------------------------#")
    print("# Model: {:s}, ".format(module))
    print("#----------------------------------------#")
    # Model Running
    n_latent = 10
    n_hidden = n_dim
    model = AE(module = module, window = window, n_latent = n_latent, n_hidden = n_hidden)
    model.compile(optimizer = "adam", loss = "mse")
    model.fit(train_X, train_X, epochs = model_epochs, batch_size = batch_size, verbose = 0)

    # Get repr
    test_X, test_y = create_assigning_dataset(data_X, data_y, window = window, shift = window)
    z = model.encode(test_X).numpy()
    z = z.reshape(-1, z.shape[-1])
    remainder = n_ts % window
    z = np.concatenate([z[:-window], z[-remainder:]])

    k = len(np.unique(data_y))

    w = 1
    kmeans_results = []
    gmmfull_results = []
    dtw_results = []
    ticc_results = []
    data, y_hat = baseline_data(z, data_y, w)
    
    for epoch in range(exp):
        print(epoch, end = "\n\t")

        #-------------------------------------------------------------------------------------------------------------------
        
        kmeans = bs_kmeans(data, k = k, random_state = epoch)
        print("kmeans |", metrics(data_y[(w-1):], kmeans[0]), end = "\n\t")
        kmeans_results.append(["{:d}_{:s}_kmeans_{:d}".format(w, module, epoch)] + list(metrics(data_y[(w-1):], kmeans[0])))

        kmeans_df = pd.DataFrame(kmeans_results, columns = ["w_model_epoch","nmi","ami","ari","f1","acc"])
        kmeans_df.to_csv("bs_outputs/{:s}_{:s}_kmeans_{:d}.csv".format(dataset, module, epoch), index = False)

        #-------------------------------------------------------------------------------------------------------------------
        
        gmmfull = bs_gmm(data, k = k, covariance_type = "full", random_state = epoch)
        print("gmmfull |", metrics(data_y[(w-1):], gmmfull[0]), end = "\n\t")
        gmmfull_results.append(["{:d}_{:s}_gmmfull_{:d}".format(w, module, epoch)] + list(metrics(data_y[(w-1):], gmmfull[0])))

        gmmfull_df = pd.DataFrame(gmmfull_results, columns = ["w_model_epoch","nmi","ami","ari","f1","acc"])
        gmmfull_df.to_csv("bs_outputs/{:s}_{:s}_gmmfull_{:d}.csv".format(dataset, module, epoch), index = False)

        #-------------------------------------------------------------------------------------------------------------------

        # dtw = bs_dtw(data, k = k, random_state = epoch)
        # print("dtw |", metrics(data_y[(w-1):], dtw[0]), end = "\n\t")
        # dtw_results.append(["{:d}_{:s}_dtw_{:d}".format(w, module, epoch)] + list(metrics(data_y[(w-1):], dtw[0])))

        # dtw_df = pd.DataFrame(dtw_results, columns = ["w_model_epoch","nmi","ami","ari","f1","acc"])
        # dtw_df.to_csv("bs_outputs/{:s}_{:s}_dtw_{:d}.csv".format(dataset, epoch, module), index = False)

        #-------------------------------------------------------------------------------------------------------------------
        
        ticc = bs_ticc(data, y_hat, k = k, l = l, b = b, random_state = epoch)
        print("ticc |", metrics(data_y[(w-1):], ticc[0]), end = "\n\t")
        ticc_results.append(["{:d}_{:s}_ticc_{:d}".format(w, module, epoch)] + list(metrics(data_y[(w-1):], ticc[0])))

        ticc_df = pd.DataFrame(ticc_results, columns = ["w_model_epoch","nmi","ami","ari","f1","acc"])
        ticc_df.to_csv("bs_outputs/{:s}_{:s}_ticc_{:d}.csv".format(dataset, module, epoch), index = False)

        #-------------------------------------------------------------------------------------------------------------------
        
        print()
        
        
        
        
