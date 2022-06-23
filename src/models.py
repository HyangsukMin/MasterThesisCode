from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, ReLU, Dropout, Flatten, Activation, AveragePooling1D, UpSampling1D, Conv1DTranspose, TimeDistributed, MaxPooling1D
from tensorflow.keras import Model
import tensorflow as tf

from TICC_solver import TICC
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from utils import *
import numpy as np

#---------------------------
# TCN
#---------------------------
class TCN(Model):
    def __init__(self, n_filters = 8, kernel_size = 3, dilation_exp = 2, n_dilation = 8, padding = 'same', dropout_rate = 0.25, window = 2048):
        super(TCN, self).__init__()
        self.n_dilation = n_dilation
        self.n_filters = n_filters
        self.tcn = []
        self.tcn.append(Conv1D(filters=n_filters, kernel_size=1, strides=1, padding = 'same'))
        for i in range(n_dilation):
            dilation_rate = min(dilation_exp ** (i), window//(kernel_size-1)-1)
            dilated_conv = []
            dilated_conv.append(Conv1D(filters = n_filters, kernel_size=kernel_size,
                                       strides=1, padding = padding, dilation_rate = dilation_rate))
            dilated_conv.append(BatchNormalization())
            dilated_conv.append(ReLU())
            dilated_conv.append(Conv1D(filters = n_filters, kernel_size = 1, strides = 1, padding = 'same'))
            dilated_conv.append(Dropout(rate = dropout_rate))
            self.tcn.append(dilated_conv)
        self.mix = Dense(1, use_bias = False, kernel_initializer = tf.keras.initializers.Constant(0.25), kernel_constraint = tf.keras.constraints.NonNeg())

        
    def call(self, inputs, training = True):
        x = inputs
        x = self.tcn[0](x, training = training)
        stack = [x] #x * 0.01
        for j in range(1, self.n_dilation + 1):
            for k in range(len(self.tcn[j])):
                x = self.tcn[j][k](x, training = training)
            stack.append(x)
        x = tf.concat(stack, axis = -1)
        x = tf.stack(stack, axis = -1)
        x = self.mix(x, training = training)
        x = tf.squeeze(x, axis = -1)
        return x
    
class CNN(Model):
    def __init__(self, n_filters = 8, kernel_size = 3, n_dilation = 3, padding = 'same', dropout_rate = 0.25):
        super(CNN, self).__init__()
        self.n_dilation = n_dilation
        self.n_filters = n_filters
        self.cnn = []
        for i in range(n_dilation):
            conv = []
            conv.append(Conv1D(filters = n_filters, kernel_size = kernel_size,
                               strides = 1, padding = padding))
            conv.append(BatchNormalization())
            conv.append(ReLU())
            conv.append(Dropout(rate = dropout_rate))
            self.cnn.append(conv)
        self.mix = TimeDistributed(Dense(1, use_bias = False, kernel_initializer = tf.keras.initializers.Constant(0.33), kernel_constraint = tf.keras.constraints.NonNeg()))
        
    def call(self, x, training = True):
        stack = []
        for j in range(0, self.n_dilation):
            for k in range(len(self.cnn[j])):
                x = self.cnn[j][k](x, training=training)
            stack.append(x)
        x = tf.stack(stack, axis = -1)
        x = self.mix(x, training = training)
        x = tf.squeeze(x, axis = -1)
        return x

class PCN(Model):
    def __init__(self, n_dilation, n_hidden, window):
        super(PCN, self).__init__()
        self.results = {}
        self.n_dilation = n_dilation
        self.W = Dense(n_hidden, activation = "relu")
        
        self.pcn = []
        for i in range(n_dilation):
            size = max(window//(4**i) , 4)
            self.pcn.append(MaxPooling1D(pool_size = size, strides = size))
            self.pcn.append(UpSampling1D(size = size))
                        
    def call(self, inputs , training = True):
        input = self.W(inputs, training = training)
        stack = [input]
        for i in range(0, len(self.pcn), 2):
            x = self.pcn[i](input, training = training)
            x = self.pcn[i+1](x, training = training)
            stack.append(x)
        x = tf.stack(stack, axis = -1)
        self.results["x"] = x
        x = tf.reduce_sum(x, axis = -1)
        self.results["x_last"] = x
        return x
        
#-------------------------------
# CAE
#-------------------------------
class CAE(Model):
    def __init__(self, window, shift, n_latent, n_hidden, n_filters = 8, kernel_size = 3, padding = 'causal', n_dilation = 3, dilation_exp = 4, dropout_rate = 0.25, ablation = "None", sampling_factor = 8):
        super(CAE, self).__init__()
        
        self.results = {}
        
        sampling = window // sampling_factor
    
        self.n_hidden = n_hidden
        self.ablation = ablation


        self.temporal_encoder = CNN(n_filters = n_filters, kernel_size = kernel_size, n_dilation = n_dilation, padding = padding, dropout_rate = dropout_rate)
        self.temporal_decoder = CNN(n_filters = n_filters, kernel_size = kernel_size,  n_dilation = n_dilation, padding = padding, dropout_rate = dropout_rate)

        if ablation == "None" or ablation != "noTCN":
            self.temporal_encoder = TCN(n_filters = n_filters, kernel_size = kernel_size, n_dilation = n_dilation, dilation_exp = dilation_exp, padding = padding, dropout_rate = dropout_rate, window = window)
            self.pattern_learning = PCN(n_dilation, n_hidden, window)
        
        self.latent = Conv1D(n_latent, kernel_size = 3, padding = 'same')
        self.pool = AveragePooling1D(pool_size = sampling, strides = None, padding='valid', data_format='channels_last')
        self.upsample = UpSampling1D(size = sampling)

        self.recon = Conv1D(filters = n_hidden, kernel_size = 1, padding = 'same', activation="sigmoid")
    
    def encode(self, inputs, training = True):
        n_bs, n_w, n_dim = inputs.shape
        
        x = self.temporal_encoder(inputs, training = training)
        self.results["x"] = x 
        if self.ablation != "noTCN" or self.ablation == "None":
            P = self.pattern_learning(inputs, training = training)
            x = tf.concat([x, P], axis = -1)
            self.results["x_concat"] = x
            
        z = self.latent(x, training = training)
        self.results["z"] = z
        
        return z

    def decode(self, inputs, training = True):
        x = inputs
        x = self.temporal_decoder(x, training = training)
        recon = self.recon(x, training = training)
        return recon
    
    def call(self, inputs, training = True):
        x = inputs
        
        z = self.encode(x, training = training)
        
        pool = self.pool(z)
        upsample = self.upsample(pool)
        
        recon = self.decode(upsample, training = training)
        
        return z, recon

#---------------------------
# Classification
#---------------------------
class ClassEstimation(Model):
    def __init__(self, n_clusters):
        super(ClassEstimation, self).__init__()
        self.n_clusters = n_clusters
        
        self.dense1 = Dense(n_clusters, activation = "softmax") #, bias_initializer = tf.keras.initializers.Constant(1/n_clusters))
        self.dense2 = Dense(n_clusters, activation = "softmax")
    
    def call(self, inputs, training = True):
        y = self.dense1(inputs, training = training)
        y = self.dense2(y, training = training)
        return y
    
#--------------------------
# Overall
#--------------------------
class Backbone(Model):
    def __init__(self, window,
                 shift,
                 n_clusters,
                 n_latent, 
                 n_hidden, 
                 n_filters = 8, 
                 kernel_size = 3, 
                 padding = 'same', 
                 n_dilation = 3,
                 dilation_exp = 4,
                 dropout_rate = 0.25,
                 ablation = "None"):
        
        super(Backbone, self).__init__()
        self.ablation = ablation
        self.results = {}
        self.cae = CAE(window = window, shift = shift, n_latent = n_latent, n_hidden = n_hidden, n_filters = n_filters, kernel_size = kernel_size, padding = padding, n_dilation = n_dilation, dilation_exp = dilation_exp, dropout_rate = dropout_rate, ablation = ablation)
        
        if ablation == "None" or ablation != "noPseudo":
            self.ce = ClassEstimation(n_clusters)
    
    def call(self, inputs, training = True):
        z, recon = self.cae(inputs, training = training)
        if self.ablation == "None" or self.ablation != "noPseudo":
            pred = self.ce(z, training = training)
            return z, recon, pred
        elif self.ablation != "None" and self.ablation == "noPseudo":
            return z, recon
    def predict(self, inputs, training = False):
        if self.ablation == "None" or self.ablation != "noPseudo":
            x = inputs
            z = self.cae.encode(x, training = training)
            y = self.ce(z, training = training)
            return tf.argmax(y, axis = -1)
        
        elif self.ablation != "None" and self.ablation == "noPseudo":
            print_text("No CE Model")
            pass
        
#----------------------------------
# Temporal Convolutional Networks-based Temporal Clustering
#----------------------------------
class TTC:
    def __init__(self, n_clusters, window, shift,
                 n_latent, n_hidden, epochs,
                 maxiters, # noPseudo careful
                 beta,
                 learning_rate,
                 batch_size = 32,
                 cluster_reassignment = 50,
                 random_state = 52,
                 n_filters = 8,
                 kernel_size = 3,
                 padding = "causal",
                 n_dilation = 3,
                 dilation_exp = 4,
                 dropout_rate = 0.25,
                 threshold = 0.005,
                 pretrain_epoch = 11,
                 initializer = "sequential",
                 ablation = "None"
    ):
        """
        ablation1: no Pseudo Label -> Backbone
        ablation2: no ICC -> TTC
        ablation3: no TCN -> CAE
        """    
        self.window = window
        self.shift = shift
        self.n_latent = n_latent
        self.n_clusters = n_clusters
        self.beta = beta
        self.maxiters = maxiters
        self.batch_size = batch_size
        self.cluster_reassignment = cluster_reassignment
        self.epochs = epochs
        self.threshold = threshold
        self.initializer = initializer
        self.ablation = ablation
        self.pretrain_epoch = pretrain_epoch
        self.random_state = random_state
        self.results = {}
        
        self.optimizers = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.loss_object_rec = tf.keras.losses.MeanSquaredError()
        self.loss_object_clf = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_object_tc = tf.keras.losses.MeanSquaredError()
        
        self.backbone = Backbone(window = window, shift = shift, n_clusters = n_clusters, n_latent = n_latent, n_hidden = n_hidden,  n_filters = n_filters, kernel_size = kernel_size, padding = padding, n_dilation = n_dilation, dilation_exp = dilation_exp, dropout_rate = dropout_rate, ablation = ablation)
        
        self.clustering = TICC(number_of_clusters = n_clusters, lambda_parameter = 0.2, beta = beta, maxIters = maxiters, cluster_reassignment = cluster_reassignment, random_state = random_state)
        
        if ablation != "None" and ablation == "noICC":
            self.clustering = GaussianMixture(n_components = n_clusters, covariance_type = "full")
            #self.clustering = KMeans(n_clusters = n_clusters)
    def initialize_y(self, data_X):
        n_ts, n_dim = data_X.shape
        
        if self.initializer == "sequential":
            remainder = n_ts % self.n_clusters
            data_y_hat = np.repeat(np.arange(self.n_clusters), n_ts // self.n_clusters)
            data_y_hat = np.concatenate([np.repeat(data_y_hat[0], remainder), data_y_hat])

        elif self.initializer == "repeatedsequence":
            data_y_hat_ = np.repeat(np.arange(self.n_clusters), self.cluster_reassignment)
            data_y_hat = np.tile(data_y_hat_, n_ts//len(data_y_hat_)+1 )
            data_y_hat = data_y_hat[:n_ts]
            
        elif self.initializer == "random":
            data_y_hat = np.random.randint(0, self.n_clusters, size = n_ts)

        elif self.initializer == "gmm":
            gmm = GaussianMixture(self.n_clusters, covariance_type = "diag")
            gmm.fit(data_X)
            data_y_hat = gmm.predict(data_X)
        return data_y_hat
    
    def fit(self, data_X, data_y = None):

        n_ts, n_dim = data_X.shape
        
        data_y_hat = self.initialize_y(data_X)
        
        if self.ablation == "noPseudo":
            self.epochs = self.pretrain_epoch
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            train_X, train_y = create_assigning_dataset(inputs = data_X, labels = data_y_hat, window = self.window, shift = self.shift)
            batch = create_batch(train_X, train_y, batch_size = self.batch_size, remaining = True)
            total_loss_rec = 0
            total_loss_clf = 0
            
            for iter, (data, label) in enumerate(batch):
                iter += 1
                with tf.GradientTape() as tape:
                    if (self.ablation == "None" and epoch >= self.pretrain_epoch) or self.ablation == "noTCN" or self.ablation =="noICC" :
                        # z, recon, pred
                        backbone_results = self.backbone(data, training = True)
                        loss_rec = self.loss_object_rec(data, backbone_results[1])
                        loss_clf = self.loss_object_clf(label, backbone_results[2])

                        loss = loss_rec * 0.7 + loss_clf * 0.3
                        
                    elif (self.ablation == "None" and epoch < self.pretrain_epoch) or self.ablation != "None" and self.ablation == "noPseudo":
                        backbone_results = self.backbone(data, training = True)
                        if self.ablation != "noPseudo":
                            self.backbone.ce.trainable = False
                        loss_rec = self.loss_object_rec(data, backbone_results[1])
                        loss = loss_rec
                        
                grads = tape.gradient(loss, self.backbone.trainable_weights)
                self.optimizers.apply_gradients(zip(grads, self.backbone.trainable_weights))

                total_loss_rec += loss_rec.numpy()
                
                if self.ablation != "noPseudo" and epoch >= self.pretrain_epoch:
                    total_loss_clf += loss_clf.numpy()

            if (self.ablation == "None" and epoch > 1) or self.ablation == "noTCN" or self.ablation =="noICC" :
                if (epoch >= self.pretrain_epoch-1 and epoch % 2 == 0) or epoch == self.epochs:
                    old_data_y_hat = data_y_hat.copy()
                    data_y_hat = self.updateClusters(data_X, data_y_hat, data_y)
                    print_text("Changed total {:.3f}.".format((1 - np.equal(old_data_y_hat, data_y_hat).sum()/n_ts)))
                if epoch > 10 and \
                   (1 - np.equal(old_data_y_hat, data_y_hat).sum()/n_ts) < self.threshold:
                    print_text("CONVERGED!!!")
                    if self.ablation != "noICC":
                        self.clustering = TICC(number_of_clusters = self.n_clusters, lambda_parameter = 0.11, beta = self.beta, maxIters = 10, cluster_reassignment = self.cluster_reassignment, random_state = self.random_state)
                    data_y_hat = self.updateClusters(data_X, data_y_hat, data_y)
                    self.saveResults()
                        
                    return data_y_hat
                    break
                
        if self.ablation != "None" and self.ablation == "noPseudo":
            data_y_hat  = self.updateClusters(data_X, data_y_hat, data_y)

        self.saveResults()
        return data_y_hat

    def saveResults(self):
        self.results["CAE"] = self.backbone.cae.results
        self.results["PCN"] = self.backbone.cae.pattern_learning.results
        return self.results
        
    def updateClusters(self, data_X, data_y_hat, data_y = None):
        n_ts, n_dim = data_X.shape
        test_X = create_assigning_dataset(inputs = data_X, labels = None, window = self.window, shift = self.window)
        
        backbone_results = self.backbone(test_X, training = False)
        z = backbone_results[0]
        z = z.numpy().reshape(-1, z.shape[-1])

        remainder = n_ts % self.window
        z = np.concatenate([z[:-self.window], z[-remainder:]])

        if self.ablation == "None" or self.ablation == "noTCN" or self.ablation == "noPseudo":
            y_hat, cluster_MRFs, clustered_loss, computed_covariance = self.clustering.fit(z, data_y_hat)
            
        elif self.ablation != "None" and self.ablation == "noICC":
            #if self.epoch > self.pretrain_epoch+1:
                #prev_weights = self.clustering.weights_
                #prev_means = self.clustering.means_
                #prev_precisions = self.clustering.precisions_
                #self.clustering = GaussianMixture(n_components = self.n_clusters,
                #                                  covariance_type = "full",
                #                                  weights_init = prev_weights,
                #                                  means_init = prev_means,
                #                                  precisions_init = prev_precisions
                #)
                
                prev_centroids = self.clustering.cluster_centers_
                self.clustering = KMeans(n_clusters = self.n_clusters,
                                         init = prev_centroids)
            self.clustering.fit(z)
        
            y_hat = self.clustering.labels_
            #y_hat = self.clustering.predict(z)
        
        length = np.unique(y_hat, return_counts = True)
        print_text(length)
        if data_y is not None:
            print_text("# EPOCH {} | {}".format(self.epoch, print_metrics(data_y, y_hat)))

        #-------------------
        # Implicit Ordering
        #-------------------
        data_y_hat = y_hat.copy()
        unordered = np.unique(data_y_hat, return_index = True)
        order_dict = {k:v for k, v in zip(unordered[0], unordered[1])}
        order_dict = dict(sorted(order_dict.items(), key = lambda x: x[1]))
        implicit_order = {v:k for k, v in enumerate(list(order_dict.keys()))}
        for v, k in implicit_order.items():
            data_y_hat[y_hat == v] = k     
        return data_y_hat

    def get_latent(self, data_X):
        n_ts, n_dim = data_X.shape
        test_X = create_assigning_dataset(inputs = data_X, labels = None, window = self.window, shift = self.window)
        backbone_results =  self.backbone(test_X, training = False)
        z = backbone_results[0]
        z = z.numpy().reshape(-1, z.shape[-1])

        remainder = n_ts % self.window
        z = np.concatenate([z[:-self.window], z[-remainder:]])

        return z
    
    def predict(self, data_X):
        z = self.get_latent(data_X)
        
        if self.ablation == "None" or self.ablation != "noICC":
            y_hat, _ = self.clustering.predict_clusters(z)
        elif self.ablation != "None" and self.ablation == "noICC":
            y_hat = self.clustering.predict(z)
            
        return y_hat
    
    def score(self, data_X, data_y, returns = False):
        y_hat = self.predict(data_X)
        print_text("# FINAL SCORE @ EPOCH {} | {}".format(self.epoch, print_metrics(data_y, y_hat)))
        if returns:
            return metrics(data_y, y_hat)
