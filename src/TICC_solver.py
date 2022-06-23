import numpy as np
import math, time, collections, os, errno, sys, code, random, logging
import pandas as pd
from multiprocessing import Pool
import time
from utils import *

class TICC:
    def __init__(self, number_of_clusters=5, lambda_parameter=11e-2,
                 beta = 300, maxIters = 1000, num_proc = 1, cluster_reassignment = 40, biased = False, random_state = 102):
        """
        Parameters:
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - write_out_file: (bool) if true, prefix_string is output file dir
            - prefix_string: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
            - biased: Using the biased or the unbiased covariance
        """
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.num_proc = num_proc
        self.cluster_reassignment = cluster_reassignment
        self.biased = biased
        
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(random_state)

    def fit(self, complete_D_train, clustered_points):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
#        self.log_parameters()

        # Get data into proper format
        time_series_rows_size, time_series_col_size = complete_D_train.shape

        num_train_points = len(complete_D_train)
        print_text("Data shape: {}".format(complete_D_train.shape))

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = clustered_points  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes = self.num_proc)  # multi-threading
        
        for iters in range(self.maxIters):
#            print_text("ITERATION ### {}".format(iters))
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            st = time.time()
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, time_series_col_size, pool,
                                          train_clusters_arr)
            ed = time.time()

            elapsed_time = "1 "+ f"{ed - st:.5f} sec"
#            print_text(elapsed_time)

            st = time.time()
            try:
                self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                    train_cluster_inverse)
            except:
                return clustered_points, train_cluster_inverse
            
            ed = time.time()

            elapsed_time = "2 " + f"{ed - st:.5f} sec"
#            print_text(elapsed_time)

            # update old computed covariance
            old_computed_covariance = computed_covariance
#            print_text("3")

#            print_text("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                 'computed_covariance': computed_covariance,
                                  'empirical_covariances': empirical_covariances,
                                 'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                 'complete_D_train': complete_D_train,
                                 'time_series_col_size': time_series_col_size}
            clustered_points, clustered_loss = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(list) # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()



            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)
                        #print_text("cluster that is zero is: {}, selected cluster instead is: {}".format(cluster_num, cluster_selected))
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[cluster_num] = old_computed_covariance[cluster_selected]
                            cluster_mean_stacked_info[cluster_num] = complete_D_train[
                                                                                              point_to_move, :]
                            cluster_mean_info[cluster_num] \
                                = complete_D_train[point_to_move, :][:time_series_col_size]

#            for cluster_num in range(self.number_of_clusters):
#                print_text("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))

            if np.array_equal(old_clustered_points, clustered_points):
                print_text("\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training

        if pool is not None:
            pool.close()
            pool.join()
        
        return clustered_points, train_cluster_inverse, clustered_loss, computed_covariance

    def smoothen_clusters(self, cluster_mean_info, computed_covariance,
                          cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        for cluster in range(self.number_of_clusters):
            cov_matrix = computed_covariance[cluster][:n,:n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
#        print_text("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([clustered_points_len, self.number_of_clusters])
        for point in range(clustered_points_len):
            if point < complete_D_train.shape[0]:
                for cluster in range(self.number_of_clusters):
                    cluster_mean = cluster_mean_info[cluster]
                    cluster_mean_stacked = cluster_mean_stacked_info[cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, n]),
                                 np.dot(inv_cov_matrix, x.reshape([n, 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(self, computed_covariance, len_train_clusters, log_det_values, optRes, train_cluster_inverse):
        """
        self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                               train_cluster_inverse)
        S = np.cov(np.transpose(D_train), bias=self.biased)
        empirical_covariances[cluster] = S

        rho = 1
        solver = ADMMSolver(lamb, self.window_size, size_blocks, 1, S)
        # apply to process pool
        optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
        """
        for cluster in range(self.number_of_clusters):
            # print_text(cluster)
            if optRes[cluster] == None:
                continue
            # print_text(optRes[cluster])
            val = optRes[cluster].get()
            # print_text("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
            # THIS IS THE SOLUTION
            # print_text("VAL",sum(np.isnan(val)))
            # print_text("NAN val",sum(sum(np.isnan(val))))
            S_est = upperToFull(val, 0)
            X2 = S_est
            # print_text("NAN",sum(sum(np.isnan(S_est))))
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[cluster] = cov_out
            train_cluster_inverse[cluster] = X2
        
    def train_clusters(self, cluster_mean_info, cluster_mean_stacked_info, complete_D_train, empirical_covariances,
                       len_train_clusters, n, pool, train_clusters_arr):
        """
        opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                            empirical_covariances, len_train_clusters, time_series_col_size, pool,
                            train_clusters_arr)
        """
        optRes = [None for i in range(self.number_of_clusters)]
        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                size_blocks = n #column size
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_info[cluster] = np.mean(D_train, axis=0)[:n].reshape([1, n])
                cluster_mean_stacked_info[cluster] = np.mean(D_train, axis=0)
                ##Fit a model - OPTIMIZATION
                probSize = size_blocks
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                S = np.cov(np.transpose(D_train), bias=self.biased)
                empirical_covariances[cluster] = S

                rho = 1
                solver = ADMMSolver(lamb, size_blocks, 1, S)
                # apply to process pool
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
        return optRes

    def log_parameters(self):
        print_text("lam_sparse {}".format(self.lambda_parameter))
        print_text("switch_penalty {}".format(self.switch_penalty))
        print_text("num_cluster {}".format(self.number_of_clusters))

    def predict_clusters(self, test_data = None):
        '''
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model['complete_D_train']

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['cluster_mean_info'],
                                                         self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         test_data,
                                                         self.trained_model['time_series_col_size'])

        # Update cluster points - using NEW smoothening
        clustered_points, clustered_loss = updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)
        return clustered_points, clustered_loss

#--------------------------------------------------------------------------------------------------------#
def upperToFull(a, eps=0):
        ind = (a < eps) & (a > -eps)
        a[ind] = 0
        n = int((-1 + np.sqrt(1 + 8*a.shape[0]))/2)
        A = np.zeros([n, n])
        A[np.triu_indices(n)] = a
        temp = A.diagonal()
        A = np.asarray((A + A.T) - np.diag(temp))
        return A
def updateClusters(LLE_node_vals, switch_penalty=1):
    """
    Takes in LLE_node_vals matrix and computes the path that minimizes
    the total cost over the path
    Note the LLE's are negative of the true LLE's actually!!!!!
    Note: switch penalty > 0
    """
    (T, num_clusters) = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)

    # compute future costs
    for i in range(T-2, -1, -1):
        j = i+1
        indicator = np.zeros(num_clusters)
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        for cluster in range(num_clusters):
            total_vals = future_costs + lle_vals + switch_penalty
            total_vals[cluster] -= switch_penalty
            future_cost_vals[i, cluster] = np.min(total_vals)

    # compute the best path
    path = np.zeros(T)

    # the first location
    loss = [min(future_cost_vals[0, :] + LLE_node_vals[0, :])]

    curr_location = np.argmin(future_cost_vals[0, :] + LLE_node_vals[0, :])
    path[0] = curr_location

    # compute the path
    for i in range(T-1):
        j = i+1
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        total_vals = future_costs + lle_vals + switch_penalty
        total_vals[int(path[i])] -= switch_penalty
        
        loss.append(min(total_vals))
        path[i+1] = np.argmin(total_vals)

    # return the computed path
    return path, np.array(loss)

def find_matching(confusion_matrix):
    """
    returns the perfect matching
    """
    _, n = confusion_matrix.shape
    path = []
    for i in range(n):
        max_val = -1e10
        max_ind = -1
        for j in range(n):
            if j in path:
                pass
            else:
                temp = confusion_matrix[i, j]
                if temp > max_val:
                    max_val = temp
                    max_ind = j
        path.append(max_ind)
    return path

#--------------------------------------------------------------------------------------------------------#
import numpy
import math
class ADMMSolver:
    def __init__(self, lamb, size_blocks, rho, S, rho_update_func=None):
        self.lamb = lamb
        self.numBlocks = 1
        self.sizeBlocks = size_blocks
        probSize = size_blocks
        self.length = int(probSize*(probSize+1)/2)
        self.x = numpy.zeros(self.length)
        self.z = numpy.zeros(self.length)
        self.u = numpy.zeros(self.length)
        self.rho = float(rho)
        self.S = S
        self.status = 'initialized'
        self.rho_update_func = rho_update_func

    def ij2symmetric(self, i,j,size):
        return (size * (size + 1))/2 - (size-i)*((size - i + 1))/2 + j - i

    def upper2Full(self, a):
        n = int((-1  + numpy.sqrt(1+ 8*a.shape[0]))/2)  
        A = numpy.zeros([n,n])
        A[numpy.triu_indices(n)] = a 
        temp = A.diagonal()
        A = (A + A.T) - numpy.diag(temp)             
        return A 

    def Prox_logdet(self, S, A, eta):
        d, q = numpy.linalg.eigh(eta*A-S)
        q = numpy.matrix(q)
        X_var = ( 1/(2*float(eta)) )*q*( numpy.diag(d + numpy.sqrt(numpy.square(d) + (4*eta)*numpy.ones(d.shape))) )*q.T
        x_var = X_var[numpy.triu_indices(S.shape[1])] # extract upper triangular part as update variable      
        return numpy.matrix(x_var).T

    def ADMM_x(self):    
        a = self.z-self.u
        A = self.upper2Full(a)
        eta = self.rho
        x_update = self.Prox_logdet(self.S, A, eta)
        self.x = numpy.array(x_update).T.reshape(-1)

    def ADMM_z(self, index_penalty = 1):
        a = self.x + self.u
        probSize = self.numBlocks*self.sizeBlocks
        z_update = numpy.zeros(self.length)

        # TODO: can we parallelize these?
        for i in range(self.numBlocks):
            elems = self.numBlocks if i==0 else (2*self.numBlocks - 2*i)/2 # i=0 is diagonal
            for j in range(self.sizeBlocks):
                startPoint = j if i==0 else 0
                for k in range(startPoint, self.sizeBlocks):
                    locList = [((l+i)*self.sizeBlocks + j, l*self.sizeBlocks+k) for l in range(int(elems))]
                    if i == 0:
                        lamSum = sum(self.lamb[loc1, loc2] for (loc1, loc2) in locList)
                        indices = [self.ij2symmetric(loc1, loc2, probSize) for (loc1, loc2) in locList]
                    else:
                        lamSum = sum(self.lamb[loc2, loc1] for (loc1, loc2) in locList)
                        indices = [self.ij2symmetric(loc2, loc1, probSize) for (loc1, loc2) in locList]
                    pointSum = sum(a[int(index)] for index in indices)
                    rhoPointSum = self.rho * pointSum

                    #Calculate soft threshold
                    ans = 0
                    #If answer is positive
                    if rhoPointSum > lamSum:
                        ans = max((rhoPointSum - lamSum)/(self.rho*elems),0)
                    elif rhoPointSum < -1*lamSum:
                        ans = min((rhoPointSum + lamSum)/(self.rho*elems),0)

                    for index in indices:
                        z_update[int(index)] = ans
        self.z = z_update

    def ADMM_u(self):
        u_update = self.u + self.x - self.z
        self.u = u_update

    # Returns True if convergence criteria have been satisfied
    # eps_abs = eps_rel = 0.01
    # r = x - z
    # s = rho * (z - z_old)
    # e_pri = sqrt(length) * e_abs + e_rel * max(||x||, ||z||)
    # e_dual = sqrt(length) * e_abs + e_rel * ||rho * u||
    # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
    # Returns (boolean shouldStop, primal residual value, primal threshold,
    #          dual residual value, dual threshold)
    def CheckConvergence(self, z_old, e_abs, e_rel, verbose):
        norm = numpy.linalg.norm
        r = self.x - self.z
        s = self.rho * (self.z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = math.sqrt(self.length) * e_abs + e_rel * max(norm(self.x), norm(self.z)) + .0001
        e_dual = math.sqrt(self.length) * e_abs + e_rel * norm(self.rho * self.u) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Debugging information to print(convergence criteria values)
            print('  r:', res_pri)
            print('  e_pri:', e_pri)
            print('  s:', res_dual)
            print('  e_dual:', e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    #solve
    def __call__(self, maxIters, eps_abs, eps_rel, verbose):
        num_iterations = 0
        self.status = 'Incomplete: max iterations reached'
        for i in range(maxIters):
            z_old = numpy.copy(self.z)
            self.ADMM_x()
            self.ADMM_z()
            self.ADMM_u()
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(z_old, eps_abs, eps_rel, verbose)
                if stop:
                    self.status = 'Optimal'
                    break
                new_rho = self.rho
                if self.rho_update_func:
                    new_rho = rho_update_func(self.rho, res_pri, e_pri, res_dual, e_dual)
                scale = self.rho / new_rho
                rho = new_rho
                self.u = scale*self.u
            if verbose:
                # Debugging information prints current iteration #
                print('Iteration %d' % i)
        return self.x
