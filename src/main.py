#!/home/hmin/anaconda3/envs/tensor-gpu/bin/python3.8
import argparse, pickle, logging
import numpy as np
import pandas as pd
from tf_gpu import set_gpu

from utils import *
from models import TTC


def main(args):

    # ----------------------------------------------- #1
    # Log setting
    # ----------------------------------------------- #
    logger = logging.getLogger() # 로그 생성
    logger.setLevel(logging.INFO) # 로그의 출력 기준 설정

    logFilename = "../log/{:s}_TTC_{:s}.log".format(args["dataset"], args["commit"])
    
    print_text("\nOPEN LOG FILE {}\n".format(logFilename)) 

    file_handler = logging.FileHandler(filename = logFilename, mode = "a",
                                       delay = False, encoding = None)
    logger.addHandler(file_handler)
    file_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(asctime)s - %(name)s -  - %(message)s'
    ))

    # ----------------------------------------------- #1
    # Parameter write
    # ----------------------------------------------- #
    print_text(args)
    
    # ----------------------------------------------- #1
    # GPU setting
    # ----------------------------------------------- #
    set_gpu(args["gpu_num"])
  
    # ----------------------------------------------- #
    # Dataset load
    # ----------------------------------------------- #
    data_X, data_y = load_dataset(args["dataset"])

    n_ts, n_dim = data_X.shape
    n_clusters = len(np.unique(data_y))
    args["n_clusters"] = n_clusters
    # ----------------------------------------------- #
    # Model setting
    # ----------------------------------------------- #
    
    model = TTC(n_clusters = args["n_clusters"],
            window = args["window"],
                shift = args["shift"],
                n_latent = args["n_latent"],
                n_hidden = n_dim,
                epochs = args["epochs"],
                learning_rate = args["learning_rate"],
                maxiters = args["maxIters"],
                beta = args["beta"],
                random_state = args["random_state"],
                kernel_size = args["kernel_size"],
                padding = args["padding"],
                n_dilation = args["n_dilation"],
                dilation_exp = args["dilation_exp"],
                dropout_rate = args["dropout_rate"],
                threshold = args["threshold"],
                pretrain_epoch = args["pretrain_epoch"],
                initializer = args["initializer"],
                ablation = args["ablation"]
    )
    
    pred = model.fit(data_X, data_y)
    print_text("SAVE PREDICTION.")
    pickle.dump((pred, model.clustering), open("../outputs/{:s}_TTC_{:s}.pkl".format(args["dataset"], args["commit"]),"wb"))
    
    if args["saveResults"]:
        print_text("SAVE RESULTS.")
        z = model.get_latent(data_X)
        pickle.dump((z, model.saveResults()), open("../outputs/{:s}_TTC_results_{:s}.pkl".format(args["dataset"], args["commit"]), "wb"))
        
    if args["savefig"]:
        print_text("SAVE PREDICTION PLOT.")
        fig = plotResults(data_X, data_y, pred)
        fig.savefig("../img/{:s}_TTC_{:s}.png".format(args["dataset"], args["commit"]))


    results = np.array(model.score(data_X, data_y, returns = True))
    df = pd.DataFrame([results], columns = ["nmi","ami","ari","f1","acc"])
    print_text("SAVE PREDICTION RESULTS.")
    df.to_csv("../outputs/{:s}_TTC_{:s}.csv".format(args["dataset"], args["commit"]), index = False, mode = "a")

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    print("\nCLOSE LOG FILE {}\n".format(logFilename))
     
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="train", description='TTC')

    parser.add_argument('--commit', type=str, required = True)
    
    ## EXP
    parser.add_argument('--exp', type = int, default = 1)
    parser.add_argument('--savefig', type = bool, default = True)
    parser.add_argument('--saveResults', type = bool, default = True)

    ## TTC
    parser.add_argument('--gpu_num', type = int, required = True, choices = range(5))
    parser.add_argument('--dataset', type = str, required = True, default = "mehalth", choices = ["mhealth","pamap2","skoda"])
    parser.add_argument('--window', type = int, required = True, default = 128)
    parser.add_argument('--shift', type = int, required = True, default = 128)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--optimizer', type = str, default = "Adam")
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--maxIters', type = int, default = 2)
    parser.add_argument('--beta', type = int, default = 100)
    parser.add_argument('--cluster_reassignment', type = int, default = 300)
    parser.add_argument('--random_state', type = int, default = 52)
    parser.add_argument('--threshold', type = float, default = 0.01 )
    parser.add_argument('--pretrain_epoch', type = int, default = 11)
    parser.add_argument('--initializer', type = str, default = "sequential", choices = ["sequential", "random", "gmm"])
    parser.add_argument('--ablation', type = str, default = "None", choices = ["None", "noPseudo", "noICC", "noTCN"])
    ### CAE
    parser.add_argument('--n_filters', type = int, default = 8)
    parser.add_argument('--n_latent', type = int, default = 20)
    parser.add_argument('--kernel_size', type = int, default = 9)
    parser.add_argument('--padding', type = str, default = 'causal')
    parser.add_argument('--n_dilation', type = int, default = 8)
    parser.add_argument('--dropout_rate', type = float, default = 0.25)
    parser.add_argument('--dilation_exp', type = int, default = 4)
    parser.add_argument('--sampling_factor', type = int, default = 16)

    arguments = parser.parse_args()
    args = {}
    for arg in vars(arguments):
        args[arg] = getattr(arguments, arg)
    main(args)
    
