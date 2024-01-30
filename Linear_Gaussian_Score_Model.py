# %%
import time
import numpy as np
import torch
from torch.distributions.independent import Independent
import torch.nn as nn
import Data_Generation
from Data_Generation import Linear_Gaussian_HMM, HMM_matrices
from tqdm import tqdm
import KFModels as kf_models
import core.utils as utils
import math
import subprocess
import hydra
import os
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import time
from torch.distributions import Independent, Normal, MultivariateNormal

NOTEBOOK_MODE = False

def save_np(name, x):
    if not NOTEBOOK_MODE:
        np.save(name, x)

@hydra.main(version_base=None, config_path="configurations", config_name="conf1")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

@hydra.main(config_path='configurations', config_name="conf1")
def main(cfg):
    #if not NOTEBOOK_MODE:
        #utils.save_git_hash(hydra.utils.get_original_cwd())
    device = cfg.device

    seed = np.random.randint(0, 9999999) if cfg.seed is None else cfg.seed
    print("seed", seed)
    if not NOTEBOOK_MODE:
        with open('seed.txt', 'w') as f:
            f.write(str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

    saved_models_folder_name = 'saved_models'
    if cfg.save_models and not NOTEBOOK_MODE:
        os.mkdir(saved_models_folder_name)


    # ------------------- Construct data -----------------------


    DIM = cfg.data.dim

    if not cfg.data.diagFG:
        raise NotImplementedError

    if cfg.data.path_to_data is None:
        F, G, U, V = HMM_matrices(
                                x_dim=DIM,y_dim=DIM,
                                F_eigenvals=np.random.uniform(
                                    cfg.data.F_min_eigval,
                                    cfg.data.F_max_eigval, (DIM)),
                                G_eigenvals=np.random.uniform(
                                    cfg.data.G_min_eigval,
                                    cfg.data.G_max_eigval, (DIM)),
                                U_std=cfg.data.U_std,
                                V_std=cfg.data.V_std,
                                diag=cfg.data.diagFG)

        data_gen = Linear_Gaussian_HMM(T=cfg.data.num_data, x_dim=DIM, y_dim=DIM, F=F, G=G, U=U, V=V)
        x_np, y_np = data_gen.generate_data()

        save_np('datapoints.npy', np.stack((x_np, y_np)))
        save_np('F.npy', F)
        save_np('G.npy', G)
        save_np('U.npy', U)
        save_np('V.npy', V)
    else:
        path_to_data = hydra.utils.to_absolute_path(cfg.data.path_to_data) + '/'
        F, G, U, V = np.load(path_to_data + 'F.npy'), \
                    np.load(path_to_data + 'G.npy'), \
                    np.load(path_to_data + 'U.npy'), \
                    np.load(path_to_data + 'V.npy')
        xystack = np.load(path_to_data + 'datapoints.npy')
        x_np = xystack[0, :, :]
        y_np = xystack[1, :, :]

    print("True F: ", F)
    print("True G: ", G)

    kalman_xs = np.zeros((y_np.shape[0], DIM))
    kalman_Ps = np.zeros((y_np.shape[0], DIM, DIM))

    # For t=0
    kalman_Ps[0, :, :] = np.linalg.inv(np.eye(DIM) + G.T @ np.linalg.inv(V) @ G)
    kalman_xs[0, :] = kalman_Ps[0, :, :] @ G.T @ np.linalg.inv(V) @ y_np[0, :]

    kalman_filter = kf_models.KalmanFilter(x_0=kalman_xs[0, :], P_0=kalman_Ps[0, :, :], F=F, G=G, U=U,
                                        V=V)

    for t in range(1, y_np.shape[0]):
        kalman_filter.update(y_np[t, :])
        kalman_xs[t, :] = kalman_filter.x
        kalman_Ps[t, :, :] = kalman_filter.P
    kalman_xs_pyt = torch.from_numpy(kalman_xs).float()
    kalman_Ps_pyt = torch.from_numpy(kalman_Ps).float()

    print(np.shape(kalman_xs_pyt))
    print('\n')
    print(np.shape(kalman_Ps_pyt))

    ## initialization 

    

if __name__ == "__main__":
    #os.environ["HYDRA_FULL_ERROR"] = "1"
    directory = "/home/gpalmari/OnlineLearning"
    files = os.listdir(directory)
    for file in files:
        if ".sh.o" in file or ".sh.e" in file:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
    my_app()
    main()