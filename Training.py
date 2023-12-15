# %%
import time
import numpy as np
import torch
from torch.distributions.independent import Independent
import torch.nn as nn
from core.data_generation import GaussianHMM, construct_HMM_matrices
from tqdm import tqdm
import core.nonamortised_models as models
import core.utils as utils
import math
import subprocess
import hydra
import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import time
from torch.distributions import Independent, Normal, MultivariateNormal


def main():

    seed = np.random.randint(0, 9999999) 
    print("seed", seed)
    if not NOTEBOOK_MODE:
        with open('seed.txt', 'w') as f:
            f.write(str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

    saved_models_folder_name = 'saved_models'
    os.mkdir(saved_models_folder_name)


    # ------------------- Construct data -----------------------



    DIM = 1

    if cfg.data.path_to_data is None:
        F, G, U, V = construct_HMM_matrices(dim=DIM,
                                            F_eigvals=np.random.uniform(
                                                cfg.data.F_min_eigval,
                                                cfg.data.F_max_eigval, (DIM)),
                                            G_eigvals=np.random.uniform(
                                                cfg.data.G_min_eigval,
                                                cfg.data.G_max_eigval, (DIM)),
                                            U_std=cfg.data.U_std,
                                            V_std=cfg.data.V_std,
                                            diag=cfg.data.diagFG)

        data_gen = GaussianHMM(xdim=DIM, ydim=DIM, F=F, G=G, U=U, V=V)
        x_np, y_np = data_gen.generate_data(cfg.data.num_data)

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

    kalman_filter = models.KalmanFilter(x_0=kalman_xs[0, :], P_0=kalman_Ps[0, :, :], F=F, G=G, U=U,
                                        V=V)

    for t in range(1, y_np.shape[0]):
        kalman_filter.update(y_np[t, :])
        kalman_xs[t, :] = kalman_filter.x
        kalman_Ps[t, :, :] = kalman_filter.P
    kalman_xs_pyt = torch.from_numpy(kalman_xs).float()
    kalman_Ps_pyt = torch.from_numpy(kalman_Ps).float()


    F_init, G_init, _, _ = construct_HMM_matrices(dim=DIM,
                                            F_eigvals=np.random.uniform(
                                                cfg.data.F_min_eigval,
                                                cfg.data.F_max_eigval, (DIM)),
                                            G_eigvals=np.random.uniform(
                                                cfg.data.G_min_eigval,
                                                cfg.data.G_max_eigval, (DIM)),
                                            U_std=cfg.data.U_std,
                                            V_std=cfg.data.V_std,
                                            diag=cfg.data.diagFG)

    y = torch.from_numpy(y_np).float().to(device)

    F = torch.from_numpy(F).float().to(device)
    G = torch.from_numpy(G).float().to(device)

    F_init = torch.from_numpy(F_init).float().to(device)
    G_init = torch.from_numpy(G_init).float().to(device)

    U = torch.from_numpy(U).float().to(device)
    V = torch.from_numpy(V).float().to(device)

    mean_0 = torch.zeros(DIM).to(device)
    std_0 = torch.sqrt(torch.diag(U))


    # --------------------- Construct F and G function ------------------



    class F_Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('weight',
                nn.Parameter(torch.zeros(DIM)))
            self.F_mean_fn = lambda x, t: self.weight * x
            self.F_cov_fn = lambda x, t: U
            self.F_cov = U

        def forward(self, x, t=None):
            return Independent(Normal(self.F_mean_fn(x, t),
                torch.sqrt(torch.diag(U))), 1)

    class G_Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('weight',
                nn.Parameter(torch.zeros(DIM)))
            self.G_mean_fn = lambda x, t: self.weight * x
            self.G_cov = V

        def forward(self, x, t=None):
            return Independent(Normal(self.G_mean_fn(x, t),
                torch.sqrt(torch.diag(V))), 1)

    class p_0_dist_module(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean_0 = mean_0
            self.cov_0 = torch.eye(DIM, device=device) * std_0 ** 2

        def forward(self):
            return Independent(Normal(mean_0, std_0), 1)

    F_fn = F_Module().to(device)
    G_fn = G_Module().to(device)
    p_0_dist = p_0_dist_module().to(device)

    print("Learning G")
    G_fn.weight.data = torch.diag(G_init).data

    print("Learning F")
    F_fn.weight.data = torch.diag(F_init).data

    G_theta_dim = sum([p.numel() for p in G_fn.parameters()])
    F_theta_dim = sum([p.numel() for p in F_fn.parameters()])
    print("G theta dim", G_theta_dim)
    print("F theta dim", F_theta_dim)
    theta_dim = F_theta_dim + G_theta_dim
    print("Theta dim", theta_dim)

    def get_model_parameters():
        return [*F_fn.parameters(), *G_fn.parameters()]


    # ------------------- Create phi model ------------------------




    def cond_q_mean_net_constructor():
        return torch.nn.Linear(DIM, DIM).to(device)

    
    print("Using analytic phi updates")
    phi_model = models.NonAmortizedModelBase(
        device, DIM, DIM, torch.zeros(DIM, device=device),
        torch.zeros(DIM, device=device), cond_q_mean_net_constructor,
        torch.zeros(DIM, device=device), F_fn, G_fn, p_0_dist,
        'last', 1, cfg.theta_training.window_size + 1
    )

    # ------------------ Create theta model ---------------------


    matrices_to_learn = cfg.theta_training.matrices_to_learn
    def add_theta_grads_to_params(grads):
        phi_model.F_fn.weight.grad += grads[:int(theta_dim/2)]
        phi_model.G_fn.weight.grad += grads[int(theta_dim/2):]

    if cfg.theta_training.func_type == 'neural_net':
        print("Learning theta grad with neural nets")
        h = cfg.theta_training.net_hidden_dim

        def theta_func_constructor():
            nnlist = [nn.Linear(DIM, h), nn.ReLU()]
            for i in range(cfg.theta_training.net_num_hidden_layers-1):
                nnlist += [nn.Linear(h, h), nn.ReLU()]
            nnlist += [nn.Linear(h, theta_dim)]

            return models.NNFuncEstimator(
                nn.Sequential(*nnlist), DIM, theta_dim
            ).to(device)
    
    theta_optim = torch.optim.Adam(get_model_parameters(),
        lr=cfg.theta_training.theta_lr)
    if cfg.theta_training.theta_lr_decay_type == 'exponential':
        theta_decay = torch.optim.lr_scheduler.StepLR(theta_optim,
            step_size=1, gamma=np.exp(
                (1/cfg.theta_training.num_steps_theta_lr_oom_drop) * np.log(0.1)))
    elif cfg.theta_training.theta_lr_decay_type == 'robbins-monro':
        lr_decay_rate = cfg.theta_training.robbins_monro_theta_lr_decay_rate
        lr_decay_bias = cfg.theta_training.robbins_monro_theta_lr_decay_bias
        theta_decay = torch.optim.lr_scheduler.LambdaLR(theta_optim,
            lr_lambda=lambda epoch: lr_decay_bias / (lr_decay_bias + epoch ** lr_decay_rate))
    else:
        raise NotImplementedError

    rmle = models.LinearRMLEDiagFG(np.zeros((DIM,1)), np.eye(DIM),
        F_init.detach().cpu().numpy().copy() if 'F' in cfg.theta_training.matrices_to_learn else F.detach().cpu().numpy().copy(),
        G_init.detach().cpu().numpy().copy() if 'G' in cfg.theta_training.matrices_to_learn else G.detach().cpu().numpy().copy(),
        U.cpu().detach().numpy().copy(), V.cpu().detach().numpy().copy(),
        cfg.theta_training.theta_lr,
        cfg.theta_training.matrices_to_learn)


    # ------------- Utils functions --------------------



    def estimate_joint_kl(model, num_samples, kalman_mean_T, kalman_cov_T,
                        kalman_mean_Tm1, kalman_cov_Tm1, true_F, true_U):
        """
            Estimates KL (q(x_{t-1}, x_t | y_{1:t}) || p(x_{t-1}, x_t | y_{1:t}))
            using num_samples
        """
        with torch.no_grad():
            x_samples, all_q_stats = model.sample_joint_q_t(num_samples, 1)
            x_Tm1, x_T = x_samples
            log_q_t = model.compute_log_q_t(x_T, *all_q_stats[1])
            log_q_t_1 = model.compute_log_q_t(x_Tm1, *all_q_stats[0])

            log_p_x = utils.back_1_joint_smoothing_prob(x_T, x_Tm1,
                                                        kalman_mean_T, kalman_cov_T,
                                                        kalman_mean_Tm1, kalman_cov_Tm1,
                                                        true_F, true_U)

            return torch.mean(log_q_t + log_q_t_1 - log_p_x)

    def analytic_kalman_phi_update(model, T, G, F, U, V, y_T):
        prev_mean = model.q_t_mean_list[T-1].reshape(model.xdim, 1)
        prev_cov = torch.diag(torch.exp(2*model.q_t_log_std_list[T-1]))
        y_T = y_T.reshape(model.ydim, 1)

        xp = F @ prev_mean
        Pp = F @ prev_cov @ F.T + U
        S = G @ Pp @ G.T + V
        K = Pp @ G.T @ torch.inverse(S)
        z = y_T - G @ xp

        new_mean = xp + K @ z
        new_cov = (torch.eye(model.xdim).to(device) - K @ G) @ Pp

        cond_cov = torch.inverse(torch.inverse(prev_cov) + \
            F.T @ torch.inverse(U) @ F) 
        cond_weight = cond_cov @ F.T @ torch.inverse(U)
        cond_bias = cond_cov @ torch.inverse(prev_cov) @ prev_mean

        model.q_t_mean_list[T].data = new_mean.reshape(model.xdim)
        model.q_t_log_std_list[T].data = 0.5 * torch.log(torch.diag(new_cov))
        model.cond_q_t_mean_net_list[T].weight.data = cond_weight
        model.cond_q_t_mean_net_list[T].bias.data = cond_bias.reshape(model.xdim)
        model.cond_q_t_log_std_list[T].data = 0.5 * torch.log(torch.diag(cond_cov))



    # ------------------ Start training ----------------------



    Gs = []
    Fs = []
    rmle_Gs = []
    rmle_Fs = []
    joint_kls = []
    theta_func_losses = []
    times = []
    filter_means = []
    filter_stds = []

    pbar = tqdm(range(0, cfg.data.num_data))

    for T in pbar:
        start_time = time.time()

        # ---------- Advance timesteps --------------

        phi_model.advance_timestep(y[T, :])
        theta_grad.advance_timestep()


        # ----------- Phi optimization ----------------



        
        tmpG = torch.diag(phi_model.G_fn.weight.data.clone())
        tmpF = torch.diag(phi_model.F_fn.weight.data.clone())
        analytic_kalman_phi_update(phi_model, T, tmpG, tmpF, U, V, y[T,:])


        # -------------- Theta func training ----------------


        if T >= cfg.theta_training.window_size:
            
            theta_func_optim = torch.optim.Adam(
                theta_grad.get_theta_func_TmL_parameters(),
                lr=cfg.theta_training.net_lr)
            net_inputs, net_targets = theta_grad.generate_training_dataset(
                cfg.theta_training.net_dataset_size, y
            )
            net_inputs = net_inputs.detach()
            net_targets = net_targets.detach()

            theta_grad.theta_func_TmL.update_normalization(
                net_inputs, net_targets, cfg.theta_training.net_norm_decay
            )
            for i in range(cfg.theta_training.net_iters):
                idx = np.random.choice(np.arange(net_inputs.shape[0]),
                    (cfg.theta_training.net_minibatch_size,), replace=False)
                theta_func_optim.zero_grad()
                preds = theta_grad.theta_func_TmL(net_inputs[idx,:])
                loss = torch.mean(
                    torch.sum((preds - net_targets[idx, :])**2, dim=1)
                )
                loss.backward()
                theta_func_optim.step()
                theta_func_losses.append(loss.item())


        # ---------------- Theta update ----------------


        if T > cfg.theta_training.theta_updates_start_T:
            theta_optim.zero_grad() 
            theta_grad.populate_theta_grads(
                cfg.theta_training.theta_minibatch_size, y)
            theta_optim.step()
            Gs.append(G_fn.weight.clone().detach().numpy())
            Fs.append(F_fn.weight.clone().detach().numpy())

            pbar.set_postfix({"F MAE": np.mean(np.abs(Fs[-1] - np.diag(np.array(F)))),
                              "G MAE": np.mean(np.abs(Gs[-1] - np.diag(np.array(G))))})

            rmle.step_size = theta_decay.state_dict()['_last_lr'][0]
            rmle.advance_timestep(y[T, :].detach().numpy().copy().reshape((DIM,1)))
            rmle_Gs.append(rmle.G.copy())
            rmle_Fs.append(rmle.F.copy())

            theta_decay.step()


        # -------------- Logging --------------------


        filter_means.append(phi_model.q_t_mean_list[T].detach().cpu().numpy())
        filter_stds.append(phi_model.q_t_log_std_list[T].detach().cpu().numpy())

        if T>0:
            joint_kls.append(estimate_joint_kl(phi_model, 256,
                        kalman_xs_pyt[T, :], kalman_Ps_pyt[T, :, :],
                        kalman_xs_pyt[T - 1, :], kalman_Ps_pyt[T - 1, :, :],
                        F, U).item())

        if (T % (round(max(cfg.data.num_data, cfg.theta_training.num_times_save_data)\
            / cfg.theta_training.num_times_save_data)) == 0) or\
            (T == cfg.data.num_data - 1):

            save_np('Gs.npy', np.array(Gs))
            save_np('Fs.npy', np.array(Fs))
            save_np('rmle_Gs.npy', np.array(rmle_Gs))
            save_np('rmle_Fs.npy', np.array(rmle_Fs))
            save_np('joint_kls.npy', np.array(joint_kls))
            save_np('theta_func_losses.npy', np.array(theta_func_losses))
            save_np('times.npy', np.array(times))
            save_np('filter_means.npy', np.array(filter_means))
            save_np('filter_stds.npy', np.array(filter_stds))
            if cfg.save_models:
                torch.save(phi_model.state_dict(), saved_models_folder_name + \
                    '/phi_model_{}.pt'.format(T))
                torch.save(theta_grad.theta_func_TmL.state_dict(),
                    saved_models_folder_name + '/theta_model_{}.pt'.format(T))
                torch.save(theta_optim.state_dict(), saved_models_folder_name +\
                    '/theta_optim_{}.pt'.format(T))
                torch.save(theta_decay.state_dict(), saved_models_folder_name +\
                    '/theta_decay_{}.pt'.format(T))

        times.append(time.time()-start_time)


    f, (ax1, ax2) = plt.subplots(1, 2)
    rmle_F_maes = np.mean(np.abs(np.diagonal(rmle_Fs, axis1=1, axis2=2) - np.diag(F)), 1)
    F_maes = np.mean(np.abs(Fs - np.diag(F)), 1)
    rmle_G_maes = np.mean(np.abs(np.diagonal(rmle_Gs, axis1=1, axis2=2) - np.diag(G)), 1)
    G_maes = np.mean(np.abs(Gs - np.diag(G)), 1)
    ax1.plot(rmle_F_maes)
    ax1.plot(F_maes)
    ax2.plot(rmle_G_maes)
    ax2.plot(G_maes)
    plt.show()

    print("F RMLE: ", rmle.F.copy())
    print("G RMLE: ", rmle.G.copy())
    print("F: ", Fs[-1])
    print("G: ", Gs[-1])

if __name__ == "__main__":
    main()