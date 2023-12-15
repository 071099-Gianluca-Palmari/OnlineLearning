from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import Utils as utils
from torch.distributions import MultivariateNormal, Independent, Normal, Categorical
from Utils import gaussian_posterior, sample_cov

class KalmanFilter():
    def __init__(self, x_0, P_0, F, G, U, V):
        self.x, self.P, self.F, self.G, self.U, self.V = \
            x_0, P_0, F, G, U, V

    def update(self, y):
        '''
        Update function of the Kalman Filter. 
        1-Compute xp=F*x. (initial esimation of x)
        2-Compute Pp=F*P*F^{T}+U (initial esimation of P)
        3-Compute S=G*Pp*G^{T}+V
        4-Compute K=Pp*G^{T}*S^{-1}
        5-Compute z=y-G*xp
        6-Compute x = xp + K*z (adapted estimation of x)
        7-Compute P = (I-K*G)*Pp (adapted estimation of P)
        '''
        xp = np.dot(self.F, self.x)
        Pp = np.matmul(np.matmul(self.F, self.P), np.transpose(self.F)) + self.U
        S = np.matmul(np.matmul(self.G, Pp), np.transpose(self.G)) + self.V
        K = np.matmul(np.matmul(Pp, np.transpose(self.G)), np.linalg.inv(S))
        z = y - np.dot(self.G, xp)

        self.x = xp + np.dot(K, z)
        self.P = np.matmul(np.eye(self.P.shape[0]) - np.matmul(K, self.G), Pp)

class NonAmortisedGaussianModels(nn.Module):
    '''
    NonAmortisedModels is a class that is meant to be a neural network model in the PyTorch framework. 
    You can define your neural network layers and operations within this class.
    It allows PyTorch to keep track of the network's parameters for optimization and other functionalities.
    '''

    '''
    Contains nonlinear models in the form
    x_{t+1} ~ F(x_{t})
    y_{t} ~ G(x_{t})
    F, G are nn.Modules which return a distribution and can potentially contain learnable theta parameters
    The q lists contain nonamortized variational posteriors
        q_t(x_t) = N(x_t; q_T_mean, diag(q_T_std)^2)
        q_t(x_{t-1} | x_t) = N(x_{t-1}; cond_q_t_mean_net(x_t), diag(cond_q_t_std)^2)
        These will both be factorized Gaussians.
    cond_q_t_mean_net: Linear, MLP, etc: (xdim) -> (xdim)
    '''
    def __init__(self, device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                 F_fn, G_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store=None):
        super().__init__()

        '''
        Initialize the online learning model. 
        1- Initialize the latent and the observable dimensions.
        2- Give the prior parameters for the variational smoothing distribution. Here everything is Gaussian. 
        3- Give the prior parameters for the cond variational smoothing distribution. Here everything is Gaussian.
        4- Define the window size.
        5- Give the previous mean and std that characterize the variational distrib and the cond variational distrib. 
        6- Define the functions F and G that define the generative model. 
        '''
        self.T = -1

        self.device = device
        self.xdim = xdim
        self.ydim = ydim

        # Inference model
        self.q_0_mean = q_0_mean
        self.q_0_log_std = q_0_log_std
        self.cond_q_mean_net_constructor = cond_q_mean_net_constructor
        self.cond_q_0_log_std = cond_q_0_log_std

        time_store_size = window_size + 1 if num_params_to_store is None else num_params_to_store
        self.q_t_mean_list = utils.TimeStore(None, time_store_size, 'ParameterList')
        self.q_t_log_std_list = utils.TimeStore(None, time_store_size, 'ParameterList')
        self.cond_q_t_mean_net_list = utils.TimeStore(None, time_store_size, 'ModuleList')
        self.cond_q_t_log_std_list = utils.TimeStore(None, time_store_size, 'ParameterList')

        # Generative model
        self.F_fn = F_fn
        self.G_fn = G_fn
        self.p_0_dist = p_0_dist

        self.phi_t_init_method = phi_t_init_method
        self.window_size = window_size

    def advance_timestep(self):
        '''
        Function that updates the most recent parameters when new data arrives. 
        '''
        self.T = +1
        
        if phi_t_init_method=='last':
            if self.T==0:
                self.q_t_mean_list.append(nn.Parameter(self.q_0_mean))
                self.q_t_log_std_list.append(nn.Parameter(self.q_0_log_std))
            else:
                # any operations on the detached tensor will not be tracked for autograd (automatic differentiation).
                self.q_t_mean_list.append(nn.Parameter(self.q_t_mean_list[self.T-1,:].clone().detach()))
                self.q_t_log_std_list.append(nn.Parameter(self.q_t_log_std_list[self.T-1,:].clone().detach()))

        elif phi_t_init_method=='pred':
            if self.T==0:
                self.q_t_mean_list.append(nn.Parameter(self.q_0_mean))
                self.q_t_log_std_list.append(nn.Parameter(self.q_0_log_std))
            else:
                self.q_t_mean_list.append(nn.Parameter(self.F_fn.F_mean_fn(self.q_t_mean_list[self.T-1], self.T-1)
                                                       .clone().detach()))
                # we want to compute the new value in self.q_t_log_std_list
                test_x = self.q_t_mean_list[self.T - 1].detach().clone().requires_grad_()
                # we compute the jacobian value of self.F_fn.F_mean_fn at test_x, with t fixed to T-1
                F_jac = torch.autograd.functional.jacobian(partial(self.F_fn.F_mean_fn, t=self.T-1), test_x)
                pred_cov = F_jac @ torch.diag((self.q_t_log_std_list[self.T - 1] * 2).exp()) @ F_jac.t() + F_cov
                self.q_t_log_std_list.append(nn.Parameter(pred_cov.detach().diag().log() / 2))
        else:
            assert False, "Invalid phi_t_init_method"
        
        self.cond_q_t_mean_net_list.append(self.cond_q_mean_net_constructor())
        if self.T == 0:
            self.cond_q_t_log_std_list.append(nn.Parameter(self.cond_q_0_log_std))
        else:
            self.cond_q_t_mean_net_list[self.T].load_state_dict(self.cond_q_t_mean_net_list[self.T - 1].state_dict())
            self.cond_q_t_log_std_list.append(nn.Parameter(self.cond_q_t_log_std_list[self.T - 1].clone().detach()))

        # we can optimize only the current parameters, not the previous ones. We applied detahc before, so here we apply requires_grad
        if self.T >= self.window_size:
            # variational distribution
            self.q_t_mean_list[self.T - self.window_size].requires_grad_(False)
            self.q_t_log_std_list[self.T - self.window_size].requires_grad_(False)
            # conditional variational smoothing
            self.cond_q_t_mean_net_list[self.T - self.window_size].requires_grad_(False)
            self.cond_q_t_log_std_list[self.T - self.window_size].requires_grad_(False)

    def get_phi_T_params(self):
        '''
        Function that get the parameters associated to phi_{t}, ie the Gaussian's params and the nnet. 
        '''
        params = []
        params = params + [self.q_t_mean_list[self.T], self.q_t_log_std_list[self.T],
                            *self.cond_q_t_mean_net_list[self.T].parameters(), self.cond_q_t_log_std_list[self.T]]
        for t in range(max(0, self.T - self.window_size + 1), self.T):
            params = params + [*self.cond_q_t_mean_net_list[t].parameters(), self.cond_q_t_log_std_list[t]]
        return params

    def sample_q_T(self, num_samples, detach_x=False, T=None):
        '''
        Function that sample from q_T. ie a self.xdim-Gaussian, num_samples times. The parameters are 
        q_T_mean, and the std is q_T_std. ie we do q_T_mean+q_T_std*N(0,I) where 0 is xdim-dim and I a sqaure mat
        '''
        if T is None:
            T = self.T
        assert T <= self.T
        q_T_mean = self.q_t_mean_list[T].expand(num_samples, self.xdim)
        q_T_std = self.q_t_log_std_list[T].exp().expand(num_samples, self.xdim)
        q_T_stats = [q_T_mean, q_T_std]

        eps_x_T = torch.randn(num_samples, self.xdim)#.to(self.device)
        x_T = q_T_mean + q_T_std * eps_x_T

        if detach_x:
            x_T = x_T.detach()

        return x_T, q_T_stats

    def sample_q_t_cond_T(self, x_T, num_steps_back, detach_x=False, T=None):
        '''
        Function that samples from q_T_cond. 
        '''
        if T is None:
            T = self.T
        assert T <= self.T
        num_samples = x_T.shape[0]
        if num_steps_back > T:
            print("Warning: num_steps_back > T")
        num_steps_back = min(num_steps_back, T)
        x_t_samples = [None] * num_steps_back
        all_cond_q_t_means = [None] * num_steps_back
        all_cond_q_t_stds = [None] * num_steps_back

        for t in range(T - 1, T - 1 - num_steps_back, -1):
            if t == T - 1:
                x_tp1 = x_T
            else:
                x_tp1 = x_t_samples[t - T + num_steps_back + 1]

            # We do not use the 0th entry in the conditional lists, so that at time t we are learning the
            # t-th entry in all the lists (q(x_t) and q(x_tm1|x_t))
            cond_q_t_mean = self.cond_q_t_mean_net_list[t + 1](x_tp1)
            cond_q_t_std = self.cond_q_t_log_std_list[t + 1].exp().expand(num_samples, self.xdim)

            eps_x_t = torch.randn(num_samples, self.xdim).to(self.device)
            x_t = cond_q_t_mean + cond_q_t_std * eps_x_t

            if detach_x:
                x_t = x_t.detach()

            x_t_samples[t - T + num_steps_back] = x_t
            all_cond_q_t_means[t - T + num_steps_back] = cond_q_t_mean
            all_cond_q_t_stds[t - T + num_steps_back] = cond_q_t_std

        all_cond_q_t_stats = [[mean, std] for mean, std in zip(all_cond_q_t_means, all_cond_q_t_stds)]

        return x_t_samples, all_cond_q_t_stats

    def sample_joint_q_t(self, num_samples, num_steps_back, detach_x=False, T=None):
        """
            Sample num_samples from
            q(x_T) \prod_{t= T - num_steps_back}^{T-1} q(x_t | x_{t+1})
            If detach_x is true then all x samples are detached
        """
        if T is None:
            T = self.T
        assert T <= self.T
        if num_steps_back > T:
            print("Warning: num_steps_back > T")
        num_steps_back = min(num_steps_back, T)

        x_T_samples, q_T_stats = self.sample_q_T(num_samples, detach_x=detach_x, T=T)

        x_t_samples, all_cond_q_t_stats = self.sample_q_t_cond_T(x_T_samples, num_steps_back, detach_x=detach_x, T=T)

        return x_t_samples + [x_T_samples], all_cond_q_t_stats + [q_T_stats]

    def compute_log_p_t(self, x_t, y_t, x_tm1=None, t=None):
        """
            Compute log p(x_t | x_{t-1}) and log p(y_t | x_t)
            t is set to self.T if not specified
        """
        if t is None:
            t = self.T
        if t == 0:
            log_p_x_t = self.p_0_dist().log_prob(x_t).unsqueeze(1)
        else:
            log_p_x_t = self.F_fn(x_tm1, t-1).log_prob(x_t).unsqueeze(1)
        log_p_y_t = self.G_fn(x_t, t).log_prob(y_t).unsqueeze(1)
        return {"log_p_x_t": log_p_x_t, "log_p_y_t": log_p_y_t}

    def compute_log_q_t(self, x_t, *q_t_stats):
        """
            Compute log q(x_t | x_{t+1}) (independent Gaussian inference model)
        """
        assert len(q_t_stats) == 2
        return Independent(Normal(*q_t_stats), 1).log_prob(x_t).unsqueeze(1)

    def compute_r_t(self, x_t, y_t, *q_tm1_stats, x_tm1=None, t=None):
        if t is None:
            t = self.T
        if t == 0:
            log_p_t = self.compute_log_p_t(x_t, y_t, t=0)
            log_p_x_t, log_p_y_t = log_p_t["log_p_x_t"], log_p_t["log_p_y_t"]
            r_t = log_p_x_t + log_p_y_t
        else:
            log_p_t = self.compute_log_p_t(x_t, y_t, x_tm1, t=t)
            log_p_x_t, log_p_y_t = log_p_t["log_p_x_t"], log_p_t["log_p_y_t"]
            log_q_x_tm1 = self.compute_log_q_t(x_tm1, *q_tm1_stats)
            r_t = log_p_x_t + log_p_y_t - log_q_x_tm1
        return r_t

    def sample_and_compute_r_t(self, y_t, num_samples, detach_x=False, t=None, disperse_temp=1):
        if t is None:
            t = self.T - self.window_size + 1
        x_t, q_t_stats = self.sample_joint_q_t(num_samples, self.T - t, detach_x=detach_x)
        x_t, q_t_stats = x_t[0], q_t_stats[0]
        x_t_dispersed = x_t + torch.randn_like(x_t) * np.sqrt(1/disperse_temp - 1) * q_t_stats[1]
        if t == 0:
            x_tm1 = None
            r_t = self.compute_r_t(x_t_dispersed, y_t, t=0)
        else:
            x_tm1, cond_q_tm1_stats = self.sample_q_t_cond_T(x_t_dispersed, 1, detach_x=detach_x, T=t)
            x_tm1, cond_q_tm1_stats = x_tm1[0], cond_q_tm1_stats[0]
            r_t = self.compute_r_t(x_t_dispersed, y_t, *cond_q_tm1_stats, x_tm1=x_tm1, t=t)
        return {"x_tm1": x_tm1, "x_t": x_t_dispersed, "r_t": r_t}

    def sample_and_compute_joint_r_t(self, y, num_samples, window_size, detach_x=False, only_return_first_r=False,
                                     T=None):
        """
            Sample from q(x_T) q(x_{(T-window_size):(T-1)} | x_T) and compute r_{(T-window_size+1):T}
            as well as other useful quantities
        """
        if T is None:
            T = self.T
        assert T <= self.T
        if window_size > T + 1:
            print("Warning: window_size > T + 1")
        window_size = min(window_size, T + 1)
        num_steps_back = min(window_size, T)
        x_samples, all_q_stats = self.sample_joint_q_t(num_samples, num_steps_back, detach_x=detach_x, T=T)
        ts = np.arange(T - window_size + 1, T + 1)  # correspond to r_t, length window_size
        x_t_idx = np.arange(T + 1) if window_size == T + 1 else \
            np.arange(1, window_size + 1)  # indices of x_t in x_samples to compute r_t, length window_size
        r_values = []  # r_t, length window_size

        for i, t in zip(x_t_idx, ts):
            x_t, q_t_stats = x_samples[i], all_q_stats[i]
            x_tm1, q_tm1_stats = (None, []) if t == 0 else (x_samples[i - 1], all_q_stats[i - 1])
            r_t = self.compute_r_t(x_t, y[t, :], *q_tm1_stats, x_tm1=x_tm1, t=t)
            r_values.append(r_t)

            if only_return_first_r:
                break

        log_q_x_T = self.compute_log_q_t(x_samples[-1], *all_q_stats[-1])

        return {"x_samples": x_samples, "all_q_stats": all_q_stats,
                "r_values": r_values, "log_q_x_T": log_q_x_T}

    def generate_data(self, T):
        # Generates hidden states and observations up to time T
        x = torch.zeros((T, self.xdim)).to(self.device)
        y = torch.zeros((T, self.ydim)).to(self.device)

        x[0, :] = self.p_0_dist().sample()

        for t in range(T):
            y_t = self.G_fn(x[t, :], t).sample()
            x_tp1 = self.F_fn(x[t, :], t).sample()

            y[t, :] = y_t
            if t < T-1:
                x[t+1, :] = x_tp1

        return x, y

    def compute_elbo_loss(self, y, num_samples):
        all_r_results = self.sample_and_compute_joint_r_t(y, num_samples, self.T + 1)

        r_values = all_r_results["r_values"]
        sum_r = sum(r_values)
        log_q_x_T = all_r_results["log_q_x_T"]

        loss = - (sum_r - log_q_x_T).mean()
        return loss

    def return_summary_stats(self, y, t=None, num_samples=None):
        if t is None:
            t = self.T
        if t == self.T:
            x_t_mean = self.q_t_mean_list[self.T].detach().clone()
            x_t_cov = torch.diag(torch.exp(self.q_t_log_std_list[self.T].detach().clone() * 2))
        elif t == self.T - 1:
            joint_x_samples, _ = self.sample_joint_q_t(num_samples, 1)
            x_Tm1_samples = joint_x_samples[0]
            x_t_mean = x_Tm1_samples.mean(0).detach().clone()
            x_t_cov = sample_cov(x_Tm1_samples).detach().clone()
        return x_t_mean, x_t_cov