from functools import partial 
import torch
import torch.nn as nn
import torch.nn.functional as fuctional
import numpy as np
import Utils as utils 
from torch.distributions import Independent, MultivariateNormal, Normal, Categorical
from Utils import gaussian_posterior, sample_cov 

class NonAmortisedDeepScoreModels(nn.Module):
    '''
    This class defines neural network that encode 
    x_{t} = F(x_{t},u_{t},\nabla_{1:t}), where F is a nn that encodes the latent's variable dynamics. F is F_\theta_{1}.
    y_{t} = Pi(x_{t},u_{t}), where Pi is a nn that encodes a probability distribution. Pi is Pi_\theta_{2}.
    The q lists contain nonamortized variational posteriors
            q_t(x_t) = a Dirac since we have an observation driven model.
            q_t(x_{t-1} | x_t) = N(x_{t-1}; cond_q_t_mean_net(x_t), diag(cond_q_t_std)^2)
            The last will be factorized as a Gaussian.
        cond_q_t_mean_net: Linear, MLP, etc: (xdim) -> (xdim)
    '''
    def __init__(self, device, xdim, ydim, cond_q_mean_net_constructor, cond_q_0_log_std,
                 F_fn, Pi_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store=None):
    super().__init__()

    '''
        Initialize the Deep Score online learning model. 
        1- Initialize the latent and the observable dimensions.
        2- Give the prior parameters for the conditional variational smoothing distribution which we parametrized as Gaussian, but can be something else.  
        3- Give the prior parameters for the cond variational smoothing distribution. Here everything is Gaussian.
        4- Define the window size.
        5- Give the previous mean and std that characterize the cond variational distrib. 
        6- Define the functions F and G that define the generative model. 
    '''

    # time 
    self.T = -1

    # dimensions
    self.xdim = xdim
    self.ydim = ydim

    # device, GPU or CPU 
    self.device = device 

    # nnet initializations 
    self.cond_q_mean_net_constructor = cond_q_mean_net_constructor
    self.cond_q_0_log_std = cond_q_0_log_std


    time_store_size = window_size + 1 if num_params_to_store is None else num_params_to_store
    self.cond_q_t_mean_net_list = utils.TimeStore(None, time_store_size, 'ModuleList')
    self.cond_q_t_log_std_list = utils.TimeStore(None, time_store_size, 'ParameterList')

    # Generative model
    self.F_fn = F_fn
    self.Pi_fn = Pi_fn
    self.p_0_dist = p_0_dist

    self.phi_t_init_method = phi_t_init_method
    self.window_size = window_size
    

    def advance_timestep(self):
        '''
        Function that updates the most recent parameters when new data arrives. 
        '''

        self.T = +1
        # conditional variational smoothing only since the var smoothing is a Dirac, observation driven model. 
        self.cond_q_t_mean_net_list.append(self.cond_q_mean_net_constructor())
        if self.T == 0:
            self.cond_q_t_log_std_list.append(nn.Parameter(self.cond_q_0_log_std))
        else:
            self.cond_q_t_mean_net_list[self.T].load_state_dict(self.cond_q_t_mean_net_list[self.T - 1].state_dict())
            self.cond_q_t_log_std_list.append(nn.Parameter(self.cond_q_t_log_std_list[self.T - 1].clone().detach()))

        # we can optimize only the current parameters, not the previous ones. We applied detahc before, so here we apply requires_grad
        if self.T >= self.window_size:
            self.cond_q_t_mean_net_list[self.T - self.window_size].requires_grad_(False)
            self.cond_q_t_log_std_list[self.T - self.window_size].requires_grad_(False)
    
    def get_phi_T_params(self):
        params = []
        params = params + [*self.cond_q_t_mean_net_list[self.T].parameters(), self.cond_q_t_log_std_list[self.T]]
        for t in range(max(0, self.T - self.window_size + 1), self.T):
            params = params + [*self.cond_q_t_mean_net_list[t].parameters(), self.cond_q_t_log_std_list[t]]
        return params

    def compute_log_p_y_t(self, x_t, y_t, x_tm1=None, t=None):
        """
            Compute log p(y_t | x_t)
            t is set to self.T if not specified
        """
        if t is None:
            t = self.T
        log_p_y_t = self.Pi_fn(x_t, t).log_prob(y_t).unsqueeze(1)
        return {"log_p_y_t": log_p_y_t}

    def sample_q_t_cond_T(self, x_T, num_steps_back, detach_x=False, T=None):
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

    def sample_joint_q_t(self, x_T, num_samples, num_steps_back, detach_x=False, T=None):
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

        x_T_samples = x_T.expand(num_samples, self.xdim)

        x_t_samples, all_cond_q_t_stats = self.sample_q_t_cond_T(x_T_samples, num_steps_back, detach_x=detach_x, T=T)

        return x_t_samples + [x_T_samples], all_cond_q_t_stats

    def compute_log_q_xtm1_cond_xt(self, x_t, x_tm1, *q_tm1_stats):
        """
            Compute log q(x_t | x_{t+1}) (independent Gaussian inference model)
        """
        assert len(q_tm1_stats) == 2
        return Independent(Normal(*q_tm1_stats), 1).log_prob(x_t).unsqueeze(1)
        
    def compute_r_t(self, x_t, y_t, *q_tm1_stats, x_tm1=None, t=None):
        if t is None:
            t = self.T
        if t == 0:
            log_p_y_t = self.compute_log_p_y_t(x_t, y_t, t=0)
            r_t = log_p_y_t
        else:
            log_p_y_t = self.compute_log_p_y_t(x_t, y_t, x_tm1, t=t)
            log_m_x_t = self.compute_log_q_xtm1_cond_xt(x_t, x_tm1, *q_tm1_stats)
            r_t = log_p_y_t - log_m_x_t
        return r_t
## problem here 
    def sample_and_compute_r_t(self, y_t, num_samples, detach_x=False, t=None, disperse_temp=1):
        if t is None:
            t = self.T - self.window_size + 1
        x_t, q_tm1_stats = self.sample_joint_q_t(num_samples, self.T - t, detach_x=detach_x)
        print('q_tm1_stat:', q_tm1_stats)
        x_t, q_tm1_stats = x_t[0], q_t_stats[0]
        x_t_dispersed = x_t + torch.randn_like(x_t) * np.sqrt(1/disperse_temp - 1) * q_t_stats[1]
        if t == 0:
            x_tm1 = None
            r_t = self.compute_r_t(x_t_dispersed, y_t, t=0)
        else:
            x_tm1, cond_q_tm1_stats = self.sample_q_t_cond_T(x_t_dispersed, 1, detach_x=detach_x, T=t)
            x_tm1, cond_q_tm1_stats = x_tm1[0], cond_q_tm1_stats[0]
            r_t = self.compute_r_t(x_t_dispersed, y_t, *cond_q_tm1_stats, x_tm1=x_tm1, t=t)
        return {"x_tm1": x_tm1, "x_t": x_t_dispersed, "r_t": r_t}


    def sample_and_compute_joint_r_t(self, x_T,  y, num_samples, window_size, detach_x=False, only_return_first_r=False,
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
        x_samples, all_q_stats = self.sample_joint_q_t(x_T, num_samples, num_steps_back, detach_x=detach_x, T=T)
        ts = np.arange(T - window_size + 1, T + 1)  # correspond to r_t, length window_size
        x_t_idx = np.arange(T + 1) if window_size == T + 1 else \
            np.arange(1, window_size + 1)  # indices of x_t in x_samples to compute r_t, length window_size
        r_values = []  # r_t, length window_size

        for i, t in zip(x_t_idx, ts):
            x_t = x_samples[i]
            x_tm1, q_tm1_stats = (None, []) if t == 0 else (x_samples[i - 1], all_q_stats[i-1])
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
            y_t = self.Pi_fn(x[t, :], t).sample()
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