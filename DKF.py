import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class DeepKalmanFilter(nn.Module):
    def __init__(self, d_y, N_pos, N_ref, hidden_units, N_prime, d_x):
        super(DeepKalmanFilter, self).__init__()

        # Known parameters
        self.N_ref = N_ref
        self.N_pos = N_pos
        self.N_prime = N_prime
        self.d_x = d_x

        # Learnable parameters
        self.V = nn.Parameter(torch.randn(N_pos, d_y))
        self.U = nn.Parameter(torch.randn(N_pos, 1))  # Adjusted to have a single parameter for N_time
        self.t_n = nn.Parameter(torch.randn(1))  # Assuming a single time point for simplicity
        self.A = nn.Parameter(torch.randn(1, 1))  # Adjusted for a single parameter for N_sim
        self.B = nn.Parameter(torch.randn(1, N_ref))
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.theta_0 = nn.Parameter(torch.randn(1))
        self.theta_1 = nn.Parameter(torch.randn(1))
        
        # Additional layers
        self.mlp = nn.Sequential(
            nn.Linear(N_ref + 1, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, N_prime),
            nn.Tanh()
        )

        self.geometric_attention = nn.Sequential(
            nn.Linear(N_prime, d_x),
            nn.Tanh()
        )

    def max_norm(self, y, y_ref, t):
        norms = torch.norm(y - y_ref, dim=1)
        max_norm_value, _ = torch.max(norms, dim=0)
        return max_norm_value

    def sim_theta_0_T(self, y):
        max_norm_value = self.max_norm(y, self.y_ref, self.t_n)
        composition = self.A * F.relu(self.B * max_norm_value + self.b) + self.a
        return F.softmax(self.theta_0 * composition, dim=1)

    def pos_theta_1_T(self, y):
        y_time_points = torch.stack([y[:, int(self.t_n * 1 / 1)]])  # Adjusted for a single time point
        U_result = torch.matmul(self.U, y_time_points)
        pos_theta_1_T = U_result.t() + self.V
        return self.theta_1 * pos_theta_1_T

    def att_theta_T(self, t, y):
        sim_theta_0_T = self.sim_theta_0_T(y)
        pos_theta_1_T = self.pos_theta_1_T(y)
        att_result = torch.cat([t.view(1, 1), (sim_theta_0_T * pos_theta_1_T).view(1, -1)], dim=1)
        return att_result

    def forward(self, t, y):
        att_result = self.att_theta_T(t, y)
        mlp_output = self.mlp(att_result)
        geometric_attention_output = self.geometric_attention(mlp_output)

        # Gaussian mapping
        mean_projection = F.softplus(geometric_attention_output[:, :self.N_prime])
        var_projection = F.softplus(geometric_attention_output[:, self.N_prime:])
        
        # Sample from the Gaussian distribution
        normal_distribution = Normal(mean_projection, var_projection)
        sample = normal_distribution.sample()

        return sample

# Example usage
N_ref = 5  # Replace with the actual value
N_pos = 5  # Replace with the actual value
d_y = 3  # Replace with the actual value
hidden_units = 10  # Replace with the desired number of hidden units
N_prime = 5  # Replace with the desired value for N_prime
d_x = 2  # Replace with the desired value for d_x

# Create an instance of the AttentionNetwork
deep_kalman_filter = DeepKalmanFilter(d_y, N_pos, N_ref, hidden_units, N_prime, d_x)

# Example input
t = 0.5  # Replace with the actual value
y = torch.randn(N_pos, d_y)

# Forward pass through the network
result = attention_net(t, y)
print(result)
