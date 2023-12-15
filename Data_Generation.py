import numpy as np 

'''
We propose multiple data generating procedures to test the DKF and the DSF. 
1-Linear_Gaussian_HMM
2-Non_Linear_Gaussian_HMM
3-Non_Linear_Student_HMM
4-Non_Linear_Time_Varying_Markovian
'''

  # --------------------- Linear Case ---------------------

class Linear_Gaussian_HMM():
    '''
    Gaussian hidden markov model:
    x_t are states, y_t are observations

    x_t = F x_{t-1} + \mu_t    \mu_t ~ N(0, U)    for  t = {1, 2, ...}
    y_t = G x_t     + \nu_t    \nu_t ~ N(0, V)    for  t = {0, 1, 2, ...}

    x_0 sampled from N(0, I)
    '''

    def __init__(self, T, x_dim, y_dim, F, G, U, V):
        self.T = T
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.x0=np.random.randn(x_dim)
        self.F=F
        self.G=G
        self.U=U
        self.V=V 

    def generate_data(self):
        '''
        Generate the latent variables and the observables from time 0 to time T.
        1-Create tensors for the xs and the ys.
        2-Code a loop that defines the observable as a function of the latent variables.
        '''
        # initial tensors that will take the latent variable and the observable
        self.x = np.zeros((self.T, self.x_dim))
        self.y = np.zeros((self.T, self.y_dim))

        # initialize x0
        self.x[0,:] = self.x0

        # sample gaussian with zero mean
        def gaussian_zero_sample(dim, cov):
            '''
            Generate the noise distributed as a Gaussian. 
            1-Set the parameters: dimensions, mean, cov.
            2-Generate the Gaussian vector using numpy multivariate normal function.  
            "'''
            gaussian_sample = np.random.multivariate_normal(
                mean=np.zeros((dim)),
                cov=cov
            )
            return gaussian_sample 

        # loop that create the data using F, G, U and V
        for t in range(self.T):
            nu_t = gaussian_zero_sample(self.y_dim, self.V)
            self.y[t,:] = np.dot(self.G, self.x[t,:]) + nu_t
            if t < self.T-1:
                mu_tp1 = gaussian_zero_sample(self.x_dim, self.U)
                self.x[t+1,:] = np.dot(self.F, self.x[t,:]) + mu_tp1
            else:
                continue
        return self.x, self.y

def HMM_matrices(x_dim,y_dim,F_eigenvals,G_eigenvals,U_std=0.3,V_std=0.3, diag=False):
    '''
    Create the matrix that encodes the discretized ODE. 
    1-Assert that dimensions are correct. 
    2-Create matrices F and G with the associated eigenvalues.
    3-Create the covarince matrices with the std given by the arguments.
    '''

    # assert dimensions are correct
    try:
        assert x_dim == np.shape(F_eigenvals)[0] and y_dim == np.shape(G_eigenvals)[0]
    except AssertionError:
        raise ValueError("Assertion failed: Dimensions of x_dim and F_eigenvals, or y_dim and G_eigenvals do not match.")

    # create F and G
    def ode_encoding_matrices(x_dim,y_dim,F_eigenvals,G_eigenvals):
        '''
        Create the matrices F and G with the associated eigenvalues.
        1-Define two random matrices (by density they are non-singular).
        2-Construct F and G.
        '''

        # random matrices 
        if diag == False:
            x_random_matrix = np.random.rand(x_dim,x_dim)
            y_random_matrix = np.random.rand(y_dim,y_dim)
        else:
            x_random_matrix = np.eye(x_dim)
            y_random_matrix = np.eye(y_dim)

        # Construct F and G
        diag_F = np.diag(F_eigenvals)
        F = np.matmul(np.matmul(x_random_matrix, diag_F), np.linalg.inv(x_random_matrix))

        diag_G = np.diag(G_eigenvals)
        G = np.matmul(np.matmul(y_random_matrix, diag_G), np.linalg.inv(y_random_matrix))

        return F, G

    F, G = ode_encoding_matrices(x_dim,y_dim,F_eigenvals,G_eigenvals)

    # create U and V
    def covariance_matrices(x_dim,y_dim,U_std,V_std):
        '''
        Create the matrices U and V with the std.
        1-Define the var instead of the std
        2-Create U ad V
        '''
        U = (U_std**2) * np.eye(x_dim)
        V = (V_std**2) * np.eye(y_dim)

        return U, V

    U, V = covariance_matrices(x_dim,y_dim,U_std,V_std)

    return F, G, U, V

  # --------------------- Non Linear Gaussian Case - Additive noise ---------------------

class Non_Linear_Gaussian_HMM():
    '''
    Non-linear Gaussian hidden markov model:
    x_t are states, y_t are observations

    x_t = Func_F(x_{t-1})+ \mu_t    \mu_t ~ N(0, U)    for  t = {1, 2, ...}
    y_t = Func_G(x_t)    + \nu_t    \nu_t ~ N(0, V)    for  t = {0, 1, 2, ...}

    x_0 sampled from N(0, I)
    '''

    def __init__(self, T, x_dim, y_dim, Func_F, Func_G, U, V):
        self.T = T
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.x0=np.random.randn(x_dim)
        self.Func_F=Func_F
        self.Func_G=Func_G
        self.U=U
        self.V=V 

    def generate_data(self):
        '''
        Generate the latent variables and the observables from time 0 to time T.
        1-Create tensors for the xs and the ys.
        2-Code a loop that defines the observable as a function of the latent variables.
        '''
        # initial tensors that will take the latent variable and the observable
        self.x = np.zeros((self.T, self.x_dim))
        self.y = np.zeros((self.T, self.y_dim))

        # initialize x0
        self.x[0,:] = self.x0

        # sample gaussian with zero mean
        def gaussian_zero_sample(dim, cov):
            '''
            Generate the noise distributed as a Gaussian. 
            1-Set the parameters: dimensions, mean, cov.
            2-Generate the Gaussian vector using numpy multivariate normal function.  
            '''
            gaussian_sample = np.random.multivariate_normal(
                mean=np.zeros((dim)),
                cov=cov
            )
            return gaussian_sample 

        # loop that create the data using F, G, U and V
        for t in range(self.T):
            nu_t = nu_t = gaussian_zero_sample(self.y_dim, self.V)
            #self.y[t,:] = self.Func_G(self.x[t,:]) + nu_t
            self.y[t,:] = self.Func_G(self.x[t,:]) + nu_t
            if t < self.T-1:
                mu_tp1 = nu_t = gaussian_zero_sample(self.x_dim, self.U)
                #self.x[t+1,:] = self.Func_F(self.x[t,:]) + mu_tp1
                self.x[t+1,:] = self.Func_F(self.x[t,:]) + mu_tp1
            else:
                continue
        return self.x, self.y

# --------------------- Non Linear Case - Student Additive noise ---------------------

class Non_Linear_Student_HMM():
    '''
    Non-linear Student hidden markov model:
    x_t are states, y_t are observations

    x_t = Func_F(x_{t-1})+ \mu_t    \mu_t ~ Law(Student)     for  t = {1, 2, ...}
    y_t = Func_G(x_t)    + \nu_t    \nu_t ~ Law(Student)     for  t = {0, 1, 2, ...}

    x_0 sampled from N(0, I)
    '''

    def __init__(self, T, x_dim, y_dim, Func_F, Func_G, mu_degrees_of_f, nu_degrees_of_f):
        self.T = T
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.x0=np.random.randn(x_dim)
        self.Func_F=Func_F
        self.Func_G=Func_G
        self.mu_degrees_of_f=mu_degrees_of_f
        self.nu_degrees_of_f=nu_degrees_of_f

    def generate_data(self):
        '''
        Generate the latent variables and the observables from time 0 to time T.
        1-Create tensors for the xs and the ys.
        2-Code a loop that defines the observable as a function of the latent variables.
        '''
        # initial tensors that will take the latent variable and the observable
        self.x = np.zeros((self.T, self.x_dim))
        self.y = np.zeros((self.T, self.y_dim))

        # initialize x0
        self.x[0,:] = self.x0

        # sample noise with zero mean
        def noise_zero_sample(dim, degrees_of_f):
            '''
            Generate the noise distributed as a student. 
            1-Set the parameters: dimensions, mean, degrees of freedom.
            2-Generate the student using the chi-squared parametrization with the scaling factor. 
            '''
            # set the parameters
            mean = np.zeros(dim)  
            df = degrees_of_f      
            covariance_matrix = np.eye(dim)  
            # Generate multivariate Student's t-distributed random vector
            random_vector = np.random.multivariate_normal(mean, covariance_matrix)

            # Apply the t-distribution scaling
            scaling_factor = np.sqrt((df - 2) / df)
            student_sample = mean + scaling_factor * random_vector / np.sqrt(np.random.chisquare(df) / df)

            return student_sample

        # loop that create the data using F, G, U and V
        for t in range(self.T):
            nu_t = noise_zero_sample(self.y_dim, self.nu_degrees_of_f)
            self.y[t,:] = self.Func_G(self.x[t,:]) + nu_t
            if t < self.T-1:
                mu_tp1 = noise_zero_sample(self.x_dim, self.mu_degrees_of_f)
                self.x[t+1,:] = self.Func_F(self.x[t,:]) + mu_tp1
            else:
                continue
        return self.x, self.y


# --------------------- Non Linear Case - Time Varying - Markovian ---------------------

class Non_Linear_Time_Varying_Markovian():
    '''
    NonLinear model, time varying:
    x_t are states, y_t are observations

    x_t = Func_F(t,x_{t-1},\mu_t)  \mu_t ~ Law(noise)    for  t = {1, 2, ...}
    y_t = \pi(Func_G(t,x_t))   \pi ~ Law(parametrized distribution)   for  t = {0, 1, 2, ...}

    x_0 sampled from N(0, I)
    '''

    def __init__(self, T, x_dim, y_dim, Func_F, Func_G, U, V):
        self.T = T
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.x0=np.random.randn(x_dim)
        self.Func_F=Func_F
        self.Func_G=Func_G
        self.U=U
        self.V=V 

    def generate_data(self):
        '''
        Generate the latent variables and the observables from time 0 to time T.
        1-Create tensors for the xs and the ys.
        2-Code a loop that defines the observable as a function of the latent variables.
        '''
        # initial tensors that will take the latent variable and the observable
        self.x = np.zeros((self.T, self.x_dim))
        self.y = np.zeros((self.T, self.y_dim))

        # initialize x0
        self.x[0,:] = self.x0

        # sample gaussian with zero mean
        def noise_zero_sample(dim, degrees_of_f):
            '''
            Generate the noise distributed as a student. 
            1-Set the parameters: dimensions, mean, degrees of freedom.
            2-Generate the student using the chi-squared parametrization with the scaling factor. 
            '''
            # set the parameters
            mean = np.zeros(dim)  
            df = degrees_of_f      
            covariance_matrix = np.eye(dim)  
            # Generate multivariate Student's t-distributed random vector
            random_vector = np.random.multivariate_normal(mean, covariance_matrix)

            # Apply the t-distribution scaling
            scaling_factor = np.sqrt((df - 2) / df)
            student_sample = mean + scaling_factor * random_vector / np.sqrt(np.random.chisquare(df) / df)

            return student_sample

        # loop that create the data using F, G, U and V
        for t in range(self.T):
            self.y[t,:] = self.pi(self.y_dim, self.x[t,:], distribution_type='levy')
            if t < self.T-1:
                mu_tp1 = noise_zero_sample(self.x_dim, self.degrees_of_f)
                self.x[t+1,:] = self.Func_F(t,self.x[t,:],mu_tp1) 
            else:
                continue
        return self.x, self.y

# define the density that generates the observations
from scipy.stats import levy

def pi_density(y_dim, parameters, distribution_type='levy'):
    '''
    Data generating function. 
    1-Define the type of distribution.
    2-Set the parameters.
    3-Generate a sample with dimension y_dim.
    '''
    try:
        assert distribution_type in ['gaussian', 'student', 'levy'], "Invalid distribution type"
        
        if distribution_type == 'levy':
            # Set the parameters 
            location = parameters[0]  # Location parameter
            scale = parameters[1]     # Scale parameter
            # Generate a sample from the Lévy distribution
            levy_sample = levy.rvs(loc=location, scale=scale, size=y_dim)
            binomial_signs = np.random.choice([-1, 1], size=y_dim)
            levy_sample *= binomial_signs
            return levy_sample

        elif distribution_type == 'student':
            # Set the parameters
            mean = parameters[0] 
            df = parameters[1]      
            covariance_matrix = np.eye(y_dim)  # Adjusted to y_dim
            # Generate multivariate Student's t-distributed random vector
            student_sample = np.random.multivariate_normal(mean, covariance_matrix)
            
            # Apply the t-distribution scaling
            scaling_factor = np.sqrt((df - 2) / df)
            student_sample = mean + scaling_factor * student_sample / np.sqrt(np.random.chisquare(df) / df)
            return student_sample

        elif distribution_type == 'gaussian':
            # Set the parameters
            mean = parameters[0] 
            covariance_matrix = parameters[1] 
            # Generate multivariate Gaussian sample
            gaussian_sample = np.random.multivariate_normal(mean=mean, cov=covariance_matrix, size=y_dim)
            return gaussian_sample

    except AssertionError as e:
        print(f"AssertionError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")