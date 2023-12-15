'''
We generate the data:
1-Linear_Gaussian_HMM
2-Non_Linear_Gaussian_HMM
3-Non_Linear_Student_HMM
'''

import numpy as np 
import Data_Generation as dg 
import os

directory = "/home/gpalmari/OnlineLearning"
files = os.listdir(directory)
for file in files:
    if ".sh.o" in file or ".sh.e" in file:
        file_path = os.path.join(directory, file)
        os.remove(file_path)

def main():
    generative_dictionary = {
    'Linear_Gaussian_HMM': True,
    'Non_Linear_Gaussian_HMM': True,
    'Non_Linear_Student_HMM': True,
    'Non_Linear_Time_Varying_Markovian': False
    }
    try:
        if generative_dictionary is None or not isinstance(generative_dictionary, dict):
            raise ValueError("The dictionary is not well-initialized.")

        print("Dictionary is well-initialized.")
        if generative_dictionary['Linear_Gaussian_HMM']:
            '''
            Linear_Gaussian_HMM.
            0- Set the parameters. 
            1-Define F, G, U, V.
            2-Create an element from Linear_Gaussian_HMM.
            3-Plot the data that has been generated.
            '''
            # Parameters
            T = 10
            x_dim=2
            y_dim=2
            F_eigenvals = [-4,3]
            G_eigenvals = [-1,3]
            F,G,U,V = dg.HMM_matrices(x_dim,y_dim,F_eigenvals,G_eigenvals,U_std=0.3,V_std=0.3)
            print(F)
            print(G)
            print(U)
            print(V)
            L_G_HMM = dg.Linear_Gaussian_HMM(T, x_dim, y_dim, F, G, U, V)
            x_L_G_HMM, y_L_G_HMM = L_G_HMM.generate_data()
            print(x_L_G_HMM)
            print(y_L_G_HMM)
        if generative_dictionary['Non_Linear_Gaussian_HMM']:
            '''
            Non_Linear_Gaussian_HMM.
            0- Set the parameters. 
            1-Define F, G, U, V.
            2-Create an element from Non_Linear_Gaussian_HMM.
            3-Plot the data that has been generated.
            '''
            def Func_F(x):
                '''
                Function encoding the latent equation.
                x_{t+1} = Func_F(x_{t}) + noise_x
                '''
                dim = np.shape(x)[0]
                return x**2

            def Func_G(x):
                '''
                Function encoding the observable equation.
                y_{t+1} = Func_G(x_{t}) + noise_y
                '''
                return np.sin(x)
            # Parameters
            T = 10
            x_dim=1
            y_dim=1
            F_eigenvals = [0]
            G_eigenvals = [0]
            F,G,U,V = dg.HMM_matrices(x_dim,y_dim,F_eigenvals,G_eigenvals,U_std=0.3,V_std=0.3)
            print(F)
            print(G)
            print(U)
            print(V)
            NL_G_HMM = dg.Non_Linear_Gaussian_HMM(T, x_dim, y_dim, Func_F, Func_G, U, V)
            x_NL_G_HMM, y_NL_G_HMM = NL_G_HMM.generate_data()
            print(x_NL_G_HMM)
            print(y_NL_G_HMM)
            
        if generative_dictionary['Non_Linear_Student_HMM']:
            '''
            Non_Linear_Student_HMM.
            0- Set the parameters. 
            1-Define F, G, U, V.
            2-Create an element from Non_Linear_Gaussian_HMM.
            3-Plot the data that has been generated.
            '''
            def Func_F(x):
                '''
                Function encoding the latent equation.
                x_{t+1} = Func_F(x_{t}) + noise_x
                '''
                return np.cos(x)

            def Func_G(x):
                '''
                Function encoding the observable equation.
                y_{t+1} = Func_G(x_{t}) + noise_y
                '''
                return np.sin(x)
            # Parameters
            T = 10
            x_dim=1
            y_dim=1
            F_eigenvals = [0]
            G_eigenvals = [0]
            F,G,U,V = dg.HMM_matrices(x_dim,y_dim,F_eigenvals,G_eigenvals,U_std=0.3,V_std=0.3)
            print(F)
            print(G)
            print(U)
            print(V)
            mu_degrees_of_f, nu_degrees_of_f = 2, 3
            NL_S_HMM = dg.Non_Linear_Student_HMM(T, x_dim, y_dim, Func_F, Func_G, mu_degrees_of_f, nu_degrees_of_f)
            x_NL_S_HMM, y_NL_S_HMM = NL_S_HMM.generate_data()
            print(x_NL_S_HMM)
            print(y_NL_S_HMM)
        return 0

    except KeyError as ke:
        print(f"KeyError: {ke}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
if __name__ == "__main__":
    main()
    print('Everything has been executed.')
else:
    print("main() did not run")
