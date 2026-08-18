import numpy as np
import pandas as pd
from pyomo.environ import *
count = 0
   
def update_func(A_0, A, D_r, D_x):
    
    df = pd.DataFrame(A)
    df = pd.DataFrame(D_r)
    R_d = np.matmul(np.linalg.inv(A), D_r)#, np.linalg.inv(A.T))
    df = pd.DataFrame(R_d)
    # df.to_csv('R_d.csv', index=False)
    R_d_1 = np.matmul(np.matmul(np.linalg.inv(A), D_r), np.linalg.inv(A.T)) * 2
    df = pd.DataFrame(R_d_1)
    # df.to_csv('R_d_1.csv', index=False)
    df = pd.DataFrame(np.linalg.inv(A.T))
    # df.to_csv('inv_A_T.csv', index=False)
    X_d = np.matmul(np.linalg.inv(A), D_x)#, np.linalg.inv(A.T))
    df = pd.DataFrame(X_d)
    # df.to_csv('X_d.csv', index=False)
    X_d_1 = np.matmul(np.matmul(np.linalg.inv(A), D_x), np.linalg.inv(A.T)) *2
    df = pd.DataFrame(X_d_1)
    # df.to_csv('X_d_1.csv', index=False)
    A_dash = np.matmul(np.linalg.inv(A), A_0)
    
    return A_dash, R_d, X_d

def initialize_param():

    global count
    count = 0