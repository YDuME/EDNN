# Yifan Du             dyifan1@jhu.edu
# Tamer A. Zaki        t.zaki@jhu.edu
# Johns Hokpins University
# US patent submitted 03/08/2021

# EDNN solver of Kuramoto-Sivashinsky equation

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import matplotlib.pyplot as plt
import time
import sys
from pdb import set_trace as keyboard
from ednn import EvolutionalDNN
from marching_schemes import *
from rhs import *


def SinInit(X): 
    u = - np.sin(np.pi*X/10)
    return u

def main():
    # -----------------------------------------------------------------------------
    # Parameters for simulation configuration
    # -----------------------------------------------------------------------------
    # NN and solution directory
    case_name = "KSNN/"
    # Numer of collocation points
    Nx = 1000
    # if Initial == True, train the neural network for initial condition
    # if Initial == False, march the initial network stored in case_namei
    if sys.argv[1] == '0':
        Initial = True
    elif sys.argv[1] == '1':
        Initial = False
    else:
        sys.exit("Wrong flag specified")
    # Physical domain
    x1 = -10.0
    x2 =  10.0
    # Other parameters
    dt = 1e-3
    Nt = 100000
    # ------------------------------------------------------------------------------
    # Generate the collocation points and initial condition array
    # ------------------------------------------------------------------------------
    X  = np.linspace(x1,x2,num=Nx, dtype=np.float32)
    Input = X.reshape(Nx,-1)
    InitU = SinInit(X)
    Init = InitU.reshape(Nx,-1)
    
    try:
        nrestart = int(np.genfromtxt(case_name + 'nrestart'))
    except OSError:
        nrestart = 0
    
    # -----------------------------------------------------------------------------
    # Initialize PINN
    # -----------------------------------------------------------------------------
    lr = keras.optimizers.schedules.ExponentialDecay(1e-4, 10000000, 0.9)
    layers  = [2] + 4*[20] + [1]
    
    EDNN = EvolutionalDNN(layers,
                             rhs = rhs_Kuramoto_Sivashinsky, 
                             marching_method = Runge_Kutta,
                             dest=case_name,activation = 'tanh',
                             optimizer=keras.optimizers.Adam(lr),
                             eq_params=[0.0],
                             restore=True)
    print('Learning rate:', EDNN.optimizer._decayed_lr(tf.float32))
    
    if Initial: 
        alpha = 0.0
        t0 = time.time()
        tot_eps = 100
        # Train the initial condition tot_eps epochs, 
        for i in range(tot_eps):
            EDNN.train(Input, Init,epochs=1,
                   batch_size=100, verbose=False, timer=False)
        Input = tf.convert_to_tensor(Input)
        u = EDNN.output(Input)
        u = u[0].numpy().reshape(Nx)
        X.dump(case_name+'X')
        u.dump(case_name+'U')
    else:
        Input = tf.convert_to_tensor(Input)
        nbatch = 100
        params_marching = [dt,nbatch]
        # March the EDNN class till Nt time steps.  
        for n in range(0,Nt):
            print('time step', n)
            EDNN.Marching(Input,params_marching)
            # The solution field is stored every time step.
            [U] = EDNN.output(Input)
            U   = U.numpy().reshape(Nx)
            X.dump(case_name+'X'+str(n))
            U.dump(case_name+'U'+str(n))
     
            EDNN.save_NN()

if __name__ == "__main__":
    main()



