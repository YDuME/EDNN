# Yifan Du             dyifan1@jhu.edu
# Tamer A. Zaki        t.zaki@jhu.edu
# Johns Hokpins University
# US patent submitted 03/08/2021

# EDNN solver of 2D Navier Stokes equation with periodicity

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


def Kflowinit(X,Y):
    U =   0.0 * Y
    V =   np.sin(X)
    return [U,V]

def main():
    # -----------------------------------------------------------------------------
    # Parameters for simulation configuration
    # -----------------------------------------------------------------------------
    # NN and solution directory
    case_name = "KflowNN/"
    # Numer of collocation points
    Nx = 65
    Ny = 65
    # if Initial == True, train the neural network for initial condition
    # if Initial == False, march the initial network stored in case_name
    if sys.argv[1] == '0':
        Initial = True
    elif sys.argv[1] == '1':
        Initial = False
    else:
        sys.exit("Wrong flag specified")
    # Physical domain
    x1 = 0.0
    x2 = 2*np.pi
    y1 = 0.0
    y2 = 2*np.pi
    # Other parameters
    nu = 1e-2
    chi = 1e-1
    Nt = 1000
    dt = 1e-2
    tot_eps = 10
    # ------------------------------------------------------------------------------
    # Generate the collocation points and initial condition array
    # ------------------------------------------------------------------------------
    x  = np.linspace(x1,x2,num=Nx, dtype=np.float32)
    y  = np.linspace(y1,y2,num=Ny, dtype=np.float32)
    X,Y = np.meshgrid(x,y,indexing = 'ij')
    Xi = X[1:-1,1:-1]
    Yi = Y[1:-1,1:-1]
    Input = np.concatenate((X.reshape((Nx)*(Ny),-1),Y.reshape((Nx)*(Ny),-1)),axis = 1)
    InputPeriodic = np.concatenate((np.sin(X).reshape((Nx)*(Ny),-1),
                                    np.cos(X).reshape((Nx)*(Ny),-1),
                                    np.sin(Y).reshape((Nx)*(Ny),-1),
                                    np.cos(Y).reshape((Nx)*(Ny),-1)),axis = 1)
    #Initial condition
    InitU, InitV = Kflowinit(X,Y)
    InputInit = np.concatenate((InitU.reshape((Nx)*(Ny),-1),InitV.reshape((Nx)*(Ny),-1)),axis = 1)
    
    Index = np.arange(Nx*Ny).reshape(Nx,Ny)
    
    try: 
        nrestart = int(np.genfromtxt(case_name + 'nrestart'))
    except OSError: 
        nrestart = 0
    
    # -----------------------------------------------------------------------------
    # Initialize EDNN
    # -----------------------------------------------------------------------------
    lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 10000000, 0.9)
    layers  =[4] + 4*[20] + [1]
    
    EDNN = EvolutionalDNN(layers,
                             rhs = rhs_2d_adv_diff_eqs, 
                             marching_method = Runge_Kutta,
                             dest=case_name,activation = 'tanh',
                             optimizer=keras.optimizers.Adam(lr),
                             eq_params=[nu,chi],
                             restore=True)
    print('Learning rate:', EDNN.optimizer._decayed_lr(tf.float32))
    
    #print('Learning rate:', PINN.optimizer._decayed_lr(tf.float32))
    
    
    if Initial: 
        t0 = time.time()
        # Train the initial condition tot_eps epochs, 
        for i in range(tot_eps):
            EDNN.train(Input,  InputInit, epochs=1,
                   batch_size=100, verbose=False, timer=False)
        # Evaluate and output the initial condition 
        Input = tf.convert_to_tensor(Input)
        [U,V] = EDNN.output(Input)
        U = U.numpy().reshape((Nx,Ny))
        V = V.numpy().reshape((Nx,Ny))
        X.dump(case_name+'X')
        Y.dump(case_name+'Y')
        U.dump(case_name+'U')
        V.dump(case_name+'V')
    
    
    else:
        Input = tf.convert_to_tensor(Input)
    
        nbatch = 1 * 65
        params_marching = [dt,nbatch]
        # March the EDNN class till Nt time steps. 
        for n in range(nrestart+1,Nt):
            print('time step', n)
            EDNN.Marching(Input,params_marching)
            # The velocity field is stored every time step.
            [U,V] = EDNN.output(Input)
            Phi = EDNN.model(InputPeriodic)
            Omega = EDNN.output_vorticity(Input)
            U = U.numpy().reshape((Nx, Ny))
            V = V.numpy().reshape((Nx, Ny))
            Phi = Phi[0].numpy().reshape((Nx, Ny))
            Omega = Omega.numpy().reshape((Nx, Ny))
            Phi = Phi - np.mean(Phi)
    
            X.dump(case_name+'X'+str(n))
            Y.dump(case_name+'Y'+str(n))
            U.dump(case_name+'U'+str(n))
            V.dump(case_name+'V'+str(n))
            Phi.dump(case_name+'Phi'+str(n))
            Omega.dump(case_name+'Omega'+str(n))
            EDNN.save_NN()
            
            np.savetxt(case_name+'nrestart',np.array([n]))

if __name__ == "__main__":
    main()


