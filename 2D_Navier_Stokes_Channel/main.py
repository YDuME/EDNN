# Yifan Du             dyifan1@jhu.edu
# Tamer A. Zaki        t.zaki@jhu.edu
# Johns Hokpins University
# US patent submitted 03/08/2021

# EDNN solver of heat equation

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



def main():
    # -----------------------------------------------------------------------------
    # Parameters for simulation configuration
    # -----------------------------------------------------------------------------
    # NN and solution directory
    case_name = "Channel/"
    # Numer of collocation points
    Nx = 32
    Ny = 32
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
    y2 = 2
    # Other parameters
    nu = 1
    dPdx = 1
    Nt = 1000000
    dt = 1e-3
    # ------------------------------------------------------------------------------
    # Generate the collocation points and initial condition array
    # ------------------------------------------------------------------------------
    x  = np.linspace(x1,x2,num=Nx, dtype=np.float32)
    y  = np.linspace(y1,y2,num=Ny, dtype=np.float32)
    X,Y = np.meshgrid(x,y,indexing = 'ij')
    Xi = X[1:-1,1:-1]
    Yi = Y[1:-1,1:-1]
    #Initial condition
    U0 = 0.0 * Xi
    V0 = 0.0 * Yi
    Input = np.concatenate((X.reshape((Nx)*(Ny),-1),Y.reshape((Nx)*(Ny),-1)),axis = 1)
    InputInterior = np.concatenate((Xi.reshape((Nx-2)*(Ny-2),-1),Yi.reshape((Nx-2)*(Ny-2),-1)),axis = 1)
    InitInterior = np.concatenate((U0.reshape((Nx-2)*(Ny-2),-1),V0.reshape((Nx-2)*(Ny-2),-1)),axis = 1)
    
    Index = np.arange(Nx*Ny).reshape(Nx,Ny)
    
    Index = np.arange(Nx*Ny).reshape(Nx,Ny)
    IE = (0.0 * Index + Index[-1,:].reshape(1,Ny)).astype(np.int).reshape((Nx)*(Ny),-1)
    IW = (0.0 * Index + Index[0,:].reshape(1,Ny)).astype(np.int).reshape((Nx)*(Ny),-1)
    IN = (0.0 * Index + Index[:,-1].reshape(Nx,1)).astype(np.int).reshape((Nx)*(Ny),-1)
    IS = (0.0 * Index + Index[:,0].reshape(Nx,1)).astype(np.int).reshape((Nx)*(Ny),-1)
    BI = np.concatenate((IE,IW,IN,IS),axis = 1)

    #Extract the index of boundary points for the enforcement of B.C.     
    IEInterior = (0.0 * Index[1:-1,1:-1] + Index[-1,1:-1].reshape(1,Ny-2)).astype(np.int).reshape((Nx-2)*(Ny-2),-1)
    IWInterior = (0.0 * Index[1:-1,1:-1] + Index[0,1:-1].reshape(1,Ny-2)).astype(np.int).reshape((Nx-2)*(Ny-2),-1)
    INInterior = (0.0 * Index[1:-1,1:-1] + Index[1:-1,-1].reshape(Nx-2,1)).astype(np.int).reshape((Nx-2)*(Ny-2),-1)
    ISInterior = (0.0 * Index[1:-1,1:-1] + Index[1:-1,0].reshape(Nx-2,1)).astype(np.int).reshape((Nx-2)*(Ny-2),-1)
    BIInterior = np.concatenate((IEInterior,IWInterior,INInterior,ISInterior),axis = 1)
    
    
    
    
    try: 
        nrestart = int(np.genfromtxt(case_name + 'nrestart'))
    except OSError: 
        nrestart = 0
    
    # -----------------------------------------------------------------------------
    # Initialize EDNN
    # -----------------------------------------------------------------------------
    lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 10000000, 0.9)
    layers  = [3] + 4*[10] + [1]
    
    EDNN = EvolutionalDNN(layers,
                             rhs = rhs_2d_adv_diff_eqs, 
                             marching_method = Runge_Kutta,
                             dest=case_name,activation = 'tanh',
                             optimizer=keras.optimizers.Adam(lr),
                             eq_params=[nu,dPdx],
                             restore=True)
    print('Learning rate:', EDNN.optimizer._decayed_lr(tf.float32))
   
    
    if Initial: 
        alpha = 0.0
        t0 = time.time()
        tot_eps = 500
        # Train the initial condition tot_eps epochs, 
        for i in range(tot_eps):
            InputInteriorBoundary = Input[BIInterior]
            EDNN.train(InputInterior, InputInteriorBoundary, InitInterior, epochs=1,
                   batch_size=100, verbose=False, timer=False)
        # Evaluate and output the initial condition 
        InputBoundary = tf.convert_to_tensor(Input[BI])
        Input = tf.convert_to_tensor(Input)
        [U,V] = BNN.output(Input,InputBoundary)
        U = U.numpy().reshape((Nx,Ny))
        V = V.numpy().reshape((Nx,Ny))
        X.dump(boundary_dir+'X')
        Y.dump(boundary_dir+'Y')
        U.dump(boundary_dir+'U')
        V.dump(boundary_dir+'V')
    
    else:
        InputInteriorBoundary = tf.convert_to_tensor(Input[BIInterior])
        InputBoundary = tf.convert_to_tensor(Input[BI])
        InputInterior = tf.convert_to_tensor(InputInterior)
        Input = tf.convert_to_tensor(Input)
    
        nbatch = 30 * 10
        params_marching = [dt,nbatch]
        # March the EDNN class till Nt time steps. 
        for n in range(nrestart,Nt):
            print('time step', n)
            EDNN.Marching(InputInterior,InputInteriorBoundary,params_marching)
            [Uh, Vh] = EDNN.output(Input,InputBoundary)
            # The solution field is stored every time step. 
            U = Uh.numpy().reshape((Nx, Ny))
            V = Vh.numpy().reshape((Nx, Ny))
            X.dump(case_name+'X'+str(n))
            Y.dump(case_name+'Y'+str(n))
            U.dump(case_name+'U'+str(n))
            V.dump(case_name+'V'+str(n))
            EDNN.save_NN()
            
            np.savetxt(case_name+'nrestart',np.array([n]))
if __name__ == "__main__":
    main()
   




