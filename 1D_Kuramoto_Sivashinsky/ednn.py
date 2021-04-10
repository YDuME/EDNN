# Yifan Du             dyifan1@jhu.edu
# Tamer A. Zaki        t.zaki@jhu.edu
# Johns Hokpins University
# US patent submitted 03/08/2021

# EDNN class

# Requires Python 3.* and Tensorflow 2.0

import os
import copy
import numpy as np
import tensorflow as tf
from   tensorflow import keras
import time
from pdb import set_trace as keyboard
#tf.config.experimental_run_functions_eagerly(True)
class EvolutionalDNN:
    """
    General EDNN class

    A primary implementation of evolutional deep neural network (EDNN). 

    The EvolutionalDNN derived class containts the neural network and all
    related parameters and functions of EDNN. 

    The EDNN solved a evolutional PDE of the following type: 

                    \frac{\partial u}{\partial t} = N(u)              (*)

    A few definitions before proceeding to the initialization parameters:
    
    din   = input dims
    dout  = output dims
    dpar  = number of parameters used by the pde

    Parameters
    ----------

    layers : list
        Shape of the NN. The first element must be din, and the last one must
        be dout.
    rhs : tensorflow decorated function. 
          The right-hand-side N(u) in the equation (*). 
    marching_method: tensorflow decorated function. 
          The time-marching scheme for the neural network evolution
    dest : str [optional]
        Path for the neural network. 
    activation : str [optional]
        Activation function to be used. Default is 'tanh'.
    optimizer : keras.optimizer instance [optional]
        Optimizer to be used in the gradient descent. Default is Adam with
        fixed learning rate equal to 5e-4. Specifically it is used for the 
        optimization of initial condition. 
    eq_params : list [optional]
        List of parameters to be used in rhs.
    restore : bool [optional]
        If True, it checks if a checkpoint exists in dest. If a checkpoint
        exists it restores the modelfrom there. Default is True.
    """
    # Initialize the class
    def __init__(self,
                 layers,
                 rhs,
                 marching_method, 
                 dest='./',
                 activation='tanh',
                 optimizer=keras.optimizers.Adam(lr=5e-4),
                 eq_params=[],
                 restore=True):

        # Numbers and dimensions
        self.din  = layers[0]
        self.dout = layers[-1]
        depth     = len(layers)-2
        width     = layers[1]

        # Extras
        self.dpar            = len(eq_params)
        self.dest            = dest
        self.eq_params       = eq_params
        self.eval_params     = copy.copy(eq_params)
        self.restore         = restore
        self.optimizer       = optimizer
        self.activation      = activation
        self.rhs             = rhs
        self.marching_method = marching_method
        # Activation function
        if activation=='tanh':
            self.act_fn = keras.activations.tanh
        elif activation=='relu':
            self.act_fn = keras.activations.relu
        elif activation == 'adaptive_global':
            self.act_fn = AdaptiveAct()

        # Input definition
        coords = keras.layers.Input(self.din, name='coords')

        # Normalzation
        hidden  = coords

        # Hidden layers
        for ii in range(depth):
            hidden = keras.layers.Dense(width)(hidden)
            if activation=='adaptive_layer':
                self.act_fn = AdaptiveAct()
            hidden = self.act_fn(hidden)

        # Output definition
        fields = keras.layers.Dense(self.dout, name='fields')(hidden)

        cte   = keras.layers.Lambda(lambda x: 0*x[:,0:1]+1)(coords)
        dummy = keras.layers.Dense(1, use_bias=False)(cte)
        self.inv_outputs = [dummy]

        # Create model
        model = keras.Model(inputs=coords, outputs=[fields]+self.inv_outputs)
        self.model = model
        self.num_trainable_vars = np.sum([np.prod(v.shape)
                                          for v in self.model.trainable_variables])

        # Can be modified from the outside before calling PINN.train

        # Create save checkpoints / Load if existing previous
        self.ckpt    = tf.train.Checkpoint(step=tf.Variable(0),
                                           model=self.model,
                                           optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.dest, max_to_keep=5)
        if self.restore:
            self.ckpt.restore(self.manager.latest_checkpoint)


    def train(self,
              X_data, Y_data, 
              epochs, batch_size,
              flags=None,
              rnd_order_training=True,
              verbose=False,
              print_freq=1,
              save_freq=1,
              timer=False):
        """
        Train function

        Loss functions are written to output.dat

        Parameters
        ----------

        X_data : ndarray
            Coordinates where the initial data are located. 
            Must have shape (:, din).
        Y_data : ndarray
            The initial data at the corresponding lodation. 
            Must have shape (:, dout). First dimension must be the same as
            X_data. 
        epochs : int
            Number of epochs to train
        batch_size : int
            Size of batches
        rnd_order_training: bool [optional]
            If True points are taking randomly from each group when forming a
            batch, if False points are taking in order, following the order the
            data was supplied. Default is True.
        verbose : bool [optional]
            Verbose output or not. Default is False.
        print_freq : int [optional]
            Print status frequency. Default is 1.
        save_freq : int [optional]
            Save model frequency. Default is 1.
        timer : bool [optional]
            If True, print time per batch for the first 10 batches. Default is
            False.
        """

        len_data = X_data.shape[0]
        batches = len_data // batch_size


        # Expand flags
        if flags is None:
            flags = [1 for _ in range(len_data)]
        flags     = np.array(flags)
        flag_idxs = [np.where(flags==f)[0] for f in np.unique(flags)]

        # Run epochs
        ep0     = int(self.ckpt.step)
        for ep in range(ep0, ep0+epochs):
            for ba in range(batches):

                # Create batches and cast to TF objects
                (X_batch,
                 Y_batch) = get_mini_batch(X_data,
                                          Y_data,
                                          ba,
                                          batches,
                                          flag_idxs,
                                          random=rnd_order_training)
                X_batch = tf.convert_to_tensor(X_batch)
                Y_batch = tf.convert_to_tensor(Y_batch)
                ba_counter = tf.constant(ba)

                if timer: t0 = time.time()
                loss_data = self.training_step(X_batch, Y_batch)
                print (ep, ba, loss_data.numpy())
                if timer:
                    print("Time per batch:", time.time()-t0)
                    if ba>10: timer = False

            #Print status
            if ep%print_freq==0:
                self.print_status(ep,
                                  loss_data,
                                  verbose=verbose)
            # Save progress
            self.ckpt.step.assign_add(1)
            if ep%save_freq==0:
                self.manager.save()

    # EDNN function to save the network state. 
    def save_NN(self):
        self.manager.save()
        return None
        

    @tf.function
    def eval_NN_grad(self, X):
        '''
        Evaluate the Jacobian of solution variable with respect to network 
        parameters at a set of given spatial locations
        X : a set of spatial coordinates corresponding to the collocation 
            points to enforce the equation. 
        '''
        with tf.GradientTape(persistent=True) as tape:
            Y = self.output(X)
        gradients = [tape.jacobian(y,
                    self.model.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO) for y in Y]
        del tape
        return gradients
        
    def eval_rhs(self,Input,nbatch,w=None):
        '''
        Evaluate the RHS of the following evolution equation of network parameters: 

                    dW/dt = gamma(t)
        
        A least square problem is solved to find the time derivative of network 
        parameters. 

        Input: coordinates of spatial collocation points
        nbatch: number of batches. The value of nbatch depends on the scale of 
                given problem and available memory. 
        '''

        if w is not None:
            wtmp = self.get_weights_np()
            self.set_weights_np(w)
        Ju = []
        for x in range(int(len(Input)/nbatch)):
            JUV = self.eval_NN_grad(tf.reshape(Input[x*nbatch:(x+1)*nbatch,:],[nbatch,-1]))
            J = JUV[0]
            indk = [i for i in range(len(J))][::2]
            indb = [i for i in range(len(J))][1::2]
            Jn = [j.numpy() for j in J]
            #Jn = [jn.reshape(-1,*jn.shape[-2:]) for jn in Jn]
            Jn = [jn.reshape(jn.shape[0],-1) for jn in Jn]
            Jn = np.concatenate(Jn,axis = 1)
            Ju = Ju + [Jn]

        Ju = np.concatenate(Ju,axis = 0)
        JJ = Ju
        dudt = self.rhs(self.output, Input, self.eq_params)
        dudt = dudt.numpy().reshape((-1))
        sol = np.linalg.lstsq(JJ,dudt,rcond = 1e-3)
        dwdt = sol[0]
        if w is not None:
            self.set_weights_np(wtmp)

        return dwdt

    # Set network weights from numpy array
    def set_weights_np(self,Wnew):
        W = self.model.get_weights()
        k = 0
        for i in range(len(W)):
            sp = W[i].shape
            sz = W[i].size
            W[i] = Wnew[k:k+sz].reshape(sp)
            k = k + sz
        self.model.set_weights(W)
        return None
    # Evaluate network weights on numpy array
    def get_weights_np(self):
        W = self.model.get_weights()
        Wnp = np.concatenate([w.reshape(-1) for w in W])
        return Wnp

    # Higher level function to execute the time marching.         
    def Marching(self, Input, params):
        W = self.get_weights_np()
        W = self.marching_method(W,self.eval_rhs,Input,params)
        self.set_weights_np(W)
        return None

    # A wrapper to enforce periodicity on a given 
    # neural network self.model 
    @tf.function
    def output(self, X):

        k = tf.constant(np.pi/10.0)
        sinX = tf.reshape(tf.sin(k*X[:,0]),[-1,1])
        cosX = tf.reshape(tf.cos(k*X[:,0]),[-1,1])
        XT = tf.concat([sinX,cosX],axis = 1)
        Ypred = self.model(XT)[0]
        u = Ypred[:,0]
        u   = tf.reshape(u,[-1])

        return [u]

    # For training of the EDNN at initial time on a batch of data. 
    @tf.function
    def training_step(self, X_batch, Y_batch):
        with tf.GradientTape(persistent=True) as tape:
            Ypred = self.output(X_batch)
            aux = [tf.reduce_mean(tf.square(Ypred[i] - Y_batch[:,i])) for i in range(len(Ypred))]
            loss_data = tf.add_n(aux)
            loss = loss_data
        gradients_data = tape.gradient(loss_data,
                    self.model.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)
        del tape
        gradients = [x for x in gradients_data]
        self.optimizer.apply_gradients(zip(gradients,
                    self.model.trainable_variables))

        return loss_data

    def print_status(self, ep, lu, verbose=False):
        """ Print status function """

        # Loss functions
        output_file = open(self.dest + 'output.dat', 'a')
        print(ep, f'{lu}', 
              file=output_file)
        output_file.close()
        if verbose:
            print(ep, f'{lu}', f'{lf}')

# Get a mini batch of data from the full dataset. 
def get_mini_batch(X, Y, ba, batches, flag_idxs, random=True):
    idxs = []
    for fi in flag_idxs:
        if random:
            sl = np.random.choice(fi, len(fi)//batches)
            idxs.append(sl)
        else:
            flag_size = len(fi)//batches
            sl = slice(ba*flag_size, (ba+1)*flag_size)
            idxs.append(fi[sl])
    idxs = np.concatenate(idxs)
    return X[idxs], Y[idxs]

