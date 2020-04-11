'''
Jinsung Yoon (0*/13/2018)
PATEGAN
'''

#%% Packages
import tensorflow as tf
import numpy as np
from tqdm import tqdm

#%% Function Start

def PATE_GAN(X_train, Y_train, X_test, Y_test, epsilon, delta, niter, num_teachers):

    #%% Parameters
    # Batch size    
    mb_size = 128
    
    # Feature no
    X_dim = len(X_train[0,:])
    
    # Sample no
    no = len(X_train[:,0])
        
    # Random variable dimension
    z_dim = int(X_dim/4)
    
    # Hidden unit dimensions
    h_dim = int(X_dim)
    
    C_dim = 1
            
    # WGAN-GP Parameters
    lam = 10
    lr = 1e-4    
    
    lamda =np.sqrt(2*np.log(1.25*(10^(delta))))/epsilon

    #%% Data Preprocessing
    X_train = np.asarray(X_train)
    
    #%% Data Normalization
    Min_Val = np.min(X_train,0)
    
    X_train = X_train - Min_Val
    
    Max_Val = np.max(X_train,0)
    
    X_train = X_train / (Max_Val + 1e-8)
    
    #%% Algorithm Start

    #%% Necessary Functions

    # Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape = size, stddev = xavier_stddev)    
        
    # Sample from uniform distribution
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size = [m, n])
        
    # Sample from the real data
    def sample_X(m, n):
        return np.random.permutation(m)[:n]  
     
    #%% Placeholder
    
    # Feature
    X = tf.placeholder(tf.float32, shape = [None, X_dim])      
    # Label
    Y = tf.placeholder(tf.float32, shape = [None, C_dim])  
    # Random Variable    
    Z = tf.placeholder(tf.float32, shape = [None, z_dim])
    # Conditional Variable
    M = tf.placeholder(tf.float32, shape = [None, C_dim])
      
#%% Discriminator
    # Discriminator
    D_W1 = tf.Variable(xavier_init([X_dim + C_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim,h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    D_W3 = tf.Variable(xavier_init([h_dim,1]))
    D_b3 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
    
    #%% Generator

    G_W1 = tf.Variable(xavier_init([z_dim + C_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim,h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim,X_dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[X_dim]))
    
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    #%% Functions
    def generator(z, y):
        inputs = tf.concat([z,y], axis = 1)
        G_h1 = tf.nn.tanh(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
        G_log_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        
        return G_log_prob
    
    def discriminator(x, y):
        inputs = tf.concat([x,y], axis = 1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        out = (tf.matmul(D_h2, D_W3) + D_b3)
        
        return out
      
    #%% 
    # Structure
    G_sample = generator(Z, Y)
    D_real = discriminator(X, Y)
    D_fake = discriminator(G_sample, Y)

    #%%
    D_entire = tf.concat(axis = 0, values = [D_real, D_fake])    
    
    #%%

    # Replacement of Clipping algorithm to Penalty term
    # 1. Line 6 in Algorithm 1
    eps = tf.random_uniform([mb_size, 1], minval = 0., maxval = 1.)
    X_inter = eps*X + (1. - eps) * G_sample

    # 2. Line 7 in Algorithm 1
    grad = tf.gradients(discriminator(X_inter, Y), [X_inter, Y])[0]
    grad_norm = tf.sqrt(tf.reduce_sum((grad)**2 + 1e-8, axis = 1))
    grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

    # Loss function
    D_loss = tf.reduce_mean((1-M) * D_entire) - tf.reduce_mean(M * D_entire) + grad_pen
    G_loss = -tf.reduce_mean(D_fake)

    # Solver
    D_solver = (tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(D_loss, var_list = theta_D))
    G_solver = (tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(G_loss, var_list = theta_G))
            
    #%%
    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
        
    #%%
    # Iterations
    for it in tqdm(range(niter)):

        for _ in range(num_teachers):
            #%% Teacher Training            
            Z_mb = sample_Z(mb_size, z_dim)            
            
            # Teacher 1
            X_idx = sample_X(no,mb_size)        
            X_mb = X_train[X_idx,:]  
            
            Y_mb = np.reshape(Y_train[X_idx], [mb_size,1])
            
            #%%
            
            M_real = np.ones([mb_size,])
            M_fake = np.zeros([mb_size,])

            M_entire = np.concatenate((M_real, M_fake),0)
                        
            Normal_Add = np.random.normal(loc=0.0, scale=lamda, size = mb_size*2)
    
            M_entire = M_entire + Normal_Add
            
            M_entire = (M_entire > 0.5)            
            
            M_mb = np.reshape(M_entire.astype(float), (2*mb_size,1))
            
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, Z: Z_mb, M: M_mb, Y: Y_mb})
            
                    
        #%% Generator Training
                    
        Z_mb = sample_Z(mb_size, z_dim)   
        
        X_idx = sample_X(no,mb_size)        
        X_mb = X_train[X_idx,:]  
            
        Y_mb = np.reshape(Y_train[X_idx], [mb_size,1])
                    
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {Z: Z_mb, Y: Y_mb})

    #%%       

    #%% Output Generation
    
    New_X_train = sess.run([G_sample], feed_dict = {Z: sample_Z(len(X_train[:,0]), z_dim), Y: np.reshape(Y_train, [len(Y_train),1])})

    New_X_train = New_X_train[0]
    
    #### Renormalization
        
    New_X_train = New_X_train * (Max_Val + 1e-8)
    
    New_X_train = New_X_train + Min_Val
    
    ## Testing
    
    New_X_test = sess.run([G_sample], feed_dict = {Z: sample_Z(len(X_test[:,0]), z_dim), Y: np.reshape(Y_test, [len(Y_test),1])})

    New_X_test = New_X_test[0]
    
    #### Renormalization
        
    New_X_test = New_X_test * (Max_Val + 1e-8)
    
    New_X_test = New_X_test + Min_Val
    
    return New_X_train, Y_train, New_X_test, Y_test
