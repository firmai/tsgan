import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True


def L_cholesky(x, DD):

    jitter = 1e-15

    L_matrix = tf.cholesky(x + tf.constant(jitter,dtype=tf.float32) *tf.eye(DD, dtype=tf.float32))
    return L_matrix


class DKLITE(object):
    def __init__(self, input_dim, num_sample, num_sample_u,num_hidden=256, num_layers =2, learning_rate=0.001,
                 reg_var=50,reg_rec=25,output_dim=1,outcome_type='continuous'):


        self.learning_rate = learning_rate
        self.num_sample_u = num_sample_u
        self.num_layers = num_layers
        self.num_sample = num_sample
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.reg_var = reg_var
        self.reg_rec = reg_rec
        self.size_z = input_dim
        self.ml_primal = {}
        self.ker_inv = {}
        self.params = {}
        self.mean = {}
        self.beta = {}
        self.lam = {}
        self.num = {}
        self.r = {}

        ''' Initialize parameter weight '''
        self.params = self.initialize_weights()

        self.mu = tf.reduce_mean(self.T)

        zero_set = tf.where(self.T < 0.5)[:, 0]

        one_set = tf.where(self.T > 0.5)[:, 0]

        self.num['0'] = tf.reduce_sum(1 - self.T)
        self.num['1'] = tf.reduce_sum(self.T)

        self.Z_train, self.Prob_train_0,self.Prob_train_1  = self.Encoder(self.X)

        self.Z_test, self.Prob_test_0, self.Prob_test_1 = self.Encoder(self.X_u)

        self.Z = tf.concat([self.Z_train, self.Z_test], axis=0)

        X_h = self.Decoder(self.Z)

        #X_h_test = self.Decoder(self.Z_test)
        #

        self.loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.concat([self.X,self.X_u],axis=0) - X_h), axis=1))


        if outcome_type == 'continuous':


            Z_0 = tf.gather(self.Z_train, zero_set)
            Y_0 = tf.gather(self.Y, zero_set)

            Z_1 = tf.gather(self.Z_train, one_set)
            Y_1 = tf.gather(self.Y, one_set)

            mean_0 = tf.reduce_mean(Y_0)
            std_0 = tf.math.reduce_std(Y_0)
            mean_1 = tf.reduce_mean(Y_1)
            std_1 = tf.math.reduce_std(Y_1)

            Y_0 = (Y_0-mean_0)/std_0
            Y_1 = (Y_1 - mean_1) / std_1


            self.GP_NN(Y_0, Z_0, 0)


            self.GP_NN(Y_1, Z_1,1)




            self.var_0 = tf.reduce_mean(tf.diag_part(tf.matmul(Z_1,tf.matmul(self.ker_inv['0'], tf.transpose(Z_1)))))/self.lam['0']



            self.var_1 = tf.reduce_mean(tf.diag_part(tf.matmul(Z_0,tf.matmul(self.ker_inv['1'], tf.transpose(Z_0)))))/self.lam['1']


            self.ele_var_0_tr = tf.diag_part(tf.matmul(self.Z_train,tf.matmul(self.ker_inv['0'], tf.transpose(self.Z_train))))/self.lam['0']
            self.ele_var_1_tr = tf.diag_part(tf.matmul(self.Z_train,tf.matmul(self.ker_inv['1'], tf.transpose(self.Z_train))))/self.lam['1']

            self.ele_var_0_te = tf.diag_part(tf.matmul(self.Z_test,tf.matmul(self.ker_inv['0'], tf.transpose(self.Z_test))))/self.lam['0']
            self.ele_var_1_te = tf.diag_part(tf.matmul(self.Z_test,tf.matmul(self.ker_inv['1'], tf.transpose(self.Z_test))))/self.lam['1']


            pred_tr_0 = tf.matmul(self.Z_train, self.mean['0'])*std_0 + mean_0
            pred_tr_1 = tf.matmul(self.Z_train, self.mean['1'])*std_1 + mean_1
            pred_te_0 = tf.matmul(self.Z_test, self.mean['0'])*std_0 + mean_0
            pred_te_1 = tf.matmul(self.Z_test, self.mean['1'])*std_1 + mean_1


            self.Y_train = tf.concat([pred_tr_0,pred_tr_1],axis=1)

            self.Y_test = tf.concat([pred_te_0,pred_te_1],axis=1)


            self.prediction_loss = self.ml_primal['0']+ self.ml_primal['1']+ self.reg_var *(self.var_0 + self.var_1)+ self.reg_rec * self.loss_1



        elif outcome_type == 'binary':


            Y_0 = tf.gather(self.Y, zero_set)
            self.Prob_train_0_f = tf.gather(self.Prob_train_0, zero_set)
            self.Prob_train_0_cf = tf.gather(self.Prob_train_0, one_set)


            Y_1 = tf.gather(self.Y, one_set)
            self.Prob_train_1_f = tf.gather(self.Prob_train_1, one_set)
            self.Prob_train_1_cf = tf.gather(self.Prob_train_1, zero_set)




            self.binary_loss_0 = -tf.reduce_mean(Y_0 * tf.log(self.Prob_train_0_f+1e-20) +(1-Y_0) * tf.log(1-self.Prob_train_0_f+1e-20))
            self.binary_loss_1 = -tf.reduce_mean(Y_1 * tf.log(self.Prob_train_1_f+1e-20) +(1-Y_1) * tf.log(1-self.Prob_train_1_f+1e-20))



            self.entropy_0 = tf.reduce_mean(self.Prob_train_0_cf*tf.log(self.Prob_train_0_cf) + (1-self.Prob_train_0_cf)*tf.log(1-self.Prob_train_0_cf))
            self.entropy_1 = tf.reduce_mean(self.Prob_train_1_cf*tf.log(self.Prob_train_1_cf) + (1-self.Prob_train_1_cf)*tf.log(1-self.Prob_train_1_cf))
            self.entropy_test = tf.reduce_mean(self.Prob_test_0*tf.log(self.Prob_test_0) + (1-self.Prob_test_0)*tf.log(1-self.Prob_test_0)) \
                             + tf.reduce_mean(self.Prob_test_1*tf.log(self.Prob_test_1) + (1-self.Prob_test_1)*tf.log(1-self.Prob_test_1))


            self.Y_train_0 = tf.cast(self.Prob_train_0>0.5,dtype=tf.float32)
            self.Y_train_1 = tf.cast(self.Prob_train_1>0.5,dtype=tf.float32)
            self.Y_test_0 = tf.cast(self.Prob_test_0>0.5,dtype=tf.float32)
            self.Y_test_1 = tf.cast(self.Prob_test_1>0.5,dtype=tf.float32)

            self.Y_train = tf.concat([self.Y_train_0,self.Y_train_1],axis=1)
            self.Y_test = tf.concat([self.Y_test_0,self.Y_test_1], axis=1)


            #self.entropy_0 + self.entropy_1+
            self.prediction_loss = self.binary_loss_0 + self.binary_loss_1 + self.reg_var *(self.entropy_0+self.entropy_1+self.entropy_test)+ self.reg_rec * self.loss_1



        else:

            raise AssertionError('Error: wrong outcome type')



        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.prediction_loss)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def element_var(self, X, Y, T, X_u):

        var_0_tr, var_1_tr,var_0_te, var_1_te = self.sess.run([self.ele_var_0_tr,self.ele_var_1_tr,self.ele_var_0_te,self.ele_var_1_te],
                feed_dict={self.X: X, self.X_u: X_u, self.Y: Y, self.T: T})


        return var_0_tr,var_1_tr,var_0_te,var_1_te





    def embed(self, X, Y, T, X_u):

        Z= self.sess.run(self.Z_train, feed_dict={self.X: X, self.X_u: X_u, self.Y: Y, self.T: T})

        return Z




    def fit(self, X, Y, T, X_u, num_iteration):


        for i in range(num_iteration):

            self.sess.run(self.optimizer, feed_dict={self.X: X, self.X_u: X_u, self.Y: Y, self.T: T})


    def pred(self, X, Y, T, X_u):


        Y_hat_train, Y_hat_test = self.sess.run([self.Y_train, self.Y_test], feed_dict={self.X: X, self.X_u: X_u, self.Y: Y, self.T: T})


        return Y_hat_train, Y_hat_test




    def destroy_graph(self):
        tf.reset_default_graph()






    def Encoder(self, X):

        X_h = tf.nn.tanh(tf.matmul(X, self.params['e_w_in']) + self.params['e_b_in'])

        for layer_i in range( self.num_layers):
            X_h = tf.nn.tanh(tf.matmul(X_h, self.params['e_w_' + str(layer_i)])+self.params['e_b_' + str(layer_i)])

        Z =  tf.nn.tanh(tf.matmul(X_h, self.params['e_w_' + str(self.num_layers)]) + self.params['e_b_' + str(self.num_layers)])

        Prob_0 = tf.nn.sigmoid(tf.matmul(Z, self.params['e_w_out_0']) + self.params['e_b_out_0'])

        Prob_1 = tf.nn.sigmoid(tf.matmul(Z, self.params['e_w_out_1']) + self.params['e_b_out_1'])

        return Z, Prob_0, Prob_1




    def Decoder(self,Z):

        Z_pred = tf.nn.tanh(tf.matmul(Z, self.params['d_w_in']) + self.params['d_b_in'])

        for layer_i in range(self.num_layers):
            Z_pred = tf.nn.tanh(tf.matmul(Z_pred, self.params['d_w_' + str(layer_i)]) + self.params['d_b_' + str(layer_i)])

        X_p = tf.matmul(Z_pred, self.params['d_w_' + str(self.num_layers)]) + self.params['d_b_' + str(self.num_layers)]

        return X_p






    def GP_NN(self, Y_f, Z_f,index):

        self.beta[str(index)] = tf.nn.softplus(self.params['beta_' + str(index)])

        self.lam[str(index)] = tf.nn.softplus(self.params['lambda_' + str(index)])

        self.r[str(index)] = self.beta[str(index)] / self.lam[str(index)]

        self.DD = tf.shape(Z_f)[1]


        phi_phi = self.r[str(index)] * tf.matmul(tf.transpose(Z_f), Z_f)

        Ker = phi_phi + tf.eye(self.DD, dtype=tf.float32)

        L_matrix = L_cholesky(Ker, self.DD)

        L_inv_reduce = tf.linalg.triangular_solve(L_matrix, rhs=tf.eye(self.DD, dtype=tf.float32))

        L_y = tf.matmul(L_inv_reduce, tf.matmul(tf.transpose(Z_f), Y_f))

        term1 = tf.constant(0.5, dtype=tf.float32) * self.beta[str(index)] * (tf.reduce_sum(tf.square(Y_f)) - self.r[str(index)] * tf.reduce_sum(tf.square(L_y)))

        term2 = tf.reduce_sum(tf.log(tf.linalg.diag_part(L_matrix)))

        term3 = -tf.constant(0.5, dtype=tf.float32) * tf.cast(self.num[str(index)], dtype=tf.float32) * tf.log(self.beta[str(index)])

        self.ml_primal[str(index)] = (term1+ term2+ term3)/tf.cast(self.num[str(index)], dtype=tf.float32)

        self.ker_inv[str(index)] = tf.matmul(tf.transpose(L_inv_reduce), L_inv_reduce)

        self.mean[str(index)] = self.r[str(index)] * tf.matmul(tf.transpose(L_inv_reduce), L_y)



    def initialize_weights(self):


        self.X = tf.placeholder(tf.float32, [None, self.input_dim])

        self.X_u = tf.placeholder(tf.float32, [None, self.input_dim])

        self.Y = tf.placeholder(tf.float32, [None, 1])

        self.T = tf.placeholder(tf.float32, [None, 1])



        all_weights = {}

        name_beta = 'beta_0'
        all_weights[name_beta] = tf.Variable(0.00001*tf.ones([1,1],tf.float32),
                                             name = name_beta, trainable=True)

        name_lambda = 'lambda_0'
        all_weights[name_lambda] = tf.Variable(0.00001*tf.ones([1,1],tf.float32),
                                             name = name_lambda, trainable=True)

        name_beta = 'beta_1'
        all_weights[name_beta] = tf.Variable(0.00001*tf.ones([1,1],tf.float32),
                                             name = name_beta, trainable=True)

        name_lambda = 'lambda_1'
        all_weights[name_lambda] = tf.Variable(0.00001*tf.ones([1,1],tf.float32),
                                             name = name_lambda, trainable=True)



        ''' Input layer of the encoder '''
        name_wi = 'e_w_in'
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.input_dim, self.num_hidden], trainable=True)
        name_bi = 'e_b_in'
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.num_hidden], trainable=True)


        ''' Hidden layer of the encoder '''
        for layer_i in range(self.num_layers):

            name_wi = 'e_w_' + str(layer_i)
            all_weights[name_wi ] = tf.get_variable(name =name_wi,  shape=[self.num_hidden,self.num_hidden],  trainable=True)

            name_bi = 'e_b_' + str(layer_i)
            all_weights[name_bi] = tf.get_variable(name =name_bi, shape = [self.num_hidden], trainable=True)

        ''' Final layer of the encoder '''
        name_wi = 'e_w_' + str(self.num_layers)
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.num_hidden, self.size_z], trainable=True)

        name_bi = 'e_b_' + str(self.num_layers)
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.size_z], trainable=True)


        name_wi = 'e_w_out_0'
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.size_z, self.output_dim], trainable=True)

        name_bi = 'e_b_out_0'
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.output_dim], trainable=True)

        name_wi = 'e_w_out_1'
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.size_z, self.output_dim], trainable=True)

        name_bi = 'e_b_out_1'
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.output_dim], trainable=True)



        ''' Input layer of the decoder '''
        name_wi = 'd_w_in'
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.size_z, self.num_hidden],trainable=True)

        name_bi = 'd_b_in'
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[self.num_hidden],trainable=True)


        ''' Hidden layer of the decoder '''
        for layer_i in range(self.num_layers):


            name_wi = 'd_w_' + str(layer_i)
            all_weights[name_wi ] = tf.get_variable(name =name_wi,  shape=[self.num_hidden,self.num_hidden],trainable=True)

            name_bi = 'd_b_' + str(layer_i)
            all_weights[name_bi] = tf.get_variable(name =name_bi, shape = [self.num_hidden],trainable=True)



        ''' Final layer of the decoder '''
        name_wi = 'd_w_' + str(self.num_layers)
        all_weights[name_wi] = tf.get_variable(name=name_wi, shape=[self.num_hidden, self.input_dim],trainable=True)

        name_bi = 'd_b_' + str(self.num_layers)
        all_weights[name_bi] = tf.get_variable(name=name_bi, shape=[(self.input_dim)],trainable=True)


        return all_weights