#!/bin/env python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from enum import Enum
import numpy as np
import uuid, time, sys, os, math

class MLP( object ):
    '''
    A Multi-Layer Perceptron (MLP) Object
    '''

    ##############################################################
    ##############################################################
    class Optimizer( Enum ):
        '''
        optimization methods
        '''
        SGD     = 1     # Stochastic Gradient Descent
        ADAM    = 2     # Adam
        SNGD    = 3     # Relative Natural Gradient Descent

    class Arch( Enum ):
        '''
        different architectures
        '''
        PLAIN   = 1     # Plain MLP
        BNA     = 2     # BN after activation
        BNB     = 3     # BN before activation
    ##############################################################
    ##############################################################

    def __batch_inv( self, m, dim ):
        '''
        matrix inverse for natural gradient optimization
        to avoid singular m, compute instead

            (m + epsilon I)^{-1}

        where epsilon is a small positive number

        m: 3D tensor of size b x dim x dim
        '''

        def regular():
            epsilonI = self.ngd_rel_eps * tf.expand_dims( tf.expand_dims( tf.trace( m ), -1 ), -1 ) * tf.eye( dim, batch_shape=[1] )
            epsilonI += self.ngd_abs_eps * tf.eye( dim, batch_shape=[1] )
            return tf.matrix_inverse( tf.add( m, epsilonI ) )

        def singular():
            '''
            inverse the RFIM may cause exception
            this is the fallback procedure
            '''
            #epsilonI = tf.expand_dims( tf.expand_dims( tf.trace( m ), -1 ), -1 ) * tf.eye( dim, batch_shape=[1] )
            #return tf.matrix_inverse( tf.add( m, epsilonI ) )
            #eigs = tf.self_adjoint_eigvals( m )
            #with tf.control_dependencies( [ tf.Print( eigs, [eigs], 'this is eigs:', summarize=1000 ) ] ):
            return tf.eye( dim, batch_shape=[ tf.shape(m)[0] ] )

        return tf.cond( self.fallback, singular, regular )

    def __metric( self, current, new, current_dim, new_dim ):
        '''
        get the EMA-updated inverse metric

        current: batch_size x current_dim
        new    : batch_size x new_dim
        current_dim: number of input neurons
        new_dim:     number of output neurons

        return the inverse metric variable
        the metric updating op will be added to the list self.metric_ops
        the inverse metric op will be added to the list self.inv_ops
        '''
        metric     = tf.Variable( tf.eye( current_dim, batch_shape=[new_dim] ),
                              trainable=False, name="metric" )

        inv_metric = tf.Variable( tf.eye( current_dim, batch_shape=[new_dim] ),
                     trainable=False, name="inv_metric" )

        # optionally do a soft bounding of the feature vector
        # to avoid overflow of the metric computation
        #if max_feature > 0:
        #    x = max_feature * tf.tanh( current / max_feature )
        #else:
        #    x = current

        # compute x x^T  !!!!1) compute cov of X*X^T (?,1,dim)*(?,dim,1)=(?,dim,dim)
        cov = tf.expand_dims( current, 1 ) * tf.expand_dims( current, -1 )

        if new is None:
            # new is None means to compute the linear metric for simplicity
            # in this case new_dim==1
            self.metric_ops.append(
                metric.assign( self.metric_decay * metric
                             + (1-self.metric_decay) * tf.reduce_mean( cov, 0, keep_dims=True ) ) )

        else:
            # compute the RFIM of RELU
            sel = tf.pow( tf.sigmoid( new ), 2 )
            self.metric_ops.append(
                metric.assign( self.metric_decay * metric
                     + (1-self.metric_decay) * tf.reduce_mean(
                     tf.expand_dims( tf.expand_dims( sel, -1 ), -1 )
                     * tf.expand_dims( cov, 1 ), 0 ) ) )

        #tf.Print()
        self.inv_ops.append( inv_metric.assign( self.__batch_inv( metric, current_dim ) ) )

        return inv_metric

    def __apply_ng( self, grads ):
        '''
        apply (relative) natural gradient
        grads: [(gradient, variable),...]
        '''
        for grad_v, v in grads:
            if not v in self.natural:
                print( '{} is based on regular gradient'.format(v) )
                new_grad_v = grad_v

            elif self.natural[v].get_shape()[0] == 1:
                print( '{} is based on the linear RFIM'.format(v) )
                # 1xIxI IxO = IxO
                new_grad_v = tf.matmul( tf.squeeze( self.natural[v], [0] ), grad_v )

            else:
                print( '{} is based on RFIM'.format(v) )
                # OxIxI OxIx1 = OxIx1 -> OxI -> IxO
                new_grad_v = tf.matmul( self.natural[v],
                             tf.expand_dims( tf.transpose(grad_v), -1 ) )
                new_grad_v = tf.transpose( tf.squeeze( new_grad_v, [2] ) )

            yield ( new_grad_v, v )

    def __layer_name( self, layer_idx ):
        if layer_idx == -1:
            return "layerL"
        else:
            return ( "layer{}".format( layer_idx ) )

    def __layer_linear( self, current, s1, s2, drop ):
        '''
        a vanilla linear layer
        '''
        if self.init_stddev > 0:
            W = tf.get_variable( "weights", shape=(s1,s2),
                initializer=tf.truncated_normal_initializer(stddev=self.init_stddev) )
            b = tf.get_variable( "biases", shape=(s2,),
                initializer=tf.truncated_normal_initializer(stddev=self.init_stddev) )

        else: # use default initializer
            W = tf.get_variable( "weights", shape=(s1,s2) )
            b = tf.get_variable( "biases", shape=(s2,) )

        self.cost += self.l2_beta * tf.nn.l2_loss( W )

        if drop is not None: current = tf.nn.dropout( current, drop )
        return tf.add( tf.matmul( current, W ), b ) # y = x*W +b

    def __layer_bna( self, current, s1, s2, layer_idx, drop ):
        '''
        BN AFTER each activation of all hidden layers
        '''
        if layer_idx == 0:
            # first layer is a vanilla layer
            return self.__layer_linear( current, s1, s2, drop )

        else:
            # no bias is necessary here because BN already has a bias term beta
            if self.init_stddev > 0:
                W = tf.get_variable( "weights", shape=(s1,s2),
                    initializer=tf.truncated_normal_initializer(stddev=self.init_stddev) )
            else:
                W = tf.get_variable( "weights", shape=(s1,s2) )

            self.cost += self.l2_beta * tf.nn.l2_loss( W )

            if drop is not None: current = tf.nn.dropout( current, drop )

            return tf.matmul( tf.contrib.layers.batch_norm( current,
                        center=True, scale=True,
                        is_training=self.phase_train,
                        updates_collections=None ), W )

    def __layer_bnb( self, current, s1, s2, layer_idx, drop ):
        '''
        BN BEFORE each activation
        '''
        if layer_idx >= 0:
            if self.init_stddev > 0:
                W = tf.get_variable( "weights", shape=(s1,s2),
                    initializer=tf.truncated_normal_initializer(stddev=self.init_stddev) )
            else:
                W = tf.get_variable( "weights", shape=(s1,s2) )

            self.cost += self.l2_beta * tf.nn.l2_loss( W )

            if drop is not None: current = tf.nn.dropout( current, drop )

            return tf.contrib.layers.batch_norm( tf.matmul( current, W ),
                        center=True, scale=True,
                        is_training=self.phase_train,
                        updates_collections=None )

        else:
            # last layer is a vanilla layer
            return self.__layer_linear( current, s1, s2, drop )

    def __layer_sngd( self, current, s1, s2, layer_idx, drop, with_bn=False ):
        '''
        a SNGD layer

        current   -- the input op
        s1        -- input dimension
        s2        -- output dimension
        layer_idx -- index of the layer 
        with_bn   -- with batch normalization or not

        return the layer output op
        '''
        if with_bn and layer_idx != 0:
            # BN after activation of all hidden layers
            current = tf.contrib.layers.batch_norm( current,
                        center=True, scale=True,
                        is_training=self.phase_train,
                        updates_collections=None )

        # augment the current tensor with a column of ones
        # then apply a linear transformation
        o = tf.ones( shape=tf.stack( [ tf.shape(current)[0], 1 ] ) )
        current = tf.concat( [ current, o ], 1 )

        if drop is None:
            ocurrent = current
        else:
            ocurrent = tf.nn.dropout( current, drop )

        if self.init_stddev > 0:
            W = tf.get_variable( "weights", shape=(s1+1,s2),
                initializer=tf.truncated_normal_initializer(stddev=self.init_stddev) )
        else:
            W = tf.get_variable( "weights", shape=(s1+1,s2) )
        new = tf.matmul( ocurrent, W )

        # use exactly the same regualarization term as the other methods
        self.cost += self.l2_beta * tf.nn.l2_loss( tf.slice( W, [0,0], [s1,s2] ) )

        if layer_idx <= 0:
            # first and last layer, compute the simplified metric
             self.natural[W] = self.__metric( current, None, s1+1, 1 )
        else:
            self.natural[W] = self.__metric( current, new, s1+1, s2 )

        return new

    def __build_graph( self, nn_shape, dropout ):
        '''
        build the computational graph of a MLP

        nn_shape -- shape of all layers
         dropout -- the dropout rates

        return the output of the computational graph
        '''

        # initialize "current" with the input placeholder
        # current will be the latest op in building the MLP graph
        current  = self.x
        self.natural = {} ## dict type

        layer_shapes = list( zip( nn_shape, nn_shape[1:], dropout[1:] ) )
        for layer_idx, ( s1, s2, drop ) in enumerate( layer_shapes ):
            if layer_idx == len(layer_shapes)-1: layer_idx = -1

            with tf.variable_scope( self.__layer_name(layer_idx) ):
                # each layer has its own variable scope
                # this will be the operations inside layer_idx
                # the last layer is indexed by -1

                if self.optimizer == MLP.Optimizer.SNGD:
                    if self.arch == MLP.Arch.PLAIN:
                        current = self.__layer_sngd( current, abs(s1), abs(s2), layer_idx, drop, with_bn=False )

                    elif self.arch == MLP.Arch.BNA:
                        current = self.__layer_sngd( current, abs(s1), abs(s2), layer_idx, drop, with_bn=True )

                    else:
                        raise NotImplementedError( 'architecture not implemented for SNGD' )

                else:
                    if self.arch == MLP.Arch.PLAIN:
                        current = self.__layer_linear( current, abs(s1), abs(s2), drop )

                    elif self.arch == MLP.Arch.BNA:
                        current = self.__layer_bna( current, abs(s1), abs(s2), layer_idx, drop )

                    elif self.arch == MLP.Arch.BNB:
                        current = self.__layer_bnb( current, abs(s1), abs(s2), layer_idx, drop )

                    else:
                        raise RuntimeError( 'unknown architecture' )

                # IMPORTANT: must activate before the next layer
                if layer_idx >= 0: current = self.activate( current )

        return current

    def __cost_and_train( self, out ):
        '''
        build cost ops and train ops
        '''

        # add a final softmax layer 
        # and then measure the average cross entropy with the input y
        self.cost += tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits=out, labels=self.y ) )

        # measure accuracy in correct rate (0~1)
        _correct = tf.equal( tf.argmax(out, 1), tf.argmax(self.y, 1) )
        self.accuracy = tf.reduce_mean( tf.cast( _correct, tf.float32 ) )

        if self.optimizer == MLP.Optimizer.SGD:
            self.train_op = tf.train.GradientDescentOptimizer(
                                self.lrate ).minimize( self.cost )
            ###minimize() include two steps: 1)compute_gradients  2)apply_gradients(update-param))
        elif self.optimizer == MLP.Optimizer.ADAM:
            self.train_op = tf.train.AdamOptimizer(
                                self.lrate ).minimize( self.cost )

        elif self.optimizer == MLP.Optimizer.SNGD:
            optimizer = tf.train.GradientDescentOptimizer( self.lrate )
            #1) compute_gradients return [(gradient, variable),...]
            grads = optimizer.compute_gradients( self.cost, tf.trainable_variables() )
            #2) modify_new_gradsXX
            #new_grads = self.__apply_ng( grads )
            #3) update_varialbe
            #self.train_op = optimizer.apply_gradients( new_grads )
            self.train_op = optimizer.apply_gradients(self.__apply_ng( grads ))
        else:
            raise RuntimeError( 'unknown method' )

    def __init__( self,
              nn_shape,             # shape of all layers, from x to y
               dropout,             # dropout rates of all layers (None for dropout)
                  arch,             # architecture
             optimizer,             # optimizer
                 lrate,             # learning rate
              activate=tf.nn.relu,  # activation function
            batch_size=50,          # batch size
           init_stddev=0.1,         # variance of initial weights
          metric_decay=.995,        # (NG) metric
           ngd_rel_eps=1e-4,        # (NG) relative epsilon
           ngd_abs_eps=0,           # (NG) absolute epsilon
        ngd_inv_update=100,         # (NG) update interval of inverse metric
       early_stop_itv1=3,           # for early stop
       early_stop_itv2=-1,          # for early stop (e.g. 10),
                                    # set <0 to disable early stop
           output_intv=1,           # output every number of epochs
              l2_beta=0.001,        # L2 regularization strength
         ):
        '''
        record hyper parameters, build the graph
        '''

        self.arch           = arch
        self.optimizer      = optimizer

        self.lrate          = lrate
        self.activate       = activate
        self.batch_size     = batch_size
        self.init_stddev    = init_stddev

        self.metric_decay   = metric_decay
        self.ngd_rel_eps    = ngd_rel_eps
        self.ngd_abs_eps    = ngd_abs_eps
        self.ngd_inv_update = ngd_inv_update

        self.early_stop_itv1 = early_stop_itv1
        self.early_stop_itv2 = early_stop_itv2
        self.output_intv    = output_intv
        self.l2_beta        = l2_beta

        self.x = tf.placeholder( tf.float32, [ None, nn_shape[0] ], name="x" )
        self.y = tf.placeholder( tf.float32, [ None, nn_shape[-1]], name="y" )
        self.phase_train = tf.placeholder( tf.bool, name="phase_train" )
        self.fallback    = tf.placeholder( tf.bool, name="fall_back" )

        self.metric_ops = []
        self.inv_ops    = []
        self.cost = 0
        self.__cost_and_train( self.__build_graph( nn_shape, dropout ) )

        # print trainable variables
        print( 'all trainable variables are listed as follows' )
        for v in tf.trainable_variables():
            print( v.name, v.get_shape(), v.dtype )

        # setup IO variables
        save_path = 'checkpoints/'
        if not os.path.exists( save_path ): os.makedirs( save_path )

        self.filename = os.path.join( save_path,
            "model_{}_{}_{}_{}_{:.0f}_{}.ckpt"
            .format( arch.name, optimizer.name, lrate, batch_size,
                     time.time(), uuid.uuid4().hex ) )

        self.saver = tf.train.Saver()

    def fetch_batches( self, allidx, bs ):
        '''
        split allidx into batches, each of size batchsize
        '''
        N = allidx.size
        nbatch = int( np.ceil( N / bs ) )
        for itr in range( nbatch ):
            start_idx = itr * bs
            end_idx   = min( start_idx + bs, N )
            yield allidx[range( start_idx, end_idx )]

    def test( self, x, y, sess=None, large_batchsize=5000 ):
        '''
        test cost function and accuracy on a given dataset

           x -- features
           y -- one-hot labels
        sess -- the user may provide an existing tf session,
                otherwise a new session will be created

        return a pair (cost, accuracy)
        '''
        if sess is None:
            sess = tf.Session()
            self.saver.restore( sess, self.filename )
            newsess = True
        else:
            newsess = False

        N = x.shape[0]
        allidx = np.arange( N )

        _cost = 0
        _acc  = 0
        for batch in self.fetch_batches( allidx, large_batchsize ):
            tmp1, tmp2 = sess.run( [self.cost, self.accuracy],
                                 feed_dict={ self.x: x[batch],
                                             self.y: y[batch],
                                   self.phase_train: False,
                                      self.fallback: False } )
            _cost += tmp1 * batch.size
            _acc  += tmp2 * batch.size

        if newsess: sess.close()

        return (_cost/N),  (_acc/N)

    def train( self, train_x, train_y, valid_x, valid_y, num_epochs ):
        '''
        start a training session

        train_x -- training set (x)
        train_y -- training set (y)
        valid_x -- validation set (x)
        valid_y -- validation set (y)
        num_epochs -- #epochs to train

        return the three learning curves representing
        the training cross-entropy, validation cross-entropy, validation classificatioon accuracy
        '''

        train_curve     = np.zeros( num_epochs )
        valid_curve     = np.zeros( num_epochs )
        valid_acc_curve = np.zeros( num_epochs )

        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run( init_op )

            start_time = time.time()
            allidx = np.arange( train_x.shape[0] )
            for epoch in range( num_epochs ):

                # shuffle
                np.random.shuffle( allidx )

                # going through mini-batches and learning
                for itr, batch in enumerate( self.fetch_batches( allidx, self.batch_size ) ):
                    if itr % 10 == 0: print( '#', end='' )

                    sess.run( self.train_op,
                              feed_dict={ self.x: train_x[batch],
                                          self.y: train_y[batch],
                                          self.phase_train: True,
                                          self.fallback: False } )

                    # update the metric in every iteration
                    if itr % 1 == 0:
                        sess.run( self.metric_ops,
                                  feed_dict={ self.x: train_x[batch],
                                              self.y: train_y[batch],
                                              self.phase_train: False,
                                              self.fallback: False } )

                    try:
                        # recompute the inverse metric
                        # this may cause exception
                        if itr % self.ngd_inv_update == 0:
                            sess.run( self.inv_ops,
                                      feed_dict={ self.x: train_x[batch],
                                                  self.y: train_y[batch],
                                                  self.phase_train: False,
                                                  self.fallback: False } )

                    except tf.errors.InvalidArgumentError:
                        print( '###### singularity catched #######' )
                        sess.run( self.train_op,
                                  feed_dict={ self.x: train_x[batch],
                                              self.y: train_y[batch],
                                              self.phase_train: False,
                                              self.fallback: True } )

                print( '' )

                # re-assess the model after each epoch
                train_cost, train_acc = self.test( train_x, train_y, sess=sess )
                valid_cost, valid_acc = self.test( valid_x, valid_y, sess=sess )
                train_curve[epoch]     = train_cost
                valid_curve[epoch]     = valid_cost
                valid_acc_curve[epoch] = valid_acc

                if epoch % self.output_intv == 0:
                    speed = ( time.time() - start_time ) / (epoch+1)
                    print( "[{:04d}] train_cost={:.3f} train_acc={:.3f} "
                           " valid_cost={:.3f}, valid_acc={:.3f} ({:.0f}s/epoch)"
                           .format( epoch+1, train_cost, train_acc, valid_cost, valid_acc, speed ) )

                if np.isinf( train_curve[epoch] ) \
                   or np.isnan( train_curve[epoch] ) \
                   or train_curve[epoch] > 1e3 * train_curve[0]:
                    # training failed, terminate
                    print( 'trainging failed' )
                    break

                if ( self.early_stop_itv1 > 0 ) and \
                   ( self.early_stop_itv2 > 0 ) and \
                   ( epoch > self.early_stop_itv2 ) and \
                   ( np.mean( valid_curve[epoch-self.early_stop_itv1+1:epoch+1] )
                    >= np.mean( valid_curve[epoch-self.early_stop_itv2+1:epoch+1] ) ):

                    # simple heuristic for early stoping based on moving average:
                    # quit when validation cost start to rise
                    print( 'early stoping' )
                    break

            train_curve[(epoch+1):]     = train_curve[epoch]
            valid_curve[(epoch+1):]     = valid_curve[epoch]
            valid_acc_curve[(epoch+1):] = valid_acc_curve[epoch]

            save_path = self.saver.save( sess, self.filename )
            print( "Model saved in file: %s" % save_path )
            sess.close()

        return train_curve, valid_curve, valid_acc_curve

