#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from MLP import MLP
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data


import numpy as np
import sys, os, time, gc, itertools, argparse

# supress debuging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_mnist( size_valid=10000, size_test=10000, shuffle=True ):
    '''
    shuffle and split the MNIST dataset
    '''
    mnist = input_data.read_data_sets( "data/", one_hot=True )
    tf.logging.set_verbosity(old_v)

    X = np.vstack( [ mnist.train.images, mnist.validation.images, mnist.test.images ] ).astype( np.float32 )
    Y = np.vstack( [ mnist.train.labels, mnist.validation.labels, mnist.test.labels ] ).astype( np.float32 )
    XY = np.hstack( [X, Y] )

    if shuffle: np.random.shuffle( XY )
    assert( XY.shape[0] == 70000 )

    X = XY[:,:784]
    Y = XY[:,784:]

    train_x = X[:-(size_valid+size_test)]
    valid_x = X[-(size_valid+size_test):-size_test]
    test_x  = X[-size_test:]

    train_y = Y[:-(size_valid+size_test)]
    valid_y = Y[-(size_valid+size_test):-size_test]
    test_y  = Y[-size_test:]

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def batch_exp( arch,                        # architecture
               optimizer,                   # optimizer
               nn_shape,                    # shape of the NN
               dropout,                     # dropout rates
               lrate,                       # learning rates to try
               batchsize,                   # batch sizes to try
               stddev,                      # standard deviation of initial weights
               beta,                        # regularization strength
               nepochs,                     # number of epochs
               repeat,                      # repeat for each config
             ):
    '''
    perform a batch run of one single method on MNIST
    based on a grid of configurations
    '''

    print( arch.name, optimizer.name )
    print( 'MLP shape', nn_shape )
    print( 'dropout rates', dropout )
    print( 'learning rate:', lrate )
    print( 'batch size:', batchsize )
    print( 'stddev:', stddev )
    print( 'beta:', beta )
    print( 'num epochs: {}'.format( nepochs ) )
    print( 'repeat: {}'.format( repeat ) )

    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist()

    configs = list( itertools.product( lrate, batchsize, stddev, beta ) )
    print( 'trying {} configurations'.format( len(configs) ) )

    train_results     = np.zeros( [ len(configs), repeat, nepochs ] )
    valid_results     = np.zeros( [ len(configs), repeat, nepochs ] )
    valid_acc_results = np.zeros( [ len(configs), repeat, nepochs ] )
    test_results      = np.zeros( [ len(configs), repeat ] )
    test_acc_results  = np.zeros( [ len(configs), repeat ] )

    for i, (lr, bs, sd, _beta) in enumerate( configs ):
        for j in range( repeat ):
            gc.collect() ##clean memory
            tf.reset_default_graph()
            print( 'lrate={} batchsize={} stddev={} trial{:2d}/{:2d}'.format( lr, bs, sd, j+1, repeat ) )

            start_t = time.time()

            mlp = MLP( nn_shape, dropout, arch, optimizer, lr, batch_size=bs, init_stddev=sd, l2_beta=_beta )
            train_curve, valid_curve, valid_acc_curve = mlp.train( train_x, train_y, valid_x, valid_y, nepochs )

            train_results[i,j,:]     = train_curve
            valid_results[i,j,:]     = valid_curve
            valid_acc_results[i,j,:] = valid_acc_curve

            train_cost, train_acc = mlp.test( train_x, train_y )
            valid_cost, valid_acc = mlp.test( valid_x, valid_y )
            test_cost,  test_acc  = mlp.test( test_x,  test_y )
            print( 'train: cost={:.4f} acc={:.4f}'.format( train_cost, train_acc ) )
            print( 'valid: cost={:.4f} acc={:.4f}'.format( valid_cost, valid_acc ) )
            print( 'test:  cost={:.4f} acc={:.4f}'.format( test_cost,  test_acc ) )

            test_results[i,j]     = test_cost
            test_acc_results[i,j] = test_acc

            min_per_run = ( time.time() - start_t ) / 60.
            rest_runs = ( len(configs)-i-1 )*repeat + ( repeat-j-1 )
            if rest_runs > 0:
                print( 'speed={:.1f}m/experiment'.format( min_per_run ) )
                print( '{:.1f}m to go!'.format( min_per_run*rest_runs ) )

    # estimate the achieved accuracy on the validation set in the last epoch
    avg_acc = valid_acc_results[:,:,-1].mean( -1 )
    avg_acc = [ np.inf if ( np.isnan(_) or np.isinf(_) ) else _ for _ in avg_acc ]
    idx = np.argmax( avg_acc )
    print( '#######################################' )
    print( 'best configuration based on validation:' )
    print( 'lrate {} batchsize {} stddev {} beta {}'.format( *configs[idx] ) )
    print( 'validation: cost={} acc={}'.format( valid_results[idx,:,-1].mean(), avg_acc[idx] ) )
    print( 'testing:    cost={} acc={}'.format( test_results[idx].mean(), test_acc_results[idx].mean() ) )
    print( '#######################################' )

    # save the learning curves
    outfile = ( 'mnist_full_%s+%s_%g_%g' % ( arch.name, optimizer.name,
                                             lrate[0], lrate[-1] ) )
    np.savez( outfile,
              configs=configs,
              train_results=train_results,
              valid_results=valid_results,
              valid_acc_results=valid_acc_results,
              test_results=test_results,
              test_acc_results=test_acc_results )
    print( 'results saved to %s' % outfile )

def main():
    ARCHS      = {
                 'plain': MLP.Arch.PLAIN,
                   'bna': MLP.Arch.BNA, ##bn after
                   'bnb': MLP.Arch.BNB, ##bn before
                 }
    OPTIMIZERS = {
                   'sgd': MLP.Optimizer.SGD,
                  'adam': MLP.Optimizer.ADAM,
                  'sngd': MLP.Optimizer.SNGD,
                 }
    SHAPES     = {
                    'a': ( ( 784, 80, 80, 80, 10 ),
                           ( None, None, None, None, None) ),
                    'b': ( ( 784, 80, 60, 40, 40, 40, 20, 10 ),
                           ( None, None, None, None, None, None, None, None) ),
                    'c': ( ( 784, 20, 20, 20, 10 ),
                           ( None, None, None, None, None ) ),
                    'd': ( ( 784, 80, 80, 80, 80, 80, 80, 10 ),
                           ( None, None, None, None, None, None, None, None ) )
                 }

    parser = argparse.ArgumentParser()
    parser.add_argument( 'arch',        type=str,   choices=ARCHS.keys(),
                         help='architecture' )
    parser.add_argument( 'optimizer',   type=str,   choices=OPTIMIZERS.keys(),
                         help='optimizer' )
    parser.add_argument( 'shape',       type=str,   choices=SHAPES.keys(),
                         help='network shape' )
    parser.add_argument( '--lrate',     type=float, nargs='+', default=[1e-2, 5e-3],
                         help='learning rate' )
    parser.add_argument( '--batchsize', type=int,   nargs='+', default=[50],
                         help='size of mini batch' )
    parser.add_argument( '--stddev',    type=float, nargs='+', default=[-1],
                         help='standard deviation of  initial weights. Use -1 for default' )
    parser.add_argument( '--beta',      type=float, nargs='+', default=[1e-3],
                         help='l2 regularization strength' )
    parser.add_argument( '--nepochs',   type=int,   default=100,
                         help='number of epochs to run' )
    parser.add_argument( '--repeat',    type=int,   default=1, #default=40,
                         help='number of repeats for each configuration' )
    args = parser.parse_args()

    batch_exp( ARCHS[args.arch],
               OPTIMIZERS[args.optimizer],
               SHAPES[args.shape][0],
               SHAPES[args.shape][1],
               args.lrate, 
               args.batchsize,
               args.stddev,
               args.beta,
               args.nepochs,
               args.repeat )

if __name__ == '__main__':
    sys.stdout = sys.stderr
    start_t = time.time()
    np.random.seed( 2017 )
    tf.set_random_seed( 2017 )
    main()
    print( '%.1f hours, WOW' % ( ( time.time() - start_t ) / 3600 ) )

