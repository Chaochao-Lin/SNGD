#!/usr/bin/env python 

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.size'] = 14

import numpy as np
import sys, os, argparse, math

def smooth( x, intv ):
    '''
    x -- 1d numpy array
    average x over every intv items
    '''
    #print( 'smoothing over {0} iterations'.format( intv ) )

    assert( intv >= 1 )
    if intv == 1: return x

    num_intvs = int( math.ceil( x.size / intv ) )
    return np.array( [ np.mean( x[i*intv : (i+1)*intv] )
                       for i in range( num_intvs ) ] )

def final_mean_std( results, L ):
    '''
    results -- 3D array of experimental results (accuracy)

    return the mean and std of the final results
    along the first dimension of results (exclude invalid entries)
    '''
    _mean = []
    _std  = []
    for arr in results:
        # remove invalid runs caused by numecial instability, large learning rates, etc.
        valid_runs = [ run for run in arr
                       if not ( np.isnan( np.mean(run) )
                             or np.isinf( np.mean(run) ) ) ]
        if len(valid_runs) < arr.shape[0]:
            print( 'found {} invalid runs'.format( arr.shape[0] - len(valid_runs) ) )

        if len( valid_runs ) > 0:
            valid_runs = np.array( valid_runs )
            _mean.append( np.mean( valid_runs[:,-L:] ) )
            _std.append(  np.std(  valid_runs[:,-L:] ) )
        else:
            _mean.append( np.inf )
            _std.append( np.inf )

    return np.array( _mean ), np.array( _std )

def visualize( filenames, top, smooth_intv, ncurves, last, alpha=0.3 ):
    '''
    analyse npz files, output statistics and figures

      filenames -- input log files
            top -- plot top k results for each method
    smooth_intv -- compute learning curves by averaging every N iterations
        ncurves -- number of learning curves to show
    '''

    fig = plt.figure( figsize=[4,4], frameon=True, dpi=300 )
    ax1  = plt.Axes( fig, [ 0., 0., 1., 1. ] )
    ax2  = ax1.twinx()
    fig.add_axes( ax1, ax2 )

    patterns = [
               ( 'PLAIN+SGD' , 'r-.', 'r-' ),
               ( 'PLAIN+ADAM', 'g-.', 'g-' ),
               ( 'PLAIN+SNGD', 'b-.', 'b-' ),
               ( 'BNA+SGD'   , 'r-.', 'r-' ),
               ( 'BNA+ADAM'  , 'g-.', 'g-' ),
               ( 'BNA+RNGD'  , 'b-.', 'b-' ),
               ]

    # load results from disk into "curves"
    curves = {}
    for filename in filenames:
        dirname = os.path.dirname( filename )
        dataset = '_'.join( os.path.basename( filename ).split( '_' )[:2] )
        method  = os.path.basename( filename ).split( '_' )[2]

        _raw                = np.load( filename )
        configs             = _raw[ 'configs' ]
        train_results       = _raw[ 'train_results' ]
        valid_results       = _raw[ 'valid_results' ]
        valid_acc_results   = _raw[ 'valid_acc_results' ]
        test_results        = _raw[ 'test_results' ]
        test_acc_results    = _raw[ 'test_acc_results' ]

        curves[ method ] = [ configs, train_results, valid_results, valid_acc_results, test_results, test_acc_results ]

    # scan all existing methods
    lines = []
    for method, pat1, pat2 in patterns:
        if not method in curves: continue

        configs           = curves[method][0]
        train_results     = curves[method][1]
        valid_results     = curves[method][2]
        valid_acc_results = curves[method][3]
        test_acc_results  = curves[method][5]

        # order the configuration based on validation accuracy
        final_mean, final_std = final_mean_std( valid_acc_results, last )
        best_idx = np.argsort( final_mean )[::-1]

        print( '--==== %-10s ====--' % method )
        for rank, idx in enumerate( best_idx ):
            print( 'lrate=%-10g' % configs[idx,0], end=" " )
            print( ' '.join( ['%-10g' % _ for _ in configs[idx,1:] ] ), end=" " )
            print( 'valid_acc={:.4f}-{:.4f}'.format( final_mean[idx], final_std[idx] ), end=" " )
            print( 'test_acc={:.4f}'.format( test_acc_results[idx].mean() ) )
            if rank >= top: continue

            # select the best 'ncurves' runs (to factor out local optimums)
            # based on the final training error
            acc = np.mean( valid_acc_results[idx,:,-last:], -1 )
            best_runs = np.argsort( acc )[-ncurves:]

            for i, train_curve in enumerate( train_results[idx,best_runs,:] ):
                smoothed_curve = smooth( train_curve, smooth_intv )
                x_plot = [ j*smooth_intv for j in range( len( smoothed_curve ) ) ]
                line = ax1.plot( x_plot, smoothed_curve, pat1, label=method+' (train)', alpha=alpha )[0]
                if i == 0: lines.append( line )

            # valid curve only show average (plotting all curves is visually unclear)
            valid_curve = valid_acc_results[idx,:,:].mean(0)
            smoothed_curve = smooth( valid_curve, smooth_intv )
            x_plot = [ j*smooth_intv for j in range( len( smoothed_curve ) ) ]
            line = ax2.plot( x_plot, smoothed_curve, pat2, label=method+' (valid)' )[0]
            lines.append( line )

        print( '' )

    # plot legend
    labs = [ l.get_label() for l in lines ]
    #leg = plt.legend( lines, labs, loc=(0.4,0.1) )
    leg = plt.legend( lines, labs, loc=5 )
    leg.get_frame().set_alpha( alpha )

    ax1.tick_params( top='off' )
    ax2.tick_params( top='off', right='on' )

    ax1.set_xlim( [0,train_results.shape[-1]] )
    ax1.set_xlabel( '#epochs' )
    ax1.set_ylabel( 'error' )
    #ax1.set_ylim( [0.09,0.4] ) # PLAIN
    #ax1.set_ylim( [0.03,0.5] ) # BNA

    ax2.set_ylabel( 'accuracy' )
    #ax2.set_ylim( [0.965,0.977] ) # PLAIN
    #ax2.set_ylim( [0.965,0.978] ) # PLAIN 100
    #ax2.set_ylim( [0.97,0.978] )  # BNA
    #ax2.set_ylim( [0.97,0.979] )  # BNA 100

    ofilename = os.path.join( dirname, '%s_curves' % dataset ).replace( '.', '_' )
    fig.savefig( ofilename+'.pdf', bbox_inches='tight', pad_inches=0, transparent=True )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'filenames',     type=str,   nargs='+' )
    parser.add_argument( '--top',         type=int,   default=1 )
    parser.add_argument( '--smooth',      type=int,   default=1 )
    parser.add_argument( '--ncurves',     type=int,   default=20,
                         help='number of learning curves to show' )
    parser.add_argument( '--last',        type=int,   default=10,
                         help='select curves based on validation performance over LAST epochs')
    args = parser.parse_args()

    visualize( args.filenames, args.top, args.smooth, args.ncurves, args.last )

