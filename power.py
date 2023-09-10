#!/usr/bin/env python

'''
power method
'''

from __future__ import print_function
import numpy as np
import scipy.linalg
import sys, time

def __largest( A, init_v, orth_v, max_iterations, tol ):
    '''
    extract the largest value/eigenvector of A that
    are orthognal to all unit vectors in orth_v
    '''
    y = init_v
    for i in range( max_iterations ):
        v = y / np.linalg.norm( y )

        if orth_v is not None and ( orth_v.size > 0 ):
            y = np.dot( A, v-( ( orth_v * v ).sum( 1 )[:,None] * orth_v ).sum( 0 ) )
        else:
            y = np.dot( A, v )

        w = ( v * y ).sum()
        if ( i < max_iterations-1 ) and \
           ( np.linalg.norm( y - w * v ) < tol * abs( w ) ): break

    return w, v

def power( A, init_V, max_iterations=100, tol=1e-3, fast=False ):
    '''
    power method to compute the largest eigevalues
    of a square matrix A
    as well as the corresponding eigehvectors

    init_V is for initializing the eigenvectors
    each row an eigenvector
    '''
    assert( A.ndim == 2 )
    assert( init_V.ndim == 2 )
    assert( A.shape[0] == A.shape[1] )
    assert( A.shape[0] == init_V.shape[1] )

    W = np.array( [] )
    V = np.zeros( [ 0, A.shape[1] ] )

    _A = A.copy() if fast else A

    for _ in init_V:
        if fast:
            w, v = __largest( _A, _, None, max_iterations, tol )
        else:
            w, v = __largest( _A, _, V, max_iterations, tol )

        W = np.append( W, w )
        V = np.vstack( [ V, v ] )

        if fast: _A -= (w * v[:,None]) * v

    return W, V

if __name__ == "__main__":
    DIM = 500
    NEIG = 3
    REPEAT = 10

    print( 'benchmarking speed' )
    start_t = time.time()
    for i in range( REPEAT ):
        A = np.random.rand( DIM, DIM )
        A = np.dot( A, A.T )
        power_w, power_v = power( A, np.random.rand( NEIG, DIM ) )
    print( 'POWER: %d sec' % ( time.time() - start_t ) )

    start_t = time.time()
    for i in range( REPEAT ):
        A = np.random.rand( DIM, DIM )
        A = np.dot( A, A.T )
        w, v = scipy.linalg.eigh( A, eigvals=(DIM-NEIG, DIM-1) )
    print( 'EIGH: %d sec' % ( time.time() - start_t ) )

    A = np.random.rand( DIM, DIM )
    A = np.dot( A, A.T )
    power_w, power_v = power( A, np.random.rand( NEIG, DIM ) )
    w, v = scipy.linalg.eigh( A, eigvals=(DIM-NEIG,DIM-1) )

    for i in range( NEIG ):
        print( '%dth eigen value' % (i+1) )
        print( '%.1f = %.1f' % ( power_w[i], w[-(i+1)] ) )
        print( ( power_v[i] * v[:,-(i+1)] ).sum() )
        print( '' )

