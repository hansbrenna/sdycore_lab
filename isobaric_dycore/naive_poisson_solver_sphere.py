#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 12:41:13 2018

@author: hanbre
A naive implementation of a relaxation poisson solver on a 2 spherical grid
"""
from __future__ import print_function
import numpy as np
from numba import jit

@jit()
def iterative_solver_sphere(Qin,R,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,exit_status=0):
    omg = 1
    Q=Qin.copy()
    Q[:]=0.0
    Qd = np.zeros([nlambda+2,nphi+2,nP])
    Qstar = Q.copy()
    Q_old = Q[:,:,0].copy()
    cosphi_d = np.zeros([nphi+2])
    tanphi_d = np.zeros([nphi+2])
    cosphi_d[1:-1]=cosphi
    tanphi_d[1:-1]=tanphi
    
    cosphi_d[0] = -cosphi[1]
    cosphi_d[-1] = -cosphi[-2]
    
    a1 = (1./(2./(a*a*dlambda*dlambda*cosphi_d*cosphi_d)+2./(a*a*dphi*dphi)))
    a2 = (1./(a*a*dlambda*dlambda*cosphi_d*cosphi_d))
    a3 = -tanphi_d/(a*a*2*dphi)
    a4 = 1./(a*a*dphi*dphi)
    
    maxiter=1000
    epsilon = 1e-14
    delta=1

    n_iter=0
    for k in range(nP):
        n_iter = 0
        delta = 1.
        while True:
            if delta < epsilon:
                exit_status = 0
                break
            if n_iter > maxiter:
                exit_status = 1
                break
#        while np.max(np.abs(Q[:,:,k]-Q_old))>epsilon:# or n_iter<1 and n_iter<maxiter:
#            print((np.max(np.abs(Q[:,:,k]-Q_old))>epsilon))
            n_iter+=1
            #print('starting_iteration {}'.format(n_iter))
#            print(k,n_iter)
            Qd[1:-1,1:-1,k] = Q[:,:,k]
            #set periodic boundary in lambda
            Qd[0,1:-1,k] = Q[-1,:,k]
            Qd[-1,1:-1,k] = Q[0,:,k]
            #set polar boundary condition in phi
            Qd[1:-1,0,k] = np.roll(Q[:,-2,k],nlambda//2,axis=0)
            Qd[1:-1,-1,k] = np.roll(Q[:,1,k],nlambda//2,axis=0)
            Q_old = Q[:,:,k].copy()
            for i in range(1,nlambda+1):
                for j in range(1,nphi+1):
                    Qstar[i-1,j-1,k] = (a1[j]*(a2[j]*(Qd[i+1,j,k]+Qd[i-1,j,k])+a3[j]
                             *(Qd[i,j+1,k]-Qd[i,j-1,k])+a4*(Qd[i,j+1,k]+Qd[i,j-1,k])-R[i-1,j-1,k]))
#            omg = 2. / ( 1. + np.sin(np.pi/(n_iter+1)) )
            Q = omg*Qstar + (1.-omg)*Q
            delta = np.max(np.abs(Q_old-Q[:,:,k]))
#            if k == 2:
#                print(k,n_iter,delta)
#            print(n_iter,delta)
#    print(n_iter)
    return Q.copy(),exit_status
                
                         