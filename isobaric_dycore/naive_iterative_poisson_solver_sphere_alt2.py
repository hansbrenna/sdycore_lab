#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:27:05 2021

@author: hansb
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 12:41:13 2018
@author: hanbre
A naive implementation of a relaxation poisson solver on a 2 spherical grid
"""
# from __future__ import print_function
import numpy as np
from numba import jit,prange
import xarray as xr
import time
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline,RectSphereBivariateSpline

def regrid(phi_old,lambda_old,nphi_new,nlambda_new,data):
    Drc = data[1:-1,1:-1,0].copy()
    lut = RectSphereBivariateSpline(phi_old+pi/2,lambda_old+pi,Drc.T)

    dlambda1 = 2*pi/nlambda_new
    dphi1 = pi/nphi_new

    phi1 = np.array([-((pi/2)-dphi1/2)+(j)*dphi1 for j in range(nphi_new)])+pi/2
    lambd1 = (np.array([-(pi)+(i)*dlambda1 for i in range(nlambda_new)]))+pi

    glambd,gphi = np.meshgrid(lambd1,phi1)

    interp = (lut.ev(gphi.ravel(),glambd.ravel())).reshape(nphi_new,nlambda_new).T
    return interp,phi1,lambd1,dphi1,dlambda1

def helper_define_trigs(phi1,nphi1):

    cosphi1 = np.zeros([nphi1+2])
    sinphi1 = np.zeros([nphi1+2])
    tanphi1 = np.zeros([nphi1+2])

    cosphi1[1:-1] = np.cos(phi1)
    cosphi1[0] = -cosphi1[1]
    cosphi1[-1] = -cosphi1[-2]
    tanphi1[1:-1] = np.tan(phi1)
    sinphi1[1:-1] = np.sin(phi1)
    return cosphi1,sinphi1,tanphi1

@jit(nopython=True)
def apply_BCs(data,nlambda,k):
    #set periodic boundary in lambda
    data[0,:,k] = data[-1,:,k]
    data[-1,:,k] = data[0,:,k]
    #set polar boundary condition in phi
    SP = np.roll(data[1:-1,1,k],nlambda//2) #index 0
    NP = np.roll(data[1:-1,-2,k],nlambda//2) #index -1
    data[1:-1,0,k] = SP
    data[1:-1,-1,k] = NP
    return data

@jit(nopython=True,parallel=True)
def iterative_solver_sphere(Qin,R,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,exit_status=0):
    """
    Successive over-relaxation solver for Poisson equation in spherical geometry

    Parameters
    ----------
    Qin : TYPE
        DESCRIPTION.
    R : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    tanphi : TYPE
        DESCRIPTION.
    cosphi : TYPE
        DESCRIPTION.
    dlambda : TYPE
        DESCRIPTION.
    dphi : TYPE
        DESCRIPTION.
    nlambda : TYPE
        DESCRIPTION.
    nphi : TYPE
        DESCRIPTION.
    nP : TYPE
        DESCRIPTION.
    exit_status : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.
    stati : TYPE
        DESCRIPTION.
    ns : TYPE
        DESCRIPTION.
    deltas : TYPE
        DESCRIPTION.

    """
    Q=Qin.copy()

    a1 = (1./(2./(a*a*dlambda*dlambda*cosphi*cosphi)+2./(a*a*dphi*dphi)))
    a2 = (1./(a*a*dlambda*dlambda*cosphi*cosphi))
    a3 = -tanphi/(a*a*2*dphi)
    a4 = 1./(a*a*dphi*dphi)
    
    maxiter=30000
    epsilon = 0.000001
    delta=1

    stati = []
    ns = []
    deltas = []
    n_iter=0
    for k in prange(nP):
        n_iter = 0
        delta = 1
        while True:
            if delta < epsilon:
                exit_status = 0
                stati.append(exit_status)
                ns.append(n_iter)
                deltas.append(delta)
                break
            if n_iter > maxiter:
                exit_status = 1
                stati.append(exit_status)
                ns.append(n_iter)
                deltas.append(delta)
                break
            delta = 0
            n_iter+=1
            Q_temp = apply_BCs(Q,nlambda,k)[:,:,k].copy()
            Q_old = Q_temp[:,:].copy()
            # Q_temp = Q_old.copy()
            omg = 2. / ( 1. + np.sin(np.pi/(n_iter+1)) )
            for i in prange(1,nlambda+1):
                for j in prange(1,nphi+1):
                    Q[i,j,k] = (a1[j]*(a2[j]*(Q_temp[i+1,j] + Q_temp[i-1,j]) +
                            a3[j]*(Q_temp[i,j+1] - Q_temp[i,j-1]) +
                            a4*(Q_temp[i,j+1] + Q_temp[i,j-1]) - R[i,j,k]))
                    Q[i,j,k] = omg*Q[i,j,k] + (1-omg)*Q_old[i,j]
                    Q_temp[i,j] = Q[i,j,k]
                    delta  += np.abs(Q[i,j,k] - Q_old[i,j])
            delta/=(nlambda*nphi*a*a)
    return Q.copy(),stati,ns,deltas

if __name__ == '__main__':
    pi = np.pi
    nphi = 61#72 #36
    nlambda = 120#144 #72
    nP = 20
    dlambda = 2*pi/nlambda
    dlambda_deg = np.rad2deg(dlambda)
    dphi = pi/nphi
    dphi_deg = np.rad2deg(dphi)#2.5 #5.

    a = 6.371e+6

    phi = np.array([-((pi/2)-dphi/2)+(j)*dphi for j in range(nphi)])
    lambd = (np.array([-(pi)+(i)*dlambda for i in range(nlambda)]))
    cosphi = np.zeros([nphi+2])
    sinphi = np.zeros([nphi+2])
    tanphi = np.zeros([nphi+2])

    cosphi[1:-1] = np.cos(phi)
    cosphi[0] = -cosphi[1]
    cosphi[-1] = -cosphi[-2]
    tanphi[1:-1] = np.tan(phi)
    sinphi[1:-1] = np.sin(phi)

    Qin = np.zeros([nlambda+2,nphi+2,nP])
    R = (np.random.rand(nlambda+2,nphi+2,nP)-0.5)*100
    Ra = (np.random.rand(nlambda+2,nphi+2,nP)-0.5)*100
    #alt
    R[1:-1,1:-1,:] = xr.open_dataset('history_out_hour678.nc')['U'].isel(time=0).values
    Ra[1:-1,1:-1,:] = xr.open_dataset('history_out_hour678.nc')['V'].isel(time=0).values

    start = time.time()
    Qout,status,n_iter,delta = iterative_solver_sphere(Qin,R,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,exit_status=0)
    print(status,n_iter,delta)
    Qouta,status,n_iter,delta = iterative_solver_sphere(Qin,Ra,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,exit_status=0)
    print(status,n_iter,delta)


    # Rrc = R[1:-1,1:-1,0].copy()
    # # test = RectSphereBivariateSpline(phi+pi/2,lambd+pi,Rrc.T)

    # nphi1 = 4
    # nlambda1 = 8

    # interp,phi1,lambd1,dphi1,dlambda1 = regrid(phi,lambd,nphi1,nlambda1,R)
    # lambd1 = lambd1-pi
    # phi1 = phi1-pi/2
    # cosphi1,sinphi1,tanphi1 = helper_define_trigs(phi1,nphi1)

    # plt.contourf(lambd1,phi1,interp.T,cmap='RdBu_r',vmin=-100,vmax=100),plt.colorbar()
    # plt.show()
    # plt.contourf(lambd,phi,Rrc.T,cmap='RdBu_r',vmin=-100,vmax=100),plt.colorbar()
    # plt.show()

    # Qin1 = np.zeros([nlambda1+2,nphi1+2,nP])
    # R1 = np.zeros([nlambda1+2,nphi1+2,nP])
    # R1[1:-1,1:-1,0] = interp
    # Qout,status,n_iter,delta = iterative_solver_sphere(Qin1,R1,a,tanphi1,cosphi1,dlambda1,dphi1,nlambda1,nphi1,nP,exit_status=0)
    # print(status,n_iter,delta)

    # plt.contourf(lambd1,phi1,Qout[1:-1,1:-1,0].T,cmap='viridis'),plt.colorbar()
    # plt.show()

    # interp2,phi2,lambd2,dphi,dlambda = regrid(phi1,lambd1,nphi,nlambda,Qout)

    # Qin2 = np.zeros([nlambda+2,nphi+2,nP])
    # Qin2[1:-1,1:-1,0] = interp2
    # R2 = np.zeros([nlambda+2,nphi+2,nP])
    # R2[1:-1,1:-1,0] = R[1:-1,1:-1,0]

    # Qout2,status,n_iter,delta = iterative_solver_sphere(Qin2,R2,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,exit_status=0)
    # print(status,n_iter,delta)
    # plt.contourf(lambd,phi,Qout2[1:-1,1:-1,0].T,cmap='viridis'),plt.colorbar()
    # # Qout,status,n_iter,delta = iterative_solver_sphere(Qin,R,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,exit_status=0)
    end = time.time()
    print(end - start)