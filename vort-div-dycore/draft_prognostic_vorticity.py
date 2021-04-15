# -*- coding: utf-8 -*-
"""
Author: Hans Brenna

2021-03-27

Draft of vorticity equation for use in sdycor_lab 

GPL license
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/hansb/github/sdycore_lab/isobaric_dycore/')
import numpy as np
from numba import jit,njit,prange
from isobaric_gcm_4th_order_ghost_BC_mass_energy_fixed_time_split_H_S import *
# from naive_iterative_poisson_solver_sphere_alt2 import iterative_solver_sphere,apply_BCs

# @njit
def differentiate_phi(psi,dpsi_dphi,dphi,nlambda,nphi,nP,cosphi):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                dpsi_dphi[i,j,k] = (((psi[i,j-2,k]+8*(-psi[i,j-1,k]+psi[i,j+1,k])
                      -psi[i,j+2,k])/(12*dphi)))
    return dpsi_dphi

# @njit
def differentiate_lambda(psi,dpsi_dlambda,dlambda,nlambda,nphi,nP,cosphi,a,scaling=False):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                    # print('not scaling')
                dpsi_dlambda[i,j,k] = ((psi[i-2,j,k] +
                            8*(-psi[i-1,j,k] + psi[i+1,j,k]) +
                            psi[i+2,j,k])/(12*dlambda))
    
    if scaling:
        for j in prange(4,nphi+4):
            dpsi_dlambda[:,j,:] = (1/(a*cosphi[j]))*dpsi_dlambda[:,j,:]
    return dpsi_dlambda

def differentiate_lambda_O2(psi,dpsi_dlambda,dlambda,cosphi):
    dpsi_dlambda[:] = 0.0
    shape = psi.shape
    nlambda = shape[0]-8
    nphi = shape[1]-8
    nP = shape[2]
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            for k in range(nP):
                temp = (1/(a*cosphi[j]))*(psi[i+1,j,k] - psi[i-1,j,k])/(2*dlambda)
                # print(temp)
                dpsi_dlambda[i,j,k] = temp
    return dpsi_dlambda

@njit
def differentiate_lambda_O4(psi,dpsi_dlambda,dlambda,cosphi):
    dpsi_dlambda[:] = 0.0
    shape = psi.shape
    nlambda = shape[0]-8
    nphi = shape[1]-8
    nP = shape[2]
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            for k in range(nP):
                temp = ((1/(a*cosphi[j]))*((psi[i-2,j,k]+8*(-psi[i-1,j,k]+psi[i+1,j,k])
                      -psi[i+2,j,k])/(12*dlambda)))
                # print(temp)
                dpsi_dlambda[i,j,k] = temp
    return dpsi_dlambda
    

# @njit
def prognostic_vorticity(vort_f,vort_n,vort_p,vort_s,vort_t,div_n,u_n,v_n,us,vs,omega_n,
                         omegas_n,domegadphi,domegadlambda,domegadphi_s,domegadlambda_s,
                         Ps_n,Pf,nlambda,nphi,nP,f,cosphi,
                         dlambda,dphi,dP,dt,a,):

    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                # u * d/dlambda(\xi + f)
                u_dot_dvort = u_n[i,j,k]*(1/(a*cosphi[j])*(vort_t[i-2,j,k] +
                                8*(-vort_t[i-1,j,k] + vort_t[i+1,j,k])+vort_t[i+2,j,k])/(12*dlambda))

                # v * d/dphi(\xi + f)
                v_dot_dvort = (v_n[i,j,k]/(a)*((vort_t[i,j-2,k]+8*(-vort_t[i,j-1,k]+vort_t[i,j+1,k])
                      -vort_t[i,j+2,k])/(12*dphi)))

                #advection of vorticity in P-direction
                if k == 0:
                    adv_omegaxi =(((0.5*(vort_n[i,j,k+1]+vort_n[i,j,k])*omega_n[i,j,k+1]))/(dP[0])
                               - vort_n[i,j,k]*(omega_n[i,j,k+1])/(dP[0])) #placeholder dP must be changed for inhomogeneous p-resolution
                elif k == nP-1:
                    adv_omegaxi = (((vort_s[i,j]*omegas_n[i,j]-0.5*(vort_n[i,j,k]+vort_n[i,j,k-1])
                               *omega_n[i,j,k]))/(Ps_n[i,j]-Pf[nP-1])
                               - vort_n[i,j,k]*(omegas_n[i,j]-omega_n[i,j,k])/(Ps_n[i,j]-Pf[nP-1])) #placeholder dP must be changed for inhomogeneous p-resolution
                else:
                    adv_omegaxi = (((0.5*(vort_n[i,j,k+1]+vort_n[i,j,k])*omega_n[i,j,k+1]-0.5
                               *(vort_n[i,j,k]+vort_n[i,j,k-1])*omega_n[i,j,k]))/(dP[0])
                               - vort_n[i,j,k]*(omega_n[i,j,k+1]-omega_n[i,j,k])/(dP[0])) #placeholder dP must be changed for inhomogeneous p-resolution

                # (\xi+f) \nabla \cdot V
                vort_times_div = vort_t[i,j,k] * div_n[i,j,k]

                # k \cdot (d/dp V \cross \nabla \omega)
                #TODO
                if k == 0:
                    solenoid1 = (1/a)*((0.5*(u_n[i,j,k+1]+u_n[i,j,k])*domegadphi[i,j,k+1])/(dP[0])
                               - u_n[i,j,k]*(domegadphi[i,j,k+1])/(dP[0]))
                    solenoid2 = (1/a*cosphi[j])*(((0.5*(v_n[i,j,k+1]+v_n[i,j,k])*domegadlambda[i,j,k+1]-0.5
                               *(v_n[i,j,k]+v_n[i,j,k-1])*domegadlambda[i,j,k]))/(dP[0])
                               - v_n[i,j,k]*(domegadlambda[i,j,k+1]-domegadlambda[i,j,k])/(dP[0]))
                elif k == nP-1:
                    solenoid1 = (1/a)*(((0.5*(us[i,j]+u_n[i,j,k])*domegadphi_s[i,j]-0.5
                               *(u_n[i,j,k]+u_n[i,j,k-1])*domegadphi[i,j,k]))/(dP[0])
                               - u_n[i,j,k]*(domegadphi_s[i,j]-domegadphi[i,j,k])/(dP[0]))
                    solenoid2 = (1/a*cosphi[j])*(((0.5*(vs[i,j]+v_n[i,j,k])*domegadlambda_s[i,j]-0.5
                               *(v_n[i,j,k]+v_n[i,j,k-1])*domegadlambda[i,j,k]))/(dP[0])
                               - v_n[i,j,k]*(domegadlambda_s[i,j]-domegadlambda[i,j,k])/(dP[0]))
                else:
                    solenoid1 = (1/a)*(((0.5*(u_n[i,j,k+1]+u_n[i,j,k])*domegadphi[i,j,k+1]-0.5
                               *(u_n[i,j,k]+u_n[i,j,k-1])*domegadphi[i,j,k]))/(dP[0])
                               - u_n[i,j,k]*(domegadphi[i,j,k+1]-domegadphi[i,j,k])/(dP[0]))
                    solenoid2 = (1/a*cosphi[j])*(((0.5*(v_n[i,j,k+1]+v_n[i,j,k])*domegadlambda[i,j,k+1]-0.5
                               *(v_n[i,j,k]+v_n[i,j,k-1])*domegadlambda[i,j,k]))/(dP[0])
                               - v_n[i,j,k]*(domegadlambda[i,j,k+1]-domegadlambda[i,j,k])/(dP[0]))

                vort_f[i,j,k] = vort_p[i,j,k] + 2*dt*(-u_dot_dvort - v_dot_dvort -
                                                  adv_omegaxi + vort_times_div +
                                                  solenoid1 + solenoid2)
                if i == 64:
                    if j == 30:
                        if k == 10:
                            print(-u_dot_dvort, v_dot_dvort, adv_omegaxi, vort_times_div,
                                                  solenoid1, solenoid2)

    return vort_f

def update_BCs(tupl,vector,nlambda):
    out = []
    for psi,vec in zip(tupl,vector):
        psi = update_periodic_BC(psi)
        psi = update_polar_BC(psi, nlambda, vec)
        out.append(psi)
    return out

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

@jit(nopython=True,parallel=False)
def old_iterative_solver_sphere(Qin,R,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,epsilon=1e-5,maxiter=3e4,exit_status=0):
    """
    Successive over-relaxation solver for the 2D Poisson equation in spherical
    geometry. Even though the solver is only two dimensional the provided arrays
    must have a third dimension representing "height" which can be singleton.

    Parameters
    ----------
    Qin : np.ndarray
        DESCRIPTION.
    R : np.ndarray
        DESCRIPTION.
    
    The following arguments are the grid parameters
    a : float
        Sphere radius.
    tanphi : np.ndarray
        Array containing the numerical values of the tangent function of 
        latitude (phi)
    cosphi : np.ndarray
        Array containing the numerical values of the cosine function of 
        latitude (phi).
    dlambda : float
        Grid spacing in longitude (radians)
    dphi : float
        Grid spacing in latitude (radians)
    nlambda : int
        Number of grid points in the longitudinal direction
    nphi : int
        Number of grid points in the latitudinal direction
    nP : int
        Number of vertical layers

    exit_status : int, optional
        Given to reduce the number of varibles to initialize. The default is 0.

    Returns
    -------
    Q : TYPE
        DESCRIPTION.
    stati : TYPE
        DESCRIPTION.
    ns : TYPE
        DESCRIPTION.
    deltas : TYPE
        DESCRIPTION.

    """
    Q=Qin[4:-4,4:-4,:].copy()
    R = R[4:-4,4:-4,:].copy()

    a1 = (1./(2./(a*a*dlambda*dlambda*cosphi[4:-4]*cosphi[4:-4])+2./(a*a*dphi*dphi)))
    a2 = (1./(a*a*dlambda*dlambda*cosphi[4:-4]*cosphi[4:-4]))
    a3 = -tanphi[4:-4]/(a*a*2*dphi)
    a4 = 1./(a*a*dphi*dphi)
    
    # maxiter=30000
    # epsilon = 0.00000001
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
            Q_temp = Q[:,:,k].copy()
            Q_old = Q_temp[:,:].copy()
            # Q_temp = Q_old.copy()
            omg = 2. / ( 1. + np.sin(np.pi/(n_iter+1)) )
            for i in prange(nlambda):
                if i == nlambda-1:
                    i=-1
                for j in prange(nphi):
                    if j == 0:
                        Q[i,j,k] = (a1[j]*(a2[j]*(Q_temp[i+1,j] + Q_temp[i-1,j]) +
                            a3[j]*(Q_temp[i,j+1] - Q_temp[i-nlambda//2,j+1]) +
                            a4*(Q_temp[i,j+1] + Q_temp[i-nlambda//2,j+1]) - R[i,j,k]))
                    elif j == nphi-1:
                        Q[i,j,k] = (a1[j]*(a2[j]*(Q_temp[i+1,j] + Q_temp[i-1,j]) +
                            a3[j]*(Q_temp[i-nlambda//2,j-1] - Q_temp[i,j-1]) +
                            a4*(Q_temp[i-nlambda//2,j-1] + Q_temp[i,j-1]) - R[i,j,k]))
                    else:
                        Q[i,j,k] = (a1[j]*(a2[j]*(Q_temp[i+1,j] + Q_temp[i-1,j]) +
                            a3[j]*(Q_temp[i,j+1] - Q_temp[i,j-1]) +
                            a4*(Q_temp[i,j+1] + Q_temp[i,j-1]) - R[i,j,k]))

                    Q[i,j,k] = omg*Q[i,j,k] + (1-omg)*Q_old[i,j]
                    Q_temp[i,j] = Q[i,j,k]
                    delta  += np.abs(Q[i,j,k] - Q_old[i,j])
            delta/=(nlambda*nphi*a*a)
    return Q.copy(),stati,ns,deltas


@jit(nopython=True,parallel=False)
def iterative_solver_sphere(Qin,R,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,epsilon=1e-5,maxiter=3e4,exit_status=0,omg=None):
    """
    Successive over-relaxation solver for the 2D Poisson equation in spherical
    geometry. Even though the solver is only two dimensional the provided arrays
    must have a third dimension representing "height" which can be singleton.

    Parameters
    ----------
    Qin : np.ndarray
        DESCRIPTION.
    R : np.ndarray
        DESCRIPTION.
    
    The following arguments are the grid parameters
    a : float
        Sphere radius.
    tanphi : np.ndarray
        Array containing the numerical values of the tangent function of 
        latitude (phi)
    cosphi : np.ndarray
        Array containing the numerical values of the cosine function of 
        latitude (phi).
    dlambda : float
        Grid spacing in longitude (radians)
    dphi : float
        Grid spacing in latitude (radians)
    nlambda : int
        Number of grid points in the longitudinal direction
    nphi : int
        Number of grid points in the latitudinal direction
    nP : int
        Number of vertical layers

    exit_status : int, optional
        Given to reduce the number of varibles to initialize. The default is 0.

    Returns
    -------
    Q : TYPE
        DESCRIPTION.
    stati : TYPE
        DESCRIPTION.
    ns : TYPE
        DESCRIPTION.
    deltas : TYPE
        DESCRIPTION.

    """
    Q=Qin[4:-4,4:-4,:].copy()
    R = R[4:-4,4:-4,:].copy()

    print(Q.shape)

    a1 = (1./(a*a*cosphi[4:-4]*cosphi[4:-4]))
    a2 = 1./(dlambda*dlambda)
    a3 = cosphi[4:-4]
    a4 = -sinphi[4:-4]/(2*dphi)
    a5 = cosphi[4:-4]/(dphi*dphi)
    # a1 = (1./(2./(a*a*dlambda*dlambda*cosphi*cosphi)+2./(a*a*dphi*dphi)))
    # a2 = (1./(a*a*dlambda*dlambda*cosphi*cosphi))
    # a3 = -tanphi/(a*a*2*dphi)
    # a4 = 1./(a*a*dphi*dphi)
    
    # maxiter=30000
    # epsilon = 0.00000001
    delta=1
    
    calc_omg = False
    if omg == None:
        calc_omg = True

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
            Q_temp = Q[:,:,k].copy()
            # Q_temp[3:-3,3:-3] = apply_BCs(Q[3:-3,3:-3],nlambda,k)[:,:,k].copy()
            Q_old = Q_temp[:,:].copy()
            # Q_temp = Q_old.copy()
            if calc_omg:
                omg = 2. / ( 1. + np.sin(np.pi/(n_iter+1)))
            for i in prange(nlambda):
                if i == nlambda-1.:
                    i = -1
                for j in prange(nphi):
                    if j == 0:
                        Q[i,j,k] = 0.5*(1/(a2+a3[j]*a5[j]))*(a2*(Q_temp[i+1,j]+Q_temp[i-1,j]) +
                                                a3[j]*(a4[j]+a5[j])*Q_temp[i,j+1] +
                                                a3[j]*(a5[j]-a4[j])*Q_temp[i-nlambda//2,j] -
                                                (1/a1[j])*R[i,j,k])
                    elif j == nphi-1:
                        Q[i,j,k] = 0.5*(1/(a2+a3[j]*a5[j]))*(a2*(Q_temp[i+1,j]+Q_temp[i-1,j]) +
                                                a3[j]*(a4[j]+a5[j])*Q_temp[i-nlambda//2,j] +
                                                a3[j]*(a5[j]-a4[j])*Q_temp[i,j-1] -
                                                (1/a1[j])*R[i,j,k])
                    else:
                        Q[i,j,k] = 0.5*(1/(a2+a3[j]*a5[j]))*(a2*(Q_temp[i+1,j]+Q_temp[i-1,j]) +
                                                a3[j]*(a4[j]+a5[j])*Q_temp[i,j+1] +
                                                a3[j]*(a5[j]-a4[j])*Q_temp[i,j-1] -
                                                (1/a1[j])*R[i,j,k])

                    Q[i,j,k] = omg*Q[i,j,k] + (1-omg)*Q_old[i,j]
                    Q_temp[i,j] = Q[i,j,k]
                    delta  += np.abs(Q[i,j,k] - Q_old[i,j])
            delta/=(nlambda*nphi*a*a)
    return Q.copy(),stati,ns,deltas

@jit(nopython=True,parallel=False)
def jacobi_solver_sphere(Qin,R,a,tanphi,cosphi,dlambda,dphi,nlambda,nphi,nP,epsilon=1e-5,maxiter=3e4,exit_status=0,omg=None):
    #omg = 1
    Q=Qin[4:-4,4:-4,:].copy()
    R = R[4:-4,4:-4,:].copy()

    print(Q.shape)

    # Q[:]=0.0
    # Qd = np.zeros([nlambda+2,nphi+2,nP])
    # Qstar = Q.copy()
    # Q_old = Q[:,:,0].copy()
    # cosphi_d = np.zeros([nphi+2])
    # tanphi_d = np.zeros([nphi+2])
    # cosphi_d[1:-1]=cosphi
    # tanphi_d[1:-1]=tanphi
    

    
    a1 = (1./(2./(a*a*dlambda*dlambda*cosphi[4:-4]*cosphi[4:-4])+2./(a*a*dphi*dphi)))
    a2 = (1./(a*a*dlambda*dlambda*cosphi[4:-4]*cosphi[4:-4]))
    a3 = -tanphi[4:-4]/(a*a*2*dphi)
    a4 = 1./(a*a*dphi*dphi)
    
    # maxiter=30000
    # epsilon = 0.000001
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
#        while np.max(np.abs(Q[:,:,k]-Q_old))>epsilon:# or n_iter<1 and n_iter<maxiter:
#            print((np.max(np.abs(Q[:,:,k]-Q_old))>epsilon))
            n_iter+=1
            #print('starting_iteration {}'.format(n_iter))
#            print(k,n_iter)
            # Q[1:-1,1:-1,k] = Q[1:-1,1:-1,k]
            #set periodic boundary in lambda
            # Q[0,:,k] = Q[-1,:,k]
            # Q[-1,:,k] = Q[0,:,k]
            # #set polar boundary condition in phi
            # Q[1:-1,0,k] = np.roll(Q[1:-1,-2,k],nlambda//2,axis=0)
            # Q[1:-1,-1,k] = np.roll(Q[1:-1,1,k],nlambda//2,axis=0)
            Q_temp = Q[:,:,k].copy()
            Q_old = Q_temp[:,:].copy()
            for i in prange(nlambda):
                if i == nlambda-1.:
                    i = -1
                for j in prange(nphi):
                    if j == 0:
                        Q[i,j,k] = (a1[j]*(a2[j]*(Q_old[i+1,j]+Q_old[i-1,j])+a3[j]
                             *(Q_old[i,j+1]-Q_old[i-nlambda//2,j])+a4*(Q_old[i,j+1]+Q_old[i-nlambda//2,j])-R[i,j,k]))
                    elif j == nphi-1:
                        Q[i,j,k] = (a1[j]*(a2[j]*(Q_old[i+1,j]+Q_old[i-1,j])+a3[j]
                             *(Q_old[i-nlambda//2,j]-Q_old[i,j-1])+a4*(Q_old[i-nlambda//2,j]+Q_old[i,j-1])-R[i,j,k]))
                    else:
                        Q[i,j,k] = (a1[j]*(a2[j]*(Q_old[i+1,j]+Q_old[i-1,j])+a3[j]
                             *(Q_old[i,j+1]-Q_old[i,j-1])+a4*(Q_old[i,j+1]+Q_old[i,j-1])-R[i,j,k]))
                    delta  += np.abs(Q[i,j,k] - Q_old[i,j])
            delta/=(nlambda*nphi*a*a)
#            omg = 2. / ( 1. + np.sin(np.pi/(n_iter+1)) )
            #Q = omg*Qstar + (1.-omg)*Q
            # delta = np.max(np.abs(Q_old[1:-1,1:-1]-Q[1:-1,1:-1,k]))
            # if n_iter % 100 == 0:
                # print(n_iter,delta)
            # print(n_iter,delta)
#            if k == 2:
#                print(k,n_iter,delta)
#            print(n_iter,delta)
    # print(n_iter,delta)
    return Q.copy(),stati,ns,deltas

#### Main 

if __name__ == '__main__':
    init = True
    if init:
        exec(open('/home/hansb/github/sdycore_lab/isobaric_dycore/isobaric_gcm_4th_order_ghost_BC_mass_energy_fixed_time_split_H_S.py').read())


    vort_f = np.zeros_like(u_n)
    vort_n = vort_f.copy()
    vort_p = vort_f.copy()
    div_n = vort_f.copy()
    div_p = vort_f.copy()

    vort_s = us.copy()

    vort_p,div_p = diag_vorticity_divergence(vort_p,div_p,u_p,v_p)
    vort_n,div_n = diag_vorticity_divergence(vort_n,div_n,u_n,v_n)

    domegadlambda = np.zeros_like(omega_n)
    domegadphi = np.zeros_like(omega_n)
    domegadlambda = differentiate_lambda(omega_n,domegadlambda,dlambda,nlambda,nphi,nP,cosphi,a)
    domegadphi = differentiate_phi(omega_n,domegadphi,dphi,nlambda,nphi,nP,cosphi)

    domegadphi_s = us.copy()
    domegadlambda_s = us.copy()

    f = OMEGA*sinphi
    vort_t =  np.rollaxis(np.rollaxis(vort_n,1,3)+f,1,3)

    vort_n,vort_p,vort_t,div_n,domegadlambda,domegadphi,u_n,v_n = update_BCs(
        (vort_n,vort_p,vort_t,div_n,domegadlambda,domegadphi,u_n,v_n),
        (False,False,False,False,False,False,True,True),nlambda
        )


    vort_f = prognostic_vorticity(vort_f,vort_n,vort_p,vort_s,vort_t,div_n,u_n,v_n,us,vs,omega_n,
                         omegas_n,domegadphi,domegadlambda,domegadphi_s,domegadlambda_s,
                         Ps_n,Pf,nlambda,nphi,nP,f,cosphi,
                         dlambda,dphi,dP,dt,a,)

    stream = np.zeros_like(vort_n)
    stream[4:-4,4:-4,:],stati,ns,deltas = iterative_solver_sphere(stream,vort_n,a,tanphi,cosphi,
                                dlambda,dphi,nlambda,nphi,nP,epsilon=1e-9,maxiter=3e4,
                                exit_status=0,omg=None)

    if 1 in stati:
        print(stati,ns)
    else:
        print('all STREAM converged',ns)


    # stream = np.zeros_like(vort_n)
    # stream[4:-4,4:-4,:],stati,ns,deltas = jacobi_solver_sphere(stream,vort_n,a,tanphi,cosphi,
    #                             dlambda,dphi,nlambda,nphi,nP,epsilon=1e-14,maxiter=6e4,
    #                             exit_status=0)

    # if 1 in stati:
    #     print(stati,ns)
    # else:
    #     print('all STREAM converged',ns)

    # stream = np.zeros_like(vort_n)
    # stream[4:-4,4:-4,:],stati,ns,deltas = old_iterative_solver_sphere(stream,vort_n,a,tanphi,cosphi,
    #                             dlambda,dphi,nlambda,nphi,nP,epsilon=1e-8,maxiter=3e4,
    #                             exit_status=0)

    # if 1 in stati:
    #     print(stati,ns)
    # else:
    #     print('all STREAM converged',ns)

    pot = np.zeros_like(vort_n)
    pot[4:-4,4:-4,:],stati,ns,deltas = iterative_solver_sphere(pot,div_n,a,tanphi,cosphi,
                                dlambda,dphi,nlambda,nphi,nP,epsilon=1e-9,maxiter=3e4,
                                exit_status=0,omg=None)
    if 1 in stati:
        print(stati,ns)
    else:
        print('all POT converged',ns)

    stream,pot = update_BCs((-stream,pot),(False,False),nlambda)

    temp1 = np.zeros_like(vort_n)
    temp2 = np.zeros_like(vort_n)
    u_r = -1/a*differentiate_phi(stream,temp1,dphi, nlambda, nphi, nP, cosphi)
    u_d =  differentiate_lambda(pot,temp2,dlambda,nlambda,nphi,nP,cosphi,a,scaling=True)
    temp1 = np.zeros_like(vort_n)
    temp2 = np.zeros_like(vort_n)
    u_d2 = differentiate_lambda(pot.copy(),temp2.copy(),dlambda,nlambda,nphi,nP,cosphi,a,scaling=False)
    u_d2 = (sinphi[None,:,None]*u_d2)/a
    u_d3 = np.gradient(pot.copy(),dlambda,axis=0)
    u_d3 = (sinphi[None,:,None]*u_d3)/a
    u_d4 = differentiate_lambda_O4(pot.copy(), temp2.copy(), dlambda,cosphi)
    u_d5 = differentiate_lambda_O2(pot.copy(), temp2.copy(), dlambda,cosphi)
    # u_d4 = ((sinphi)[None,:,None]*u_d4)/a
    temp1 = np.zeros_like(vort_n)
    temp2 = np.zeros_like(vort_n)
    v_r = differentiate_lambda(stream,temp1,dlambda,nlambda,nphi,nP,cosphi,a,scaling=True)
    temp1 = np.zeros_like(vort_n)
    temp2 = np.zeros_like(vort_n)
    v_r2 = differentiate_lambda(stream,temp1,dlambda,nlambda,nphi,nP,cosphi,a,scaling=False)
    v_r2 = (cosphi[None,:,None]*v_r2)/a
    v_r3 = np.gradient(stream,dlambda,axis=0)
    v_r3 = (cosphi[None,:,None]*v_r2)/a
    v_d = 1/a*differentiate_phi(pot,temp1,dphi, nlambda, nphi, nP, cosphi)
    u_r,u_d3,v_r3,v_d = update_BCs((u_r,u_d2,v_r2,v_d),(True,True,True,True,),nlambda)
    
    ub = u_r + u_d3
    vb = v_r3 + v_d

    F = np.fft.fft(vort_n[4:-4,4:-4,10],axis=1)

    U = np.zeros_like(F)


    import scipy.sparse
    import scipy.sparse.linalg

    cophi = np.flip(phi+np.pi/2)

    for N in range(-int(nlambda/2),int(nlambda/2)):
        i = int(N+nlambda/2)
        # N=-nlambda/2
    
        n = nphi
        k1 = np.array([np.ones(n-1),-2*np.ones(n),np.ones(n-1)])
        offset = [-1,0,1]
        A1 = scipy.sparse.diags(k1,offset).toarray()
        A1[0,0] += A1[0,0]*(-1)**N
        A1[nphi-1,nphi-1] += A1[nphi-1,nphi-1]*(-1)**N
        A1 = 1/dphi**2*A1
        k2 = np.array([np.ones(n-1),0*np.ones(n),np.ones(n-1)])
        A2 = scipy.sparse.diags(k2,offset).toarray()
        A2[0,0] += A2[0,0]*(-1)**N
        A2[nphi-1,nphi-1] += A2[nphi-1,nphi-1]*(-1)**N
        A2 = 1/(np.tan(cophi)*dphi)*A2
        k3 = np.array([0*np.ones(n-1),1*np.ones(n),0*np.ones(n-1)])
        A3 = scipy.sparse.diags(k3,offset).toarray()
        A3 = (N**2)/(np.sin(cophi)*np.sin(cophi))*A3
    
        A = A1+A2+A3
        A = scipy.sparse.csc_matrix(A)
    
        F0 = F[i,:]
    
        U0=scipy.sparse.linalg.spsolve(A,F0)
    
        U[i,:] = U0

    test = np.fft.ifft(U)


    from windspharm.standard import VectorWind
    from windspharm.tools import prep_data, recover_data, order_latdim

    uwnd, uwnd_info = prep_data(u_n[4:-4,4:-4], 'xyp')

    vwnd, vwnd_info = prep_data(v_n[4:-4,4:-4], 'xyp')

    lats, uwnd, vwnd = order_latdim(phi, uwnd, vwnd)
    w = VectorWind(uwnd, vwnd)
    sf, vp = w.sfvp()
    sf = recover_data(sf, uwnd_info)
    vp = recover_data(vp, vwnd_info)
    uchi, vchi, upsi, vpsi = w.helmholtz()
    uchi = recover_data(uchi, uwnd_info)
    vchi = recover_data(vchi, uwnd_info)
    upsi = recover_data(upsi, uwnd_info)
    vpsi = recover_data(vpsi, uwnd_info)

    s=w.s
    #convert vort + div from A-grid model to u,v using Spharm
    vrt, vrt_info = prep_data(vort_n[4:-4,4:-4], 'xyp')
    div, div_info = prep_data(div_n[4:-4,4:-4], 'xyp')

    lats, vrt, div = order_latdim(phi, vrt, div)
    vrtspec = s.grdtospec(vrt)
    divspec = s.grdtospec(div)
    ugrid, vgrid = s.getuv(vrtspec,divspec)
    ugrid=recover_data(ugrid,vrt_info)
    vgrid=recover_data(vgrid,vrt_info)
    
    #convert stream + pot from iterative solver to u,v using Spharm
    pt, pt_info = prep_data(pot[4:-4,4:-4], 'xyp')
    st, st_info = prep_data(stream[4:-4,4:-4], 'xyp')
    
    lats, st, pt = order_latdim(phi, st, pt)
    
    stspec = s.grdtospec(st)
    ptspec = s.grdtospec(pt)
    vrot, urot = s.getgrad(stspec)
    udiv, vdiv = s.getgrad(ptspec)
    
    urot = recover_data(-urot, pt_info)
    vrot = recover_data(vrot, pt_info)
    udiv = recover_data(udiv, pt_info)
    vdiv = recover_data(vdiv, pt_info)

    # Q[i,j,k] = -R[i,j,k] + (a1[j]*(a2*(Q_temp[i+1,j]-2*Q_temp[i,j]+Q_temp[i-1,j]) +
    #                   a3[j]*(a4[j]*(Q_temp[i,j+1]-Q_temp[i,j-1])+
    #                          a5[j]*(Q_temp[i,j+1]-2*Q_temp[i,j]+Q_temp[i,j-1]))))