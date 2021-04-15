#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
4th order finite difference primitive equations dynamical core with isobaric vertical coordinate
    Copyright (C) 2018  Hans Brenna
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

@author: Hans Brenna
This program implements a hydrostatic dry primitive equations GCM on an isobaric 
vertical coordinate. The numerics are 4th order centered differences in 
space and 2nd order centered (leapfrog) in time on an unstaggered uniform lat, 
lon horizontal grid (A-grid). 

Subgrid scale diffusion is neglected and energy is removed from the small 
grid scale by a 4th order Shapiro filter in both horizontal directions.
Forcing will be adapted from Held & Suarez 1994 with newtonian relaxation to
radiative equilibrium and Reyleigh damping of the horizontal velocities in 
the lower layers. To control the 2dt computational mode we apply a 
Robert-Asselin filter to the prognostic variables.

Prognostic varriables: u,v,T,Ps
Diagnostic variables: z,omega,omega_s,Q,theta,Fl,Fphi

A speedup of the code was acheived using the numba just-in-time compiler @jit
on most of the functions doing for loops.

Eventually moving to shared memory parallelization?

Rewrite using Bohrium?

Some design thoughts: There are 3 directions this project can take over the
short term. 
    
    1: Implementing a C-grid version using the exact same formulation of the
    continuous equations and the same discretization philosophy using simple
    4th order centered differences for all spatial derivatives. This involves
    defining 2 more grids and rewriting some terms using averages. Could 
    possibly involve changes to pole handling, but I'm not sure.
    
    2: Implementing the Moist Idealized Test Case physics from XX in the A-grid
    model. This means translating most of a physics package into python, but
    little new development other than implementing a water vapor transport
    equation as a prognostic. Doing this first would then make it very easy to
    later couple a C-grid model to the same physics
    
    3: Rewriting the current model as a sigma-coordinate model. Doing this 
    first would later make it easier to implement a C-grid sigma-coordinate
    dycore, which would be the "best" version to eventually expand into an EMIC.
    Am I interested in this direction at all? Currently only considering Aquaplanet
    applications and ITCs, but doing general changes first might be smart.

Other ideas could be changing to a z-coordinate. Longer term plans include
implementing simplified radiation and a mixed-layer ocean making it possible 
to do "climate"-simulations for ECS for instance

Want to do the physics package first.

Planning to implement a moist version as well

"""
from __future__ import print_function
import sys
from numba import jit,prange
import numpy as np
from scipy.fftpack import rfft,irfft
from threading import Thread
import pdb
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import time


np.seterr(all='warn')
np.seterr(divide='raise',invalid='raise') #Set numpy to raise exceptions on 
#invalid operations and divide by zero

class threadWithReturn(Thread):
    def __init__(self, *args, **kwargs):
        super(threadWithReturn, self).__init__(*args, **kwargs)

        self._return = None

    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)

    def join(self, *args, **kwargs):
        super(threadWithReturn, self).join(*args, **kwargs)

        return self._return



@jit(nopython=True,nogil=True,parallel=True)
def prognostic_u(u_f,u_n,u_p,v_n,us,omega_n,z_n,Ps_n,omegas_n):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                #advection of u in u-direction
                adv_uu = ((u_n[i,j,k]/(a*cosphi[j]))*((u_n[i-2,j,k]+8*(-u_n[i-1,j,k]+u_n[i+1,j,k])
                      -u_n[i+2,j,k])/(12*dlambda)))
                #advection of u in v-direction, get points from across poles
                adv_vu = (v_n[i,j,k]/(a)*((u_n[i,j-2,k]+8*(-u_n[i,j-1,k]+u_n[i,j+1,k])
                      -u_n[i,j+2,k])/(12*dphi)))
                #advection of u in omega-direction
                if k == 0:
                    adv_omegau = (((0.5*(u_n[i,j,k+1]+u_n[i,j,k])*omega_n[i,j,k+1]))/(dP[0])
                               - u_n[i,j,k]*(omega_n[i,j,k+1])/(dP[0])) #placeholder dP must be changed for inhomogeneous p-resolution
                elif k == nP-1:
                    adv_omegau = (((us[i,j]*omegas_n[i,j]-0.5*(u_n[i,j,k]+u_n[i,j,k-1])
                               *omega_n[i,j,k]))/(Ps_n[i,j]-Pf[nP-1])
                               - u_n[i,j,k]*(omegas_n[i,j]-omega_n[i,j,k])/(Ps_n[i,j]-Pf[nP-1])) #placeholder dP must be changed for inhomogeneous p-resolution
                else:
                    adv_omegau = (((0.5*(u_n[i,j,k+1]+u_n[i,j,k])*omega_n[i,j,k+1]-0.5
                               *(u_n[i,j,k]+u_n[i,j,k-1])*omega_n[i,j,k]))/(dP[0])
                               - u_n[i,j,k]*(omega_n[i,j,k+1]-omega_n[i,j,k])/(dP[0])) #placeholder dP must be changed for inhomogeneous p-resolution
                #coriolis term
                cor_u = 2*OMEGA*sinphi[j]*v_n[i,j,k]
                #gradient of geopotential height
                gdz_u = ((g)/(a*cosphi[j])*((z_n[i-2,j,k]+8*(-z_n[i-1,j,k]+z_n[i+1,j,k])
                      -z_n[i+2,j,k])/(12*dlambda)))
                #curvature term
                curv_u = ((u_n[i,j,k]*v_n[i,j,k]*tanphi[j])/a)                
                
                u_f[i,j,k] = (u_p[i,j,k]+2*dt*(-(adv_uu+adv_vu+adv_omegau)+cor_u
                           -gdz_u+curv_u))
    return u_f

@jit(nopython=True,nogil=True,parallel=True)
def prognostic_v(v_f,v_n,v_p,u_n,vs,omega_n,z_n,Ps_n,omegas_n):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                #advection of v in u-direction
                adv_uv = ((u_n[i,j,k]/(a*cosphi[j]))*((v_n[i-2,j,k]+8*(-v_n[i-1,j,k]+v_n[i+1,j,k])
                      -v_n[i+2,j,k])/(12*dlambda)))
                #advection of v in v-direction, get points from across poles
                adv_vv = (((v_n[i,j,k]/(a))*((v_n[i,j-2,k]+8*(-v_n[i,j-1,k]+v_n[i,j+1,k])
                      -v_n[i,j+2,k])/(12*dphi))))
                if k == 0:
                    adv_omegav = (((0.5*(v_n[i,j,k+1]+v_n[i,j,k])*omega_n[i,j,k+1]))/(dP[0])
                               - v_n[i,j,k]*(omega_n[i,j,k+1]-omega_n[i,j,k])/(dP[0])) #placeholder dP must be changed for inhomogeneous p-resolution
                elif k == nP-1:
                    adv_omegav = (((vs[i,j]*omegas_n[i,j]-0.5*(v_n[i,j,k]+v_n[i,j,k-1])
                               *omega_n[i,j,k]))/(Ps_n[i,j]-Pf[nP-1])
                               - v_n[i,j,k]*(omegas_n[i,j]-omega_n[i,j,k])/(Ps_n[i,j]-Pf[nP-1])) #placeholder dP must be changed for inhomogeneous p-resolution
                else:
                    adv_omegav = (((0.5*(v_n[i,j,k+1]+v_n[i,j,k])*omega_n[i,j,k+1]-0.5
                               *(v_n[i,j,k]+v_n[i,j,k-1])*omega_n[i,j,k]))/(dP[0])
                               - v_n[i,j,k]*(omega_n[i,j,k+1]-omega_n[i,j,k])/(dP[0]))
                        
                cor_v = 2*OMEGA*sinphi[j]*u_n[i,j,k]
                gdz_v = (g/a*((z_n[i,j-2,k]+8*(-z_n[i,j-1,k]+z_n[i,j+1,k])
                      -z_n[i,j+2,k])/(12*dphi)))
                curv_v = (u_n[i,j,k]*u_n[i,j,k]*tanphi[j])/a
                
                v_f[i,j,k] = (v_p[i,j,k]+2*dt*(-(adv_uv+adv_vv+adv_omegav)-cor_v
                   -gdz_v-curv_v))             
    return v_f

@jit(nopython=True,nogil=True,parallel=True)
def prognostic_T(T_f,T_n,T_p,u_n,v_n,omega_n,theta_n,Ps_n,theta_s,omegas_n):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                adv_uT = ((u_n[i,j,k]/(a*cosphi[j]))*((T_n[i-2,j,k]+8*(-T_n[i-1,j,k]+T_n[i+1,j,k])
                      -T_n[i+2,j,k])/(12*dlambda)))
                adv_vT = (((v_n[i,j,k]/(a))*((T_n[i,j-2,k]+8*(-T_n[i,j-1,k]+T_n[i,j+1,k])
                      -T_n[i,j+2,k])/(12*dphi))))
                if k == 0:
                    adv_omegaT = (T_n[i,j,k]/theta_n[i,j,k]*(((0.5*(theta_n[i,j,k+1]+theta_n[i,j,k])*omega_n[i,j,k+1]))/(dP[0])
                               - theta_n[i,j,k]*(omega_n[i,j,k+1]-omega_n[i,j,k])/(dP[0]))) #placeholder dP must be changed for inhomogeneous p-resolution
                elif k == nP-1:
                    adv_omegaT = (T_n[i,j,k]/theta_n[i,j,k]*(((theta_s[i,j]*omegas_n[i,j]-0.5*(theta_n[i,j,k]+theta_n[i,j,k-1])
                               *omega_n[i,j,k]))/(Ps_n[i,j]-Pf[nP-1])
                               - theta_n[i,j,k]*(omegas_n[i,j]-omega_n[i,j,k])/(Ps_n[i,j]-Pf[nP-1]))) #placeholder dP must be changed for inhomogeneous p-resolution
                else:
                    adv_omegaT = (T_n[i,j,k]/theta_n[i,j,k]*(((0.5*(theta_n[i,j,k+1]+theta_n[i,j,k])*omega_n[i,j,k+1]-0.5
                               *(theta_n[i,j,k]+theta_n[i,j,k-1])*omega_n[i,j,k]))/(dP[0])
                               - theta_n[i,j,k]*(omega_n[i,j,k+1]-omega_n[i,j,k])/(dP[0])))
                        
                T_f[i,j,k] = (T_p[i,j,k]+2*dt*(-(adv_uT+adv_vT+adv_omegaT)))
    return T_f

@jit
def prognostic_Ps(Ps_f,Ps_n,Ps_p,omegas_n,us,vs,lnPs):
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            adv_P = ((us[i,j])/(a*cosphi[j])*(Ps_n[i-2,j]+8*(-Ps_n[i-1,j]+Ps_n[i+1,j])-Ps_n[i+2,j])
                      /(12*dlambda)+(vs[i,j]/(a))*((Ps_n[i,j-2]+8*(-Ps_n[i,j-1]+Ps_n[i,j+1])-Ps_n[i,j+2]))/(12*dphi))
            Ps_f[i,j] = Ps_p[i,j]+2*dt*(-adv_P+omegas_n[i,j])

#    if Ps_f.min() < 92500.:
#        Ps_f[np.where(Ps_f < 92500.)] = 92500.
#    Ps_f = mass_fixer(Ps_f,P0)
#    lnPs[4:-4,4:-4] = np.log(Ps_f[4:-4,4:-4])
    return Ps_f,lnPs

#@jit
#def prognostic_Ps(Ps_f,Ps_n,Ps_p,omegas_n,us,vs,lnPs):
#    for i in range(4,nlambda+4):
#        for j in range(4,nphi+4):
#            adv_P = ((us[i,j])/(a*cosphi[j])*(Ps_n[i+1,j]-Ps_n[i-1,j])
#                      /(2*dlambda)+(vs[i,j]/(a))*(Ps_n[i,j+1]-Ps_n[i,j-1])/(2*dphi))
#            Ps_f[i,j] = Ps_p[i,j]+2*dt*(-adv_P+omegas_n[i,j])
#
#    if Ps_f.min() < 92500.:
#        Ps_f[np.where(Ps_f < 92500.)] = 92500.
#    Ps_f = mass_fixer(Ps_f,P0)
#    lnPs[4:-4,4:-4] = np.log(Ps_f[4:-4,4:-4])
#    return Ps_f,lnPs

@jit(nopython=True,nogil=True,parallel=True)
def diag_omega(omega_n,u_n,v_n,cosphi,dP):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
             for k in range(nP):
                 int_div_u = 0.0
                 int_div_v = 0.0
                 if k == 0:
                     omega_n[i,j,k] = 0.0
                 else:
                     for m in range(0,k):
                         dP_m = dP[0]
                         int_div_v += (((cosphi[j-2]*v_n[i,j-2,m]+8*(-cosphi[j-1]
                                       *v_n[i,j-1,m]+cosphi[j+1]*v_n[i,j+1,m])
                                       -cosphi[j+2]*v_n[i,j+2,m])/(12*dphi))*dP_m)
                             
                         
                         int_div_u += (((u_n[i-2,j,m]+8*(-u_n[i-1,j,m]+u_n[i+1,j,m])
                                   -u_n[i+2,j,m])/(12*dlambda))*dP_m)
                             
                     omega_n[i,j,k] = -(1./(a*cosphi[j])*int_div_u+1./(a*cosphi[j])*int_div_v)
    return omega_n

@jit(nopython=True,nogil=True,parallel=True)
def diag_omega2(omega_n,div):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
             for k in range(nP):
                 if k == 0:
                     omega_n[i,j,k] = 0.0
                 else:
                     omega_n[i,j,k] = (omega_n[i,j,k-1]-div[i,j,k-1]*dP[0])
    return omega_n
    

@jit
def diag_omegas(omegas_n,omega_n,u_n,v_n,us,vs):
#    ub = 0.5*(u_n[:,:,nP-1]+us)
#    vb = 0.5*(v_n[:,:,nP-1]+vs)
    ub = u_n[:,:,nP-1]
    vb = v_n[:,:,nP-1]
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            int_div_v = ((cosphi[j-2]*v_n[i,j-2,nP-1]+8*(-cosphi[j-1]*v_n[i,j-1,nP-1]+cosphi[j+1]*v_n[i,j+1,nP-1])-cosphi[j+2]*v_n[i,j+2,nP-1])/(12*dphi)*dP[0]/2. 
                      + (cosphi[j-2]*vb[i,j-2]+8*(-cosphi[j-1]*vb[i,j-1]+cosphi[j+1]*vb[i,j+1])-cosphi[j+2]*vb[i,j+2])/(12*dphi)*(Ps_n[i,j]-P[nP-1]))
            int_div_u = ((u_n[i-2,j,nP-1]+8*(-u_n[i-1,j,nP-1]+u_n[i+1,j,nP-1])-u_n[i+2,j,nP-1])/(12*dlambda)*dP[0]/2. 
                      + (ub[i-2,j]+8*(-ub[i-1,j]+ub[i+1,j])-ub[i+2,j])/(12*dlambda)*(Ps_n[i,j]-P[nP-1]))
            
            omegas_n[i,j] = (omega_n[i,j,nP-1] + (-(1./(a*cosphi[j])*int_div_u+1./(a*cosphi[j])*int_div_v)))
    return omegas_n

@jit
def diag_omegas3(omegas_n,omega_n,u_n,v_n,Ps_n,us,vs):
#    ub = 0.5*(u_n[:,:,nP-1]+us)
#    vb = 0.5*(v_n[:,:,nP-1]+vs)
    ub = u_n[:,:,nP-1]
    vb = v_n[:,:,nP-1]
    dP_l = Ps_n-P[nP-1]
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            int_div_v = ((cosphi[j-2]*v_n[i,j-2,nP-1]+8*(-cosphi[j-1]*v_n[i,j-1,nP-1]+cosphi[j+1]*v_n[i,j+1,nP-1])-cosphi[j+2]*v_n[i,j+2,nP-1])/(12*dphi)*dP[0]/2. 
                      + (cosphi[j-2]*vb[i,j-2]*dP_l[i,j-2]+8*(-cosphi[j-1]*vb[i,j-1]*dP_l[i,j-1]+cosphi[j+1]*vb[i,j+1]*dP_l[i,j+1])-cosphi[j+2]*vb[i,j+2]*dP_l[i,j+2])/(12*dphi))
            int_div_u = ((u_n[i-2,j,nP-1]+8*(-u_n[i-1,j,nP-1]+u_n[i+1,j,nP-1])-u_n[i+2,j,nP-1])/(12*dlambda)*dP[0]/2. 
                      + (ub[i-2,j]*dP_l[i-2,j]+8*(-ub[i-1,j]*dP_l[i-1,j]+ub[i+1,j]*dP_l[i+1,j])-ub[i+2,j]*dP_l[i+2,j])/(12*dlambda))
            
            omegas_n[i,j] = (omega_n[i,j,nP-1] + (-(1./(a*cosphi[j])*int_div_u+1./(a*cosphi[j])*int_div_v)))
    return omegas_n

@jit
def diag_omegas4(omegas_n,omega_n,u_n,v_n,Ps_n,us,vs):
#    ub = 0.5*(u_n[:,:,nP-1]+us)
#    vb = 0.5*(v_n[:,:,nP-1]+vs)
    ub = u_n[:,:,nP-1]
    vb = v_n[:,:,nP-1]
    dP_l = Ps_n-Pf[nP-1]
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            int_div_v = ((cosphi[j-2]*vb[i,j-2]*dP_l[i,j-2]+8*(-cosphi[j-1]*vb[i,j-1]*dP_l[i,j-1]+cosphi[j+1]*vb[i,j+1]*dP_l[i,j+1])-cosphi[j+2]*vb[i,j+2]*dP_l[i,j+2])/(12*dphi))
            int_div_u = ((ub[i-2,j]*dP_l[i-2,j]+8*(-ub[i-1,j]*dP_l[i-1,j]+ub[i+1,j]*dP_l[i+1,j])-ub[i+2,j]*dP_l[i+2,j])/(12*dlambda))
            
            omegas_n[i,j] = (omega_n[i,j,nP-1] + (-(1./(a*cosphi[j])*int_div_u+1./(a*cosphi[j])*int_div_v)))
    return omegas_n

@jit
def diag_omegas2(omegas_n,omega_n,div):
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            omegas_n[i,j] = omega_n[i,j,nP-1]-div[i,j,nP-1]*(Ps_n[i,j]-Pf[nP-1])
    return omegas_n

#@jit
#def diag_omegas(omegas_n,omega_n,u_n,v_n,us,vs):
#    ub = 0.5*(u_n[:,:,nP-1]+us)
#    vb = 0.5*(v_n[:,:,nP-1]+vs)
#    for i in range(4,nlambda+4):
#        for j in range(4,nphi+4):
#            int_div_v = ((cosphi[j+1]*v_n[i,j+1,nP-1]-cosphi[j-1]*v_n[i,j-1,nP-1])/(2*dphi)*dP[0]/2. 
#                      + (cosphi[j+1]*vb[i,j+1]-cosphi[j-1]*vb[i,j-1])/(2*dphi)*(Ps_n[i,j]-P[nP-1]))
#            int_div_u = ((u_n[i+1,j,nP-1]-u_n[i-1,j,nP-1])/(2*dlambda)*dP[0]/2. 
#                      + (ub[i+1,j]-ub[i-1,j])/(2*dlambda)*(Ps_n[i,j]-P[nP-1]))
#            
#            omegas_n[i,j] = (omega_n[i,j,nP-1] + (-(1./(a*cosphi[j])*int_div_u+1./(a*cosphi[j])*int_div_v)))
#    return omegas_n

#@jit
#def diag_omegas(omegas_n,omega_n,u_n,v_n,us,vs):
#    ub = 0.5*(u_n[:,:,nP-1]+us)
#    vb = 0.5*(v_n[:,:,nP-1]+vs)
#    for i in range(4,nlambda+4):
#        for j in range(4,nphi+4):
#            int_div_v = ((cosphi[j+1]*v_n[i,j+1,nP-1]-cosphi[j-1]*v_n[i,j-1,nP-1])/(2*dphi)*dP[0]/2. 
#                      + (cosphi[j+1]*vb[i,j+1]-cosphi[j-1]*vb[i,j-1])/(2*dphi)*(Ps_n[i,j]-P[nP-1]))
#            int_div_u = ((u_n[i+1,j,nP-1]-u_n[i-1,j,nP-1])/(2*dlambda)*dP[0]/2. 
#                      + (ub[i+1,j]-ub[i-1,j])/(2*dlambda)*(Ps_n[i,j]-P[nP-1]))
#            
#            omegas_n[i,j] = (omega_n[i,j,nP-1] + (-(1./(a*cosphi[j])*int_div_u+1./(a*cosphi[j])*int_div_v)))
#    return omegas_n


@jit(nopython=True,nogil=True,parallel=True)
def diag_z(zf_n,z_n,T_n,Ps_n,dlnP):
    zf_n[:] = 0.0
    z_n[:] = 0.0
    for k in range(nP-1,-1,-1):
        if k == nP-1:
            zf_n[4:-4,4:-4,k] = 0.0+R*T_n[4:-4,4:-4,k]*np.log(Ps_n[4:-4,4:-4]/Pf[k])
            z_n[4:-4,4:-4,k] = 0.0+R*T_n[4:-4,4:-4,k]*np.log(Ps_n[4:-4,4:-4]/P[k])
        else:
            zf_n[4:-4,4:-4,k] = zf_n[4:-4,4:-4,k+1]+R*T_n[4:-4,4:-4,k]*np.log(Pf[k+1]/Pf[k])
            z_n[4:-4,4:-4,k] = zf_n[4:-4,4:-4,k+1]+R*T_n[4:-4,4:-4,k]*np.log(Pf[k+1]/P[k])
    
    zf_n = zf_n/g
    z_n = z_n/g
    # if z_n.min() < 10:
    

    return zf_n,z_n

def z_limiter(z_n,zf_n):
    z_n[np.where(z_n < 10)] = 10
    zf_n[np.where(zf_n < 11)] = 11
    return zf_n,z_n
    

@jit
def helper_dlnP(dlnP,lnPs,Pf):
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            for k in range(nP-1):
                dlnP[i,j,k] = lnP[k+1]-lnP[k]
            dlnP[i,j,nP-1] = lnPs[i,j] - lnP[nP-1]
    return dlnP

@jit(nopython=True,nogil=True,parallel=True)
def diag_Q(Q_n,T_n,Ps_n,T_eq,kT):
    sigmab = 0.7
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                function = ((P[k]/Ps_n[i,j])-sigmab)/(1.-sigmab)
                if function > 0.:
                    kT[i,j,k] = ka+(ks-ka)*function*cosphi[j]**4
                else:
                    kT[i,j,k] = ka
                    
                
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            for k in range(nP):
                Q_n[i,j,k] = -kT[i,j,k]*(T_n[i,j,k]-T_eq[i,j,k])
    return Q_n

#@jit(nopython=True,nogil=True)
def diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n):              
    theta_n[4:-4,4:-4,:] = T_n[4:-4,4:-4,:]*(P0/P)**kappa
    theta_s[4:-4,4:-4] = Ts[4:-4,4:-4]*(P0/Ps_n[4:-4,4:-4])**kappa
    return theta_n,theta_s

@jit(nopython=True,nogil=True,parallel=True)
def diag_divergence(div,u_n,v_n):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                div[i,j,k] = (1./(a*cosphi[j])*(((u_n[i-2,j,k]+8*(-u_n[i-1,j,k]+u_n[i+1,j,k])
                      -u_n[i+2,j,k])/(12*dlambda))+((cosphi[j-2]*v_n[i,j-2,k]+8*(-cosphi[j-1]
                      *v_n[i,j-1,k]+cosphi[j+1]*v_n[i,j+1,k])-cosphi[j+2]*v_n[i,j+2,k])/(12*dphi))))
    return div

#Vorticity and divergence diagnostic functions
def diag_vorticity_divergence(vort,div,u_n,v_n):
    ud = u_n.copy()
    vd = v_n.copy()
    cosphi_d = cosphi.copy()
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            for k in range(nP):
                vort[i,j,k] = (1./(a*cosphi_d[j])*(((vd[i-2,j,k]+8*(-vd[i-1,j,k]+vd[i+1,j,k])
                      -vd[i+2,j,k])/(12*dlambda))-((cosphi_d[j-2]*ud[i,j-2,k]+8*(-cosphi_d[j-1]
                      *ud[i,j-1,k]+cosphi_d[j+1]*ud[i,j+1,k])-cosphi_d[j+2]*ud[i,j+2,k])/(12*dphi))))
                div[i,j,k] = (1./(a*cosphi_d[j])*(((ud[i-2,j,k]+8*(-ud[i-1,j,k]+ud[i+1,j,k])
                      -ud[i+2,j,k])/(12*dlambda))+((cosphi_d[j-2]*vd[i,j-2,k]+8*(-cosphi_d[j-1]
                      *vd[i,j-1,k]+cosphi_d[j+1]*vd[i,j+1,k])-cosphi_d[j+2]*vd[i,j+2,k])/(12*dphi))))
    return vort,div


#surface functions
@jit(nopython=True,nogil=True)
def diag_surface_wind(us,vs,Vs,u_n,v_n,Ps_p,A):

    A = (kv_surf_w/(kv_surf_w+cv*Vs*(Ps_p-P[nP-1]))) #Vs and Ps used at previous time step
    us[4:-4,4:-4] = A[4:-4,4:-4]*u_n[4:-4,4:-4,nP-1]
    vs[4:-4,4:-4] = A[4:-4,4:-4]*v_n[4:-4,4:-4,nP-1]
    Vs[4:-4,4:-4] = np.sqrt(us[4:-4,4:-4]*us[4:-4,4:-4]+vs[4:-4,4:-4]*vs[4:-4,4:-4])
    return us,vs,Vs,A

@jit 
def diag_total_energy(gTE,TE_n,u_n,v_n,T_n):
    TE_n[:] = 0.0
    dP_l = Ps_n-Pf[nP-1]
    TE_n[4:-4,4:-4] = (np.sum((dP[0]*0.5*(u_n[4:-4,4:-4,0:-1]*u_n[4:-4,4:-4,0:-1]+v_n[4:-4,4:-4,0:-1]*v_n[4:-4,4:-4,0:-1])),axis=2)+
            np.sum(dP[0]*cp*T_n[4:-4,4:-4,0:-1]))
    TE_n[4:-4,4:-4] += dP_l[4:-4,4:-4]*(0.5*(u_n[4:-4,4:-4,-1]*u_n[4:-4,4:-4,-1]+v_n[4:-4,4:-4,-1]*v_n[4:-4,4:-4,-1])+cp*T_n[4:-4,4:-4,-1])
#    gTE = 1/g*(TE_n[4:-4,4:-4]*cosphi[4:-4]).sum()
    gTE = 1./g*(np.mean(np.average(TE_n[4:-4,4:-4],axis=1,weights=cosphi[4:-4])))*4*pi*a*a
    return gTE,TE_n

def energy_fixer(gTE_p,gTE_n,T_f,beta):
    print(cp,g,P0,a)
    print(gTE_p-gTE_n)
    beta = (gTE_p-gTE_n)/((cp/g)*(P0*4*pi*a*a))
    T_f = T_f + beta
    return T_f,beta

def energy_fixer2(gTE_p,gTE_n,T_f,beta):
    beta = gTE_p/gTE_n
    T_f=T_f*beta
    return T_f,beta

def H_S_equilibrium_temperature(T_eq,T_eqs,Ps_n):
    dTy = 60 #K
    dthtz = 10 #K
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            for k in range(nP):
                HS_T_func = ((315-dTy*sinphi[j]*sinphi[j]-dthtz
                             *np.log(P[k]/P0)*(cosphi[j])**2)*(P[k]/P0)**kappa)
                T_eq[i,j,k] = np.max([200.,HS_T_func])
        
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            HS_T_func = ((315-dTy*sinphi[j]*sinphi[j]-dthtz
                        *np.log(Ps_n[i,j]/P0)*(cosphi[j])**2)*(Ps_n[i,j]/P0)**kappa)
            T_eqs[i,j] = np.max([200.,HS_T_func])
    return T_eq, T_eqs

@jit(nopython=True,parallel=True)
def H_S_friction(Fl,Fphi,u_n,v_n,Ps_n,kv,sigmab):
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                function = ((P[k]/Ps_n[i,j])-sigmab)/(1.-sigmab)
                if function > 0:
                    kv[i,j,k] = kf*function
                else:
                    kv[i,j,k] = 0
    
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                Fl[i,j,k] = kv[i,j,k]*u_n[i,j,k]
                Fphi[i,j,k] = kv[i,j,k]*v_n[i,j,k] 
    return Fl,Fphi

#@jit
def mass_fixer(Ps_f,P0):
    Ps_f[4:-4,4:-4] = (P0/np.mean(np.average(Ps_f[4:-4,4:-4],axis=1,weights=cosphi[4:-4]))*Ps_f[4:-4,4:-4])
    return Ps_f

@jit(nopython=True,parallel=True)
def update_uvT_with_physics(u_n,v_n,T_n,Fl,Fphi,Q_n,dt_phys):
    u_n = u_n-Fl*dt_phys
    v_n = v_n-Fphi*dt_phys
    T_n = T_n+Q_n*dt_phys
    return u_n,v_n,T_n

@jit
def sponge_layer(sFl,sFphi,skv,u_n,v_n):
    for k in range(nP):
        if P[k] < Psponge:
            skv[:,:,k] = kf*((Psponge-P[k])/Psponge)**2
        else:
            skv[:,:,k] = 0.0
        
    sFl = skv*(u_n)
    sFphi = skv*(v_n)
    return sFl,sFphi

def calculate_polar_filter_coefficients():
    coeffs = np.zeros([nlambda//2,nphi])
    coeffs2 = np.zeros([nlambda,nphi])
    C = cosphi[4:-4]/np.cos(np.deg2rad(45))
    wavenumber = np.arange(1,nlambda//2)
    D = np.zeros([nlambda//2])
#    print(D.shape,C.shape)
    D[0] = 1
#    print(D[71])
#    print(C[71])
    temp=0
#    print(temp.shape)
    D[1:] = 1./(np.sin(wavenumber*pi/nlambda))
    for i in range(nlambda//2):
        for j in range(nphi):
#            print(i,j)
            temp = (C[j]*D[i])**2
#            temp = D[i]
            if temp > 1.0:
                temp = 1.0
            coeffs[i,j] = temp
    coeffs[0,:] = 1.0
    ii =[[i,i] for i in range(1,nlambda//2)]
    ii=np.array(ii)
    ii=ii.flatten()
    coeffs2[1:-1,:] = coeffs[ii,:]
    coeffs2[0,:] = coeffs[0,:]
    coeffs2[-1,:] = coeffs2[-2,:]
    return coeffs2


def new_polar_filter(psi,coeffs):
    psi_trans = rfft(psi[4:-4,4:-4,:],axis=0)
    psi_trans_filtered = np.zeros(psi_trans.shape)
    for k in range(nP):
        psi_trans_filtered[:,:,k] = psi_trans[:,:,k]*coeffs
    psi[4:-4,4:-4,:] = irfft(psi_trans_filtered,axis=0)        
    return psi

def new_polar_filter_2d(psi,coeffs):
    psi_trans = rfft(psi[4:-4,4:-4],axis=0)
    psi_trans_filtered = psi_trans.copy()
    psi_trans_filtered[:,:] = psi_trans[:,:]*coeffs
    psi[4:-4,4:-4] = irfft(psi_trans_filtered,axis=0)        
    return psi


def polar_fourier_filter(psi):
    #filter zonal wavenumbers in polar regions
    #south pole
    psi_trans_sp = rfft(psi[4:-4,4:8,:],axis=0)
    psi_trans_sp[3:,0,:] = 0.0
    psi_trans_sp[11:,1,:] = 0.0
    psi_trans_sp[19:,2,:] = 0.0
    psi_trans_sp[23:,3,:] = 0.0
#    psi_trans[12:,4,:] = 0.0
#    psi_trans[15:,5,:] = 0.0
#    psi_trans[18:,6,:] = 0.0
#    psi_trans[20:,7,:] = 0.0
    #north pole
    psi_trans_np = rfft(psi[4:-4,-8:-4,:],axis=0)
    psi_trans_np[3:,-1,:] = 0.0
    psi_trans_np[11:,-2,:] = 0.0
    psi_trans_np[19:,-3,:] = 0.0
    psi_trans_np[23:,-4,:] = 0.0
#    psi_trans[12:,nphi-5,:] = 0.0
#    psi_trans[15:,nphi-6,:] = 0.0
#    psi_trans[18:,nphi-7,:] = 0.0
#    psi_trans[20:,nphi-8,:] = 0.0
    psi[4:-4,4:8,:] = irfft(psi_trans_sp,axis=0)
    psi[4:-4,-8:-4,:] = irfft(psi_trans_np,axis=0)
    return psi

#@jit
def polar_fourier_filter_2D(psi):
    #filter zonal wavenumbers in polar regions
    #south pole
    psi_trans_sp = np.fft.rfft(psi[4:-4,4:-4],axis=0)
    psi_trans_sp[2:,0] = 0.0
    psi_trans_sp[6:,1] = 0.0
    psi_trans_sp[10:,2] = 0.0
    psi_trans_sp[12:,3] = 0.0
#    psi_trans[12:,4,:] = 0.0
#    psi_trans[15:,5,:] = 0.0
#    psi_trans[18:,6,:] = 0.0
#    psi_trans[20:,7,:] = 0.0
    #north pole
    psi_trans_sp[2:,-1] = 0.0
    psi_trans_sp[6:,-2] = 0.0
    psi_trans_sp[10:,-3] = 0.0
    psi_trans_sp[12:,-4] = 0.0
#    psi_trans[12:,nphi-5,:] = 0.0
#    psi_trans[15:,nphi-6,:] = 0.0
#    psi_trans[18:,nphi-7,:] = 0.0
#    psi_trans[20:,nphi-8,:] = 0.0
    psi[4:-4,4:-4] = np.fft.irfft(psi_trans_sp,axis=0)
    return psi

@jit
def first_order_shapiro_filter(psi):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered_lambda[i,j,k] = (0.25*(psi[i-1,j,k]+2*psi[i,j,k]+psi[i+1,j,k]))
                
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered[i,j,k] = (0.25*(psi_filtered_lambda[i,j-1,k]+2
                            *psi_filtered_lambda[i,j,k]+psi_filtered_lambda[i,j+1,k]))
    return psi_filtered

#@jit(nopython=True)
def update_tau(u_f,v_f,tau):
    if np.max(np.abs(u_f)) > 100.0 or np.max(np.abs(v_f)) > 100.0:
        tau = tau*0.9
        if tau <= 1.0:
            tau = 1.0
    else:
        tau = tau*1.001
        if tau > 10.0:
            tau = 10.0
    return tau

@jit(nopython=True,nogil=True,parallel=True)
def fourth_order_shapiro_filter(psi,vector=False,tau=10):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
#    psi_filtered_2 = psi.copy()
    subtractive_filter = psi.copy()
#    tau = 3000./dt
    #4th order shapiro in lambda direction
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                psi_filtered_lambda[i,j,k] = ((1./256)*(186.*psi[i,j,k]+56.*(psi[i-1,j,k]
                           +psi[i+1,j,k])-28*(psi[i-2,j,k]+psi[i+2,j,k])
                           +8*(psi[i-3,j,k]+psi[i+3,j,k])-(psi[i-4,j,k]+psi[i+4,j,k])))
           
    #4th order shapiro filter in phi direction
    for i in prange(4,nlambda+4):
        for j in prange(4,nphi+4):
            for k in prange(nP):
                psi_filtered[i,j,k] = ((1./256.)*(186.*psi_filtered_lambda[i,j,k]+56.*(psi_filtered_lambda[i,j-1,k]
                        +psi_filtered_lambda[i,j+1,k])-28*(psi_filtered_lambda[i,j-2,k]+psi_filtered_lambda[i,j+2,k])
                        +8*(psi_filtered_lambda[i,j-3,k]+psi_filtered_lambda[i,j+3,k])-(psi_filtered_lambda[i,j-4,k]+psi_filtered_lambda[i,j+4,k])))
    
    subtractive_filter = psi-psi_filtered
    psi_filtered_2 = psi-1.0/tau*subtractive_filter
    
    return psi_filtered_2

@jit
def fourth_order_shapiro_filter_2D(psi,vector=False):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    tau = 3000./dt
    #4th order shapiro in lambda direction
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            psi_filtered_lambda[i,j] = ((1./256)*(186.*psi[i,j]+56.*(psi[i-1,j]
                           +psi[i+1,j])-28*(psi[i-2,j]+psi[i+2,j])
                           +8*(psi[i-3,j]+psi[i+3,j])-(psi[i-4,j]+psi[i+4,j])))
           
    #4th order shapiro filter in phi direction
    for i in range(4,nlambda+4):
        for j in range(4,nphi+4):
            psi_filtered[i,j] = ((1./256.)*(186.*psi_filtered_lambda[i,j]+56.*(psi_filtered_lambda[i,j-1]
                        +psi_filtered_lambda[i,j+1])-28*(psi_filtered_lambda[i,j-2]+psi_filtered_lambda[i,j+2])
                        +8*(psi_filtered_lambda[i,j-3]+psi_filtered_lambda[i,j+3])-(psi_filtered_lambda[i,j-4]+psi_filtered_lambda[i,j+4])))
    
    subtractive_filter = psi-psi_filtered
    psi_filtered_2 = psi-1/tau*subtractive_filter
    
    return psi_filtered_2

@jit(nopython=True,nogil=True,parallel=True)
def Robert_Asselin_filter(psi_f,psi_n,psi_p):
    filter_parameter = 0.1
    psi_n_filtered = psi_n + 0.5*filter_parameter*(psi_p-2*psi_n+psi_f)
    return psi_n_filtered

def update_periodic_BC(psi):
    psi[0,:,:] = psi[-8,:,:]
    psi[1,:,:] = psi[-7,:,:]
    psi[2,:,:] = psi[-6,:,:]
    psi[3,:,:] = psi[-5,:,:]
    
    psi[-1,:,:] = psi[7,:,:]
    psi[-2,:,:] = psi[6,:,:]
    psi[-3,:,:] = psi[5,:,:]
    psi[-4,:,:] = psi[4,:,:]
    
    return psi

def update_polar_BC(psi,nlambda,vector):
    if vector == True:
        factor=-1.0
    else:
        factor=1.0
    rolln = nlambda//2
    psi[4:-4,0,:] = factor*np.roll(psi[4:-4,7,:],rolln,axis=0)
    psi[4:-4,1,:] = factor*np.roll(psi[4:-4,6,:],rolln,axis=0)
    psi[4:-4,2,:] = factor*np.roll(psi[4:-4,5,:],rolln,axis=0)
    psi[4:-4,3,:] = factor*np.roll(psi[4:-4,4,:],rolln,axis=0)
    
    psi[4:-4,-1,:] = factor*np.roll(psi[4:-4,-8,:],rolln,axis=0)
    psi[4:-4,-2,:] = factor*np.roll(psi[4:-4,-7,:],rolln,axis=0)
    psi[4:-4,-3,:] = factor*np.roll(psi[4:-4,-6,:],rolln,axis=0)
    psi[4:-4,-4,:] = factor*np.roll(psi[4:-4,-5,:],rolln,axis=0)
    return psi

def update_periodic_BC_2D(psi):
    psi[0,:] = psi[-8,:]
    psi[1,:] = psi[-7,:]
    psi[2,:] = psi[-6,:]
    psi[3,:] = psi[-5,:]
    
    psi[-1,:] = psi[7,:]
    psi[-2,:] = psi[6,:]
    psi[-3,:] = psi[5,:]
    psi[-4,:] = psi[4,:]
    
    return psi

def update_polar_BC_2D(psi,nlambda,vector):
    if vector == True:
        factor=-1.0
    else:
        factor=1.0
    rolln = nlambda//2
    psi[4:-4,0] = factor*np.roll(psi[4:-4,7],rolln,axis=0)
    psi[4:-4,1] = factor*np.roll(psi[4:-4,6],rolln,axis=0)
    psi[4:-4,2] = factor*np.roll(psi[4:-4,5],rolln,axis=0)
    psi[4:-4,3] = factor*np.roll(psi[4:-4,4],rolln,axis=0)
    
    psi[4:-4,-1] = factor*np.roll(psi[4:-4,-8],rolln,axis=0)
    psi[4:-4,-2] = factor*np.roll(psi[4:-4,-7],rolln,axis=0)
    psi[4:-4,-3] = factor*np.roll(psi[4:-4,-6],rolln,axis=0)
    psi[4:-4,-4] = factor*np.roll(psi[4:-4,-5],rolln,axis=0)
    return psi

def update_prognostic_BCs(u_n,v_n,T_n,Ps_n,nlambda):
    u_n = update_periodic_BC(u_n)
    u_n = update_polar_BC(u_n,nlambda,vector=True)
    v_n = update_periodic_BC(v_n)
    v_n = update_polar_BC(v_n,nlambda,vector=True)
    T_n = update_periodic_BC(T_n)
    T_n = update_polar_BC(T_n,nlambda,vector=False)
    Ps_n = update_periodic_BC_2D(Ps_n)
    Ps_n = update_polar_BC_2D(Ps_n,nlambda,vector=False)
    return u_n,v_n,T_n,Ps_n

def update_diagnostic_BCs(z_n,zf_n,omega_n,omegas_n,us,vs,nlambda):
    z_n = update_periodic_BC(z_n)
    z_n = update_polar_BC(z_n,nlambda,vector=False)
    zf_n = update_periodic_BC(zf_n)
    zf_n = update_polar_BC(zf_n,nlambda,vector=False)
    omega_n = update_periodic_BC(omega_n)
    omega_n = update_polar_BC(omega_n,nlambda,vector=False)
    omegas_n = update_periodic_BC_2D(omegas_n)
    omegas_n = update_polar_BC_2D(omegas_n,nlambda,vector=False)
    us = update_periodic_BC_2D(us)
    us = update_polar_BC_2D(us,nlambda,vector=True)
    vs = update_periodic_BC_2D(vs)
    vs = update_polar_BC_2D(vs,nlambda,vector=True)
    return z_n,zf_n,omega_n,omegas_n
    

#output functions
def plotter_help(psi,func=1,flip=0):
    absmax = np.max(np.abs(psi))
    if np.min(psi) < 0.:
        cmap = 'coolwarm'
        vmin = -absmax
        vmax = absmax
    else:
        cmap = 'viridis'
        vmin = psi.min()
        vmax = psi.max()
        
    if psi.shape == (nphi,nP):
        x = lat;y=P
        plt.figure(figsize=[5,3])
    elif psi.shape == (nlambda,nphi):
        x = lon;y=lat
    else:
        x=np.arange(psi.shape[0]);y=np.arange(psi.shape[1])
    
    if func == 1:
        CM = plt.pcolormesh(x,y,psi.transpose(),vmin=vmin,vmax=vmax,cmap=cmap)
    elif func == 0:
        CM = plt.contourf(x,y,psi.transpose(),vmin=vmin,vmax=vmax,cmap=cmap)
        if x.shape[0] == nphi:
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(30.))
    elif func == 2:
        CM = plt.contourf(x,y,psi.transpose(),1000,vmin=vmin,vmax=vmax,cmap=cmap)
    if flip:
        plt.gca().invert_yaxis()
        
    plt.colorbar(CM)

def print_max_min_of_field(field,name):
    maxf = np.max(field)
    minf = np.min(field)
    #posmax = np.where(field == maxf)
    #posmin = np.where(field == minf)
    print('{}: Max = {}; Min = {}'.format(name,maxf,minf))
    
def threedarray2DataArray(arr):
    dummy = np.zeros([1,nlambda,nphi,nP])
    dummy[0,:,:,:] = arr[4:-4,4:-4,:]
    da = xr.DataArray(dummy,coords = [np.array([day]),lon,lat,P],dims = ['time','lon','lat','lev'])
    return da

def twodarray2DataArray(arr):
    dummy = np.zeros([1,nlambda,nphi])
    dummy[0,:,:] = arr[4:-4,4:-4]
    da = xr.DataArray(dummy,coords = [np.array([day]),lon,lat],dims = ['time','lon','lat'])
    return da

def write_restart_file(u_n,u_p,v_n,v_p,T_n,T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs,day):
    np.savez('output/restart_day{}'.format(day),u_n=u_n,u_p=u_p,v_n=v_n,v_p=v_p,T_n=T_n,T_p=T_p,Ps_n=Ps_n,Ps_p=Ps_p,omega_n=omega_n,z_n=z_n,theta_n=theta_n,omegas_n=omegas_n,theta_s=theta_s,us=us,vs=vs)

def read_restart_file(u_n,u_p,v_n,v_p,T_n,T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs,day):
    data = np.load('output/restart_day{}.npz'.format(day))
    u_n = data['u_n']
    u_p = data['u_p']
    v_n = data['v_n']
    v_p = data['v_p']
    T_n = data['T_n']
    T_p = data['T_p']
    Ps_n = data['Ps_n']
    Ps_p = data['Ps_p']
    omega_n = data['omega_n']
    z_n = data['z_n']
    theta_n = data['theta_n']
    omegas_n = data['omegas_n']
    theta_s = data['theta_s']
    us = data['us']
    vs = data['vs']
    return u_n,u_p,v_n,v_p,T_n,T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs

def add_fields_to_history(u_h,u_n,v_h,v_n,Ps_h,Ps_n,T_h,T_n,omega_h,omega_n
                          ,omegas_h,omegas_n,Q_h,Q_n,theta_h,theta_n):
    u_h += u_n
    v_h += v_n
    Ps_h += Ps_n
    T_h += T_n
    omega_h += omega_n
    omegas_h += omegas_n
    Q_h += Q_n
    theta_h += theta_n
#    if t % 86400 == 0:
#        u_h /= steps_per_day
#        v_h /= steps_per_day
#        T_h /= steps_per_day
#        Ps_h /= steps_per_day
#        omega_h /= steps_per_day
#        omegas_h /= steps_per_day
#        Q_h /= steps_per_day
#        theta_h /= steps_per_day
    return u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h

def write_output(u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h,u_n,u_p,v_n,v_p,T_n,
                 T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs,t,hour):
    print('writing history file hour {}'.format(hour))
    ds = xr.Dataset()
    ds['U'] = threedarray2DataArray(u_h)
    ds['V'] = threedarray2DataArray(v_h)
    ds['Ps'] = twodarray2DataArray(Ps_h)
    ds['T'] = threedarray2DataArray(T_h)
    ds['OMEGA'] = threedarray2DataArray(omega_h)
    ds['OMEGAS'] = twodarray2DataArray(omegas_h)
    ds['Q'] = threedarray2DataArray(Q_h)
    ds['THETA'] = threedarray2DataArray(theta_h)
    ds.to_netcdf('output/history_out_hour{}.nc'.format(hour),unlimited_dims = ['time'],engine='scipy')
    if t%864000 == 0:
        print('writing restart file day {}'.format(day))
        write_restart_file(u_n,u_p,v_n,v_p,T_n,T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs,day)
    del ds
    u_h[:] = 0;v_h[:] = 0;Ps_h[:] = 0;T_h[:] = 0;omega_h[:] = 0;omegas_h[:] = 0;
    Q_h[:] = 0;theta_h[:] = 0;

def print_diagnostics(u_n,v_n,T_n,Ps_n,omega_n,omegas_n,z_n,n,t,day):
    if n%100 == 0:
        print('\nValues for time step {}, time {}, day {}'.format(n,t,day))
        print('\ntau = {}'.format(tau))
        print_max_min_of_field(u_n,'U')
        print_max_min_of_field(v_n,'V')
        print_max_min_of_field(T_n,'T')
        print_max_min_of_field(Ps_n,'Ps')
        print('Ps.mean: {}'.format(np.average(Ps_n[4:-4,4:-4],axis=1,weights=cosphi[4:-4]).mean()))
        print('global TE: {}'.format(diag_total_energy(gTE_n,TE_n,u_n,v_n,T_n)[0]))
    #    Ps_mean.append(np.average(Ps_n[:,1:nphi-1],axis=1,weights=np.cos(phi[1:nphi-1])).mean())
        print_max_min_of_field(omega_n,'omega')
        print_max_min_of_field(omegas_n,'omegas')
        print_max_min_of_field(z_n[4:-4,4:-4,19],'z_l')


### Main
if __name__ == '__main__':
    start = time.time()
    start_day = time.time()
    #define constants
    pi = np.pi
    nphi = 61#72 #36
    nlambda = 120#144 #72    
    nP = 20 
    g = 9.81
    cp = 1004.
    kappa = 2./7
    R = kappa*cp
    a = 6.371e+6
    OMEGA = 7.292e-5
    kf = 1./86400.
    ka = 1/40.*kf
    ks = 1/4.*kf
    P0 = 100000. #Pa
    kv_surf_w = 24
    cv = 0.01
    sigmab = 0.7
    dlambda = 2*pi/nlambda
    dlambda_deg = np.rad2deg(dlambda)
    dphi = pi/nphi
    dphi_deg = np.rad2deg(dphi)#2.5 #5.
    dt = 300 #seconds
    dt_phys = dt*6 #seconds
    #tstop = 86400*100
    tstop = 25*86400
    t = 0
    steps_per_day = 86400/4.
    #define fields
    phi = np.array([-((pi/2)-dphi/2)+(j)*dphi for j in range(nphi)])
    cosphi = np.zeros([4+nphi+4])
    sinphi = np.zeros([4+nphi+4])
    tanphi = np.zeros([4+nphi+4])
    cosphi[4:-4] = np.cos(phi)
    cosphi[0] = -1.*cosphi[7]
    cosphi[1] = -1.*cosphi[6]
    cosphi[2] = -1.*cosphi[5]
    cosphi[3] = -1.*cosphi[4]
    cosphi[-1] = -1.*cosphi[-8]
    cosphi[-2] = -1.*cosphi[-7]
    cosphi[-3] = -1.*cosphi[-6]
    cosphi[-4] = -1.*cosphi[-5]
    tanphi[4:-4] = np.tan(phi)
    sinphi[4:-4] = np.sin(phi)
    fcor = 2*OMEGA*np.sin(phi)
    lambd = (np.array([-(pi)+(i)*dlambda for i in range(nlambda)]))
    Pf = np.linspace(1,900,nP)*100 #Pa
    dP = np.zeros([nP])+(Pf[2]-Pf[1]) #Pa
    P = Pf+dP/2
    revnP = reversed(range(nP))
    dP_stag = dP.copy() #Pa
    Psponge = 14300
    lon = np.rad2deg(lambd)
    lat = np.rad2deg(phi)
    lnP = np.log(Pf)
    #tau = 3000./dt
    tau = 10.
    init = True
    restart = False
    rest_day = 90
    out = 0
    coeffs = calculate_polar_filter_coefficients()
    if init:
        #prognostic fields
        Ps_f = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        Ps_n = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        Ps_p = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        Ps_h = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        
        u_f = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        u_n = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        u_p = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        u_h = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        u_mean = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        v_f = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        v_n = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        v_p = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        v_h = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        v_mean = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        T_f = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        T_n = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        T_p = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        T_h = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        T_mean = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        #diagnostic fields
        z_n = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        zf_n = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        z_h = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        omega_n = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        omega_p = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        omega_h = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        omegas_n = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        omegas_p = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        omegas_h = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        
        Q_n = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        Q_p = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        Q_h = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        theta_n = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        theta_p = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        theta_s = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        theta_h = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        Fl = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        Fphi = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        sFl = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        sFphi = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        vort = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        div = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        stream = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        pot = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        TE_p = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        TE_n = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        gTE_n = 0.0
        gTE_p = 0.0
        beta = 0.0
        
        
        #surface fields
        Ts = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        us = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        vs = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        A = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        Vs = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        
        #helper fields
        dlnP = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        T_m = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        kv = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        kT = np.zeros(shape=np.array([4+nlambda+4,4+nphi+4,nP]))
        skv = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        Ps_mean = []
        dummy_psi = np.zeros([nlambda,nphi,nP+8])
        zeros = np.zeros(shape=[nlambda,nphi,nP])
        
        
        #tendency fields
        tend1 = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        tend2 = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        tend3 = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        tend4 = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        tend5 = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        tend6 = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        
        #Equilibrium temperature field
        T_eq = np.zeros(shape=[4+nlambda+4,4+nphi+4,nP])
        T_eqs = np.zeros(shape=[4+nlambda+4,4+nphi+4])
        #initialize
        Ps_n[:,:] = 100000.
        Ps_p[:,:] = 100000.
        T_eq,T_eqs = H_S_equilibrium_temperature(T_eq,T_eqs,Ps_n)
        T_n = T_eq + np.random.random(size=[nlambda+8,nphi+8,nP])/1e2 #Random perturbations to break hemispheric and zonal symmetry
        T_p = T_eq
        Ts = T_eqs
        #T_m = helper_Tm(T_m,T_n,Ts)
        theta_n,theta_s = diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n)
        lnPs = np.log(Ps_n)
        dlnp = helper_dlnP(dlnP,lnPs,Pf)
        #sys.exit()
        zf_n,z_n = diag_z(zf_n,z_n,T_n,Ps_n,dlnP)
        z_n,zf_n,omega_n,omegas_n = update_diagnostic_BCs(z_n.copy(),zf_n.copy(),omega_n.copy(),omegas_n.copy(),us,vs,nlambda)
        n = 0
        umean_n = 0
        day = 1
        hour = 0
        if restart:
            u_n,u_p,v_n,v_p,T_n,T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs = read_restart_file(u_n,u_p,v_n,v_p,T_n,T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs,rest_day)
            day = rest_day
            us[:] = 0.0
            vs[:] = 0.0
        gTE_p = diag_total_energy(gTE_p,TE_p,u_n,v_n,T_n)[0]
    #time stepping loop
#    sys.exit()
    while t<tstop:
        n += 1
        #print('Begin timestep: {}'.format(n))
        
 
        u_f = prognostic_u(u_f,u_n,u_p,v_n,us,omega_n,z_n,Ps_n,omegas_n)
        v_f = prognostic_v(v_f,v_n,v_p,u_n,vs,omega_n,z_n,Ps_n,omegas_n)
        T_f = prognostic_T(T_f,T_n,T_p,u_n,v_n,omega_n,theta_n,Ps_n,theta_s,omegas_n)
        Ps_f,lnPs = prognostic_Ps(Ps_f,Ps_n,Ps_p,omegas_n,us,vs,lnPs)

    
        u_f,v_f,T_f,Ps_f = update_prognostic_BCs(u_f.copy(),v_f.copy(),T_f.copy(),Ps_f.copy(),nlambda)
        
        tau = update_tau(u_f,v_f,tau)
        
        u_f = fourth_order_shapiro_filter(u_f,True,tau)
        v_f = fourth_order_shapiro_filter(v_f,True,tau)
        T_f = fourth_order_shapiro_filter(T_f,False,tau)
        Ps_f = fourth_order_shapiro_filter_2D(Ps_f)
        
        u_f,v_f,T_f,Ps_f = update_prognostic_BCs(u_f.copy(),v_f.copy(),T_f.copy(),Ps_f.copy(),nlambda)

        #polar filter
        u_f = new_polar_filter(u_f,coeffs)
        v_f = new_polar_filter(v_f,coeffs)
        T_f = new_polar_filter(T_f,coeffs)
        Ps_f = new_polar_filter_2d(Ps_f,coeffs)
#        u_f = polar_fourier_filter(u_f)
#        v_f = polar_fourier_filter(v_f)
#        T_f = polar_fourier_filter(T_f)    
#        Ps_f = polar_fourier_filter_2D(Ps_f)
        
#        Apply Robert-Asselin filter to prognostic variables at time n
        u_n = Robert_Asselin_filter(u_f,u_n,u_p)
        v_n = Robert_Asselin_filter(v_f,v_n,v_p)
        T_n = Robert_Asselin_filter(T_f,T_n,T_p)
        Ps_n = Robert_Asselin_filter(Ps_f,Ps_n,Ps_p)
        #print('Finished prognostic equations')
        t = t+dt
        #print('Advance time to: t={}'.format(t))
        #Apply mass and energy fixers
        Ps_f = mass_fixer(Ps_f,P0)
        lnPs[4:-4,4:-4] = np.log(Ps_f[4:-4,4:-4])
        if t%(dt_phys) == 0:
            gTE_n = diag_total_energy(gTE_n,TE_n,u_f,v_f,T_f)[0]
            T_f,beta = energy_fixer2(gTE_p,gTE_n,T_f,beta)
            gTE_n = diag_total_energy(gTE_n,TE_n,u_f,v_f,T_f)[0]
            # print(beta,gTE_n/gTE_p)
        u_p = u_n.copy()
        u_n = u_f.copy()    
        v_p = v_n.copy()
        v_n = v_f.copy()
        T_p = T_n.copy()
        T_n = T_f.copy()
        Ps_p = Ps_n.copy()
        Ps_n = Ps_f.copy()
        u_n,v_n,T_n,Ps_n = update_prognostic_BCs(u_n.copy(),v_n.copy(),T_n.copy(),Ps_n.copy(),nlambda)
        #print('Diagnostic equations')
#        omega_n = diag_omega(omega_n,u_n,v_n,cosphi,dP)
        div = diag_divergence(div,u_n,v_n)
        omega_n = diag_omega2(omega_n,div)
        zf_n,z_n = diag_z(zf_n,z_n,T_n,Ps_n,dlnP)
        zf_n,z_n = z_limiter(z_n,zf_n)
        theta_n,theta_s = diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n)
        #us,vs,Vs,A = diag_surface_wind(us,vs,Vs,u_n,v_n,Ps_p,A)
        z_n,zf_n,omega_n,omegas_n = update_diagnostic_BCs(z_n.copy(),zf_n.copy(),omega_n.copy(),omegas_n.copy(),us,vs,nlambda)
#        omegas_n = diag_omegas(omegas_n,omega_n,u_n,v_n,us,vs)
        omegas_n = diag_omegas4(omegas_n,omega_n,u_n,v_n,Ps_n,us,vs)
#        omegas_n = diag_omegas2(omegas_n,omega_n,div)
        z_n,zf_n,omega_n,omegas_n = update_diagnostic_BCs(z_n.copy(),zf_n.copy(),omega_n.copy(),omegas_n.copy(),us,vs,nlambda)
        
        # Time split physics
        # Call physics adjustments if physics time step has been completed
        if t%dt_phys == 0:
#            sys.exit()
            Q_n = diag_Q(Q_n,T_n,Ps_n,T_eq,kT)
            Fl,Fphi = H_S_friction(Fl,Fphi,u_n,v_n,Ps_n,kv,sigmab)
            u_n,v_n,T_n = update_uvT_with_physics(u_n,v_n,T_n,Fl,Fphi,Q_n,dt_phys)
            #recompute z
            zf_n,z_n = diag_z(zf_n,z_n,T_n,Ps_n,dlnP)
            zf_n,z_n = z_limiter(z_n,zf_n)
            #Update boundary conditions
            u_n,v_n,T_n,Ps_n = update_prognostic_BCs(u_n.copy(),v_n.copy(),T_n.copy(),Ps_n.copy(),nlambda)
            z_n,zf_n,omega_n,omegas_n = update_diagnostic_BCs(z_n.copy(),zf_n.copy(),omega_n.copy(),omegas_n.copy(),us,vs,nlambda)
            #Define new energy level to be adiabatically conserved
            gTE_p = diag_total_energy(gTE_p,TE_p,u_n,v_n,T_n)[0]
            
        print_diagnostics(u_n,v_n,T_n,Ps_n,omega_n,omegas_n,z_n,n,t,day)

    
        
    #handle output
        
        if t%21600 == 0:
            hour += 6
            u_mean += u_n
            v_mean += v_n
            T_mean += T_n
            umean_n += 1
            u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h = (add_fields_to_history(u_h
                                                   ,u_n,v_h,v_n,Ps_h,Ps_n,T_h,T_n
                                                   ,omega_h,omega_n,omegas_h
                                                   ,omegas_n,Q_h,Q_n,theta_h,theta_n))
            try:
                t_out.join()
            except:
                1+1
            t_out = Thread(target=write_output,args=(u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h,u_n,u_p,v_n,v_p,T_n,
                 T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs,t,hour))
            t_out.start()
#            print(u_h.max())
#            out+=1
        if t % 86400 == 0:
            
#            write_output(u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h,u_n,u_p,v_n,v_p,T_n,
#                 T_p,Ps_n,Ps_p,omega_n,z_n,theta_n,omegas_n,theta_s,us,vs,t,day)
            plotter_help(np.mean(u_n[4:-4,4:-4,:],axis=0),0,1)
            plt.show()
            day += 1
            end_day = time.time()
            print('This day took {}'.format(end_day-start_day))
            start_day = time.time()
            
    #handle errors and exceptions
    
    end = time.time()
    print(end - start)
