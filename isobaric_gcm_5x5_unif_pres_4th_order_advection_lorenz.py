#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:05:48 2018

@author: Hans Brenna
@license: MIT
This program implements a hydrostatic primitive equations GCM on an isobaric 
vertical coordinate. The numerics will be 2nd order centered differences in 
space and time on a uniform lat, lon horizontal grid. Subgrid scale diffusion 
is neglected and energy will be removed from the grid scale by a 4th order
Shapiro filter in both horizontal directions. Forcing will be adapted from 
Held & Suarez 1994 with newtonian relaxation to radiative equilibrium and 
Reyleigh damping of the horizontal velocities in the lower layers. To control 
the 2dt computational mode I will apply a Robert-Asselin filter to the 
prognostic variables.

Prognostic varriables: u,v,T,Ps
Diagnostic variables: z,omega,omega_s,Q,theta,Fl,Fphi

A speedup of the code was acheived using the numba just-in-time compiler @jit
on most of the functions doing for loops.

This early version uses a low time step (~50-100 s) to ensure linear stability. I will add
polar filtering using fft to control the short waves close to the poles to 
increase the time step to ~300 s

Current issues: Does not conserve total mass. drift is small ~0.1% over 10 days
considering potential fixes

adding 4th order advection of velocity

I believe lorenz grid will be needed to avoid explicit vertical mixing, which I think 
destroys the solution slightly. This time without helper fields. More transparent that
way (for me)

Need to change vertical advection of u,v,T, and algorithms for omega, omegas, us?, vs? and z.
"""
from __future__ import print_function
import sys
from numba import jit
import numpy as np
import pdb
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import time
#import timing

np.seterr(all='warn')
np.seterr(divide='raise',invalid='raise') #Set numpy to raise exceptions on 
#invalid operations and divide by zero



@jit
def prognostic_u(u_f,u_n,u_p,v_n,us,omega_n,z_n,Ps_n,omegas_n,Fl):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        for j in range(1,nphi-1):
            for k in range(nP):
                #advection of u in u-direction
                adv_uu = ((u_n[i,j,k]/(a*cosphi[j]))*((u_n[i-2,j,k]+8*(-u_n[i-1,j,k]+u_n[i+1,j,k])
                      -u_n[i+2,j,k])/(12*dlambda)))
                #advection of u in v-direction
                if j in (1,nphi-2):
                    adv_vu = (v_n[i,j,k]/(a)*(u_n[i,j+1,k]
                           -u_n[i,j-1,k])/(2*dphi))
                else:
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
                cor_u = 2*OMEGA*np.sin(phi[j])*v_n[i,j,k]
                #gradient of geopotential height
                gdz_u = ((g)/(a*np.cos(phi[j]))*(z_n[i+1,j,k]-z_n[i-1,j,k])
                      /(2*dlambda))
                #curvature term
                curv_u = ((u_n[i,j,k]*v_n[i,j,k]*np.tan(phi[j]))/a)
                
                u_f[i,j,k] = (u_p[i,j,k]+2*dt*(-(adv_uu+adv_vu+adv_omegau)+cor_u
                           -gdz_u+curv_u-Fl[i,j,k]))
    #handle poles            
    u_f[:,0,:] = 0.0
    u_f[:,nphi-1,:] = 0.0
    
    #alternative upper BC
    #u_f[:,:,0] = 0.0
                
    return u_f

@jit
def prognostic_v(v_f,v_n,v_p,u_n,vs,omega_n,z_n,Ps_n,omegas_n,Fphi):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        for j in range(1,nphi-1):
            for k in range(nP):
                #advection of v in u-direction
                adv_uv = ((u_n[i,j,k]/(a*np.cos(phi[j])))*((v_n[i-2,j,k]+8*(-v_n[i-1,j,k]+v_n[i+1,j,k])
                      -v_n[i+2,j,k])/(12*dlambda)))
                #advection of v in v-direction
                if j in (1,nphi-2):
                    adv_vv = (((v_n[i,j,k]/(a))*((v_n[i,j+1,k]
                           -v_n[i,j-1,k])/(2*dphi))))
                else:
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
                        
                cor_v = 2*OMEGA*np.sin(phi[j])*u_n[i,j,k]
                gdz_v = g/a*(z_n[i,j+1,k]-z_n[i,j-1,k])/(2*dphi)
                curv_v = (u_n[i,j,k]*u_n[i,j,k]*np.tan(phi[j]))/a
                
                v_f[i,j,k] = (v_p[i,j,k]+2*dt*(-(adv_uv+adv_vv+adv_omegav)-cor_v
                   -gdz_v-curv_v-Fphi[i,j,k]))
    
    #handle poles            
    v_f[:,0,:] = 0.0
    v_f[:,nphi-1,:] = 0.0
    
    #alternative upper BC
    #v_f[:,:,0] = 0.0
             
    return v_f

@jit
def prognostic_T(T_f,T_n,T_p,u_n,v_n,omega_n,theta_n,Ps_n,theta_s,omegas_n,Q_n):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                adv_uT = ((u_n[i,j,k]/(a*np.cos(phi[j])))*((T_n[i+1,j,k]
                       -T_n[i-1,j,k])/(2*dlambda)))
                adv_vT = (v_n[i,j,k]/(a))*((T_n[i,j+1,k]-T_n[i,j-1,k])/(2*dphi))
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
                        
                T_f[i,j,k] = (T_p[i,j,k]+2*dt*(-(adv_uT+adv_vT+adv_omegaT)+Q_n[i,j,k]))
    
    #pole handling
    for k in range(nP):
        T_f[:,0,k] = T_f[:,1,k].mean()
        T_f[:,nphi-1,k] = T_f[:,nphi-2,k].mean()
    return T_f

@jit
def prognostic_Ps(Ps_f,Ps_n,Ps_p,omegas_n,us,vs,lnPs):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            adv_P = ((us[i,j])/(a*np.cos(phi[j]))*(Ps_n[i+1,j]-Ps_n[i-1,j])
                  /(2*dlambda)+(vs[i,j]/(a))*(Ps_n[i,j+1]-Ps_n[i,j-1])/(2*dphi))
            Ps_f[i,j] = Ps_p[i,j]+2*dt*(-adv_P+omegas_n[i,j])
    #pole handling
    Ps_f[:,0] = Ps_f[:,1].mean()
    Ps_f[:,nphi-1] = Ps_f[:,nphi-2].mean()
    lnPs = np.log(Ps_f)
#    
#    if Ps_f.min() < 95500.:
#        Ps_f[np.where(Ps_f < 95500.)] = 95500.
    return Ps_f,lnPs

@jit
def diag_omega(omega_n,u_n,v_n):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
             for k in range(nP):
                 int_div_u = 0.0
                 int_div_v = 0.0
                 if k == 0:
                     omega_n[i,j,k] = 0.0
                 else:
                     for m in range(0,k):
                         dP_m = dP[0]
                         int_div_u += (u_n[i+1,j,m]-u_n[i-1,j,m])/(2*dlambda)*dP_m
                         int_div_v += (cosphi[j+1]*v_n[i,j+1,m]-cosphi[j-1]*v_n[i,j-1,m])/(2*dphi)*dP_m
                     omega_n[i,j,k] = -(1./(a*np.cos(phi[j]))*int_div_u+1./(a*cosphi[j])*int_div_v)
    return omega_n

@jit
def diag_omegas(omegas_n,omega_n,u_n,v_n,us,vs):
    ub = 0.5*(u_n[:,:,nP-1]+us)
    vb = 0.5*(v_n[:,:,nP-1]+vs)
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            int_div_u = ((u_n[i+1,j,nP-1]-u_n[i-1,j,nP-1])/(2*dlambda)*dP[0]/2. 
                      + (ub[i+1,j]-ub[i-1,j])/(2*dlambda)*(Ps_n[i,j]-P[nP-1]))
            int_div_v = ((cosphi[j+1]*v_n[i,j+1,nP-1]-cosphi[j-1]*v_n[i,j-1,nP-1])/(2*dphi)*dP[0]/2. 
                      + (cosphi[j+1]*vb[i,j+1]-cosphi[j-1]*vb[i,j-1])/(2*dphi)*(Ps_n[i,j]-P[nP-1]))
            omegas_n[i,j] = (omega_n[i,j,nP-1] + (-(1./(a*np.cos(phi[j]))*int_div_u+1./(a*cosphi[j])*int_div_v)))
#            omegas_n[i,j] = (omega_n[i,j,nP-1]+(-(1./(a*np.cos(phi[j])))
#                          *(ub[i+1,j]-ub[i-1,j])/(2*dlambda)+(1./(a*cosphi[j]))*(cosphi[j+1]*vb[i,j+1]
#                          -cosphi[j-1]*vb[i,j-1])/(2*dphi))*(Ps_n[i,j]-Pf[nP-1]))
    return omegas_n

@jit
def diag_z(zf_n,z_n,T_n,Ps_n,dlnP):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP-1,-1,-1):
                if k == nP-1:
                    zf_n[i,j,k] = 0.0+R*T_n[i,j,k]*np.log(Ps_n[i,j]/Pf[k])
                    z_n[i,j,k] = 0.0+R*T_n[i,j,k]*np.log(Ps_n[i,j]/P[k])
                else:
                    zf_n[i,j,k] = zf_n[i,j,k+1]+R*T_n[i,j,k]*np.log(Pf[k+1]/Pf[k])
                    z_n[i,j,k] = zf_n[i,j,k+1]+R*T_n[i,j,k]*np.log(Pf[k+1]/P[k])
    
    zf_n = zf_n/g
    z_n = z_n/g
    #handle poles:
    for k in range(nP):
        zf_n[:,0,k] = zf_n[:,1,k].mean()
        zf_n[:,nphi-1,k] = zf_n[:,nphi-2,k].mean()
        z_n[:,0,k] = z_n[:,1,k].mean()
        z_n[:,nphi-1,k] = z_n[:,nphi-2,k].mean()
        
    if z_n.min() < 0.0:
        print('Negative values of z_n. Layer are crossing. Exiting...')
        sys.exit()
    return zf_n,z_n

#@jit
#def diag_z(zf_n,z_n,T_n,dlnP):
#    alpha = np.zeros([nlambda,nphi,nP])
#    for i in range(nlambda):
#        if i == nlambda-1:
#            i=-1
#        for j in range(1,nphi-1):
#            for k in range(nP):
#                int_Tdlnp = 0.0
#                for m in range(k,nP):
#                    int_Tdlnp += T_n[i,j,m]*dlnP[i,j,m] 
#                zf_n[i,j,k] = (R/g)*int_Tdlnp
#                
#    #handle poles:
#    for k in range(nP):
#        zf_n[:,0,k] = zf_n[:,1,k].mean()
#        zf_n[:,nphi-1,k] = zf_n[:,nphi-2,k].mean()
#    
##    if z_n.min() < 100.:
##        z_n[np.where(z_n < 100.)] = 100.
#    if z_n.min() < 0.0:
#        print('Negative values of z_n. Layer are crossing. Exiting...')
#        sys.exit()
#    return zf_n,z_n

@jit
def helper_Tm(T_m,T_n,Ts):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP-1):
                T_m[i,j,k] = (T_n[i,j,k+1]+T_n[i,j,k])*0.5
            T_m[i,j,nP-1] = (Ts[i,j]+T_n[i,j,nP-1])*0.5
    return T_m

@jit
def helper_dlnP(dlnP,lnPs,Pf):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP-1):
                dlnP[i,j,k] = lnP[k+1]-lnP[k]
            dlnP[i,j,nP-1] = lnPs[i,j] - lnP[nP-1]
    return dlnP

@jit
def diag_Q(Q_n,T_n,Ps_n,T_eq):
    sigmab = 0.7
    kT = np.zeros(shape=[nlambda,nphi,nP])
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                function = ((P[k]/Ps_n[i,j])-sigmab)/(1.-sigmab)
                if function > 0.:
                    kT[i,j,k] = ka+(ks-ka)*function*np.cos(phi[j])**4
                else:
                    kT[i,j,k] = ka
                    #kT[i,j,k] = ka+(ks-ka)*np.max([0,function])*np.cos(phi[j])**4
                
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                Q_n[i,j,k] = -kT[i,j,k]*(T_n[i,j,k]-T_eq[i,j,k])
    return Q_n

@jit
def diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                theta_n[i,j,k] = T_n[i,j,k]*(P0/P[k])**(R/cp)  
        
        for i in range(nlambda):
            if i == nlambda-1:
                i=-1
            for j in range(1,nphi-1):
                theta_s[i,j] = Ts[i,j]*(P0/Ps_n[i,j])**(R/cp)
        #handle poles:
    for k in range(nP):
        theta_n[:,0,k] = theta_n[:,1,k].mean()
        theta_n[:,nphi-1,k] = theta_n[:,nphi-2,k].mean()
    theta_s[:,0] = theta_s[:,1].mean()
    theta_s[:,nphi-1] = theta_s[:,nphi-2].mean()
    return theta_n,theta_s

#surface functions
@jit
def diag_surface_wind(us,vs,Vs,u_n,v_n,Ps_p,A):

    A = (kv_surf_w/(kv_surf_w+cv*Vs*(Ps_p-P[nP-1]))) #Vs and Ps used at previous time step
    us = A*u_n[:,:,nP-1]
    vs = A*v_n[:,:,nP-1]
    Vs = np.sqrt(us*us+vs*vs)
    return us,vs,Vs 
    

def H_S_equilibrium_temperature(T_eq,T_eqs,Ps_n):
    dTy = 60 #K
    dthtz = 10 #K
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                HS_T_func = ((315-dTy*np.sin(phi[j])*np.sin(phi[j])-dthtz
                             *np.log(P[k]/P0)*(np.cos(phi[j]))**2)*(P[k]/P0)**kappa)
                T_eq[i,j,k] = np.max([200.,HS_T_func])
    #handle poles
    for k in range(nP):
        T_eq[:,0,k] = T_eq[:,1,k].mean()
        T_eq[:,nphi-1,k] = T_eq[:,nphi-2,k].mean()
        
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            HS_T_func = ((315-dTy*np.sin(phi[j])*np.sin(phi[j])-dthtz
                        *np.log(Ps_n[i,j]/P0)*(np.cos(phi[j]))**2)*(Ps_n[i,j]/P0)**kappa)
            T_eqs[i,j] = np.max([200.,HS_T_func])
    #handle poles
    T_eqs[:,0] = T_eqs[:,1].mean()
    T_eqs[:,nphi-1] = T_eqs[:,nphi-2].mean()
    return T_eq, T_eqs

@jit(nopython=True)
def H_S_friction(Fl,Fphi,u_n,v_n,Ps_n,kv,sigmab):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                function = ((P[k]/Ps_n[i,j])-sigmab)/(1.-sigmab)
                if function > 0:
                    kv[i,j,k] = kf*function
                else:
                    kv[i,j,k] = 0
    
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                Fl[i,j,k] = kv[i,j,k]*u_n[i,j,k]
                Fphi[i,j,k] = kv[i,j,k]*v_n[i,j,k] 
    return Fl,Fphi

@jit
def sponge_layer(sFl,sFphi,skv,u_n,v_n):
    for k in range(nP):
        if P[k] < Psponge:
            skv[:,:,k] = kf*((Psponge-P[k])/Psponge)**2
        else:
            skv[:,:,k] = 0.0
        
    sFl = skv*(u_n)
    #sFphi = skv*(v_n-v_n.mean(axis=0))
    sFphi = skv*(v_n)
    return sFl,sFphi

#filter functions
@jit
def polar_fourier_filter(psi):
    #filter zonal wavenumbers in polar regions
    #south pole
    psi_trans = np.fft.rfft(psi,axis=0)
    psi_trans[5:,1,:] = 0.0
    psi_trans[8:,2,:] = 0.0
    psi_trans[10:,3,:] = 0.0
    psi_trans[12:,4,:] = 0.0
    psi_trans[15:,5,:] = 0.0
    psi_trans[18:,6,:] = 0.0
    psi_trans[20:,7,:] = 0.0
    #north pole
    psi_trans[5:,nphi-2,:] = 0.0
    psi_trans[8:,nphi-3,:] = 0.0
    psi_trans[10:,nphi-4,:] = 0.0
    psi_trans[12:,nphi-5,:] = 0.0
    psi_trans[15:,nphi-6,:] = 0.0
    psi_trans[18:,nphi-7,:] = 0.0
    psi_trans[20:,nphi-8,:] = 0.0
    psi = np.fft.irfft(psi_trans,axis=0)
    return psi


def filter_4dx_components(psi):
    #filter meridional components
    psi_trans = np.fft.rfft(psi,axis=1)
    if len(psi.shape) == 2:
        psi_trans[:,11:] = 0.0+0.0j
    else:    
        psi_trans[:,11:,:] = 0.0+0.0j

    psi = np.fft.irfft(psi_trans,axis=1)
    psi_complex = psi.copy()
    psi = psi.real
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


@jit
def first_order_shapiro_filter_2D(psi):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            psi_filtered_lambda[i,j] = (0.25*(psi[i-1,j]+2*psi[i,j]+psi[i+1,j]))
                
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            psi_filtered[i,j] = (0.25*(psi_filtered_lambda[i,j-1]+2
                        *psi_filtered_lambda[i,j]+psi_filtered_lambda[i,j+1]))
    return psi_filtered


@jit
def second_order_shapiro_filter(psi):
    order = 2
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    #2nd order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered_lambda[i,j,k] = ((1./16.)*(-psi[i-2,j,k]+4*psi[i-1,j,k]
                               +10*psi[i,j,k]+4*psi[i+1,j,k]-psi[i+2,j,k])) 
           
    #2nd order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            for k in range(nP):  
                if order == 2:
                    psi_filtered[i,j,k] = ((1./16.)*(-psi_filtered_lambda[i,j-2,k]+4*psi_filtered_lambda[i,j-1,k]
                               +10*psi_filtered_lambda[i,j,k]+4*psi_filtered_lambda[i,j+1,k]-psi_filtered_lambda[i,j+2,k])) 
                elif order == 1:
                    psi_filtered[i,j,k] = ((0.25)*(psi_filtered_lambda[i,j-1,k]
                    +2*psi_filtered_lambda[i,j,k]+psi_filtered_lambda[i,j+1,k]))
    return psi_filtered

@jit
def alternative_second_order_shapiro_filter(psi,tau):
    order = 2
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    #2nd order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered_lambda[i,j,k] = psi[i,j,k] + tau*((1./16.)*(-psi[i-2,j,k]+4*psi[i-1,j,k]
                               -6*psi[i,j,k]+4*psi[i+1,j,k]-psi[i+2,j,k])) 
           
    #2nd order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            for k in range(nP):  
                if order == 2:
                    psi_filtered[i,j,k] = psi_filtered_lambda[i,j,k] + tau*((1./16.)*(-psi_filtered_lambda[i,j-2,k]+4*psi_filtered_lambda[i,j-1,k]
                               -6*psi_filtered_lambda[i,j,k]+4*psi_filtered_lambda[i,j+1,k]-psi_filtered_lambda[i,j+2,k])) 
                elif order == 1:
                    psi_filtered[i,j,k] = psi_filtered_lambda[i,j,k] + tau*((0.25)*(psi_filtered_lambda[i,j-1,k]
                    -2*psi_filtered_lambda[i,j,k]+psi_filtered_lambda[i,j+1,k]))
    return psi_filtered

@jit
def second_order_shapiro_filter_2D(psi):
    order = 2
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    #2nd order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        for j in range(1,nphi-1):
            psi_filtered_lambda[i,j] = ((1./16.)*(-psi[i-2,j]+4*psi[i-1,j]
                               +10*psi[i,j]+4*psi[i+1,j]-psi[i+2,j])) 
           
    #2nd order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            if order == 2:
                psi_filtered[i,j] = ((1./16.)*(-psi_filtered_lambda[i,j-2]+4*psi_filtered_lambda[i,j-1]
                           +10*psi_filtered_lambda[i,j]+4*psi_filtered_lambda[i,j+1]-psi_filtered_lambda[i,j+2])) 
            elif order == 1:
                psi_filtered[i,j] = ((0.25)*(psi_filtered_lambda[i,j-1]
                +2*psi_filtered_lambda[i,j]+psi_filtered_lambda[i,j+1]))
    return psi_filtered
                
@jit
def fourth_order_shapiro_filter(psi):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    tau = np.zeros([nP])+36.
    tau[-4:] = 36.
    #4th order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        elif i == nlambda-3:
            i=-3
        elif i == nlambda-4:
            i=-4
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered_lambda[i,j,k] = ((1./256)*(186.*psi[i,j,k]+56.*(psi[i-1,j,k]
                           +psi[i+1,j,k])-28*(psi[i-2,j,k]+psi[i+2,j,k])
                           +8*(psi[i-3,j,k]+psi[i+3,j,k])-(psi[i-4,j,k]+psi[i+4,j,k])))
           
    #4th order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            elif j in (nphi-3,nphi-4,2,3):
                order = 2
            else:
                order = 4
            for k in range(nP):  
                if order == 4:
                    psi_filtered[i,j,k] = ((1./256.)*(186.*psi_filtered_lambda[i,j,k]+56.*(psi_filtered_lambda[i,j-1,k]
                               +psi_filtered_lambda[i,j+1,k])-28*(psi_filtered_lambda[i,j-2,k]+psi_filtered_lambda[i,j+2,k])
                               +8*(psi_filtered_lambda[i,j-3,k]+psi_filtered_lambda[i,j+3,k])-(psi_filtered_lambda[i,j-4,k]+psi_filtered_lambda[i,j+4,k])))
                elif order == 2:
                    psi_filtered[i,j,k] = ((1./16.)*(-psi_filtered_lambda[i,j-2,k]+4*psi_filtered_lambda[i,j-1,k]
                               +10*psi_filtered_lambda[i,j,k]+4*psi_filtered_lambda[i,j+1,k]-psi_filtered_lambda[i,j+2,k])) 
                elif order == 1:
                    psi_filtered[i,j,k] = ((0.25)*(psi_filtered_lambda[i,j-1,k]+2*psi_filtered_lambda[i,j,k]+psi_filtered_lambda[i,j+1,k]))
    
    subtractive_filter = psi-psi_filtered
    psi_filtered_2 = psi-1/tau*subtractive_filter
    
    return psi_filtered_2

@jit
def filter_for_vertical_mixing(psi,dummy_psi,psi_s,psi_t,tau):
    psi_filtered = psi.copy()
#    tau =16.
    dummy_psi[:,:,4:4+nP] = psi.copy()
    dummy_psi[:,:,-1] = psi_s
    dummy_psi[:,:,-2] = psi_s
    dummy_psi[:,:,-3] = psi_s
    dummy_psi[:,:,-4] = psi_s
    dummy_psi[:,:,0] = psi_t[:,:,0]
    dummy_psi[:,:,1] = psi_t[:,:,0]
    dummy_psi[:,:,2] = psi_t[:,:,0]
    dummy_psi[:,:,3] = psi_t[:,:,0]
    
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(4,nP+4):  
                psi_filtered[i,j,k-4] = ((1./256.)*(186.*dummy_psi[i,j,k]+56.*(dummy_psi[i,j,k-1]
                           +dummy_psi[i,j,k+1])-28*(dummy_psi[i,j,k-2]+dummy_psi[i,j,k+2])
                           +8*(dummy_psi[i,j,k-3]+dummy_psi[i,j,k+3])-(dummy_psi[i,j,k-4]+dummy_psi[i,j,k+4])))
                    
    subtractive_filter = psi-psi_filtered
    psi_filtered_2 = psi-1./tau*subtractive_filter
    
    return psi_filtered_2
#def filter_for_vertical_mixing(psi):
#    psi_filtered = psi.copy()
#    for i in range(nlambda):
#        if i == nlambda-1:
#            i=-1
#        for j in range(1,nphi-1):
#            for k in range(1,nP-1):  
#                if k in (nP-2,1):
#                    order = 1
#                elif k in (nP-3,nP-4,2,3):
#                    order = 2
#                else:
#                    order = 4
#                
#                if order == 4:
#                    psi_filtered[i,j,k] = ((1./256.)*(186.*psi[i,j,k]+56.*(psi[i,j,k-1]
#                               +psi[i,j,k+1])-28*(psi[i,j,k-2]+psi[i,j,k+2])
#                               +8*(psi[i,j,k-3]+psi[i,j,k+3])-(psi[i,j,k-4]+psi[i,j,k+4])))
#                elif order == 2:
#                    psi_filtered[i,j,k] = ((1./16.)*(-psi[i,j,k-2]+4*psi[i,j,k-1]
#                               +10*psi[i,j,k]+4*psi[i,j,k+1]-psi[i,j,k+2])) 
#                elif order == 1:
#                    psi_filtered[i,j,k] = ((0.25)*(psi[i,j,k-1]+2*psi[i,j,k]+psi[i,j,k+1]))
#                    
#    return psi_filtered

@jit
def fourth_order_shapiro_filter_2D(psi):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    tau = 36.
    #4th order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        elif i == nlambda-3:
            i=-3
        elif i == nlambda-4:
            i=-4
        for j in range(1,nphi-1):
            psi_filtered_lambda[i,j] = ((1./256)*(186.*psi[i,j]+56.*(psi[i-1,j]
                       +psi[i+1,j])-28*(psi[i-2,j]+psi[i+2,j])
                       +8*(psi[i-3,j]+psi[i+3,j])-(psi[i-4,j]+psi[i+4,j])))
           
    #4th order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            elif j in (nphi-3,nphi-4,2,3):
                order = 2
            else:
                order = 4
            for k in range(nP):  
                if order == 4:
                    psi_filtered[i,j] = ((1./256.)*(186.*psi_filtered_lambda[i,j]+56.*(psi_filtered_lambda[i,j-1]
                               +psi_filtered_lambda[i,j+1])-28*(psi_filtered_lambda[i,j-2]+psi_filtered_lambda[i,j+2])
                               +8*(psi_filtered_lambda[i,j-3]+psi_filtered_lambda[i,j+3])-(psi_filtered_lambda[i,j-4]+psi_filtered_lambda[i,j+4])))
                elif order == 2:
                    psi_filtered[i,j] = ((1./16.)*(-psi_filtered_lambda[i,j-2]+4*psi_filtered_lambda[i,j-1]
                               +10*psi_filtered_lambda[i,j]+4*psi_filtered_lambda[i,j+1]-psi_filtered_lambda[i,j+2])) 
                elif order == 1:
                    psi_filtered[i,j] = ((0.25)*(psi_filtered_lambda[i,j-1]+2*psi_filtered_lambda[i,j]+psi_filtered_lambda[i,j+1]))
    
    subtractive_filter = psi-psi_filtered
    psi_filtered_2 = psi-1/tau*subtractive_filter
    
    return psi_filtered_2

def Robert_Asselin_filter(psi_f,psi_n,psi_p):
    filter_parameter = 0.1
    psi_n_filtered = psi_n + 0.5*filter_parameter*(psi_p-2*psi_n+psi_f)
    return psi_n_filtered

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
    elif func == 2:
        CM = plt.contourf(x,y,psi.transpose(),1000,vmin=vmin,vmax=vmax,cmap=cmap)
    if flip:
        plt.gca().invert_yaxis()
        
    plt.colorbar(CM)
    
def compare_filters(psi):
    psif4 = fourth_order_shapiro_filter(psi)
    psif2 = second_order_shapiro_filter(psi)
    psif1 = first_order_shapiro_filter(psi)
    psiff = filter_4dx_components(psi)
    plt.plot(phi,psi[30,:,8],phi,psif4[30,:,8],phi,psif2[30,:,8],phi,psif1[30,:,8],phi,psiff[30,:,8])
    plt.legend(['UF','S4','S2','S1','FF'])
    plt.figure()
    plt.plot(lambd,psi[:,8,8],lambd,psif4[:,8,8],lambd,psif2[:,8,8],lambd,psif1[:,8,8],lambd,psiff[:,8,8])
    plt.legend(['UF','S4','S2','S1','FF'])
    

def print_max_min_of_field(field,name):
    maxf = np.max(field)
    minf = np.min(field)
    #posmax = np.where(field == maxf)
    #posmin = np.where(field == minf)
    print('{}: Max = {}; Min = {}'.format(name,maxf,minf))
    
def threedarray2DataArray(arr):
    dummy = np.zeros([nlambda,nphi,nP,1])
    dummy[:,:,:,0] = arr
    da = xr.DataArray(dummy,coords = [lon,lat,P,np.array([day])],dims = ['lon','lat','lev','time'])
    return da

def twodarray2DataArray(arr):
    dummy = np.zeros([nlambda,nphi,1])
    dummy[:,:,0] = arr
    da = xr.DataArray(dummy,coords = [lon,lat,np.array([day])],dims = ['lon','lat','time'])
    return da

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
    if t % 86400 == 0:
        u_h /= steps_per_day
        v_h /= steps_per_day
        T_h /= steps_per_day
        Ps_h /= steps_per_day
        omega_h /= steps_per_day
        omegas_h /= steps_per_day
        Q_h /= steps_per_day
        theta_h /= steps_per_day
    return u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h

start = time.time()
#define constants
pi = np.pi
nphi = 46
nlambda = 72    
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
dlambda_deg = 5.
dlambda = np.deg2rad(dlambda_deg)
dphi_deg = 4.
dphi = np.deg2rad(dphi_deg)
dt = 100 #seconds
tstop = 86400*40
#tstop = 50000
t = 0
steps_per_day = 86400/dt
#define fields
phi = np.array([-(pi/2)+(j)*dphi for j in range(nphi)])
cosphi = np.cos(phi)
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
init = False
if init:
    #prognostic fields
    Ps_f = np.zeros(shape=[nlambda,nphi])
    Ps_n = np.zeros(shape=[nlambda,nphi])
    Ps_p = np.zeros(shape=[nlambda,nphi])
    Ps_h = np.zeros(shape=[nlambda,nphi])
    
    u_f = np.zeros(shape=[nlambda,nphi,nP])
    u_n = np.zeros(shape=[nlambda,nphi,nP])
    u_p = np.zeros(shape=[nlambda,nphi,nP])
    u_h = np.zeros(shape=[nlambda,nphi,nP])
    
    v_f = np.zeros(shape=[nlambda,nphi,nP])
    v_n = np.zeros(shape=[nlambda,nphi,nP])
    v_p = np.zeros(shape=[nlambda,nphi,nP])
    v_h = np.zeros(shape=[nlambda,nphi,nP])
    
    T_f = np.zeros(shape=[nlambda,nphi,nP])
    T_n = np.zeros(shape=[nlambda,nphi,nP])
    T_p = np.zeros(shape=[nlambda,nphi,nP])
    T_h = np.zeros(shape=[nlambda,nphi,nP])
    
    #diagnostic fields
    z_n = np.zeros(shape=[nlambda,nphi,nP])
    zf_n = np.zeros(shape=[nlambda,nphi,nP])
    z_p = np.zeros(shape=[nlambda,nphi,nP])
    z_h = np.zeros(shape=[nlambda,nphi,nP])
    
    omega_n = np.zeros(shape=[nlambda,nphi,nP])
    omega_p = np.zeros(shape=[nlambda,nphi,nP])
    omega_h = np.zeros(shape=[nlambda,nphi,nP])
    
    omegas_n = np.zeros(shape=[nlambda,nphi])
    omegas_p = np.zeros(shape=[nlambda,nphi])
    omegas_h = np.zeros(shape=[nlambda,nphi])
    
    Q_n = np.zeros(shape=[nlambda,nphi,nP])
    Q_p = np.zeros(shape=[nlambda,nphi,nP])
    Q_h = np.zeros(shape=[nlambda,nphi,nP])
    
    theta_n = np.zeros(shape=[nlambda,nphi,nP])
    theta_p = np.zeros(shape=[nlambda,nphi,nP])
    theta_s = np.zeros(shape=[nlambda,nphi])
    theta_h = np.zeros(shape=[nlambda,nphi,nP])
    
    Fl = np.zeros(shape=[nlambda,nphi,nP])
    Fphi = np.zeros(shape=[nlambda,nphi,nP])
    sFl = np.zeros(shape=[nlambda,nphi,nP])
    sFphi = np.zeros(shape=[nlambda,nphi,nP])
    
    #surface fields
    Ts = np.zeros(shape=[nlambda,nphi])
    us = np.zeros(shape=[nlambda,nphi])
    vs = np.zeros(shape=[nlambda,nphi])
    A = np.zeros(shape=[nlambda,nphi])
    Vs = np.zeros(shape=[nlambda,nphi])
    
    #helper fields
    dlnP = np.zeros(shape=[nlambda,nphi,nP])
    T_m = np.zeros(shape=[nlambda,nphi,nP])
    kv = np.zeros(shape=[nlambda,nphi,nP])
    skv = np.zeros(shape=[nlambda,nphi,nP])
    Ps_mean = []
    dummy_psi = np.zeros([nlambda,nphi,nP+8])
    zeros = np.zeros(shape=[nlambda,nphi,nP])
    
    #Equilibrium temperature field
    T_eq = np.zeros(shape=[nlambda,nphi,nP])
    T_eqs = np.zeros(shape=[nlambda,nphi])
    #initialize
    Ps_n[:,:] = 100000.
    Ps_p[:,:] = 100000.
    T_eq,T_eqs = H_S_equilibrium_temperature(T_eq,T_eqs,Ps_n)
    T_n = T_eq + np.random.random(size=[nlambda,nphi,nP])/1e2 #Random perturbations to break hemispheric and zonal symmetry
    T_p = T_eq
    Ts = T_eqs
    T_m = helper_Tm(T_m,T_n,Ts)
    theta_n,theta_s = diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n)
    lnPs = np.log(Ps_n)
    dlnp = helper_dlnP(dlnP,lnPs,Pf)
    #sys.exit()
    zf_n,z_n = diag_z(zf_n,z_n,T_n,Ps_n,dlnP)
    n = 0
    day = 1
    
#time stepping loop

while t<tstop:
    n += 1
    #print('Begin timestep: {}'.format(n))
    Fl,Fphi = H_S_friction(Fl,Fphi,u_n,v_n,Ps_n,kv,sigmab)
#    sFl,sFphi = sponge_layer(sFl,sFphi,skv,u_n,v_n)
    Fl += sFl; Fphi += sFphi
    u_f = prognostic_u(u_f,u_n,u_p,v_n,us,omega_n,z_n,Ps_n,omegas_n,Fl)
    v_f = prognostic_v(v_f,v_n,v_p,u_n,vs,omega_n,z_n,Ps_n,omegas_n,Fphi)
    T_f = prognostic_T(T_f,T_n,T_p,u_n,v_n,omega_n,theta_n,Ps_n,theta_s,omegas_n,Q_n)
    Ps_f,lnPs = prognostic_Ps(Ps_f,Ps_n,Ps_p,omegas_n,us,vs,lnPs)
#    sFl,sFphi = sponge_layer(sFl,sFphi,skv,u_f,v_f)
#    u_f = u_f - 2*dt*sFl
#    v_f = v_f - 2*dt*sFphi
    # Apply shapiro filters of order 2 and 4 to the solution.
#    if n % 31 == 0:
#        u_f = second_order_shapiro_filter(u_f)
#        v_f = second_order_shapiro_filter(v_f)
#        T_f = second_order_shapiro_filter(T_f)
#        #Ps_f = second_order_shapiro_filter_2D(Ps_f)
    u_f = fourth_order_shapiro_filter(u_f)
    #u_f = filter_for_vertical_mixing(u_f,dummy_psi,us,zeros,90)
    v_f = fourth_order_shapiro_filter(v_f)
    #v_f = filter_for_vertical_mixing(v_f,dummy_psi,vs,zeros,90)
    T_f = fourth_order_shapiro_filter(T_f)
    #T_f = filter_for_vertical_mixing(T_f,dummy_psi,Ts,T_f,60)
    Ps_f = fourth_order_shapiro_filter_2D(Ps_f)

    #increased diffusion in top layers
#    u_f[:,:,1] = alternative_second_order_shapiro_filter(u_f,1./8)[:,:,1]
#    u_f[:,:,2] = alternative_second_order_shapiro_filter(u_f,1./16)[:,:,2]
#    u_f[:,:,3] = alternative_second_order_shapiro_filter(u_f,1./64)[:,:,3]
#    v_f[:,:,1] = alternative_second_order_shapiro_filter(v_f,1./8)[:,:,1]
#    v_f[:,:,2] = alternative_second_order_shapiro_filter(v_f,1./16)[:,:,2]
#    v_f[:,:,3] = alternative_second_order_shapiro_filter(v_f,1./64)[:,:,3]
    #polar filter
#    u_f = polar_fourier_filter(u_f)
#    v_f = polar_fourier_filter(v_f)
#    T_f = polar_fourier_filter(T_f)    
#    Ps_f = polar_fourier_filter(Ps_f)
    
    #if T_f.max()>320.:
    #    sys.exit()
    #Apply Robert-Asselin Frequency filter to prognostic variables at time n
    u_n = Robert_Asselin_filter(u_f,u_n,u_p)
    v_n = Robert_Asselin_filter(v_f,v_n,v_p)
    T_n = Robert_Asselin_filter(T_f,T_n,T_p)
    Ps_n = Robert_Asselin_filter(Ps_f,Ps_n,Ps_p)
    #print('Finished prognostic equations')
    t = t+dt
    #print('Advance time to: t={}'.format(t))
    u_p = u_n.copy()
    u_n = u_f.copy()    
    v_p = v_n.copy()
    v_n = v_f.copy()
    T_p = T_n.copy()
    T_n = T_f.copy()
    Ps_p = Ps_n.copy()
    Ps_n = Ps_f.copy()
    #print('Diagnostic equations')
    omega_n = diag_omega(omega_n,u_n,v_n)
    omegas_n = diag_omegas(omegas_n,omega_n,u_n,v_n,us,vs)
    T_m = helper_Tm(T_m,T_n,Ts)
    dlnP = helper_dlnP(dlnP,lnPs,P)
    zf_n,z_n = diag_z(zf_n,z_n,T_n,Ps_n,dlnP)
    theta_n,theta_s = diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n)
    Q_n = diag_Q(Q_n,T_n,Ps_n,T_eq)
    us,vs,Vs = diag_surface_wind(us,vs,Vs,u_n,v_n,Ps_p,A)
    if n%100 == 0:
        print('\nValues for time step {}, time {}, day {}'.format(n,t,day))
        print_max_min_of_field(u_n,'U')
        print_max_min_of_field(v_n,'V')
        print_max_min_of_field(T_n,'T')
        print_max_min_of_field(Ps_n,'Ps')
        print('Ps.mean: {}'.format(np.average(Ps_n[:,1:nphi-1],axis=1,weights=np.cos(phi[1:nphi-1])).mean()))
        Ps_mean.append(np.average(Ps_n[:,1:nphi-1],axis=1,weights=np.cos(phi[1:nphi-1])).mean())
        print_max_min_of_field(omega_n,'omega')
        print_max_min_of_field(omegas_n,'omegas')
        print_max_min_of_field(zf_n,'z')

    
#handle output    
    u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h = (add_fields_to_history(u_h,
                                                u_n,v_h,v_n,Ps_h,Ps_n,T_h,T_n
                                                ,omega_h,omega_n,omegas_h
                                                ,omegas_n,Q_h,Q_n,theta_h,theta_n))
    if t % 86400 == 0:
        print('writing history file')
        ds = xr.Dataset()
        ds['U'] = threedarray2DataArray(u_h)
        ds['V'] = threedarray2DataArray(v_h)
        ds['Ps'] = twodarray2DataArray(Ps_h)
        ds['T'] = threedarray2DataArray(T_h)
        ds['OMEGA'] = threedarray2DataArray(omega_h)
        ds['OMEGAS'] = twodarray2DataArray(omegas_h)
        ds['Q'] = threedarray2DataArray(Q_h)
        ds['THETA'] = threedarray2DataArray(theta_h)
        ds.to_netcdf('history_out_day{}.nc'.format(day),unlimited_dims = ['time'])
        plotter_help(ds['U'][:,:,:,0].mean(dim='lon'),0,1)
        plt.show()
        plotter_help(ds['T'][:,:,:,0].mean(dim='lon'),0,1)
        plt.show()
        day += 1
        del ds
        u_h[:] = 0;v_h[:] = 0;Ps_h[:] = 0;T_h[:] = 0;omega_h[:] = 0;omegas_h[:] = 0;
        Q_h[:] = 0;theta_h[:] = 0;
    
#handle errors and exceptions

end = time.time()
print(end - start)