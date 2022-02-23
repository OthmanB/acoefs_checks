#!/usr/bin/env python
import numpy as np
from scipy import special, integrate
import itertools
import matplotlib.pyplot as plt
from activity import *
from acoefs import *
from fit_a2sig import Qlm
from fit_a2sig import eta0_fct
from fit_a2sig import a_model_cpp

def Lorentzian(nu, nu_nlm, H_nlm, W_nlm):
    L=H_nlm/(1. + 4.*(nu - nu_nlm)**2 / W_nlm**2)
    return L

def nu_nlm_AR(l, m, nu_nl, a1, rho, epsilon_nl, theta0, delta, do_CF=True, do_AR=True, ftype='gate'):
    '''
        Applies the perturbation due to the centrifugal force (through a1) if do_CF=True
        Applies the perturbation due to the activity if do_AR = True
        REMARK: WE DO NOT APPLY ANY SYMETRICAL SPLITTING HERE : a1, a3, a5, ... are not added to nu_nlm
                The centrifugal force will add a pure a2 term
                The activity will add terms in a2, a4, a6, ...
    '''
    #Dnl=0.75
    #G=6.667e-8
    eta0=eta0_fct(rho=rho)
    if do_CF == True:
        dnu_CF=eta0*nu_nl * (a1*1e-6)**2 * Qlm(l,m)
        #print('         dnu_CF =', dnu_CF)
    else:
        dnu_CF=0
    if do_AR == True:
        dnu_AR=nu_nl*epsilon_nl*Alm_cpp(l,m, theta0=theta0, delta=delta, ftype=ftype)
        #print('         dnu_AR =', dnu_AR)
    else:
        dnu_AR=0
        
    nu_nlm=nu_nl + dnu_CF + dnu_AR
    return nu_nlm

def test_profile(l=1, acoefs=[1,0,0,0,0,0], theta0=np.pi/2, delta=2*np.pi*8.4/180, epsilon_nl=1e-4, Dnu=100, H_nl=1, W_nl=0.5, Vlm=[0.25, 0.5, 0.25], do_CF=True, ftype='gate'):
    ''' 
        Show the profile of a Lorentzian mode given acoeficients (symetrical ones, acoefs=[a1,0, a3,0, a5, 0]) on which
        are added a2 coefficient from the centrifugal force and the activity effect
    '''
    Dnu_sun=135.1
    numax_sun=3150.
    R_sun=6.96342e5 #in km
    M_sun=1.98855e30 #in kg
    rho_sun=M_sun*1e3/(4*np.pi*(R_sun*1e5)**3/3) #in g.cm-3
    #
    a1=acoefs[0]
    a3=acoefs[2]
    a5=acoefs[4]

    nu_nl=numax_sun
    rho=(Dnu/Dnu_sun)**2 * rho_sun
    #
    Ndata=1000
    nu=np.linspace(nu_nl-2*W_nl-(2*l+1)*a1/2, nu_nl+2*W_nl+(2*l+1)*a1/2, Ndata)
    profile=0.
    profile_ref=0.
    for m in range(-l, l+1):
        nu_nlm=nu_nlm_AR(l,m, nu_nl, a1, rho, epsilon_nl, theta0, delta, ftype=ftype, do_CF=do_CF)
        nu_nlm=nu_nlm + Pslm(1,l,m)*a1 + Pslm(3,l,m)*a3 + Pslm(5,l,m)*a5
        nu_nlm_ref=nunlm_from_acoefs(nu_nl, l, a1=a1, a2=0, a3=0,a4=0,a5=0,a6=0)
        profile=profile+Lorentzian(nu, nu_nlm, H_nl*Vlm[m+l], W_nl)
        profile_ref=profile_ref+Lorentzian(nu, nu_nlm_ref, H_nl*Vlm[m+l], W_nl)

    plt.plot(nu, profile, color='blue')
    plt.plot(nu, profile_ref, color='red')
    plt.show()


def acoefs_epsilon(l, a1=2500, Dnu=135.1, nu_nl=3150., ftype='gate'):
    '''
        For a given degree l, rotation a1 (nHz) and Dnu (translates into density), compute the
        odds a-acoeficients in presence of a centrifugal distorsion and of an activity
        at the pole or at the equator
    '''
    eps_min=0.0
    eps_max=0.0015
    epsilon_nl=np.linspace(eps_min, eps_max,14)

    #theta0=np.pi/2
    #delta=2*np.pi*8.4/180.
    theta0_eq=(90-8.4/2)*np.pi/180.
    delta_eq=8.4*np.pi/180.
    #
    #theta_polar=0.
    #delta_polar=np.pi*45./180.
    theta_polar=(45./2.)*np.pi/180.
    delta_polar=np.pi*45/180.
    ftype='gate'
    # -----------------
    #
    #rho=(Dnu/Dnu_sun)**2 * rho_sun
    #
    acoefs_eq=[]
    acoefs_polar=[]
    print(' acoefs:')
    for epsi in epsilon_nl:
        #print('theta0=', theta0_eq, '  delta=', delta_eq)
        anl_eq=a_model_cpp(nu_nl, Dnu, a1, epsi, theta0_eq, delta_eq, ftype, l, Alm_vals=None) # CF + AR
        anl_polar=a_model_cpp(nu_nl, Dnu, a1, epsi, theta_polar, delta_polar, ftype, l, Alm_vals=None) # CF + AR
        acoefs_eq.append(anl_eq)
        acoefs_polar.append(anl_polar)
        print('  epsi =', epsi,  '    anl_eq:', anl_eq ,  '    anl_polar:', anl_polar)
    return epsilon_nl, acoefs_eq, acoefs_polar

def a2_epsilon_plot(lrange=[1,2, 3], colors=['Blue', 'Orange', 'Red'], ftype='gate'):
    '''
        Make same plots than Fig3 of Gizon2004 AN 323, 251
        We impose basically the same values as what Gizon2004 did
    '''
    a1=2500 # nHz
    j=0
    #lrange[1]=lrange[1]
    for l in lrange:
        print('l =', l)
        epsilon_nl, acoefs_eq, acoefs_polar=acoefs_epsilon(l, a1=a1, Dnu=135.1, nu_nl=3150.,ftype=ftype)
        a2_eq=[row[1] for row in acoefs_eq]
        #a4_eq=[row[3] for row in acoefs_eq]
        #a6_eq=[row[5] for row in acoefs_eq]
        a2_polar=[row[1] for row in acoefs_polar]
        #a4_polar=[row[3] for row in acoefs_polar]
        #a6_polar=[row[5] for row in acoefs_polar]
        #
        a2_CF=a2_eq[0]
        a2_AR_eq=a2_eq[1:]
        a2_AR_polar=a2_polar[1:]
        #print('a2_CF:', a2_CF)
        #print('a2_eq:', a2_eq)
        #print('a2_polar:', a2_polar)
        plt.plot(epsilon_nl[1:], a2_AR_eq, color=colors[j], linestyle='--')
        plt.plot(epsilon_nl[1:], a2_AR_polar, color=colors[j], linestyle='dashdot')
        plt.plot(epsilon_nl, np.repeat(a2_CF,len(epsilon_nl)), linestyle='solid', color=colors[j])
        j=j+1
    #
    plt.ylabel='a2 (microHz)'
    plt.xlabel='epsilon'
    plt.xscale('log')
    plt.xlim=[0.0001, 0.001]
    #plt.ylim=[-.5, 0.2]
    plt.show()
