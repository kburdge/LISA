#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:37:47 2020

@author: kburdge
"""

import numpy as np
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5, BarycentricTrueEcliptic  # Low-level frames
import matplotlib.pyplot as plt

"Constants, in cgs"

msun=1.98847e33
kpc=3.09e21
G=6.67408e-8
c=2.99792458e10


"Relevant papers: Robson et al 2019. https://ui.adsabs.harvard.edu/abs/2019CQGra..36j5011R/abstract and Cornish and Larson 2003: https://ui.adsabs.harvard.edu/abs/2003PhRvD..67j3001C/abstract"

def SNR(RA,Dec,m1,m2,i,d,p):
    i=i*np.pi/180
    f=2/p
    chirp=(m1*m2)**(3/5)/((m1+m2)**(1/5))

    "Dealing with coordinates"

    coord = SkyCoord(RA,Dec, unit="deg")
    ecl=coord.transform_to(BarycentricTrueEcliptic)
    ecliptic_latitude=ecl.lat.deg
    ecliptic_longitude=ecl.lon.deg

    "Using Cornish 2003 conventions, where the RA is phi [0,pi], and Dec is theta [0,2*pi]"

    phi=ecliptic_longitude* np.pi / 180
    theta=np.pi/2-(ecliptic_latitude* np.pi / 180)

    "LISA Specs using Robson et al."

    "Robson right after Equation 1"
    L=2500000000
    f_star=19.09/1000.0
    
    "Number of LISA Channels"
    NC=2

    "Robson Equation 10"
    P_OMS=(1.5e-11)**2*(1+(0.002/f)**4)

    "Robson Equation 11"
    P_acc=(3e-15)**2*(1+(0.0004/f)**2)*(1+(f/0.008)**4)

    "Robson Equation 12"
    P=P_OMS/(L**2)+2*(1+np.cos(f/f_star)**2)*P_acc/((2*np.pi*f)**4*L**2)

    "Robson Equation 9"
    Transfer_Function=NC*(3/20)*1/(1+6/10*(f/f_star)**2)

    "Robson Equation 2"
    S_n=P/Transfer_Function

    "LISA Confusion Noise using Robson et al. using 4 yr values from Table 1"

    Amp=1.8e-44/NC
    alpha=0.138
    beta=-221
    kappa=521
    gamma=1680
    f_k=0.00113

    S_c=Amp*f**(-7/3)*np.exp(-f*alpha+beta*f*np.sin(kappa*f))*(1+np.tanh(gamma*(f_k-f)))

    "LISA Sensitivity (Robson Equation 1)"

    S=S_n+S_c

    "LISA Power Sensitivity (divide by transfer function, but multiply by 2 to keep the benefit of having two independent channels"

    "Inverting Robson Equation 2, but keeping factor of 2 for 2 independent channels"
    P_LISA=S*Transfer_Function/NC

    "Using Robson formalism"

    "Robson Equation 20"
    A=(5/24)**0.5*(G*chirp/c**3)**(5/6)*f**(-7/6)/(np.pi**(2/3)*(d/c))

    "Robson Equation 15"
    hplus=A*(1+np.cos(i)**2)/2
    hcross=A*np.cos(i)

    "T=4 years"
    T=4*3.154*10**7

    "Marginalize over polarization angle (range is [0,pi])"
    psi=np.arccos(np.random.uniform(0,1,10000))

    "Compute Orbit Averaged Detector Response Patterns using Cornish 2003"

    "Cornish Equation 44"
    DplusDx=(243.0/512.0)*(np.cos(theta)*np.sin(2*phi)*(2*(np.cos(phi))**2-1)*(1+np.cos(theta)**2))
    DxDx=3.0/512.0*(120.0*np.sin(theta)**2+np.cos(theta)**2+162*(np.sin(2*phi)**2)*np.cos(theta)**2)
    DplusDplus=3.0/2048*(487+158*np.cos(theta)**2+7*np.cos(theta)**4-162*np.sin(2*phi)**2*(1+np.cos(theta)**2)**2)


    "Cornish Equation 43 (left out FplusFx because vanishes during orbit averaging)"
    FplusFplus=1/4*(np.cos(2*psi)**2*DplusDplus-np.sin(4*psi)*DplusDx+np.sin(2*psi)**2*DxDx)
    FxFx=1/4*(np.cos(2*psi)**2*DxDx+np.sin(4*psi)*DplusDx+np.sin(2*psi)**2*DplusDplus)

    "Cornish Equation 41"
    "Important! The 1/2 out front of this expression was dropped intentionally. In Cornish 2003, it comes from time averaging over a cos^2. In Robson, and most other lituerature, the gravitational waveform is expressed as h=A(f)*e^(2*i*psi(f)), and when one takes <h|h>, this becomes a 1, not a 1/2. Dropping this makes the sky and inclination averaged SNR computed here agree with Robson's estimate in Eq 27."
    amp_sigma=np.std((FxFx*hcross*hcross+FplusFplus*hplus*hplus))
    average_amp=np.mean((FxFx*hcross*hcross+FplusFplus*hplus*hplus))

    "Compute frequency change over T_obs"

    "Robson Equation 25"
    fdot=96/5*np.pi**(8/3)*(G*chirp/c**3)**(5/3)*f**(11/3)

    delta_f=T*fdot
    
    signal=1/2*(FxFx*hcross*hcross+FplusFplus*hplus*hplus)*4*delta_f
    avg_signal=np.mean(signal)
    std_signal=np.std(signal)

    "Now finally, an SNR using Robson Equation 36"
    SNR=(4*average_amp*delta_f/P_LISA)**0.5
    h_gb=8*T**0.5*(G*chirp/c**3)**(5/3)*(np.pi*f)**(2/3)/(5**0.5*(d/c))

    return [np.mean(SNR), avg_signal**0.5, std_signal**0.5, f]


print(SNR(102.8889135563,28.7398158196,0.49*msun,0.247*msun,86.9,0.933*kpc,765.5))