#!/usr/bin/env python
# coding: utf-8
"""
This file contains a series of functions used for tilt and compliance corrections for OBS data. For the theoretical backgrund of tilt and compiance corrections, please read Bell et al. (2015, BSSA).

### References:
1. Bell, S. W., D. W. Forsyth, & Y. Ruan (2015), Removing Noise from the Vertical Component Records of Ocean-Bottom Seismometers: Results from Year One of the Cascadia Initiative, Bull. Seismol. Soc. Am., 105(1), 300-313, doi:10.1785/0120140054.
2. Tian, Y., & M. H. Ritzwoller (2017), Improving ambient noise cross-correlations in the noisy ocean bottom environment of the Juan de Fuca plate, Geophys. J. Int., 210(3), 1787-1805, doi:10.1093/gji/ggx281.
"""
############################################
##functions used by the main routine
############################################
#import needed packages.
import sys
import time
import scipy
import obspy
import pyasdf
import datetime
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from scipy.fftpack import fft,fftfreq,ifft
from scipy.fftpack import rfft,rfftfreq,irfft

# Modified from Zhitu Ma. Modified by Xiaotao Yang
# 1. Reduce input arguments to only read in window length. Windows will be calculated.
# 2. Future: Need to revert back to getting windows by excluding earthquakes
# 3. The original code had bugs when computing the admittance and phases. Q and C were not computed.
# 4. The original code hard-coded sample interval when computing the frequencies, which caused wrong
# results if not changed.
def gettransfer(x,y,delta,winlen=2000,iplot=False,figname="debug_transfer.png",coherence_only=False):
    """ calculate the transfer function from x to y
    return the coherence, admittance, phase and their corresponding error
    """
    winlen=int(winlen/delta)
#     iopt=1
    nsamp=len(x)
    wins=np.arange(0,nsamp-winlen,winlen)
    nd=len(wins)
    ff=np.split(fftfreq(winlen,delta),2)[0]

    coh_debug=[]
    win_debug=[]
    taper=scipy.signal.tukey(winlen,0.05)
    for k,win in enumerate(wins):
        xtmp=x[win:win+winlen]
        ytmp=y[win:win+winlen]

        x_fft=np.split(fft(taper*xtmp),2)[0]
        y_fft=np.split(fft(taper*ytmp),2)[0]
        if k==0:
            Gxy=np.conj(x_fft)*y_fft
            Gxx=np.conj(x_fft)*x_fft
            Gyy=np.conj(y_fft)*y_fft

            Cxy=np.real(x_fft*np.real(y_fft)+np.imag(x_fft*np.imag(y_fft)))
            Qxy=np.real(x_fft*np.imag(y_fft)-np.imag(x_fft*np.real(y_fft)))
        else:
            Gxy=Gxy+np.conj(x_fft)*y_fft
            Gxx=Gxx+np.conj(x_fft)*x_fft
            Gyy=Gyy+np.conj(y_fft)*y_fft

            Cxy=Cxy+np.real(x_fft*np.real(y_fft)+np.imag(x_fft*np.imag(y_fft)))
            Qxy=Qxy+np.real(x_fft*np.imag(y_fft)-np.imag(x_fft*np.real(y_fft)))

        cohtmp=np.abs(Gxy)**2/np.real(Gxx)/np.real(Gyy)
        cohtmp=np.sqrt(cohtmp)
        coh_debug.append(np.mean(cohtmp))
        win_debug.append(win)

    #normalize by number of windows
    Gxy=Gxy/nd
    Gxx=Gxx/nd
    Gyy=Gyy/nd
    Cxy=Cxy/nd
    Qxy=Qxy/nd

#     coh=np.abs(Gxy)**2/np.real(Gxx)/np.real(Gyy)
    coh=np.real(np.abs(Gxy)**2/(Gxx*Gyy))
    coh=np.sqrt(coh)
    if coherence_only:
        adm=0.
        phs=0.
        adm_err=0.
        phs_err=0.
    else:
#         adm=np.abs(Gxy)/np.real(Gxx)
        adm=np.real(np.abs(Gxy)/Gxx)
#         phs=np.angle(Gxy)
        phs=np.arctan(Qxy/Cxy)
        adm_err=np.sqrt(1.-coh**2)/np.abs(coh)/np.sqrt(2*nd)
        coh_err=np.sqrt((1.-coh**2)*np.sqrt(2.0)/np.abs(coh)/np.sqrt(nd))
        phs_err=adm_err


    if iplot:
        plt.figure(figsize=(8,4))
        plt.plot(win_debug,coh_debug,'o')
        plt.xlabel("window")
        plt.ylabel("coherence")
        plt.title("Debug transfer function")
        plt.savefig(figname,orientation='landscape')
        plt.show()
        plt.close()

    return ff,coh,adm,phs,adm_err,phs_err

# From Zhitu Ma. Modified by Xiaotao Yang
# 1. Use freqmin and freqmax, instead of f1 and f2
# 2. Changed to include the two ending frequencies.
#
def docorrection(tr1,tr2,adm,adm_err,phs,phs_err,freqmin,freqmax,ff,iplot=0):
    """ calculate a quadratic fit to adm and phs
    use this information to predict from tr1, then remove this from tr2
    returning two trace (obspy class), one is the prediction
    one is this prediction removed from tr2 """
    delta=1.0/tr1.stats.sampling_rate
    idx=(ff>=freqmin) & (ff<=freqmax)
    ff_select=ff[idx]
    adm_select=adm[idx]
    adm_err_select=adm_err[idx]
    w=1./adm_err_select
    apol=np.polyfit(ff_select,adm_select,2,w=w)
    phs_select=phs[idx]
    phs_err_select=phs_err[idx]
    w=1./phs_err_select
    ppol=np.polyfit(ff_select,phs_select,2,w=w)

    if (iplot==1):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        adm_fit=apol[0]*ff_select**2+apol[1]*ff_select+apol[2]
        plt.plot(ff_select,adm_select)
        plt.plot(ff_select,adm_fit)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("admittance")
        plt.subplot(1,2,2)
        phs_fit=ppol[0]*ff_select**2+ppol[1]*ff_select+ppol[2]
        plt.plot(ff_select,phs_select)
        plt.plot(ff_select,phs_fit)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("phase shift")
        plt.savefig(tr1.stats.network+"."+tr1.stats.station+"."+tr1.stats.channel+"_"+                    tr2.stats.network+"."+tr2.stats.station+"."+tr2.stats.channel+"_adm_phs.png",                    orientation='landscape')
        plt.show()
        plt.close()


    ffr=fftfreq(len(tr1.data),delta)
    tr_pred=tr1.copy()
    tr_left=tr1.copy()
    Htmp_spec=rfft(tr1.data)
    Htmp_spec[0]=0
    Htmp_spec[-1]=0
    for i in np.arange(1,len(ffr)-1,2):
        rp=Htmp_spec[i]
        ip=Htmp_spec[i+1]
        if(ffr[i]>freqmax or ffr[i]<freqmin):
            Htmp_spec[i]=0.
            Htmp_spec[i+1]=0.
            continue
        amp=apol[0]*ffr[i]**2+apol[1]*ffr[i]+apol[2]
        phs=ppol[0]*ffr[i]**2+ppol[1]*ffr[i]+ppol[2]
        c=amp*np.cos(phs)
        d=amp*np.sin(phs)
        Htmp_spec[i]=rp*c-ip*d
        Htmp_spec[i+1]=ip*c+rp*d
    Htmp=irfft(Htmp_spec)
    tr_pred.data=Htmp
    tr_left.data=tr2.data-Htmp
    return tr_pred,tr_left

def maxcompfreq(d,iplot=False,figname="waterdepth_maxcompfreq.png"):
    """
    computes the maximum compliance frequency based on eq-7 of Tian and Ritzwoller, GJI, 2017
    """
#     d=np.arange(1,5051,50) #water depth
    f=np.sqrt(9.8/1.6/np.pi/d)
    if iplot:
        plt.figure(figsize=(10,5))
        plt.plot(d,f)
        plt.yscale("log")
        plt.grid(which="both")
        plt.xlabel("water depth (m)")
        plt.ylabel("frequency (Hz)")
        plt.text(1.2*np.mean(d),0.5*np.max(f),r'$\sqrt{(\frac{g}{1.6 \pi d})}$',fontsize=20)
        plt.savefig(figname,orientation='landscape')
        plt.show()
        plt.close()

    return f
