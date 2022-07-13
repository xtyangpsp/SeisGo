import os,glob,copy,obspy,scipy,time,pycwt,pyasdf,datetime
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression
from obspy.core.util.base import _get_function_from_entry_point
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.filter import bandpass,lowpass
"""
This is a planned module, to be developed.
"""
################################################################
################ DISPERSION EXTRACTION FUNCTIONS ###############
################################################################
def get_dispersion_waveforms_cwt(d, dt,fmin,fmax,dj=1/12, s0=-1, J=-1, wvn='morlet'):
    """
    Produce dispersion wavefroms with continuous wavelet tranform.

    ===parameters===
    d: 1-d array data.
    df: time interval.
    fmin, fmax: frequency range.
    dj=1/12, s0=-1, J=-1, wvn='morlet': pycwt.cwt parameters.

    ==returns===
    dout, fout: narrowband-filtered waveforms and the frequency vector.
    """
    ds_cwt, sj, f, coi, _, _ = pycwt.cwt(d, dt, dj, s0, J, wvn)
    f_ind = np.where((f >= fmin) & (f <= fmax))[0]
    dout=[]
    fout=[]
    for ii in range(len(f_ind)):
        if ii>0 and ii<len(f_ind)-1: f_ind_temp=f_ind[ii-1:ii+1]
        elif ii==len(f_ind)-1: f_ind_temp=f_ind[ii-1:ii]
        elif ii==0:f_ind_temp=f_ind[ii:ii+1]
        fout.append(np.mean(f[f_ind_temp]))
        rds_cwt=np.real(pycwt.icwt(ds_cwt[f_ind_temp], sj[f_ind_temp], dt, dj, wvn))
        ds_win=np.power(rds_cwt,2)
        dout.append(ds_win/np.max(ds_win))
    return np.flip(np.array(dout),axis=0), np.flip(fout)

def get_dispersion_waveforms(d, dt,pmin,pmax,dp=None,pscale='ln',extend=10):
    """
    Produce dispersion wavefroms with narrowband filters.

    ===parameters===
    d: 1-d array data.
    dt: sampling interval.
    pmin, pmax: period range.
    dp: period increment in seconds. default 1 s.
    pscale: period scales. "ln" for linear [default]. "nln" for non-linear scale.
    extend: extend individual period value to form a band range. default: 5 scale steps.

    ==returns===
    dout, pout: narrowband-filtered waveforms and the period vector.
    """
    period=np.array([pmin - extend*dp,pmax + extend*dp])
    if period[0] < 2*dt: period[0]=2.01*dt
    if dp is None: dp=1

    if pscale=="ln":
        # f_all=np.arange(fmin-extend*df,fmax+extend*df,df)
        ptest=np.arange(period.min(),period.max(),dp)
    elif pscale=="nln":
        ptest=2 ** np.arange(np.log2(0.1*period.min()),
                    np.log2(2*period.max()),dp)
    f_all=np.flip(1/ptest)
    fout_temp=[]
    dout_temp=[]
    din=d.copy()

    for ii in range(len(f_all)-extend):
        if f_all[ii]>=1/(2*dt) or f_all[ii+extend]>=1/(2*dt): continue
        ds_win=bandpass(din,f_all[ii],f_all[ii+extend],1/dt,corners=4, zerophase=True)
        dout_temp.append(ds_win/np.max(np.abs(ds_win)))
        fout_temp.append(np.mean([f_all[ii],f_all[ii+extend]])) #center frequency
    fout_temp=np.array(fout_temp)
    f_ind=np.where((fout_temp>=1/pmax) & (fout_temp<=1/pmin))[0]
    fout=fout_temp[f_ind]
    dout_temp=np.array(dout_temp)
    dout = dout_temp[f_ind]
    pout = 1/fout
    return dout, pout
##
def get_dispersion_image(g,t,d,pmin,pmax,vmin,vmax,dp=1,dv=0.1,window=1,pscale='ln',pband_extend=5,
                        verbose=False):
    """
    Uses phase-shift method. Park et al. (1998): http://www.masw.com/files/DispersionImaingScheme-1.pdf

    =====PARAMETERS====
    g: waveform gather for all distances (traces). It should be a numpy array.
    t: time vector.
    d: distance vector corresponding to the waveforms in `g`
    pmin: minimum period.
    pmax: maximum period.
    vmin: minimum phase velocity to search.
    vmax: maximum phase velocity to search.
    dp: period increment. default is 1.
    dv: velocity increment in searching. default is 0.1
    window: number of wavelength when slicing the time segments in computing summed energy. default is 1.
    pscale: period vector scale in applying narrowband filters. default is 'ln' for linear scale.
    pband_extend: number of period increments to extend in filtering. defult is 5.

    =====RETURNS====
    dout: dispersion information showing the normalized energy for each velocity value for each frequency.
    vout: velocity vector used in searching.
    pout: period vector.
    """

    dt=np.abs(t[1]-t[0])
    if t[0]<-1.0*dt and t[-1]> dt: #two sides.
        side='a'
        zero_idx=int((len(t)-1)/2)
    elif t[0]<-1.0*dt:
        side='n'
        zero_idx=len(t)-1
    elif t[-1]>dt:
        side='p'
        zero_idx=0
    if verbose: print('working on side: '+side)
    dfiltered_all=[]
    dist_final=[]
    for k in range(g.shape[0]):
        dtemp,pout=get_dispersion_waveforms(g[k]/np.max(np.abs(g[k])),dt,pmin,
                                        pmax,dp=dp,pscale=pscale,extend=pband_extend)
        dfiltered_all.append(dtemp)
    dfiltered_all=np.array(dfiltered_all)
    vout=np.arange(vmin,vmax+0.5*dv,dv)
    dout_n_all=[]
    dout_p_all=[]
    for k in range(len(pout)):
        win_length=window*pout[k]
        win_len_samples=int(win_length/dt)+1
        dout_n=[]
        dout_p=[]

        d_in=dfiltered_all[:,k,:]
        for i,v in enumerate(vout):
            #subset by distance
            mindist=1.5*v*pout[k] #at least 1.5 wavelength.
            dist_idx=np.where((d >= mindist))[0]
            if len(dist_idx) >5:
                if side=='a' or side=='n':
                    dvec=[]
                    for j in dist_idx: #distance, loop through traces
                        tmin=d[j]/v
                        tmin_idx=zero_idx - int(tmin/dt)
                        dsec=d_in[j][tmin_idx - win_len_samples : tmin_idx]
                        if not any(np.isnan(dsec)):
                            dvec.append(dsec)
                    dout_n.append(np.sum(np.power(np.mean(dvec,axis=1),2)))

                if side=='a' or side=='p':
                    dvec=[]
                    for j in dist_idx: #distance, loop through traces
                        tmin=d[j]/v
                        tmin_idx=zero_idx + int(tmin/dt)
                        dsec=d_in[j][tmin_idx : tmin_idx + win_len_samples]
                        if not any(np.isnan(dsec)):
                            dvec.append(dsec)
                    dout_p.append(np.sum(np.power(np.mean(dvec,axis=1),2)))
        dout_n /= np.max(dout_n)
        dout_p /= np.max(dout_p)
        dout_n_all.append(dout_n)
        dout_p_all.append(dout_p)
    #
    dout=np.squeeze(np.array([dout_n_all,dout_p_all]))


    return dout,vout,pout
# function to extract the dispersion from the image
# modified from NoisePy.
def extract_dispersion_curve(amp,vel):
    '''
    this function takes the dispersion image as input, tracks the global maxinum on
    the spectrum amplitude

    PARAMETERS:
    ----------------
    amp: 2D amplitude matrix of the wavelet spectrum
    vel:  vel vector of the 2D matrix
    RETURNS:
    ----------------
    gv:   group velocity vector at each frequency
    '''
    nper = amp.shape[0]
    gv   = np.zeros(nper,dtype=np.float32)
    dv = vel[1]-vel[0]

    # find global maximum
    for ii in range(nper):
        maxvalue = np.max(amp[ii],axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]

    return gv
