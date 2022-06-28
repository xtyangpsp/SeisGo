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

def get_dispersion_waveforms(d, dt,fmin,fmax,dp=None,fscale='ln',fextend=10):
    """
    Produce dispersion wavefroms with narrowband filters.

    ===parameters===
    d: 1-d array data.
    dt: sampling interval.
    fmin, fmax: frequency range.
    dp: period increment in seconds. default 1 s.
    fscale: frequency scales. "ln" for linear [default]. "nln" for non-linear scale.
    fextend: extend individual frequency value to form a band range. default: 5 scale steps.

    ==returns===
    dout, fout: narrowband-filtered waveforms and the frequency vector.
    """
    period=np.array([1/fmax - fextend*dp,1/fmin + fextend*dp])
    if period[0] < 2*dt: period[0]=2.01*dt
    if dp is None: dp=1

    if fscale=="ln":
        # f_all=np.arange(fmin-fextend*df,fmax+fextend*df,df)
        ptest=np.arange(period.min(),period.max(),dp)
    elif fscale=="nln":
        ptest=2 ** np.arange(np.log2(0.1*period.min()),
                    np.log2(2*period.max()),dp)
    f_all=np.flip(1/ptest)
    fout_temp=[]
    dout_temp=[]
    din=d.copy()

    for ii in range(len(f_all)-fextend):
        if f_all[ii]>=1/(2*dt) or f_all[ii+fextend]>=1/(2*dt): continue
        ds_win=bandpass(din,f_all[ii],f_all[ii+fextend],1/dt,corners=4, zerophase=True)
        dout_temp.append(ds_win/np.max(np.abs(ds_win)))
        fout_temp.append(np.mean([f_all[ii],f_all[ii+fextend]])) #center frequency
    fout_temp=np.array(fout_temp)
    f_ind=np.where((fout_temp>=fmin) & (fout_temp<=fmax))[0]
    fout=fout_temp[f_ind]
    dout_temp=np.array(dout_temp)
    dout = dout_temp[f_ind]
    return dout, fout

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
