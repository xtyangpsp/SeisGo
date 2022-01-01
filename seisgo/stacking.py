import os
import glob
import copy
import obspy
import scipy
import time
import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft,ifft,next_fast_len
"""
Modified from NoisePy stacking functions. Originally written by Chengxin Jiang and Marine Donolle.

Adapted for SeisGo by Xiaotao Yang
"""
def seisstack(d,method,par=None):
    """
    this is a wrapper for calling individual stacking functions.
    d: data. 2-d array
    method: stacking method, one of "linear","pws","robust","acf","nroot","selective"
    par: dictionary containing all parameters for each stacking method. defaults will
        be used if not specified.

    RETURNS:
    ds: stacked data, which may be a list depending on the method.
    """
    par0={"axis":0,"p":2,"g":1,"cc_min":0.0,"epsilon":1E-5,"maxstep":10}
    if par is None:
        par=par0
    else:
        par={**par0,**par} #use par values if specified. otherwise, use defaults.

    if method.lower() == 'linear':
        ds = np.mean(d,axis=par["axis"])
    elif method.lower() == 'pws':
        ds = pws(d,p=par['p'])
    elif method.lower() == 'robust':
        ds = robust(d,epsilon=par['epsilon'],maxstep=par['maxstep'])
    elif method.lower() == 'acf':
        ds = adaptive_filter(d,g=par['g'])
    elif method.lower() == 'nroot':
        ds = nroot(d,p=par['p'])
    elif method.lower() == 'selective':
        ds = selective(d,cc_min=par['cc_min'],epsilon=par['epsilon'],maxstep=par['maxstep'])
    #
    return ds

def robust(d,epsilon=1E-5,maxstep=10,win=None):
    """
    this is a robust stacking algorithm described in Pavlis and Vernon 2010. Generalized
    by Xiaotao Yang.

    PARAMETERS:
    ----------------------
    d: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration (a small number). Default 1E-5
    maxstep: maximum iterations. default 10.
    win: [start_index,end_index] used to compute the weight, instead of the entire trace. Default None.
            When None, use the entire trace.
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle
    Modified by Xiaotao Yang
    """
    if d.ndim == 1:
        print('2D matrix is needed')
        return d
    res  = 9E9  # residuals
    w = np.ones(d.shape[0])
    nstep=0
    newstack = np.median(d,axis=0)
    if win is None:
        win=[0,-1]
    while res > epsilon and nstep <=maxstep:
        stack = newstack
        for i in range(d.shape[0]):
            dtemp=d[i,win[0]:win[1]]
            crap = np.multiply(stack[win[0]:win[1]],dtemp.T)
            crap_dot = np.sum(crap)
            di_norm = np.linalg.norm(dtemp)
            ri_norm = np.linalg.norm(dtemp -  crap_dot*stack[win[0]:win[1]])
            w[i]  = np.abs(crap_dot) /di_norm/ri_norm#/len(cc_array[:,1])
        # print(w)
        w =w /np.sum(w)
        newstack =np.sum( (w*d.T).T,axis=0)#/len(cc_array[:,1])
        res = np.linalg.norm(newstack-stack,ord=1)/np.linalg.norm(newstack)/len(d[:,1])
        nstep +=1

    return newstack, w, nstep

def adaptive_filter(d,g=1):
    '''
    the adaptive covariance filter to enhance coherent signals. Fellows the method of
    Nakata et al., 2015 (Appendix B)

    the filtered signal [x1] is given by x1 = ifft(P*x1(w)) where x1 is the ffted spectra
    and P is the filter. P is constructed by using the temporal covariance matrix.

    PARAMETERS:
    ----------------------
    d: numpy.ndarray contains the 2D traces of daily/hourly cross-correlation functions
    g: a positive number to adjust the filter harshness [default is 1]
    RETURNS:
    ----------------------
    narr: numpy vector contains the stacked cross correlation function
    '''
    if d.ndim == 1:
        print('2D matrix is needed')
        return d
    N,M = d.shape
    Nfft = next_fast_len(M)

    # fft the 2D array
    spec = scipy.fftpack.fft(d,axis=1,n=Nfft)[:,:M]

    # make cross-spectrm matrix
    cspec = np.zeros(shape=(N*N,M),dtype=np.complex64)
    for ii in range(N):
        for jj in range(N):
            kk = ii*N+jj
            cspec[kk] = spec[ii]*np.conjugate(spec[jj])

    S1 = np.zeros(M,dtype=np.complex64)
    S2 = np.zeros(M,dtype=np.complex64)
    # construct the filter P
    for ii in range(N):
        mm = ii*N+ii
        S2 += cspec[mm]
        for jj in range(N):
            kk = ii*N+jj
            S1 += cspec[kk]

    p = np.power((S1-S2)/(S2*(N-1)),g)

    # make ifft
    narr = np.real(scipy.fftpack.ifft(np.multiply(p,spec),Nfft,axis=1)[:,:M])
    newstack=np.mean(narr,axis=0)

    #
    return newstack

def pws(d,p=2):
    '''
    Performs phase-weighted stack on array of time series. Modified on the noise function by Tim Climents.
    Follows methods of Schimmel and Paulssen, 1997.
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    PARAMETERS:
    ---------------------
    d: N length array of time series data (numpy.ndarray)
    p: exponent for phase stack (int). default is 2

    RETURNS:
    ---------------------
    newstack: Phase weighted stack of time series data (numpy.ndarray)
    '''

    if d.ndim == 1:
        print('2D matrix is needed')
        return d
    N,M = d.shape
    analytic = hilbert(d,axis=1, N=next_fast_len(M))[:,:M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j*phase),axis=0)
    phase_stack = np.abs(phase_stack)**(p)

    weighted = np.multiply(d,phase_stack)

    newstack=np.mean(weighted,axis=0)
    return newstack

def nroot(d,p=2):
    '''
    this is nth-root stacking algorithm translated based on the matlab function
    from https://github.com/xtyangpsp/SeisStack (by Xiaotao Yang; follows the
    reference of Millet, F et al., 2019 JGR)

    Parameters:
    ------------
    d: numpy.ndarray contains the 2D cross correlation matrix
    p: np.int, nth root for the stacking. Default is 2.

    Returns:
    ------------
    newstack: np.ndarray, final stacked waveforms

    Written by Chengxin Jiang @ANU (May2020)
    '''
    if d.ndim == 1:
        print('2D matrix is needed for nroot_stack')
        return d
    N,M = d.shape
    dout = np.zeros(M,dtype=np.float32)

    # construct y
    for ii in range(N):
        dat = d[ii,:]
        dout += np.sign(dat)*np.abs(dat)**(1/p)
    dout /= N

    # the final stacked waveform
    newstack = dout*np.abs(dout)**(p-1)

    return newstack


def selective(d,cc_min,epsilon=1E-5,maxstep=10):
    '''
    this is a selective stacking algorithm developed by Jared Bryan/Kurama Okubo.

    PARAMETERS:
    ----------------------
    d: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    cc_min: numpy.float, threshold of correlation coefficient to be selected

    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation
    nstep: np.int, total number of iterations for the stacking

    Originally ritten by Marine Denolle
    Modified by Chengxin Jiang @Harvard (Oct2020)
    '''
    if d.ndim == 1:
        print('2D matrix is needed for selective stacking')
        return d
    N,M = d.shape

    res  = 9E9  # residuals
    cof  = np.zeros(N,dtype=np.float32)
    newstack = np.mean(d,axis=0)

    nstep = 0
    # start iteration
    while res>epsilon and nstep<=maxstep:
        for ii in range(N):
            cof[ii] = np.corrcoef(newstack, d[ii,:])[0, 1]

        # find good waveforms
        indx = np.where(cof>=cc_min)[0]
        if not len(indx): raise ValueError('cannot find good waveforms inside selective stacking')
        oldstack = newstack
        newstack = np.mean(d[indx],axis=0)
        res = np.linalg.norm(newstack-oldstack)/(np.linalg.norm(newstack)*M)
        nstep +=1

    return newstack, nstep
