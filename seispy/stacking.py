import os
import glob
import copy
import obspy
import scipy
import time
import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft,ifft,next_fast_len

def robust_stack(cc_array,epsilon=1E-5):
    """
    this is a robust stacking algorithm described in Palvis and Vernon 2010

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle
    Modified by Xiaotao Yang
    """
    res  = 9E9  # residuals
    w = np.ones(cc_array.shape[0])
    nstep=0
    newstack = np.median(cc_array,axis=0)
    while res > epsilon:
        stack = newstack
        for i in range(cc_array.shape[0]):
            crap = np.multiply(stack,cc_array[i,:].T)
            crap_dot = np.sum(crap)
            di_norm = np.linalg.norm(cc_array[i,:])
            ri = cc_array[i,:] -  crap_dot*stack
            ri_norm = np.linalg.norm(ri)
            w[i]  = np.abs(crap_dot) /di_norm/ri_norm#/len(cc_array[:,1])
        # print(w)
        w =w /np.sum(w)
        newstack =np.sum( (w*cc_array.T).T,axis=0)#/len(cc_array[:,1])
        res = np.linalg.norm(newstack-stack,ord=1)/np.linalg.norm(newstack)/len(cc_array[:,1])
        nstep +=1
        if nstep>10:
            return newstack, w, nstep
    return newstack, w, nstep


def adaptive_filter(arr,g):
    '''
    the adaptive covariance filter to enhance coherent signals. Fellows the method of
    Nakata et al., 2015 (Appendix B)

    the filtered signal [x1] is given by x1 = ifft(P*x1(w)) where x1 is the ffted spectra
    and P is the filter. P is constructed by using the temporal covariance matrix.

    PARAMETERS:
    ----------------------
    arr: numpy.ndarray contains the 2D traces of daily/hourly cross-correlation functions
    g: a positive number to adjust the filter harshness
    RETURNS:
    ----------------------
    narr: numpy vector contains the stacked cross correlation function
    '''
    if arr.ndim == 1:
        return arr
    N,M = arr.shape
    Nfft = next_fast_len(M)

    # fft the 2D array
    spec = scipy.fftpack.fft(arr,axis=1,n=Nfft)[:,:M]

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
    return np.mean(narr,axis=0)

def pws(arr,sampling_rate,power=2,pws_timegate=5.):
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
    arr: N length array of time series data (numpy.ndarray)
    sampling_rate: sampling rate of time series arr (int)
    power: exponent for phase stack (int)
    pws_timegate: number of seconds to smooth phase stack (float)

    RETURNS:
    ---------------------
    weighted: Phase weighted stack of time series data (numpy.ndarray)
    '''

    if arr.ndim == 1:
        return arr
    N,M = arr.shape
    analytic = hilbert(arr,axis=1, N=next_fast_len(M))[:,:M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j*phase),axis=0)
    phase_stack = np.abs(phase_stack)**(power)

    # smoothing
    #timegate_samples = int(pws_timegate * sampling_rate)
    #phase_stack = moving_ave(phase_stack,timegate_samples)
    weighted = np.multiply(arr,phase_stack)
    return np.mean(weighted,axis=0)


def nroot_stack(cc_array,power):
    '''
    this is nth-root stacking algorithm translated based on the matlab function
    from https://github.com/xtyangpsp/SeisStack (by Xiaotao Yang; follows the
    reference of Millet, F et al., 2019 JGR)

    Parameters:
    ------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    power: np.int, nth root for the stacking

    Returns:
    ------------
    nstack: np.ndarray, final stacked waveforms

    Written by Chengxin Jiang @ANU (May2020)
    '''
    if cc_array.ndim == 1:
        print('2D matrix is needed for nroot_stack')
        return cc_array
    N,M = cc_array.shape
    dout = np.zeros(M,dtype=np.float32)

    # construct y
    for ii in range(N):
        dat = cc_array[ii,:]
        dout += np.sign(dat)*np.abs(dat)**(1/power)
    dout /= N

    # the final stacked waveform
    nstack = dout*np.abs(dout)**(power-1)

    return nstack


def selective_stack(cc_array,epsilon,cc_th):
    '''
    this is a selective stacking algorithm developed by Jared Bryan/Kurama Okubo.

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    cc_th: numpy.float, threshold of correlation coefficient to be selected

    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation
    nstep: np.int, total number of iterations for the stacking

    Originally ritten by Marine Denolle
    Modified by Chengxin Jiang @Harvard (Oct2020)
    '''
    if cc_array.ndim == 1:
        print('2D matrix is needed for nroot_stack')
        return cc_array
    N,M = cc_array.shape

    res  = 9E9  # residuals
    cof  = np.zeros(N,dtype=np.float32)
    newstack = np.mean(cc_array,axis=0)

    nstep = 0
    # start iteration
    while res>epsilon:
        for ii in range(N):
            cof[ii] = np.corrcoef(newstack, cc_array[ii,:])[0, 1]

        # find good waveforms
        indx = np.where(cof>=cc_th)[0]
        if not len(indx): raise ValueError('cannot find good waveforms inside selective stacking')
        oldstack = newstack
        newstack = np.mean(cc_array[indx],axis=0)
        res = np.linalg.norm(newstack-oldstack)/(np.linalg.norm(newstack)*M)
        nstep +=1

    return newstack, nstep
