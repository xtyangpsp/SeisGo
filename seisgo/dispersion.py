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
def get_dispersion_image(data,t,dist,freq,velocity=[0.5,5],dv=0.1):
    """
    Produce the dispersion image with given data ensemble.

    ==PARAMETERS===
    data: 2-D matrix of the surface wave data.
    t: time vector for the data.
    dist: distance vector corresponding to the data matrix, must be in the
            same order as data.
    freq: [min,max] frequency for dispersion image.
    velocity: velocity range in [min,max] for the dispersion image. Default: [0.5,5]
    dv: velocity increment. Default: 0.1
    """
    print("Place holder function.")

    aout=[] #amplitude
    pout=[] # phase
    return aout,pout

# function to extract the dispersion from the image
# modified from NoisePy.
def extract_dispersion(amp,vel):
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
