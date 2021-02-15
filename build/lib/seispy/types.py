#define key classes
import numpy as np
from seispy.plotting import plot_corrdata
######
class Station(object):
    """
    Container for basic station information. Doesn't intend to replace the inventory class in ObsPy.

    Attributes
    -----------
    net: network name
    sta: station name
    loc: location code
    lon: longitude
    lat: latitude
    ele: elevation
    """
    def __init__(station, net=None,sta=None,loc=None,lon=None, lat=None, ele=None):
        station.net = net
        station.sta = sta
        station.loc = loc
        station.lon = lon
        station.lat = lat
        station.ele = ele

class CorrData(object):
    def __init__(self,net=None,sta=None,loc=None,chan=None,cc_comp=None,lag=None,dt=None,\
                    dist=None,ngood=None,time=None,data=None,substack:bool=False):
        self.net=net
        self.sta=sta
        self.loc=loc
        self.chan=chan
        self.cc_comp=cc_comp
        self.lag=lag
        self.dt=dt
        self.dist=dist
        self.ngood=ngood
        self.time=time
        self.data=data
        self.substack=substack
    def append(self,c):
        """
        Append will merge new object. The idea is to merge multiple sets of CorrData at
        different time chunks. Therefore, this function will merge the following attributes only:
        <ngood>,<time>,<data>

        **Note: substack will be set to True after merging, regardless the value in the original object.**
        """
        if not self.substack:
            self.ngood=np.reshape(self.ngood,(1,1))
            self.time=np.reshape(self.time,(1,1))
            self.data=np.reshape(self.data,(1,self.data.shape[0]))

        if not c.substack:
            c.ngood=np.reshape(c.ngood,(1,1))
            c.time=np.reshape(c.time,(1,1))
            c.data=np.reshape(c.data,(1,c.data.shape[0]))

        self.ngood =np.concatenate((self.ngood,c.ngood))
        self.time=np.concatenate((self.time,c.time))
        self.data=np.concatenate((self.data,c.data),axis=0)

        self.substack=True

    def plot(self,freqmin=None,freqmax=None,lag=None,save=False,figdir=None,figsize=(10,8)):
        """
        Plotting method for CorrData. It basically calls seispy.plotting.plot_corrdata(), with exactly the same arguments.
        """
        plot_corrdata(self,freqmin=freqmin,freqmax=freqmax,lag=lag,save=save,figsize=figsize,figdir=figdir)

class Power(object):
    """
    Container for power spectra for each component, with any shape

    Attributes
    ----------
    c11 : :class:`~numpy.ndarray`
        Power spectral density for component 1 (any shape)
    c22 : :class:`~numpy.ndarray`
        Power spectral density for component 2 (any shape)
    cZZ : :class:`~numpy.ndarray`
        Power spectral density for component Z (any shape)
    cPP : :class:`~numpy.ndarray`
        Power spectral density for component P (any shape)
    """

    def __init__(spectra, c11=None, c22=None, cZZ=None, cPP=None, window=None,
                overlap=None,freq=None):
        spectra.c11 = c11
        spectra.c22 = c22
        spectra.cZZ = cZZ
        spectra.cPP = cPP
        spectra.window = window
        spectra.overlap = overlap
        spectra.freq = freq


class Cross(object):
    """
    Container for cross-power spectra for each component pairs, with any shape

    Attributes
    ----------
    c12 : :class:`~numpy.ndarray`
        Cross-power spectral density for components 1 and 2 (any shape)
    c1Z : :class:`~numpy.ndarray`
        Cross-power spectral density for components 1 and Z (any shape)
    c1P : :class:`~numpy.ndarray`
        Cross-power spectral density for components 1 and P (any shape)
    c2Z : :class:`~numpy.ndarray`
        Cross-power spectral density for components 2 and Z (any shape)
    c2P : :class:`~numpy.ndarray`
        Cross-power spectral density for components 2 and P (any shape)
    cZP : :class:`~numpy.ndarray`
        Cross-power spectral density for components Z and P (any shape)
    """

    def __init__(spectra, c12=None, c1Z=None, c1P=None, c2Z=None, c2P=None,
                 cZP=None, window=None,overlap=None,freq=None):
        spectra.c12 = c12
        spectra.c1Z = c1Z
        spectra.c1P = c1P
        spectra.c2Z = c2Z
        spectra.c2P = c2P
        spectra.cZP = cZP
        spectra.window = window
        spectra.overlap = overlap
        spectra.freq = freq


class Rotation(object):
    """
    Container for rotated spectra, with any shape

    Attributes
    ----------
    cHH : :class:`~numpy.ndarray`
        Power spectral density for rotated horizontal component H (any shape)
    cHZ : :class:`~numpy.ndarray`
        Cross-power spectral density for components H and Z (any shape)
    cHP : :class:`~numpy.ndarray`
        Cross-power spectral density for components H and P (any shape)
    coh : :class:`~numpy.ndarray`
        Coherence between horizontal components
    ph : :class:`~numpy.ndarray`
        Phase of cross-power spectrum between horizontal components
    direc :: class: `~numpy.ndarray`
        All directions considered when computing the coh and ph.
    tilt : float
        Angle (azimuth) of tilt axis
    admt_value : : class :`~numpy.ndarray`
        Admittance between rotated horizontal at the tilt direction and vertical.
    coh_value : float
        Maximum coherence
    phase_value : float
        Phase at maximum coherence
    """

    def __init__(spectra, cHH=None, cHZ=None, cHP=None, coh=None, ph=None,direc=None,
                 tilt=None, admt_value=None,coh_value=None, phase_value=None,
                 window=None,overlap=None,freq=None):
        spectra.cHH = cHH
        spectra.cHZ = cHZ
        spectra.cHP = cHP
        spectra.coh = coh
        spectra.ph = ph
        spectra.direc = direc
        spectra.tilt = tilt
        spectra.admt_value = admt_value
        spectra.coh_value = coh_value
        spectra.phase_value = phase_value
        # spectra.angle = angle
        spectra.window = window
        spectra.overlap = overlap
        spectra.freq = freq
