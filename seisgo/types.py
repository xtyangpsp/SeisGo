#define key classes
import os,sys,pickle,obspy,scipy,pyasdf,h5py, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.core import Trace,Stream
from obspy.io.sac.sactrace import SACTrace
from obspy.signal.filter import bandpass,highpass,lowpass
from scipy.fftpack import fft,ifft,fftfreq,next_fast_len
from seisgo import utils,stacking,helpers
from obspy import UTCDateTime
from scipy import signal
from cartopy.io.img_tiles import GoogleTiles
######
class ShadedReliefESRI(GoogleTiles):
    #Modified from one of the answers on this page: https://stackoverflow.com/questions/37423997/cartopy-shaded-relief
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg').format(
               z=z, y=y, x=x)
        return url
class SeismicEngine(object):
    """
    Engine to interactively display time series data.
    """
    def __init__(self):
        self.type="Seismic Engine"

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
    def __init__(self, net=None,sta=None,loc=None,chan=None,lon=None, lat=None, ele=None):
        self.net = net
        self.sta = sta
        self.loc = loc
        self.chan = chan
        self.lon = lon
        self.lat = lat
        self.ele = ele

    def __repr__(self):
        """
        Display key content of the object.
        """
        lines = []
        lines.append("network      :   "+str(self.net))
        lines.append("station      :   "+str(self.sta))
        lines.append("location     :   "+str(self.loc))
        lines.append("channel      :   "+str(self.chan))
        lines.append("longitude   :   "+str(self.lon))
        lines.append("latitude    :   "+str(self.lat))
        lines.append("elevation   :   "+str(self.ele))

        lines.append("")

        return "<Station object>\n" + "\n".join(lines)

    __str__ = __repr__

class RawData(object):
    """
    Object to store seismic waveforms. When in three components, there is an option
    to do rotation from ENZ system to RTZ or LQT systems. The component labels will be
    renewed after rotation. This object is useful particularly in receiver function
    processing.
    """
    def __init__(self,trlist,stlo,stla,stel,stloc=None,stainv=None,evlo=None,evlat=None,evdp=None,evmag=None,evmagtype=None,
                    quake_ml=None,misc=dict()):
        """
        Initialize the object.

        trlist: a list of obspy.core.Trace object or a Stream object. Please make sure the list is for
                different channels when more than one trace in the list, NOT the segments with gaps for
                one station-channel pair.
        """
        if stainv is not None:
            self.sta,self.net,self.lon,self.lat,self.ele,self.loc = utils.sta_info_from_inv(stainv)
        elif None not in [stlo,stla,stel]:
            self.net=trlist[0].stats.network
            self.sta=trlist[0].stats.station
            self.stlo=stlo
            self.stla=stla
            self.stel=stel
            if stloc is None:
                self.stloc=''
            else: self.stloc=stloc

class RFData(object):
    """
    Reciever function data.
    """
    def __init__(self):
        self.type='Receiver Function Data'

class FFTData(object):
    """
    Object to store FFT data. The idea of having a FFTData data type
    was originally designed by Tim Clements for SeisNoise.jl (https://github.com/tclements/SeisNoise.jl).
    """
    def __init__(self,trace=None,win_len=None,step=None,stainv=None,
                id=None,net=None,sta=None,loc=None,chan=None,lon=None,lat=None,ele=None,
                dt=None,std=None,time=None,Nfft=None,data=None,
                 freqmin=None,freqmax=None,time_norm='no',freq_norm='no',smooth=20,
                 smooth_spec=None,misc=dict(),taper_frac=0.05,df=None):
        if trace is None:
            self.type='FFT Data'
            self.id=id
            self.net=net
            self.sta=sta
            self.loc=loc
            self.chan=chan
            self.lon=lon
            self.lat=lat
            self.ele=ele
            self.dt=dt
            self.freqmin=freqmin
            self.freqmax=freqmax
            self.time_norm=time_norm
            self.freq_norm=freq_norm
            self.df=df
            self.smooth=smooth
            self.smooth_spec=smooth_spec
            self.win_len=win_len
            self.step=step
            self.std=std
            self.time=time
            self.Nfft=Nfft
            self.misc=misc
            self.data=data
        else:
            self.construct(trace,win_len,step,stainv=stainv,
                         freqmin=freqmin,freqmax=freqmax,time_norm=time_norm,
                         freq_norm=freq_norm,smooth=smooth,
                         smooth_spec=smooth_spec,misc=misc,taper_frac=taper_frac,df=df)

    def construct(self,trace,win_len,step,stainv=None,
                     freqmin=None,freqmax=None,time_norm='no',freq_norm='no',smooth=20,
                     smooth_spec=None,misc=dict(),taper_frac=0.05,df=None):
        """
        Constructure the FFTData object. Will do whitening if specicied in freq_norm.

        trace: obspy.core.Trace or Stream object.
        """
        self.type='FFT Data'
        if isinstance(trace,Trace):trace=Stream([trace])

        if stainv is not None:
            self.sta,self.net,self.lon,self.lat,self.ele,self.loc = utils.sta_info_from_inv(stainv)
        else:
            self.net=trace[0].stats.network
            self.sta=trace[0].stats.station
            self.lon=0.0
            self.lat=0.0
            self.ele=0.0
            self.loc=''
        if isinstance(self.sta,list):self.sta=self.sta[0]
        if isinstance(self.net,list):self.net=self.net[0]
        if isinstance(self.lon,list):self.lon=self.lon[0]
        if isinstance(self.lat,list):self.lat=self.lat[0]
        if isinstance(self.ele,list):self.ele=self.ele[0]
        if isinstance(self.loc,list):self.loc=self.loc[0]

        self.chan=trace[0].stats.channel
        self.id=self.net+'.'+self.sta+'.'+self.loc+'.'+self.chan
        self.dt = 1/trace[0].stats.sampling_rate
        self.freqmin=freqmin
        self.freqmax=freqmax
        self.df = df
        if df is None and self.freqmin is not None:
            self.df = self.freqmin/4

        self.time_norm=time_norm
        self.freq_norm=freq_norm
        self.smooth=smooth
        if smooth_spec is None:
            self.smooth_spec=self.smooth
        else:
            self.smooth_spec=smooth_spec
        self.win_len=win_len
        self.step=step
        self.misc=misc

        fft_white=[]
        tr=trace[0].copy()
        if time_norm == 'ftn':
            if self.freqmin is not None:
                if self.freqmax is None:self.freqmax=0.499/self.dt
                tr.data=utils.ftn(trace[0].data,self.dt,self.freqmin,self.freqmax,df=self.df)
            else:
                raise ValueError("freqmin must be specified with ftn normalization.")
        # cut data into smaller segments (dataS always in 2D)
        trace_stdS,dataS_t,dataS = utils.slicing_trace([tr],win_len,step,
                                                        taper_frac=taper_frac)        # optimized version:3-4 times faster

        if len(dataS)>0:
            N=dataS.shape[0]
            self.std=trace_stdS
            self.time=dataS_t
            #------to normalize in time or not------
            if time_norm != 'no':
                if time_norm == 'one_bit': 	# sign normalization
                    white = np.sign(dataS)
                elif time_norm == 'rma': # running mean: normalization over smoothed absolute average
                    white = np.zeros(shape=dataS.shape,dtype=dataS.dtype)
                    for kkk in range(N):
                        white[kkk,:] = dataS[kkk,:]/utils.moving_ave(np.abs(dataS[kkk,:]),smooth)
                elif time_norm == 'ftn':
                    white = dataS
                else:
                    raise ValueError("The input "+time_norm+" is not recoganizable. "+
                            "Could only be: no, one_bit, ftn, or rma.")
            else:	# don't normalize
                white = dataS

            #-----to whiten or not------

            if white.ndim == 1:
                axis = 0
            elif white.ndim == 2:
                axis = 1

            Nfft = int(next_fast_len(int(dataS.shape[axis])))
            fft_white = fft(white, Nfft, axis=axis) # return FFT

            ##
            self.data=fft_white
            self.Nfft=Nfft

            if freq_norm != 'no' and freqmin is not None:
                print('Constructing FFTData with whitening ...')
                self.whiten()  # whiten and return FFT
        else:
            self.std=None
            self.time=None
            self.data=None
            self.Nfft=None

    ##### method for whitening
    def whiten(self,freq_norm=None,smooth=None):
        """
        Whiten FFTData
        """
        if freq_norm is None: freq_norm=self.freq_norm
        if smooth is None: smooth=self.smooth_spec
        if self.freqmin is None:
            raise ValueError('freqmin has to be specified as an attribute in FFTData!')

        if self.freqmax is None:
            self.freqmax=0.499/self.dt
            print('freqmax not specified, use default as 0.499*samp_freq.')

        if self.data.ndim == 1:
            axis = 0
        elif self.data.ndim == 2:
            axis = 1

        Nfft = int(self.Nfft)

        Napod = 100
        freqVec = fftfreq(Nfft, d=self.dt)[:Nfft // 2]
        J = np.where((freqVec >= self.freqmin) & (freqVec <= self.freqmax))[0]
        low = J[0] - Napod
        if low <= 0:
            low = 1

        left = J[0]
        right = J[-1]
        high = J[-1] + Napod
        if high > Nfft/2:
            high = int(Nfft//2)

        FFTRawSign = self.data
        # Left tapering:
        if axis == 1:
            FFTRawSign[:,0:low] *= 0
            FFTRawSign[:,low:left] = np.cos(
                np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
                1j * np.angle(FFTRawSign[:,low:left]))
            # Pass band:
            if freq_norm == 'phase_only':
                FFTRawSign[:,left:right] = np.exp(1j * np.angle(FFTRawSign[:,left:right]))
            elif freq_norm == 'rma':
                for ii in range(self.data.shape[0]):
                    tave = utils.moving_ave(np.abs(FFTRawSign[ii,left:right]),smooth)
                    FFTRawSign[ii,left:right] = FFTRawSign[ii,left:right]/tave
            # Right tapering:
            FFTRawSign[:,right:high] = np.cos(
                np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
                1j * np.angle(FFTRawSign[:,right:high]))
            FFTRawSign[:,high:Nfft//2] *= 0

            # Hermitian symmetry (because the input is real)
            FFTRawSign[:,-(Nfft//2)+1:] = np.flip(np.conj(FFTRawSign[:,1:(Nfft//2)]),axis=axis)
        else:
            FFTRawSign[0:low] *= 0
            FFTRawSign[low:left] = np.cos(
                np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
                1j * np.angle(FFTRawSign[low:left]))
            # Pass band:
            if freq_norm == 'phase_only':
                FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
            elif freq_norm == 'rma':
                tave = utils.moving_ave(np.abs(FFTRawSign[left:right]),smooth)
                FFTRawSign[left:right] = FFTRawSign[left:right]/tave
            # Right tapering:
            FFTRawSign[right:high] = np.cos(
                np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
                1j * np.angle(FFTRawSign[right:high]))
            FFTRawSign[high:Nfft//2] *= 0

            # Hermitian symmetry (because the input is real)
            FFTRawSign[-(Nfft//2)+1:] = FFTRawSign[1:(Nfft//2)].conjugate()[::-1]
        ##re-assign back to self.data.
        self.data=FFTRawSign

    def __repr__(self):
        """
        Display key content of the object.
        """
        lines = []
        lines.append("id           :   "+str(self.id))
        lines.append("net          :   "+str(self.net))
        lines.append("sta          :   "+str(self.sta))
        lines.append("loc          :   "+str(self.loc))
        lines.append("chan         :   "+str(self.chan))
        lines.append("lon          :   "+str(self.lon))
        lines.append("lat          :   "+str(self.lat))
        lines.append("ele          :   "+str(self.ele))
        lines.append("dt           :   "+str(self.dt))
        lines.append("freqmin      :   "+str(self.freqmin))
        lines.append("freqmax      :   "+str(self.freqmax))
        lines.append("time_norm    :   "+self.time_norm)
        lines.append("freq_norm    :   "+self.freq_norm)
        lines.append("smooth       :   "+str(self.smooth))
        lines.append("win_len      :   "+str(self.win_len))
        lines.append("step         :   "+str(self.step))
        if self.std is not None:
            lines.append("std          :   "+str(len(self.std)))
        else:
            lines.append("std          :   none")
        if self.time is not None and len(self.time)>0:
            lines.append("time         :   "+str(obspy.UTCDateTime(self.time[0]))+" to "+str(obspy.UTCDateTime(self.time[-1])))
        else:
            lines.append("time         :   none")
        lines.append("Nfft         :   "+str(self.Nfft))
        lines.append("misc         :   "+str(self.misc))
        if self.data is not None and len(self.data)>0:
            lines.append("data         :   "+str(self.data.shape))
        else:
            lines.append("data         :   none")
        lines.append("")
        return "<FFTData object>\n" + "\n".join(lines)

    __str__ = __repr__

    def __add__(f1,f2):
        """
        Merge two FFTData objects with the same id. Only merge [time],[std],[data] attributes.
        """
        if f1.id != f2.id:
            raise ValueError('The objects to be merged have different IDs (net.sta.loc.chan): %s and %s. Cannot merge!'%(f1.id,f2.id))

        time1=f1.time
        time2=f2.time
        std1=f1.std
        std2=f2.std
        data1=f1.data
        data2=f2.data

        time=np.concatenate((time1,time2))
        std=np.concatenate((std1,std2))
        data=np.concatenate((data1,data2),axis=0)

        return FFTData(win_len=f1.win_len,step=f1.step,id=f1.id,net=f1.net,
                        sta=f1.sta,loc=f1.loc,chan=f1.chan,lon=f1.lon,lat=f1.lat,ele=f1.ele,dt=f1.dt,
                        std=std,time=time,Nfft=f1.Nfft,data=data,freqmin=f1.freqmin,freqmax=f1.freqmax,
                        time_norm=f1.time_norm,freq_norm=f1.freq_norm,smooth=f1.smooth,
                        smooth_spec=f1.smooth_spec,misc=f1.misc,df=f1.df)
    def plot(self,xrange=None,time_format='%Y-%m-%dT%H',db=True,normalize=True,cmap='jet',figsize=(6,4),
            crange=None):
        """
        Plot amplitude spectrum of the FFTData.

        ====PARAMETERS====
        Default options:
        time_format='%Y-%m-%dT%H'
        db=True
        normalize=True
        cmap='jet'
        figsize=(6,4)
        crange: color range (default is None, automatically determined.)
        """
        ydata=self.time
        dt=self.dt
        Nfft2=int(self.Nfft/2)
        nwin=self.data.shape[0]
        if nwin>10:
            tick_inc = int(nwin/5)
        else:
            tick_inc = 2
        f=np.linspace(0,0.5/dt,Nfft2)
        if db:
            amp=10*np.log10(np.abs(self.data[:,:Nfft2]))
        else:
            amp=np.abs(self.data[:,:Nfft2])
        ampN=np.ndarray((amp.shape[0],amp.shape[1]))
        tmarks=[]
        for i in range(amp.shape[0]):
            if normalize: ampN[i,:]=amp[i,:]/np.max(np.abs(amp[i,:]))
            else: ampN[i,:]=amp[i,:]
            tmarks.append(obspy.UTCDateTime(ydata[i]).strftime(time_format))
        plt.figure(figsize=figsize,facecolor='w')
        ax=plt.subplot(111)
        plt.imshow(ampN,aspect='auto',extent=[f.min(),f.max(),ampN.shape[0],0],cmap=cmap)
        plt.xscale('log')
        if xrange is not None:
            plt.xlim(xrange)
        else:
            plt.xlim([f[1],f[-1]])
        plt.xlabel('frequency (Hz)', fontsize=12)
        plt.title('PSD for '+self.id,fontsize=13)
        ax.set_yticks(np.arange(0,nwin,tick_inc))
        ax.set_yticklabels(tmarks[0:nwin:tick_inc], fontsize=12)
        if crange is not None:plt.clim(crange)
        if normalize:
            if db:
                plt.colorbar(label='normalized amplitudes (dB)')
            else:
                plt.colorbar(label='normalized amplitudes')
        else:
            if db:
                plt.colorbar(label='amplitudes (dB)')
            else:
                plt.colorbar(label='amplitudes')
        plt.show()

class CorrData(object):
    """
    Object to store cross-correlation data. The idea of having a CorrData data type
    was originally designed by Tim Clements for SeisNoise.jl (https://github.com/tclements/SeisNoise.jl).
    The CorrData class in SeisGo differrs from that in SeisNoise by adding the internal methods
    for merging, plotting, and saving.
    ======= Attributes ======
    net=[None,None],sta=[None,None],loc=[None,None],chan=[None,None],lon=[None,None],
    lat=[None,None],ele=[None,None],cc_comp=None,
    lag=None,dt=None,dist=None,time=None,data=None,substack:bool=False
    cc_len,cc_step: cc parameters.
    az,baz: azimuth and back-azimuth of the two stations.
    side: A [Default]- both negative and positive sides, N - negative sides only, P - positive side only.
    misc=dict().

    misc is a dictionary that stores additional parameters.

    ======= Methods ======
    merge(): Merge with another object.
    to_sac(): convert and save to sac file, using obspy SACTrace object.
    plot(): simple plotting function to display the cross-correlation data.
    """
    def __init__(self,net=['',''],sta=['',''],loc=['',''],chan=['',''],\
                    lon=[0.0,0.0],lat=[0.0,0.0],ele=[0.0,0.0],cc_comp='',lag=0.0,\
                    dt=0.0,cc_len=None,cc_step=None,dist=0.0,az=0.0,baz=0.0,\
                    time=[],data=None,stack_method=None,substack:bool=False,side="A",misc=dict()):
        self.type='Correlation Data'
        self.id=net[0]+'.'+sta[0]+'.'+loc[0]+'.'+chan[0]+'_'+net[1]+'.'+sta[1]+'.'+loc[1]+'.'+chan[1]
        self.net=net
        self.sta=sta
        self.loc=loc
        self.chan=chan
        self.lon=lon
        self.lat=lat
        self.ele=ele
        if cc_comp is None:
            self.cc_comp=chan[0][-1]+chan[1][-1]
        else:
            self.cc_comp=cc_comp
        self.lag=lag
        self.dt=dt
        self.cc_len=cc_len
        self.cc_step=cc_step
        self.dist=dist
        self.az=az
        self.baz=baz
        self.time=time
        self.data=data
        self.stack_method=stack_method
        if side.lower() not in helpers.xcorr_sides():
            raise ValueError("Wrong side attribute value [%s], which has to be one of [%s]."%(side,str(helpers.xcorr_sides())))
        else:
            self.side=side
        self.substack=substack
        #if ndim is > 1, it means there might be only 1 trace but in the format of substack.
        if self.data is not None:
            if self.data.ndim > 1:
                self.substack=True
        self.misc=misc

    def __repr__(self):
        """
        Display key content of the object.
        """
        lines = []
        lines.append("type     :   "+str(self.type))
        lines.append("id       :   "+str(self.id))
        lines.append("net      :   "+str(self.net))
        lines.append("sta      :   "+str(self.sta))
        lines.append("loc      :   "+str(self.loc))
        lines.append("chan     :   "+str(self.chan))
        lines.append("lon      :   "+str(self.lon))
        lines.append("lat      :   "+str(self.lat))
        lines.append("ele      :   "+str(self.ele))
        lines.append("cc_comp  :   "+str(self.cc_comp))
        lines.append("lag      :   "+str(self.lag))
        lines.append("dt       :   "+str(self.dt))
        lines.append("cc_len   :   "+str(self.cc_len))
        lines.append("cc_step  :   "+str(self.cc_step))
        lines.append("dist     :   "+str(self.dist))
        lines.append("az       :   "+str(self.az))
        lines.append("baz      :   "+str(self.baz))
        lines.append("side     :   "+str(self.side))
        if self.time is not None and len(self.time)>0:
            if self.substack:
                lines.append("time     :   "+str(obspy.UTCDateTime(self.time[0]))+" to "+str(obspy.UTCDateTime(self.time[-1])))
            else:
                lines.append("time     :   "+str(obspy.UTCDateTime(self.time)))
        else:
            lines.append("time     :   none")
        lines.append("substack :   "+str(self.substack))
        if self.stack_method is not None:
            lines.append("stack_method:"+str(self.stack_method))
        if self.data is not None:
            lines.append("data     :   "+str(self.data.shape))
            lines.append(str(self.data))
        else:
            lines.append("data     :   none")
        lines.append("")

        return "<CorrData object>\n" + "\n".join(lines)

    __str__ = __repr__

    def __add__(c1,c2):
        """
        Merge with another object for the same station pair. The idea is to merge multiple sets
        of CorrData at different time chunks. Therefore, this function will merge the following
        attributes only: <time>,<data>

        **Note: substack will be set to True after merging, regardless the value in the original object.**
        """
        #sanity check: stop merging and raise error if the two objects have different IDs.
        if c1.id != c2.id:
            print("IDs: "+c1.id+" + "+c2.id)
            raise ValueError('The objects to be merged have different IDs (net.sta.loc.chan): %s and %s. Cannot merge!'%(c1.id,c2.id))
        if c1.side != c2.side:
            print("sides: "+c1.side+" + "+c2.side)
            raise ValueError('The object to be merged has a different side values. Cannot merge!')
        if not c1.substack:
            time1=np.reshape(c1.time,(1))
            data1=np.reshape(c1.data,(1,c1.data.shape[0]))
        else:
            time1=c1.time
            data1=c1.data
        if not c2.substack:
            time2=np.reshape(c2.time,(1))
            data2=np.reshape(c2.data,(1,c2.data.shape[0]))
        else:
            time2=c2.time
            data2=c2.data

        time=np.concatenate((time1,time2))
        data=np.concatenate((data1,data2),axis=0)

        cout=c1.copy(dataless=True)
        cout.time=time
        cout.substack=True
        cout.data=data

        return cout

    def merge(self,c,ignore_channel_type=False):
        """
        Merge with another object for the same station pair. The idea is to merge multiple sets
        of CorrData at different time chunks. Therefore, this function will merge the following
        attributes only: <time>,<data>

        **Note: substack will be set to True after merging, regardless the value in the original object.**

        ===PARAMETERS===
        c: the other CorrData object to merge with.
        ignore_channel_type: when True, only check the component but ignore the type of channels. For example,
            NET1.STA1.LOC1.BHZ_NET2.STA2.LOC2.BHZ can then be merged with NET1.STA1.LOC1.EHZ_NET2.STA2.LOC2.EHZ
            for the same station pair. The merged id will be tagged as NET1.STA1.LOC1.XXZ_NET2.STA2.LOC2.XXZ.
            If False, the channel and component have to be exactly match (e.g.,BHZ_BHZ and BHZ_BHZ). Default False.
        """
        #sanity check: stop merging and raise error if the two objects have different IDs.
        id_self=self.id
        id_c = c.id
        chanpair_self=self.chan[0]+'_'+self.chan[1]
        chanpair_c=c.chan[0]+'_'+c.chan[1]
        # print([chanpair_self, chanpair_c])
        if ignore_channel_type and chanpair_self != chanpair_c:
            print("Merging IDs with different channel types: "+id_self+" + "+id_c+'. Channel types are ignored.')
            id_self = self.net[0]+'.'+self.sta[0]+'.'+self.loc[0]+'.XX'+self.chan[0][2:]+'_'+\
                        self.net[1]+'.'+self.sta[1]+'.'+self.loc[1]+'.XX'+self.chan[1][2:]
            id_c = c.net[0]+'.'+c.sta[0]+'.'+c.loc[0]+'.XX'+c.chan[0][2:]+'_'+\
                        c.net[1]+'.'+c.sta[1]+'.'+c.loc[1]+'.XX'+c.chan[1][2:]
            #update IDs and chan information. 
            self.id = id_self
            # #update chan. This is required. Otherwise, the next round of merging will get different IDs even
            # if the channels might be the same.
            self.chan=['XX'+self.chan[0][2:],'XX'+self.chan[1][2:]]

        if id_self != id_c:
            print("IDs: "+id_self+" + "+id_c)
            raise ValueError('The objects to be merged have different IDs (net.sta.loc.chan): %s and %s. Cannot merge!'%(id_self,id_c))
        if not self.substack:
            stime=np.reshape(self.time.copy(),(1))
            sdata=np.reshape(self.data.copy(),(1,self.data.shape[0]))
        else:
            stime=self.time.copy()
            sdata=self.data.copy()
        if not c.substack:
            ctime=np.reshape(c.time.copy(),(1))
            if np.ndim(c.data)==1:
                cdata=np.reshape(c.data.copy(),(1,c.data.shape[0]))
        else:
            ctime=c.time.copy()
            cdata=c.data.copy()

        try:
            self.data=np.concatenate((sdata,cdata),axis=0)
            self.time=np.concatenate((stime,ctime)) #these two attributes need to be updated after the data attribute, which may throw errors.
            self.substack=True
        except Exception as e:
            print("error in merging. skipped.")
            print(e)

       

    #subset method
    def subset(self,starttime=None,endtime=None,overwrite=False):
        """
        Subset the xcorr data by time.
        starttime: Start time in string, with the format of "2021_09_05_0_0_0" or an obspy UTCDateTime object.
        endtime: End time in the same format as the "starttime"
        overwrite: overwrite the data or return the new subset CorrData object. Default: False.


        """
        if isinstance(starttime,str):
            sdatetime = obspy.UTCDateTime(starttime)
        else:
            sdatetime = starttime
        if isinstance(endtime,str):
            edatetime = obspy.UTCDateTime(endtime)
        else:
            edatetime = endtime
        if not self.substack:
            pass
        else:
            if sdatetime is None and edatetime is None:
                print("starttime and endtime are both None. Nothing to do with subset.")
            elif sdatetime is None:
                sdatetime=self.time[0]
            elif edatetime is None:
                edatetime=self.time[-1]
            idx=np.where((self.time >= sdatetime) & (self.time<= edatetime))[0]
            subtime=self.time[idx]
            subdata=self.data[idx,:]

            if overwrite:
                self.time=subtime
                self.data=subdata
            else:
                cdata=self.copy()
                cdata.time=subtime
                cdata.data=subdata
                return cdata

    #copy method.
    def copy(self,dataless=False):
        """
        This method returns a copy of the object.

        ====PARAMETER====
        dataless: only copies the metadata if True. Default is False.

        ====RETURN===
        cout: a copy of the object.
        """

        cout=CorrData(net=self.net,sta=self.sta,loc=self.loc,chan=self.chan,\
                        lon=self.lon,lat=self.lat,ele=self.ele,cc_comp=self.cc_comp,lag=self.lag,\
                        dt=self.dt,cc_len=self.cc_len,cc_step=self.cc_step,dist=self.dist,az=self.az,\
                        baz=self.baz,time=self.time.copy(),substack=self.substack,\
                        side=self.side,misc=self.misc)
        if not dataless:
            cout.data=self.data.copy()

        return cout

    def stack(self,win_len=None,method='linear',overwrite=True,ampcut=20,verbose=False,
                demean=True,stack_par=None):
        '''
        This function stacks the cross correlation data. It will overwrite the
        [data] attribute with the stacked trace, if overwrite is True. Substack will
        be set to False if win_len is None or there is only one trace left.

        PARAMETERS:
        ----------------------
        win_len: windown length in seconds for the substack, over which all the
                corrdata.data subset will be stacked. If None [default],it stacks
                all data into one single trace.
        method: stacking method, could be: linear, robust, pws, acf, or nroot.
        overwrite: if True, it replaces the data attribute in CorrData. Otherwise,
                    it returns the stacked data as a vector. Default: True.
        ampcut: used in QC, only stack traces that satisfy ampmax<ampcut*np.median(ampmax)).
                Default: 20. Use None to disable cutting by amplitudes.
        demean: demean before stacking. Default is True.
        stack_par: Defautl None. parameter dictionary to conduct stacking.

        RETURNS:
        -----------------------
        Only returns when overwrite is False.

        ds: stacked data.
        ts: timeflag of the substacks, only returns when win_len is NOT None.
        '''
        if isinstance(method,list):method=method[0]
        if win_len is None:
            if self.substack:
                if demean:
                    cc_temp = utils.demean(self.data)
                else:
                    cc_temp = self.data
                ampmax = np.max(cc_temp,axis=1)
                if ampcut is None:
                    tindx  = np.where( (ampmax<ampcut*np.median(ampmax)) & (ampmax>0))[0]
                else:
                    tindx  = np.where((np.abs(ampmax)>0))[0]
                nstacks=len(tindx)
                if nstacks >0:
                    cc_array = cc_temp[tindx,:]

                    # do stacking
                    ds = np.zeros((self.data.shape[1]),dtype=self.data.dtype)
                    if nstacks==1: ds=cc_array
                    else:
                        ds = stacking.seisstack(cc_array,method=method,par=stack_par)
                    if overwrite:
                        #overwrite the data attribute.
                        self.substack=False
                        self.time  = self.time[tindx[0]]
                        self.data=ds
                        self.stack_method=method
                    else:
                        return ds
                if verbose: print('stacked CorrData '+self.id+' with '+str(nstacks)+' traces.')
            else:
                print('substack is set to: False or has only 1 trace. No stacking applicable.')
                pass
        else: #### stacking over segments of time windows.
            if np.ndim(self.data)>1:
                if verbose: print('Stacking with given windown len %f'%(win_len))
                if self.time[-1] - self.time[0] >= win_len:
                    win=np.arange(self.time[0],self.time[-1]+0.5*win_len,win_len)  #all time chunks
                else:
                    win=np.array([self.time[0]])
                ts_temp=[]
                ds=np.ndarray((len(win),self.data.shape[1]),dtype=self.data.dtype)
                ds.fill(np.nan)
                ngood=[]
                if len(win) == 1:
                    nwin = 1
                else:
                    nwin = len(win) - 1
                for i in range(nwin):
                    widx=np.where((self.time>=win[i]) & (self.time<win[i]+win_len))[0]
                    if len(widx) >0:
                        if demean:
                            cc0 = utils.demean(self.data[widx,:])
                        else:
                            cc0 = self.data[widx,:]
                        ampmax = np.max(cc0,axis=1)
                        if ampcut is None:
                            tindx  = np.where( (ampmax<ampcut*np.median(ampmax)) & (ampmax>0))[0]
                        else:
                            tindx  = np.where((np.abs(ampmax)>0))[0]
                        nstacks=len(tindx)
                        dstack = np.zeros((self.data.shape[1]),dtype=self.data.dtype)
                        if nstacks>0:
                            cc_array = cc0[tindx,:]

                            # do stacking
                            if nstacks==1: dstack=cc_array[0, :]
                            else:
                                dstack = stacking.seisstack(cc_array,method=method,par=stack_par)

                            ds[i,:]=dstack
                            ngood.append(i)
                            ts_temp.append(win[i])

                #
                ts=np.array(ts_temp)
                ds=ds[ngood,:]

                if overwrite:
                    if len(ngood) == 1:
                        self.data = ds[0, :]
                        self.time = ts[0]
                        self.substack = False
                    else:
                        self.data = ds
                        self.time = ts
                        self.substack = True
                    self.stack_method=method
                else:
                    return ts,ds
                
            else:
                self.substack = False
                if overwrite: pass
                else:
                    return [],[]
    # stack positive and negative sides after splitting.
    def stack_sides(self,taper=True, taper_frac=0.01, taper_maxlen=10, overwrite=False,verbose=False,
                    demean=True,weighted=False):
        """
        This method stacks the positive and negative sides after splitting. It will overwrite the
        [data] attribute with the stacked trace, if overwrite is True. Substack will
        be set to False after stacking.

        PARAMETERS:
        taper: if True, applies taper to the data before stacking. Default True.
        taper_frac=0.01,taper_maxlen=10: taper parameters.
        overwrite: if True, it replaces the data attribute in CorrData. Otherwise,
                    it returns the stacked data as a vector. Default: False.
        demean: demean before stacking. Default is True.
        weighted: if True, stack with weights determined by the SNR (maximum_abs_amplitude/median_abs_amplitude) of each trace. 
                Default False. 

        RETURNS:
        Only returns when overwrite is False.
        ds: stacked data.
        """
        if verbose: print("Stacking the positive and negative sides after splitting.")
        if verbose and overwrite:
            print("overwrite is set to True. Stacked data will replace the data in the input CorrData object.")
        #check if side is 'a'. Only stack when side is 'a'. Otherwise, return the original data.
        try: #older version didn't have "side" attribute.
            side=self.side
        except Exception as e:           
            side="A"
        if side.lower()=="a":
            #call split method to split the positive and negative sides.
            cout=self.split(taper=taper,taper_frac=taper_frac,taper_maxlen=taper_maxlen,verbose=verbose)
            if len(cout) != 2:
                print("The split method did not return two sides. No stacking applied.")
                return self.data
            else:
                c_n=cout[0]
                c_p=cout[1]
                if demean:
                    c_n.data=utils.demean(c_n.data)
                    c_p.data=utils.demean(c_p.data)
                if weighted:
                    w_n=np.max(np.abs(c_n.data))/np.median(np.abs(c_n.data))
                    w_p=np.max(np.abs(c_p.data))/np.median(np.abs(c_p.data))
                    ds=(c_n.data*w_n+c_p.data*w_p)/(w_n+w_p)
                else:
                    ds=(c_n.data+c_p.data)/2

                if overwrite:
                    self.data=ds
                    self.side="o"  #o for one-sided CorrData.
                else:
                    cout_oneside=self.copy()
                    cout_oneside.data=ds
                    cout_oneside.side="o"
                    return cout_oneside
        else:
            print("side attribute is %s. Only stacks when side is A."%(self.side))
            return self.data
        
    #split the negative and positive sides
    def split(self,taper=False,taper_frac=0.01,taper_maxlen=10,verbose=False):
        """
        This method splits the positive and negative sides of the <data> attribute in CorrData object.
        This method will assign the <side> attribute for each side.

        ========PARAMETERS===========
        taper: if True, applies taper to the data after splitting. Default False.
        taper_frac=0.01,taper_maxlen=10: taper parameters.

        ========RETURNS==============
        cout: the list of two CorrData objects.
        """
        cout=[]
        try: #older version didn't have "side" attribute.
            side=self.side
        except Exception as e:
            side="A"
        if side.lower()=="a":
            if verbose: print("Splitting negative and positive sides.")
        else:
            print("side attribute is %s. Only splits when side is A."%(self.side))
            return cout

        dt=self.dt
        #
        #initiate as zeros

        if self.substack:
            nhalfpoint=int(self.data.shape[1]/2)

            d_p=np.zeros((nhalfpoint+1),dtype=self.data.dtype)
            d_n=np.zeros((nhalfpoint+1),dtype=self.data.dtype)

            if taper:
                d_p=utils.taper(self.data[:,nhalfpoint:],
                                            fraction=taper_frac,maxlen=taper_maxlen)
                d_n=np.flip(utils.taper(self.data[:,:nhalfpoint+1],
                                            fraction=taper_frac,maxlen=taper_maxlen),axis=1)
            else:
                d_p=self.data[:,nhalfpoint:]
                d_n=np.flip(self.data[:,:nhalfpoint+1],axis=1)
        else:
            nhalfpoint=int(self.data.shape[0]/2)
            d_p=np.zeros((nhalfpoint+1),dtype=self.data.dtype)
            d_n=np.zeros((nhalfpoint+1),dtype=self.data.dtype)

            if taper:
                d_p=utils.taper(self.data[nhalfpoint:],
                                            fraction=taper_frac,maxlen=taper_maxlen)
                d_n=np.flip(utils.taper(self.data[:nhalfpoint+1],
                                            fraction=taper_frac,maxlen=taper_maxlen))
            else:
                d_p=self.data[nhalfpoint:]
                d_n=np.flip(self.data[:nhalfpoint+1])

        c_n=self.copy()
        c_n.side="N"
        c_n.data=d_n
        cout.append(c_n)

        c_p=self.copy()
        c_p.side="P"
        c_p.data=d_p
        cout.append(c_p)

        return cout
    #shaping
    def shaping(self,width,shift,wavelet='gaussian',overwrite=True,trim_end=False):
        """
        convolve with a shaping wavelet.

        ====PARAMETERS====
        width: if gaussian, sigma of the shaping wavelet. if ricker: distance
            between the two side lobes.
        shift: half length of the wavelet. This will determine the shift of
                the wavelet center.
        wavelet: type of wavelet. default gaussian. Options: gaussian or ricker.
        trim_end: trim the end of the result. otherwise, trim both start and end.
        """
        if wavelet.lower() not in helpers.wavelet_labels():
            raise ValueError(wavelet+" not supported.")
        dt=self.dt
        if wavelet.lower() == "gaussian":
            t,w=utils.gaussian(dt,width,shift)
        elif wavelet.lower() == "ricker":
            t,w=utils.ricker(dt,1/width,shift)
        else:
            raise ValueError(wavelet+" not supported.")
        nt=len(t)
        dout=np.ndarray(self.data.shape)

        if self.substack:
            npts=self.data.shape[1]
            for ii in range(self.data.shape[0]):
                dtemp=signal.convolve(self.data[ii],w)
                if trim_end:
                    dout[ii]=dtemp[:npts]
                else:
                    dout[ii]=dtemp[int(nt/2):int(nt/2)+npts]
        else:
            npts=self.data.shape[0]
            dtemp=signal.convolve(self.data,w)
            if trim_end:
                dout=dtemp[:npts]
            else:
                dout=dtemp[int(nt/2):int(nt/2)+npts]

        if not overwrite:
            cdataout=self.copy()
            cdataout.data=dout
            return cdataout
        else:
            self.data=dout
    #convert to EGF by taking the netagive time derivative of the noise correlation functions.
    def to_egf(self,taper_frac=0.01,taper_maxlen=10,verbose=False):
        """
        This function converts the CorrData correlaiton results to EGF by taking
        the netagive time derivative of the noise correlation functions.

        The positive and negative lags are converted seperatedly but merged afterward.

        =======PARAMETERS=========
        taper_frac: default 0.01. taper fraction when process the two sides seperatedly.
        taper_maxlen: default 10. taper maximum number of points.
        """
        if verbose: print("Converting to empirical Green's functions.")
        #type flag here
        egf_flag="Empirical Green's Functions"
        if self.type.lower() == egf_flag.lower():
            print("It seems the data is already EGFs. self.type="+self.type+". Skip without converting!")
            pass
        else:
            dt=self.dt
            try:
                side=self.side
            except Exception as e:
                side="A"
            #
            #initiate as zeros
            egf=np.zeros(self.data.shape,dtype=self.data.dtype)
            if self.substack:
                if side.lower()=="a":
                    nhalfpoint=int(self.data.shape[1]/2)
                    #positive side
                    egf[:,nhalfpoint:]=utils.taper(-1.0*np.gradient(self.data[:,nhalfpoint:],axis=1)/dt,
                                                    fraction=taper_frac,maxlen=taper_maxlen)
                    #negative side
                    egf[:,:nhalfpoint+1]=np.flip(utils.taper(-1.0*np.gradient(np.flip(self.data[:,:nhalfpoint+1],axis=1),
                                                    axis=1)/dt,fraction=taper_frac,maxlen=taper_maxlen),axis=1)
                    egf[:,[0,-1]]=0
                    egf[:,nhalfpoint]=np.mean(egf[:,nhalfpoint-1:nhalfpoint+1],axis=1)
                else:
                    egf=utils.taper(-1.0*np.gradient(self.data,axis=1)/dt,
                                                    fraction=taper_frac,maxlen=taper_maxlen)
            else:
                if side.lower()=="a":
                    nhalfpoint=int(self.data.shape[0]/2)
                    #positive side
                    egf[nhalfpoint:]=utils.taper(-1.0*np.gradient(self.data[nhalfpoint:])/dt,
                                                fraction=taper_frac,maxlen=taper_maxlen)
                    #negative side
                    egf[:nhalfpoint+1]=np.flip(utils.taper(-1.0*np.gradient(np.flip(self.data[:nhalfpoint+1]))/dt,
                                                fraction=taper_frac,maxlen=taper_maxlen))
                    egf[[0,-1]]=0
                    egf[nhalfpoint]=np.mean(egf[nhalfpoint-1:nhalfpoint+1])
                else:
                    egf=utils.taper(-1.0*np.gradient(self.data)/dt,
                                                fraction=taper_frac,maxlen=taper_maxlen)
            self.data=egf
            self.type=egf_flag

    #
    def filter(self,fmin=None,fmax=None,corners=4,zerophase=True):
        """
        Apply filter to CorrData.data. The parameters are same as for obspy.signal.filter filters.

        ==PARAMETERS==
        fmin, fmax: frequency range. if fmin is None, it will apply a lowpass filter.
                    if fmax is None, it will apply a highpass filter.
        corners: number of corners, default is 4.
        zerophase: default is True.
        """
        if fmin is None and fmax is None:
            raise ValueError("fmin and fmax CAN NOT all be None.")
        if self.substack:
            for i in range(self.data.shape[0]):
                if fmin is not None and fmax is not None:
                    self.data[i,:]=bandpass(self.data[i],fmin,fmax,1/self.dt,corners=corners, zerophase=zerophase)
                elif fmin is None:
                    self.data[i,:]=lowpass(self.data[i],fmax,1/self.dt,corners=corners, zerophase=zerophase)
                elif fmax is None:
                    self.data[i,:]=highpass(self.data[i],fmin,1/self.dt,corners=corners, zerophase=zerophase)
        else:
            if fmin is not None and fmax is not None:
                self.data=bandpass(self.data,fmin,fmax,1/self.dt,corners=corners, zerophase=zerophase)
            elif fmin is None:
                self.data=lowpass(self.data,fmax,1/self.dt,corners=corners, zerophase=zerophase)
            elif fmax is None:
                self.data=highpass(self.data,fmin,1/self.dt,corners=corners, zerophase=zerophase)
#
    def save(self,format,file=None,outdir=None,v=True):
        """
        Wrapper to save CorrData to file.
        format: required from user. "sac" or "asdf".
        file: filename. Will automatically determine one if None.
        outdir: output directory. Default None.
        v: verbose. Default True.
        """
        if format.lower() == "sac":
            self.to_sac(file=file,outdir=outdir,v=v)
        elif format.lower() == "asdf":
            self.to_asdf(file=file,outdir=outdir,v=v)
#
    def to_asdf(self,file=None,outdir=None,v=True):
        """
        Save CorrData object to asdf file.
        file: file name, which is required.
        """
        cc_comp = self.cc_comp
        # source-receiver pair
        netsta_pair = self.net[0]+'.'+self.sta[0]+'_'+\
                        self.net[1]+'.'+self.sta[1]
        chan_pair = self.chan[0]+'_'+self.chan[1]

        #save to asdf
        lonS,lonR = self.lon
        latS,latR = self.lat
        eleS,eleR = self.ele

        if "cc_method" in list(self.misc.keys()):
            cc_method = self.misc['cc_method']
        else:
            cc_method = ''
        if "dist_unit" in list(self.misc.keys()):
            dist_unit=self.misc['dist_unit']
        else:
            dist_unit=''
        parameters = {
            'net':self.net,
            'sta':self.sta,
            'chan':self.chan,
            'loc':self.loc,
            'dt':self.dt,
            'maxlag':np.float32(self.lag),
            'dist':np.float32(self.dist),
            'dist_unit':dist_unit,
            'azi':np.float32(self.az),
            'baz':np.float32(self.baz),
            'lonS':np.float32(lonS),
            'latS':np.float32(latS),
            'eleS':np.float32(eleS),
            'lonR':np.float32(lonR),
            'latR':np.float32(latR),
            'eleR':np.float32(eleR),
            'cc_method':cc_method,
            'cc_len':self.cc_len,
            'cc_step':self.cc_step,
            'time':self.time,
            'substack':self.substack,
            'comp':self.cc_comp,
            'type':self.type,
            'side':self.side,
            'stack_method':str(self.stack_method)}

        #check time size to avoid error. make sure it is not > 64kb
        #this is a temporary fix, though the ultimate fix will rely on HDF to lift the limit.
        if sys.getsizeof(self.time)/1024 > 64: #64k is the limit of HDF attribute.
            parameters['time']=np.float32(self.time-np.mean(self.time))
            parameters['time_mean']=np.mean(self.time)
            if sys.getsizeof(parameters['time'])/1024 > 64:
                print('Warning: Even after getting the variations, the time attribute might be still too large to save as ')
                print('the ASDF header [%fkb'%(sys.getsizeof(parameters['time'])/1024))

        #
        if file is None:
            if not self.substack:
                corrtime=obspy.UTCDateTime(self.time)
            else:
                corrtime=obspy.UTCDateTime(self.time[0])
            file=str(corrtime).replace(':', '-')+'_'+self.id+'_'+self.cc_comp+'_'+self.side+'.h5'
            if outdir is None:
                outdir="."
            file=os.path.join(outdir,file)
        elif outdir is not None:
            file=os.path.join(outdir,file)

        fhead=os.path.split(file)[0]
        if len(fhead) >0 and not os.path.isdir(fhead): os.makedirs(fhead,exist_ok = True)

        with pyasdf.ASDFDataSet(file,mpi=False) as ccf_ds:
            ccf_ds.add_auxiliary_data(data=self.data, data_type=netsta_pair, path=chan_pair, parameters=parameters)
        if v: print('CorrData saved to: '+file)

    def to_sac(self,file=None,outdir=None,v=True):
        """
        Save CorrData object to sac file.

        ====PARAMETERS====
        outdir: output file directory. default is the current folder.
        file: specify file name, ONLY when there is only one trace. i.e., substack is False.
        v: verbose, default is True.
        """
        if outdir is None:
            outdir="."
        try:
            if not os.path.isdir(outdir):os.makedirs(outdir,exist_ok = True)
        except Exception as e:
            print(e)

        try:
            side=self.side
        except Exception as e:
            side="A"
        slon,rlon=self.lon
        slat,rlat=self.lat
        sele,rele=self.ele
        if side.lower()=="a":
            b=-self.lag
        else:
            b=0.0
        #
        network=self.net[1] #network for receiver.
        station=self.sta[1]
        evname=self.net[0]+"."+self.sta[0]
        comp = self.cc_comp
        if not self.substack:
            corrtime=obspy.UTCDateTime(self.time)
            nzyear=corrtime.year
            nzjday=corrtime.julday
            nzhour=corrtime.hour
            nzmin=corrtime.minute
            nzsec=corrtime.second
            nzmsec=corrtime.microsecond

            if file is None:
                file=str(corrtime).replace(':', '-')+'_'+self.id+'_'+self.cc_comp+'_'+side+'.sac'
            sac = SACTrace(nzyear=nzyear,nzjday=nzjday,nzhour=nzhour,nzmin=nzmin,nzsec=nzsec,nzmsec=nzmsec,
                           b=b,delta=self.dt,stla=rlat,stlo=rlon,stel=sele,evla=slat,evlo=slon,evdp=rele,
                           evel=rele,dist=self.dist,az=self.az,baz=self.baz,data=self.data,
                           kevnm=evname,knetwk=network,kstnm=station,kcmpnm=comp)

            sacfile  = os.path.join(outdir,file)
            sac.write(sacfile,byteorder='big')
            if v: print('saved sac to: '+sacfile)
        else:
            nwin=self.data.shape[0]
            for i in range(nwin):
                corrtime=obspy.UTCDateTime(self.time[i])
                nzyear=corrtime.year
                nzjday=corrtime.julday
                nzhour=corrtime.hour
                nzmin=corrtime.minute
                nzsec=corrtime.second
                nzmsec=corrtime.microsecond
                if file is None:
                    ofile=str(corrtime).replace(':', '-')+'_'+self.id+'_'+self.cc_comp+'_'+side+'.sac'
                    sacfile  = os.path.join(outdir,ofile)
                else:
                    sacfile  = os.path.join(outdir,file)
                sac = SACTrace(nzyear=nzyear,nzjday=nzjday,nzhour=nzhour,nzmin=nzmin,nzsec=nzsec,nzmsec=nzmsec,
                               b=b,delta=self.dt,stla=rlat,stlo=rlon,stel=sele,evla=slat,evlo=slon,evdp=rele,
                               evel=rele,dist=self.dist,az=self.az,baz=self.baz,data=self.data[i,:],
                               kevnm=evname,knetwk=network,kstnm=station,kcmpnm=comp)

                sac.write(sacfile,byteorder='big')
                if v: print('saved sac to: '+sacfile)

    def plot(self,freqmin=None,freqmax=None,lag=None,save=False,figdir=None,figsize=(10,8),
            figname=None,format='png',stack_method='linear',get_stack=False,stack_par=None,
            time_format='%Y-%m-%dT%H:%M:%S'):
        """
        Plotting method for CorrData. It is the same as seisgo.plotting.plot_corrdata(), with exactly the same arguments.
        Display the 2D matrix of the cross-correlation functions for a certain time-chunck.
        PARAMETERS:
        --------------------------
        freqmin: min frequency to be filtered
        freqmax: max frequency to be filtered
        lag: time ranges for display
        save: Save figure, default is False
        figdir: only applies when save is True.
        figsize: Matplotlib figsize, default is (10,8).
        format: figure format when saving. default png. Use pyplot's convention.
        stack_method: method to get the stack, default is 'linear'
        get_stack: returns the sacked trace if True. Default is False.
        stack_par: dictionary to store stacking parameters. Default None.
        time_format: format when labeling the individual traces. Default: '%Y-%m-%dT%H:%M:%S'
        """
        # open data for read
        if save:
            if figdir==None:print('no path selected! save figures in the default path')

        netstachan1 = self.net[0]+'.'+self.sta[0]+'.'+self.loc[0]+'.'+self.chan[0]
        netstachan2 = self.net[1]+'.'+self.sta[1]+'.'+self.loc[1]+'.'+self.chan[1]

        dt,maxlag,dist,ttime,substack = [self.dt,self.lag,self.dist,\
                                                self.time,self.substack]
        try:
            side=self.side
        except Exception as e:
            side="A"

        dreturn=[]
       # lags for display
        if not lag:lag=maxlag
        if lag>maxlag:raise ValueError('lag excceds maxlag!')
        lag0=np.min([1.0*lag,maxlag])

        # t is the time labels for plotting
        if lag>=5:
            tstep=int(int(lag)/5)
            if side.lower()=="a":
                t1=np.arange(-int(lag),0,step=tstep);t2=np.arange(0,int(lag+0.5*tstep),step=tstep)
                t=np.concatenate((t1,t2))
            else:
                t=np.arange(0,int(lag+0.5*tstep),step=tstep)
        else:
            tstep=lag/5
            if side.lower()=="a":
                t1=np.arange(-lag,0,step=tstep);t2=np.arange(0,lag+0.5*tstep,step=tstep)
                t=np.concatenate((t1,t2))
            else:
                t=np.arange(0,lag+0.5*tstep,step=tstep)

        if side.lower()=="a":
            indx1 = int((maxlag-lag0)/dt);indx2 = indx1+2*int(lag0/dt)+1
        else:
            indx1 = 0
            indx2 = int(lag0/dt)+1


        # cc matrix
        if substack:
            data = np.ndarray.copy(self.data[:,indx1:indx2])
            meanall=np.mean(np.abs(data))
            timestamp = np.empty(ttime.size,dtype='datetime64[s]')
            # print(data.shape)
            nwin = data.shape[0]
            amax = np.zeros(nwin,dtype=np.float32)
            if nwin==0:
                print('continue! no enough trace to plot!')
                return

            tmarks = []
            data_normalizd=np.zeros(data.shape)
            data_normalizd.fill(np.nan)
            # load cc for each station-pair
            for ii in range(nwin):
                if freqmin is not None and freqmax is not None:
                    data[ii] = bandpass(data[ii],freqmin,freqmax,1/dt,corners=4, zerophase=True)
                data[ii] = utils.taper(data[ii]-np.mean(data[ii]),maxlen=10)
                amax[ii] = np.max(np.abs(data[ii]))
                timestamp[ii] = obspy.UTCDateTime(ttime[ii])
                tmarks.append(obspy.UTCDateTime(ttime[ii]).strftime(time_format))
                if np.isnan(data[ii]).any() or amax[ii] < meanall/100000:continue
                data_normalizd[ii] = data[ii]/amax[ii]

            dstack = stacking.seisstack(data,method=stack_method,par=stack_par)
            del data
    #         dstack_robust=stack.robust(data)[0]

            # plotting
            if nwin>10:
                tick_inc = int(nwin/5)
            else:
                tick_inc = 2

            fig = plt.figure(figsize=figsize,facecolor='w')
            ax = fig.add_subplot(6,1,(1,4))
            if side.lower()=="a":
                extent=[-lag0,lag0,nwin,0]
            else:
                extent=[0,lag0,nwin,0]
            ax.imshow(data_normalizd,cmap='seismic',extent=extent,aspect='auto')
            ax.plot((0,0),(nwin,0),'k-')
            if freqmin is not None and freqmax is not None:
                ax.set_title('%s-%s: dist=%5.2f km: %4.2f-%4.2f Hz: %s' % (netstachan1,netstachan2,
                                                                           dist,freqmin,freqmax,side))
            else:
                ax.set_title('%s-%s: dist=%5.2f km: unfiltered: %s' % (netstachan1,netstachan2,dist,side))
            ax.set_xlabel('time [s]')
            ax.set_xticks(t)
            ax.set_yticks(np.arange(0,nwin,step=tick_inc))
            ax.set_yticklabels(tmarks[0:nwin:tick_inc])
            if side.lower()=="a":
                ax.set_xlim([-lag,lag])
            else:
                ax.set_xlim([0,lag])
            ax.xaxis.set_ticks_position('bottom')

            ax1 = fig.add_subplot(6,1,(5,6))
            if freqmin is not None and freqmax is not None:
                ax1.set_title('stack at %4.2f-%4.2f Hz: %s'%(freqmin,freqmax,side))
            else:
                ax1.set_title('stack: unfiltered: %s'%(side))
            if side.lower()=="a":
                tstack=np.arange(-lag0,lag0+0.5*dt,dt)
            else:
                tstack=np.arange(0,lag0+0.5*dt,dt)
            if len(tstack)>len(dstack):tstack=tstack[:-2]
            ax1.plot(tstack,dstack,'b-',linewidth=1,label=stack_method)
    #         ax1.plot(tstack,dstack_robust,'r-',linewidth=1,label='robust')
            ax1.set_xlabel('time [s]')
            ax1.set_xticks(t)
            if side.lower()=="a":
                ax1.set_xlim([-lag,lag])
            else:
                ax1.set_xlim([0,lag])
            ylim=ax1.get_ylim()
            ax1.plot((0,0),ylim,'k-')

            ax1.set_ylim(ylim)
            ax1.legend(loc='upper right')
            ax1.grid()

            fig.tight_layout()

            dreturn=dstack

            tmark_figname=obspy.UTCDateTime(ttime[0]).strftime('%Y-%m-%dT%H-%M-%S')
        else: #only one trace available
            data = np.ndarray.copy(self.data[indx1:indx2])

            # load cc for each station-pair
            if freqmin is not None and freqmax is not None:
                data = bandpass(data,freqmin,freqmax,1/dt,corners=4, zerophase=True)
            data = utils.taper(data-np.mean(data),maxlen=10)
            amax = np.max(np.abs(data))
            data /= amax
            timestamp = obspy.UTCDateTime(ttime)
            tmarks=obspy.UTCDateTime(ttime).strftime(time_format)

            if side.lower()=="a":
                tx=np.arange(-lag0,lag0+0.5*dt,dt)
            else:
                tx=np.arange(0,lag0+0.5*dt,dt)
            if len(tx)>len(data):tx=tx[:-1]
            plt.figure(figsize=figsize,facecolor='w')
            ax=plt.gca()
            plt.plot(tx,data,'k-',linewidth=1)
            if freqmin is not None and freqmax is not None:
                plt.title('%s-%s: dist=%5.2f km: %4.2f-%4.2f Hz: %s: %s' % (netstachan1,netstachan2,
                                                                           dist,freqmin,freqmax,tmarks,side))
            else:
                plt.title('%s-%s: dist=%5.2f km: unfiltered: %s: %s' % (netstachan1,netstachan2,dist,tmarks,side))
            plt.xlabel('time [s]')
            plt.xticks(t)
            ylim=ax.get_ylim()
            plt.plot((0,0),ylim,'k-')

            plt.ylim(ylim)
            if side.lower()=="a":
                plt.xlim([-lag,lag])
            else:
                plt.xlim([0,lag])
            ax.grid()

            dreturn=data
            tstack=tx
            tmark_figname=obspy.UTCDateTime(ttime).strftime('%Y-%m-%dT%H-%M-%S')

        # save figure or just show
        if save:
            if figdir==None:figdir = '.'
            if not os.path.isdir(figdir):os.mkdir(figdir)
            if figname is None:
                outfname = figdir+\
                '/{0:s}_{1:s}_{2:s}-{3:s}Hz-{4:s}.{5:s}'.format(netstachan1,netstachan2,
                                                                 str(freqmin),str(freqmax),
                                                                 tmark_figname,format)
            else:
                outfname = figdir+'/'+figname
            plt.savefig(outfname, format=format, dpi=300)
            print('saved to: '+outfname)
            plt.close()
        else:
            plt.show()

        ##
        if get_stack:
            return tstack,dreturn

    ####
    def psd(self,cmap='jet',xrange=None,time_format='%Y-%m-%dT%H',normalize=True,figsize=(13,5)):
        """
        Plot the power specctral density of corrdata.data.

        =PARAMETERS=
        cmap: colormap, default is 'jet'
        time_format: format to show time marks, default is: '%Y-%m-%dT%H'
        normalize: whether normalize the PSD in plotting, default is True
        figsize: figure size, default: (13,5)
        """
        dt=self.dt
        if self.side.lower() == 'a':
            cdatan,cdatap=self.split()
            nplot=2
        elif self.side.lower() =='n' or self.side.lower() =='o':
            cdatan=self.copy()
            cdatap=None
            nplot=1
        else:
            cdatap=self.copy()
            cdatan=None
            nplot=1

        plt.figure(figsize=figsize,facecolor='w')
        cdata_all=[cdatan,cdatap]
        for ii,cdata in enumerate(cdata_all):
            if cdata is not None:
                if nplot==1:ax=plt.subplot(1,nplot,1)
                else:ax=plt.subplot(1,nplot,ii+1)

                data=cdata.data
                ydata=cdata.time
                nwin=data.shape[0]
                if nwin>10:
                    tick_inc = int(nwin/5)
                else:
                    tick_inc = 2
                f,p=utils.psd(data,1/dt)
                psdN=np.ndarray((p.shape[0],p.shape[1]))
                tmarks=[]
                for i in range(p.shape[0]):
                    if normalize: psdN[i,:]=p[i,:]/np.max(np.abs(p[i,:]))
                    else: psdN[i,:]=p[i,:]
                    tmarks.append(obspy.UTCDateTime(ydata[i]).strftime(time_format))

                plt.imshow(psdN,aspect='auto',extent=[f.min(),f.max(),psdN.shape[0],0],cmap=cmap)
                # plt.yscale('log')
                if xrange is None:plt.xlim([f[1],f[-1]])
                else:
                    plt.xlim(xrange)
                plt.xscale('log')
                ax.set_yticks(np.arange(0,nwin,step=tick_inc))
                ax.set_yticklabels(tmarks[0:nwin:tick_inc])
                if normalize: plt.colorbar(label='normalized PSD')
                else: plt.colorbar(label='PSD')
                plt.xlabel('frequency (Hz)')
                plt.title('PSD:'+cdata.id+':'+str(round(cdata.dist,2))+' km:'+cdata.side)

        plt.tight_layout()
        plt.show()
class CorrDataEnsemble(object):
    """
    Object to store cross-correlation data ensemble, from the same virtual source. This is used to make it
    easy to plot moveout data and other operations for a whole gather of CorrData. 

    ======= Attributes ======
    net=[None,None],sta=[None,None],loc=[None,None],chan=[None,None],lon=[None,None],
    lat=[None,None],ele=[None,None],cc_comp=None,
    lag=None,dt=None,dist=None,time=None,data=None,
    cc_len,cc_step: cc parameters.
    az,baz: azimuth and back-azimuth of the two stations.
    side: A [Default]- both negative and positive sides, N - negative sides only, P - positive side only.
    misc=dict().

    misc is a dictionary that stores additional parameters.

    ======= Methods ======
    to_sac(): convert and save to sac file, using obspy SACTrace object.
    plot(): simple plotting function to display the cross-correlation as moveout plot.
    """
    def __init__(self):
        self.type='Ensemble of Correlation Data'
###
class DvvData(object):
    """
    Object to store dv/v (seismic velocity change) data. This object can be initiated by directly assigning
    values to each attributes OR by giving a CorrData object, in which case some attributes will be cloned
    from the CorrData object. In the latter case, you can still assign attributes that are unique to DvvData.

    ======= Attributes ======
    STATION INFORMATION:
    net=['',''],sta=['',''],loc=['',''],chan=['',''],
    lon=[0.0,0.0],lat=[0.0,0.0],ele=[0.0,0.0],cc_comp='',
    dist=0.0,az=0.0,baz=0.0: parameters specifying the stations.

    DVV PARAMETERS:
    method: dvv measurement method, e.g., stretching or moving window cross-spectral analysis.
    stack_method: stacking method for dvv measurement, e.g., linear, phase-weighted, etc.
    side: A [Default]- both negative and positive sides, N - negative sides only, P - positive side only.
    normalize: whether normalize the cross-correlation functions before measuring dv/v. Default is False.
    window: two element list to specify the window for dv/v measurement. The unit is second. The window will be applied to the cross-correlation functions before measuring dv/v
    dt: sampling interval of the cross-correlation functions. This is needed for dv/v measurement. If initiated by giving a CorrData object, it will be cloned from the CorrData object. 
            Otherwise, it needs to be assigned when initiating the DvvData object
    time: time array of the cross-correlation functions. This is needed for dv/v measurement, especially when there are multiple windows. If initiated by giving a CorrData object, 
            it will be cloned from the CorrData object. Otherwise, it needs to be assigned when initiating the DvvData object.
    freq: frequency array of the cross-correlation functions. This is needed for dv/v measurement using moving window cross-spectral analysis. It needs to be assigned when initiating 
            the DvvData object.
    misc: a dictionary to store additional parameters. This is for flexibility, as there might be various parameters for different dv/v measurement methods. For example, for stretching 
            method, it can store the stretching factors used for measuring dv/v; for moving window cross-spectral analysis, it can store the window length and step used for measuring dv/v, etc.

    DVV DATA:
    cc1=None,cc2=None: cc1 and cc2 are the correlation coefficients arrays for negative measureemts
            and positive measurements, respectively. These are for the entire traces before stretching or any operations during dv/v measurement. 
    maxcc1=None, maxcc2=None: maximum correlation coefficients after stretching for measuring dv/v.
    error1=None, error2=None: errors when measuring the dv/v.
    data1=None,data2=None: data1 is for dvv measurement using negative side correlation data.
            data2 is for the positive side. If dvv is one-sided, only data1 will be used. data2 will stay as None.

    ======= Methods ======
    to_asdf(): save to asdf file.
    plot(): simple plotting function to display the cross-correlation data.
    """
    def __init__(self,corrdata=None,net=['',''],sta=['',''],loc=['',''],chan=['',''],\
                    lon=[0.0,0.0],lat=[0.0,0.0],ele=[0.0,0.0],cc_comp='',dist=0.0,dist_unit='',\
                    method=None,stack_method=None,window=None,dt=None,az=0.0,baz=0.0,time=None,freq=None,\
                    subfreq=True,side=None,normalize=False,cc1=None,cc2=None,maxcc1=None,maxcc2=None,\
                    error1=None,error2=None,data1=None,data2=None,misc=dict()):
        self.type='dv/v Data'
        if corrdata is None: #
            self.net=net
            self.sta=sta
            self.loc=loc
            self.chan=chan
            self.lon=lon
            self.lat=lat
            self.ele=ele
            if cc_comp is None:
                self.cc_comp=chan[0][-1]+chan[1][-1]
            else:
                self.cc_comp=cc_comp
            self.dt=dt
            self.dist=dist
            self.dist_unit=dist_unit
            self.az=az
            self.baz=baz
            self.time=time
            self.stack_method=stack_method
            self.side = side
        else: ### use CorrData metadata when possible. only extract needed attributes.
            self.net=corrdata.net
            self.sta=corrdata.sta
            self.loc=corrdata.loc
            self.chan=corrdata.chan
            self.lon=corrdata.lon
            self.lat=corrdata.lat
            self.ele=corrdata.ele
            if cc_comp is None:
                self.cc_comp=corrdata.chan[0][-1]+corrdata.chan[1][-1]
            else:
                self.cc_comp=corrdata.cc_comp
            self.dt=corrdata.dt
            self.dist=corrdata.dist
            if "dist_unit" in list(corrdata.misc.keys()):
                self.dist_unit=corrdata.misc['dist_unit']
            else:
                self.dist_unit=dist_unit
            self.az=corrdata.az
            self.baz=corrdata.baz
            self.time=corrdata.time
            if side is None:
                self.side=corrdata.side
            else:
                self.side=side

        ##
        self.id=self.net[0]+'.'+self.sta[0]+'.'+self.loc[0]+'.'+self.chan[0]+'_'+\
            self.net[1]+'.'+self.sta[1]+'.'+self.loc[1]+'.'+self.chan[1]

        self.freq=freq
        self.subfreq=subfreq
        self.stack_method=stack_method
        self.method=method
        self.window=window
        self.normalize=normalize
        self.cc1=cc1
        self.cc2=cc2
        self.maxcc1=maxcc1
        self.maxcc2=maxcc2
        self.error1=error1
        self.error2=error2
        self.data1=data1
        self.data2=data2
        self.misc=misc

    def __repr__(self):
        """
        Display key content of the object.
        """
        lines = []
        lines.append("type     :   "+str(self.type))
        lines.append("id       :   "+str(self.id))
        lines.append("net      :   "+str(self.net))
        lines.append("sta      :   "+str(self.sta))
        lines.append("loc      :   "+str(self.loc))
        lines.append("chan     :   "+str(self.chan))
        lines.append("lon      :   "+str(self.lon))
        lines.append("lat      :   "+str(self.lat))
        lines.append("ele      :   "+str(self.ele))
        lines.append("cc_comp  :   "+str(self.cc_comp))
        lines.append("dt       :   "+str(self.dt))
        lines.append("dist     :   "+str(self.dist))
        lines.append("az       :   "+str(self.az))
        lines.append("baz      :   "+str(self.baz))
        lines.append("window   :   "+str(self.window))
        lines.append("normalize:   "+str(self.normalize))
        lines.append("method   :  "+str(self.method))
        lines.append("stack    :  "+str(self.stack_method))
        lines.append("misc     :   "+str(self.misc))
        lines.append("freq     :   "+str(self.freq))
        lines.append("subfreq  :   "+str(self.subfreq))
        lines.append("side     :   "+str(self.side))

        try:
            lines.append("time     :   "+str(obspy.UTCDateTime(self.time[0]))+" to "+str(obspy.UTCDateTime(self.time[-1])))
        except Exception as e:
            lines.append("time     :   None")
        if self.cc1 is not None:
            lines.append("cc1 [N]  :  "+str(self.cc1.shape))
        else:
            lines.append("cc1 [N]:   none")
        if self.cc2 is not None:
            lines.append("cc2 [P]  :  "+str(self.cc2.shape))
        else:
            lines.append("cc2 [P]:   none")
        if self.maxcc1 is not None:
            lines.append("maxcc1 [N]  :  "+str(self.maxcc1.shape))
        else:
            lines.append("maxcc1 [N]:   none")
        if self.maxcc2 is not None:
            lines.append("maxcc2 [P]  :  "+str(self.maxcc2.shape))
        else:
            lines.append("maxcc2 [P]:   none")
        if self.error1 is not None:
            lines.append("error1 [N]  :  "+str(self.error1.shape))
        else:
            lines.append("error1 [N]:   none")
        if self.error2 is not None:
            lines.append("error2 [P]  :  "+str(self.error2.shape))
        else:
            lines.append("error2 [P]:   none")
        if self.data1 is not None:
            lines.append("data1 [N]:   "+str(self.data1.shape))
        else:
            lines.append("data1 [N]:   none")
        if self.data2 is not None:
            lines.append("data2 [P]:   "+str(self.data2.shape))
        else:
            lines.append("data2 [P]:   none")
        lines.append("")

        return "<DvvData object>\n" + "\n".join(lines)

    __str__ = __repr__

    ## method to get some info
    def get_info(self):
        """
        Get collective information.

        =====RETURNS====
        data: data matrix.
        label: data label.
        path: path label, particularly for heirachy file system.
        parameters: parameters for additional attributes
        """

        # source-receiver pair
        label = self.net[0]+'.'+self.sta[0]+'_'+\
                        self.net[1]+'.'+self.sta[1]
        path = self.chan[0]+'_'+self.chan[1]

        #save to asdf
        lonS,lonR = self.lon
        latS,latR = self.lat
        eleS,eleR = self.ele
        if self.side is not None:
            side=self.side
        else:
            if self.data1 is not None and self.data2 is not None:
                side='A'
            elif self.data1 is not None:
                side='N'
            else:
                side='P'
        #
        if side.lower() == 'a':
            odata=np.array([self.data1,self.data2])
        elif side.lower() == 'n':
            odata=self.data1
        elif side.lower() == 'p' or side.lower() == 'o':
            odata=self.data2

        parameters = {'dt':self.dt,
            'dist':np.float32(self.dist),
            'dist_unit':self.dist_unit,
            'azi':np.float32(self.az),
            'baz':np.float32(self.baz),
            'lonS':np.float32(lonS),
            'latS':np.float32(latS),
            'eleS':np.float32(eleS),
            'lonR':np.float32(lonR),
            'latR':np.float32(latR),
            'eleR':np.float32(eleR),
            'window':np.float32(self.window),
            'stack_method':self.stack_method,
            'method':self.method,
            'normalize':self.normalize,
            'subfreq':self.subfreq,
            'time':self.time,
            'comp':self.cc_comp,
            'type':self.type,
            'freq':self.freq,
            'net':self.net,
            'sta':self.sta,
            'chan':self.chan,
            'side':side,
            'cc1':np.float32(self.cc1),
            'cc2':np.float32(self.cc2),
            'maxcc1':np.float32(self.maxcc1),
            'maxcc2':np.float32(self.maxcc2),
            'error1':np.float32(self.maxcc1),
            'error2':np.float32(self.maxcc2)}

        return odata,label,path,parameters


    def to_asdf(self,outdir='.',file=None,v=True):
        """
        Save DvvData object to asdf file.
        file: file name, default is like dvv_AK.CHN..BHE_AK.CHN..BHZ_EZ.h5.
        ======parameters======
        outdir: outdirectory, default is current folder
        file: file name, file extension will be added if not already there.
        v: verbose
        """
        if file is None:
            file="dvv_"+self.id+"_"+self.cc_comp+".h5"
        elif file[-2:] != "h5":
            file = file + ".h5"
        odata,netsta_pair,chan_pair,parameters=self.get_info()

        #check time size to avoid error. make sure it is not > 64kb
        #this is a temporary fix, though the ultimate fix will rely on HDF to lift the limit.
        time_temp=parameters['time']
        if sys.getsizeof(parameters['time'])/1024 > 64: #64k is the limit of HDF attribute.
            parameters['time']=np.float32(time_temp-np.mean(time_temp))
            parameters['time_mean']=np.mean(time_temp)

        with pyasdf.ASDFDataSet(outdir+'/'+file,mpi=False) as dvv_ds:
            dvv_ds.add_auxiliary_data(data=odata, data_type=netsta_pair, path=chan_pair, parameters=parameters)
        if v: print('DvvData saved to: '+outdir+'/'+file)
    #
    def to_pickle(self,outdir='.',file=None,v=True):
        """
        Save DvvData object to a pickle file.
        file: file name, default is like dvv_AK.CHN..BHE_AK.CHN..BHZ_EZ.pk.
        ======parameters======
        outdir: outdirectory, default is current folder
        file: file name, file extension will be added if not already there.
        v: verbose
        """
        overwrite=True #appending mode is not supported yet.
        if file is None:
            file="dvv_"+self.id+"_"+self.cc_comp+".pk"
        elif file[-2:] != "pk":
            file = file + ".pk"
        odata,netsta_pair,chan_pair,parameters=self.get_info()
        dvvdict={"data":odata,"label":netsta_pair,"path":chan_pair,"parameters":parameters}
        dvvoutdict={netsta_pair:{chan_pair:dvvdict}}
        if overwrite:
            mode="wb"
        else:
            mode="ab"
        with open(os.path.join(outdir,file),mode) as dvvf:
            pickle.dump(dvvoutdict,dvvf)
        if v: print('DvvData saved to: '+outdir+'/'+file)
    #
    def save(self,outdir='.',file=None,v=True,format=None):
        """
        Save DvvData object to a file. This is a wrapper of the saving functions for different file formats.
        file: file name, default is like dvv_AK.CHN..BHE_AK.CHN..BHZ_EZ.XX.
        ASDF file will end with "h5"
        Pickle file will end with "pk"

        ======parameters======
        outdir: outdirectory, default is current folder
        file: file name, file extension will be added if not already there.
        v: verbose
        format: "asdf" or "pickle". Default is "asdf", unless specified by the file extension.
        """
        format_all=["asdf","pickle"]
        fextlist=["h5","pk"]
        fend=file[-2:]
        if format is None:
            if fend.lower() in fextlist:
                if fend.lower() == "h5":
                    format = "asdf"
                elif fend.lower() == "pk":
                    format = "pickle"
            else:
                format = "asdf"
        elif format not in format_all:
            raise ValueError(format+" is not supported yet. Use one of: "+str(format_all))

        fext="h5"
        if format.lower() == "pickle":
            fext="pk"
        if file is None:
            file="dvv_"+self.id+"_"+self.cc_comp+"."+fext
        elif fend.lower() != fext:
            file = file + "."+fext

        # call corresponding saving functions
        if format.lower() == "asdf":
            self.to_asdf(outdir=outdir,file=file,v=v)
        elif format.lower() == "pickle":
            self.to_pickle(outdir=outdir,file=file,v=v)
        else:
            raise ValueError(format+" is not supported yet.")
    ##plot
    def plot(self,cc_min=None,error_max=None,figsize=(8,5),ylim=None,save=False,nxtick=None,\
            figdir='.',format='png',figname=None,smooth=None,yinc=1.0,ytick_precision=1,
            crange=None,side="a",errorbar=True,markersize=5):
        """
        Plot DvvData.

        cc_min: minimum max-correlation-coefficient in measuring dvv.
        error_max: maximum error in dv/v measurement, applied for both negative and positive sides.
        figsize: figure size tuble
        ylim: y range for Display
        save: save figure. default False.
        figdir: directory to save figure. default is current directory.
        figname: figure name when save is True.
        smooth: box smooth options. should be a list of two elements. Defult None.
                Use same value for x and y if only one value is given.
        crange: colorbar range.
        yinc: y axis (frequency) increment.
        ytick_precision: precision of displaying frequency labels on y axis (default is 1)
        side: which side to plot. default is "a" for both negative and positive sides.
        errorbar: plot errobar or not for single frequency only. Default True.
        markersize: markersize for single frequency plots only. Default is 5.
        """
        nvdata=self.data1.copy()
        pvdata=self.data2.copy()
        if smooth is not None:
            if type(smooth) is not tuple and type(smooth) is not list:smooth=[smooth]
            if len(smooth)==1:smooth=[smooth[0],smooth[0]]
            nvdata=scipy.ndimage.filters.gaussian_filter(nvdata, smooth, mode='constant')
            pvdata=scipy.ndimage.filters.gaussian_filter(pvdata, smooth, mode='constant')
        if cc_min is None:
            cc_min=-1.0
        idx1=np.where((self.maxcc1<cc_min))
        nvdata[idx1]=np.nan

        idx2=np.where((self.maxcc2<cc_min))
        pvdata[idx2]=np.nan
        nerror=self.error1.copy()
        perror=self.error2.copy()
        idx1=[]
        idx1=np.where((self.error1<0))
        nvdata[idx1]=np.nan
        nerror[idx1]=np.nan
        idx2=[]
        idx2=np.where((self.error2<0))
        pvdata[idx2]=np.nan
        perror[idx1]=np.nan
        
        if error_max is not None:
            idx1=[]
            idx2=[]
            idx1=np.where((self.error1>error_max))
            nvdata[idx1]=np.nan

            idx2=np.where((self.error2>error_max))
            pvdata[idx2]=np.nan

        nwin=nvdata.shape[0]
        # tick inc for plotting
        if nxtick is None:
            nxtick = 5

        plt.figure(figsize=figsize, facecolor = 'white')
        # the cross-correlation coefficient

        #
        if self.subfreq: #multiple frequencies.
            # dv/v at each filtered frequency band
            xticks=np.int16(np.linspace(0,nwin-1,nxtick))
            xticklabel=[]
            for x in xticks:
                xticklabel.append(str(UTCDateTime(self.time[x]))[:10])
            period=1/self.freq
            if (side.lower()=="a" or side.lower()=="n") and self.side.lower()!="o":
                dvv_array = nvdata.T
                yrange=[np.log2(period.min()),np.log2(period.max())]
                extent=(0,nwin,yrange[1],yrange[0])
                if side.lower()=="a":ax1 = plt.subplot(211)
                else:ax1 = plt.subplot(111)
                plt.imshow(dvv_array,cmap='jet_r',aspect='auto',extent=extent)

                plt.ylabel('frequency (Hz)',fontsize=12)
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(xticklabel,fontsize=12)

                Yticks = 2 ** np.arange(np.log2(period.min()),
                                   np.log2(period.max()),yinc)
                ax1.set_yticks(np.log2(Yticks))
                ax1.set_yticklabels(np.round(1/Yticks,ytick_precision))
                if ylim is None:
                    plt.ylim(yrange)
                else:
                    plt.ylim(ylim)
                plt.yticks(fontsize=12)
                if crange is not None:plt.clim(crange)
                plt.colorbar(label='dv/v (%)')
                ax1.set_title('dv/v:'+self.id+':'+str(self.dist)+' km:negative:'+str(cc_min),fontsize=14)
                ax1.invert_yaxis()

            if side.lower()=="a" or side.lower()=="p" or side.lower()=="o" or self.side.lower()=="o":
                if side.lower() == 'o' or self.side.lower() == 'o':side_label="one-sided"
                else:side_label="positive"
                dvv_array = pvdata.T
                if side.lower()=="a":ax2 = plt.subplot(212)
                else:ax2 = plt.subplot(111)
                plt.imshow(dvv_array,cmap='jet_r',aspect='auto',extent=extent)
                plt.ylabel('frequency (Hz)',fontsize=12)
                ax2.set_xticks(xticks)
                ax2.set_xticklabels(xticklabel,fontsize=12)
                ax2.set_yticks(np.log2(Yticks))
                ax2.set_yticklabels(np.round(1/Yticks,ytick_precision))
                if ylim is None:
                    plt.ylim(yrange)
                else:
                    plt.ylim(ylim)
                plt.yticks(fontsize=12)
                if crange is not None:plt.clim(crange)
                plt.colorbar(label='dv/v (%)')
                ax2.set_title('dv/v:'+self.id+':'+str(self.dist)+' km:'+side_label+':'+str(cc_min),fontsize=14)
                ax2.invert_yaxis()
            plt.tight_layout()
        else: #only one measurement from one frequency
            xticks=np.linspace(np.min(self.time),np.max(self.time),nxtick)
            xticklabel=[]
            for x in xticks:
                xticklabel.append(str(UTCDateTime(x))[:10])
            xext=0.02*(np.max(self.time)-np.min(self.time))
            plt.hlines(0,np.min(self.time)-xext,np.max(self.time)+xext,colors='k')
            if (side.lower()=="a" or side.lower()=="n") and self.side.lower()!="o":
                if errorbar:
                    plt.errorbar(self.time,nvdata,yerr=nerror,fmt="o",markersize=markersize,
                                capsize=3,label="negative")
                else:
                    plt.plot(self.time,nvdata,".-",markersize=markersize,label="negative")
            if side.lower()=="a" or side.lower()=="p" or side.lower()=="o" or self.side.lower()=="o":
                if side.lower() == 'o' or self.side.lower() == 'o':side_label="one-sided"
                else:side_label="positive"
                if errorbar:
                    plt.errorbar(self.time,pvdata,yerr=perror,fmt="^",markersize=markersize,
                                capsize=3,label=side_label)
                else:
                    plt.plot(self.time,pvdata,".-",markersize=markersize,label=side_label)

            plt.ylabel('dv/v (%)',fontsize=12)
            plt.xlim([np.min(self.time)-xext,np.max(self.time)+xext])

            plt.xticks(xticks,labels=xticklabel,fontsize=12)
            plt.legend(fontsize=12)
            if ylim is not None:
                plt.ylim(ylim)
            plt.title('dv/v:'+self.id+':'+str(self.dist)+' km:'+side_label+':'+\
                        str(cc_min)+':'+str(np.min(self.freq))+"-"+str(np.max(self.freq))+" Hz",
                        fontsize=14)

        ###################
        ##### SAVING ######
        if save:
            if not os.path.isdir(figdir):os.mkdir(figdir)
            if figname is None: figname = 'dvv_'+self.id+'_'+self.cc_comp+'_'+side+\
                    '_'+str(cc_min)+'_'+str(np.min(self.freq))+"_"+str(np.max(self.freq))+"Hz"+'.'+format
            plt.savefig(figdir+'/'+figname, format=format, dpi=300, facecolor = 'white')
            plt.close()
        else:
            plt.show()

    #
    def dq(self, cc_min=0.75):
        """
        Dynamically computes the attenuation change (dq) from the residual 
        decorrelation using the object's internal window and frequency parameters.
        Function wrote with assistance from Gemini AI, approved and modified by Xiaotao Yang.

        Following Snieder (2006) and Larose et al. (2010), the absolute change 
        in the inverse quality factor (dq) is isolated from the post-stretching 
        residual decorrelation of the coda windows using the linear time-lapse 
        sensitivity relation: dq = (1 - CC_max) / (pi * f * t_mean).
        
        Parameters:
        -----------
        cc_min : float, optional
            Minimum maximum-CC required to trust the measurement. 
            Values below this threshold will map to NaN. Default is 0.75.
            
        Returns:
        --------
        dq1 : numpy.ndarray or None
            Attenuation change for negative lag (same dimension as data1).
            Returns None if data1 is not present.
        dq2 : numpy.ndarray or None
            Attenuation change for positive lag (same dimension as data2).
            Returns None if data2 is not present.

        References:
        -----------
        Larose, E., Planes, T., Rossetto, V., & Margerin, L. (2010). Locating a small change in a multiple scattering environment. 
                Applied Physics Letters, 96(20), 204101. https://doi.org/10.1063/1.3431269/119906
        Obermann, A., Froment, B., Campillo, M., Larose, E., Planès, T., Valette, B., Chen, J. H., & Liu, Q. Y. (2014). Seismic 
                noise correlations to image structural and mechanical changes associated with the Mw 7.9 2008 Wenchuan earthquake. 
                Journal of Geophysical Research: Solid Earth, 119(4), 3155–3168. https://doi.org/10.1002/2013JB010932
        Snieder, R. (2006). The theory of coda wave interferometry. Pure and Applied Geophysics, 163(2–3), 455–473. 
                https://doi.org/10.1007/S00024-005-0026-6/METRICS
        """
        # 1. Extract frequency parameters and calculate central frequency
        if self.freq is None or len(self.freq) < 2:
            raise ValueError("self.freq must be a two-element list/array containing [f_min, f_max].")
        central_freq = np.mean(self.freq)
        
        # 2. Extract window parameters and calculate mean absolute lapse time
        if self.window is None or len(self.window) < 2:
            raise ValueError("self.window must be a two-element list/array containing [t_start, t_end].")
        t_mean = 0.5 * (abs(self.window[0]) + abs(self.window[1]))
        
        # 3. Calculate the time-lapse sensitivity kernel (beta = pi * f * t_mean)
        beta = np.pi * central_freq * t_mean
        
        dq1 = None
        dq2 = None
        
        # Process Negative Lag (data1 / maxcc1)
        if self.data1 is not None and len(self.data1) > 0:
            maxcc1_arr = np.array(self.maxcc1)
            # Calculate raw dq: (1 - max_cc) / beta
            dq1 = (1.0 - maxcc1_arr) / beta
            # Apply Quality Control threshold
            dq1[maxcc1_arr < cc_min] = np.nan
            
        # Process Positive Lag (data2 / maxcc2)
        if self.data2 is not None and len(self.data2) > 0:
            maxcc2_arr = np.array(self.maxcc2)
            # Calculate raw dq: (1 - max_cc) / beta
            dq2 = (1.0 - maxcc2_arr) / beta
            # Apply Quality Control threshold
            dq2[maxcc2_arr < cc_min] = np.nan
            
        return dq1, dq2

class HVSRData(object):
    """
    Object to store HVSR (Horizontal-to-Vertical Spectral Ratio) data.

    Similar to DvvData, this class stores HVSR results for a single station.

    Attributes
    ----------
    Station info: net, sta, loc, chan, lon, lat, ele
    HVSR parameters: method, freqmin, freqmax, win_len_s, step_s
    Data: freqs, data (dict of methods with hvsr, hvsr_std, peaks), n_windows
    """
    def __init__(self, net=None, sta=None, loc=None, lon=None, lat=None, ele=None, 
                 method=None, freqmin=None, freqmax=None,
                 win_len_s=None, step_s=None, freqs=None, stds = None, data=None, n_windows=None, 
                 label = None, misc=dict()):
        self.type = 'HVSR Data'
        self.label = label
        
        self.net = net
        self.sta = sta
        self.loc = loc
        self.lon = lon
        self.lat = lat
        self.ele = ele
        self.method = method
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.win_len_s = win_len_s
        self.step_s = step_s
        self.freqs = freqs
        self.stds = stds #stds of hvsr for each method, same shape as data.
        self.data = data  # matrix of shape (len(freqs),len(method)).
        
        self.n_windows = n_windows
        self.id = f"{self.net}.{self.sta}.{self.loc}"
        self.misc = misc

    def __repr__(self):
        """
        Display key content of the object.
        """
        lines = []
        lines.append("type     :   " + str(self.type))
        lines.append("id       :   " + str(self.id))
        lines.append("net      :   " + str(self.net))
        lines.append("sta      :   " + str(self.sta))
        lines.append("loc      :   " + str(self.loc))
        lines.append("lon      :   " + str(self.lon))
        lines.append("lat      :   " + str(self.lat))
        lines.append("ele      :   " + str(self.ele))
        lines.append("method   :   " + str(self.method))
        lines.append("freqmin  :   " + str(self.freqmin))
        lines.append("freqmax  :   " + str(self.freqmax))
        lines.append("win_len_s:   " + str(self.win_len_s))
        lines.append("step_s   :   " + str(self.step_s))
        if self.freqs is not None:
            lines.append("freqs    :   " + f"shape {self.freqs.shape}, range {self.freqs[0]:.3f} - {self.freqs[-1]:.3f} Hz")
        else:
            lines.append("freqs    :   None")
        if self.data is not None:
            lines.append("data     :   " + f"{self.data.shape}")
        else:
            lines.append("data     :   None")
        lines.append("n_windows:   " + str(self.n_windows))
        lines.append("misc     :   " + str(self.misc))
        lines.append("")
        return "<HVSRData object>\n" + "\n".join(lines)

    __str__ = __repr__

    def get_info(self):
        """
        Get a dictionary of key attributes for saving.
        """
        info = {
            'id': self.id,
            'net': self.net,
            'sta': self.sta,
            'loc': self.loc,
            'lon': self.lon,
            'lat': self.lat,
            'ele': self.ele,
            'method': self.method,
            'freqmin': self.freqmin,
            'freqmax': self.freqmax,
            'win_len_s': self.win_len_s,
            'step_s': self.step_s,
            'n_windows': self.n_windows,
            'misc': self.misc,
        }
        return info

    def plot(self, method=None, show_std=True, figsize=(9, 5), ymax=None, xtype='frequency', 
             title=None, save=False, figname=None, fmt='png',peaks=None):
        """
        Plot the HVSR data.

        ======Parameters=====
        method: list of methods to plot. Default is all methods in self.method.
        show_std: whether to show standard deviation as shaded area. Default is True.
        figsize: figure size tuple. Default is (9, 5).
        ymax: maximum y value for the plot. Default is None (auto).
        xtype: x-axis type, either 'frequency' or 'period'. Default is 'frequency
        title: figure title. Default is None (auto).
        save: whether to save the figure. Default is False.
        figname: figure name when save is True. Default is None (auto).
        fmt: figure format when save is True. Default is 'png'.
        peaks: list of peak frequencies to highlight. Default is None (skip annotating peaks).
        """
        if self.freqs is None or self.data is None:
            raise ValueError("No HVSR data to plot.")
        x = self.freqs if xtype == 'frequency' else 1.0 / np.where(self.freqs > 0, self.freqs, np.nan)
        if method is None:
            method = self.method
        elif isinstance(method, int):
            method = [method]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(5,1))]

        #find the correct method indices since the data matrix does not necessarily have the same order as self.method list.
        method_indices = []
        for m in method:
            if m in self.method:
                method_indices.append(self.method.index(m))
            else:
                raise ValueError(f"Method {m} not found in self.method list.")
        method_indices = np.array(method_indices)
        fig, ax = plt.subplots(figsize=figsize)
        for i, m in enumerate(method):
            m_index = method_indices[i]
            hvsr = self.data[m_index, :]
            hvsr_std = self.stds[m_index,:] if self.stds is not None else None

            color = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]
            m_label = f"M{m}"
            ax.plot(x, hvsr, color=color, linestyle=ls, linewidth=1.5, label=m_label)
            if show_std and hvsr_std is not None:
                ax.fill_between(x, np.maximum(hvsr - hvsr_std, 0), hvsr + hvsr_std, color=color, alpha=0.15)
            if peaks is not None:
                best = peaks[m_label][0]
                pf = best['f0'] if xtype == 'frequency' else 1.0 / best['f0']
                ax.axvline(pf, color=color, linestyle=':', linewidth=0.8, alpha=0.7)
                ax.annotate(f"f₀={best['f0']:.3f} Hz\nA₀={best['A0']:.2f}\nscore={best['score']}/6",
                            xy=(pf, best['A0']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, color=color)
        ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)' if xtype == 'frequency' else 'Period (s)')
        ax.set_ylabel('HVSR')
        if ymax is not None:
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(bottom=0)
        ttl = title if title else f"HVSR – {self.id}"
        ttl += f"\n(n_windows={self.n_windows})"
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        if save:
            if figname is None:
                sid = self.id.replace('.', '_')
                figname = f"{sid}_HVSR.{fmt}"
            fig.savefig(figname, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {figname}")
        return fig, ax

    def to_asdf(self, outdir='.', file=None, v=True):
        """
        Save HVSR data to ASDF format.
        """
        if file is None:
            file = self.id + '_hvsr.h5'
        ds = pyasdf.ASDFDataSet(outdir + '/' + file, compression="gzip-3")
        # Add freqs
        ds.add_auxiliary_data(data=self.freqs, data_type="HVSRData", path="freqs", parameters={})
        # Add data for each method
        for m, sub in self.data.items():
            path_hvsr = f"hvsr/M{m}"
            ds.add_auxiliary_data(data=sub['hvsr'], data_type="HVSRData", path=path_hvsr, parameters=self.get_info())
            if sub['hvsr_std'] is not None:
                path_std = f"hvsr_std/M{m}"
                ds.add_auxiliary_data(data=sub['hvsr_std'], data_type="HVSRData", path=path_std, parameters={})
            # Peaks are not arrays, skip for now
        if v:
            print(f"Saved HVSR data to {outdir}/{file}")

    def to_pickle(self, outdir='.', file=None, v=True):
        """
        Save HVSR data to pickle format.
        """
        if file is None:
            file = self.id + '_hvsr.pkl'
        with open(outdir + '/' + file, 'wb') as f:
            pickle.dump(self, f)
        if v:
            print(f"Saved HVSR data to {outdir}/{file}")

    def save(self, outdir='.', file=None, v=True, format='asdf'):
        """
        Save HVSR data. Wrapper for to_asdf or to_pickle.
        """
        if format.lower() == 'asdf':
            self.to_asdf(outdir, file, v)
        elif format.lower() == 'pickle':
            self.to_pickle(outdir, file, v)
        else:
            raise ValueError("Format must be 'asdf' or 'pickle'")

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

################################################################
######################## DISPERSION DATA ########################
################################################################
# DispData stores the output of a dispersion-measurement method (aftan(),
# aftan_pmf(), and potentially others). The functions that actually PRODUCE DispData objects and the
# free function that reads one back from an .h5 file (read_dispdata()) stay in
# seisgo.dispersion.
def _json_safe(obj):
    """
    Recursively convert numpy scalar/array types (which can leak into DispData's
    `params` dict and `side` attribute from upstream dispersion-measurement code)
    into native Python types, so the result is safe to json.dumps() for
    DispData.save().
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class DispData(object):
    """
    Container for a surface-wave dispersion measurement. Meant to store the output of
    any dispersion-measurement method in seisgo.dispersion (aftan(), aftan_pmf(), and
    potentially others in the future), not just AFTAN specifically.

    ===Attributes===
    period: period vector (s), as sampled by the method that produced this object.
    group_velocity: picked group velocity (km/s) for each period. NaN where no
            acceptable pick could be made/tracked.
    phase_velocity: picked phase velocity (km/s) for each period, when computed. NaN
            (the whole array, if phase velocity was not requested/available) otherwise.
    inst_period: instantaneous period (s) recovered from the local phase derivative at
            the picked group-velocity arrival (slightly different from the nominal
            filter period).
    amplitude: envelope amplitude at the picked group-velocity arrival, for each period.
    snr: estimated SNR (dB) of the picked arrival relative to the trailing noise window.
    dist: source-receiver distance (km) used.
    dt: sampling interval (s) of the analyzed waveform.
    side: which lag ('p','n','sym', or as stored in the source CorrData) was analyzed.
    method: name of the function that produced this object (e.g. 'aftan','aftan_pmf').
    params: dict of the parameters the method was called with.
    arrival_time,phase_pick: internal per-period group-arrival time (s) and measured
            phase (rad) at that arrival; kept around because they are what
            _phase_velocity() needs, and so phase velocity can be (re)computed later
            (e.g. with a different reference curve) without rerunning the filtering.
    envelope,phase_matrix: the full 2-D narrow-band-filtered envelope and unwrapped
            phase matrices, shape (len(period), npts), i.e. one row per analyzed
            period, one column per time sample of the analyzed waveform (see
            _aftan_narrowband()) -- the same arrays group_velocity/phase_velocity were
            picked from. Not just the single picked value per period: this is the full
            time-period "image" needed to plot a classical AFTAN dispersion-energy
            image (period/velocity vs. amplitude) or to do other image-based analysis
            (e.g. multi-mode inspection, custom picking, ridge tracking). None unless
            the method that produced this object was called with store_image=True
            (the default in aftan()/aftan_pmf()). See get_image()/plot_image().
    src_net,src_sta,src_lon,src_lat: network/station code and longitude/latitude (deg)
            of the virtual source (station 1 of the pair). None if not supplied.
    rcv_net,rcv_sta,rcv_lon,rcv_lat: network/station code and longitude/latitude (deg)
            of the receiver (station 2 of the pair). None if not supplied.
            These eight fields are optional, purely for bookkeeping -- nothing in the
            group-/phase-velocity measurement itself uses them -- but they're what let
            a DispData ensemble be turned into an eikonal-tomography phase-velocity
            map (see seisgo.imaging.eikonal.eikonal_tomography()), which needs each
            measurement tied back to actual station locations rather than only the
            scalar `dist` used for the
            dispersion picking itself. aftan()/aftan_pmf() populate these automatically
            from `corrdata` (a seisgo.types.CorrData object stores exactly this as
            net=[net1,net2],sta=[sta1,sta2],lon=[lon1,lon2],lat=[lat1,lat2]) when one is
            given, or from the same-named keyword arguments when called with raw
            data/dt/dist instead.
    """
    def __init__(self, period, group_velocity, amplitude, snr, inst_period, dist, dt, side,
                 params, phase_velocity=None, method='aftan', arrival_time=None, phase_pick=None,
                 envelope=None, phase_matrix=None,
                 src_net=None, src_sta=None, src_lon=None, src_lat=None,
                 rcv_net=None, rcv_sta=None, rcv_lon=None, rcv_lat=None):
        self.type = 'Dispersion Data'
        self.period = np.asarray(period, dtype=np.float64)
        self.group_velocity = np.asarray(group_velocity, dtype=np.float64)
        self.amplitude = np.asarray(amplitude, dtype=np.float64)
        self.snr = np.asarray(snr, dtype=np.float64)
        self.inst_period = np.asarray(inst_period, dtype=np.float64)
        self.phase_velocity = np.full(len(self.period), np.nan) if phase_velocity is None \
                                else np.asarray(phase_velocity, dtype=np.float64)
        self.dist = dist
        self.dt = dt
        self.side = side
        self.method = method
        self.params = params
        self.arrival_time = arrival_time
        self.phase_pick = phase_pick
        self.envelope = None if envelope is None else np.asarray(envelope, dtype=np.float64)
        self.phase_matrix = None if phase_matrix is None else np.asarray(phase_matrix, dtype=np.float64)
        self.src_net = src_net
        self.src_sta = src_sta
        self.src_lon = src_lon
        self.src_lat = src_lat
        self.rcv_net = rcv_net
        self.rcv_sta = rcv_sta
        self.rcv_lon = rcv_lon
        self.rcv_lat = rcv_lat

    def __repr__(self):
        """
        Display key content of the object, formatted the same way as
        seisgo.types.CorrData.__str__() (one "label     :   value" line per
        attribute) -- but, unlike that method, built and returned as a single
        string rather than emitted via embedded print() calls. That's what makes
        typing a bare object name at a REPL/notebook prompt show the formatted
        block: Python calls __repr__() (not __str__()) to display a bare
        expression's result, so the formatting has to actually be the *returned*
        string, not a side effect of calling the method. __str__ is aliased to
        this same method below, so print(d) and str(d) show identical output too.
        """
        n = len(self.period) if self.period is not None else 0
        ng = int(np.sum(~np.isnan(self.group_velocity))) if self.group_velocity is not None else 0
        npv = int(np.sum(~np.isnan(self.phase_velocity))) if self.phase_velocity is not None else 0
        lines = []
        lines.append("type          :   " + str(self.type))
        lines.append("method        :   " + str(self.method))
        if self.src_sta is not None or self.rcv_sta is not None:
            src_id = "%s.%s" % (self.src_net, self.src_sta) if self.src_sta is not None else "?"
            rcv_id = "%s.%s" % (self.rcv_net, self.rcv_sta) if self.rcv_sta is not None else "?"
            lines.append("pair          :   %s -> %s" % (src_id, rcv_id))
        lines.append("dist          :   " + str(self.dist))
        lines.append("dt            :   " + str(self.dt))
        lines.append("side          :   " + str(self.side))
        if n > 0:
            lines.append("period        :   " + str(self.period.shape) + "  (%.3g to %.3g s)" %
                          (np.nanmin(self.period), np.nanmax(self.period)))
        else:
            lines.append("period        :   none")
        lines.append("group_velocity:   %d/%d valid pick(s)" % (ng, n))
        lines.append("phase_velocity:   %d/%d valid pick(s)" % (npv, n))
        if self.snr is not None and len(self.snr) > 0 and np.any(~np.isnan(self.snr)):
            lines.append("snr           :   min=%.2f, max=%.2f dB" %
                          (np.nanmin(self.snr), np.nanmax(self.snr)))
        else:
            lines.append("snr           :   none")
        lines.append("envelope      :   " + (str(self.envelope.shape) if self.envelope is not None else "none"))
        lines.append("phase_matrix  :   " + (str(self.phase_matrix.shape) if self.phase_matrix is not None else "none"))
        lines.append("params        :   " + str(self.params))
        return "<DispData object>\n" + "\n".join(lines)

    __str__ = __repr__

    def to_dataframe(self):
        """
        Return the dispersion measurements as a pandas DataFrame.
        """
        return pd.DataFrame({"period": self.period, "group_velocity": self.group_velocity,
                              "phase_velocity": self.phase_velocity, "inst_period": self.inst_period,
                              "amplitude": self.amplitude, "snr": self.snr})

    def continuity(self, vtype='both', step=2, stat='mean', norm='auto'):
        """
        Quality index for the picked dispersion curve(s), meant for QC'ing a curve
        (or batch-flagging many curves) before it's fed into imaging (e.g.
        seisgo.imaging.eikonal.eikonal_tomography()). A dispersion curve should trace
        one smooth physical branch; a curve whose picker has locked onto the wrong
        branch for part of its length (the classic failure mode aftan_pmf()'s
        guided-pick option was added to avoid -- see that function's docstring)
        instead shows sudden, large relative jumps between nearby periods.

        NOTE: this only measures period-to-period SMOOTHNESS of the picked curve
        itself. A curve can be perfectly smooth while still consistently sitting
        on the wrong (but locally continuous) branch of the dispersion image --
        continuity() structurally cannot see that, since it never looks at the
        underlying envelope image at all. Use peak_alignment() alongside this
        method for that complementary check (see its docstring).

        For each velocity array requested (group and/or phase), and for every pair
        of valid (non-NaN) samples that are `delta` array-index steps apart along
        the period axis (delta = 1, 2, ..., step), compute the relative jump

            rel_jump = |v[i+delta] - v[i]| / (norm_scale * delta)

        where `norm_scale` (a km/s velocity scale) is controlled by `norm` -- see
        below; scaling by `delta` reflects that two periods twice as far apart are
        allowed twice the excursion before it counts as equally "jumpy". Each
        individual rel_jump is then IMMEDIATELY converted to a per-pair continuity
        value

            pair_continuity = 1 / (1 + rel_jump)

        -- 1 for that one pair being perfectly smooth, shrinking toward 0 the
        worse that one jump is -- before anything is combined across pairs. `stat`
        is then applied to that whole pool of already-inverted, already
        "higher is better" per-pair continuity values (pooled across every delta
        from 1 to `step`), so `stat` names the aggregate exactly the way it reads:
        stat='mean' is literally the mean continuity, stat='max' is literally the
        best (highest, smoothest) local continuity found anywhere in the curve,
        and so on. (This is a different, and generally different-VALUED,
        computation than inverting a single aggregated discontinuity number --
        e.g. mean-of-inverted-values != inverse-of-mean-value -- but it's what
        makes each `stat` option mean what its name says.)

        ===PARAMETERS===
        vtype: 'group', 'phase', or 'both' [default] -- which curve(s) to score.
        step: consider relative jumps between periods up to this many array-index
                steps apart (default 2, i.e. pool both adjacent-period jumps and
                next-adjacent-period jumps). Including delta=2 lets one single bad
                pick sandwiched between two good ones (a 1-sample spike) still be
                caught by comparing its neighbors to each other, and also means a
                single NaN gap of length 1 doesn't fully block the jump from being
                scored between the samples that bracket it.
        stat: which summary statistic of the pooled per-pair continuity values
                (see above -- each already in (0, 1], higher already meaning
                smoother) is returned --
                'mean' [default]: the average per-pair continuity; a fixed QC
                    threshold behaves consistently across curves of different
                    lengths.
                'median': like 'mean' but robust to a single outlier jump -- use
                    this when you want to judge the curve's overall smoothness
                    without one bad period dominating the score.
                'min': the single WORST per-pair continuity anywhere in the curve
                    (i.e. driven by the single largest relative jump); catches one
                    badly-picked period even when the rest of the curve is smooth
                    (a 'mean'/'median' score can dilute a single bad jump away).
                    This is the option to use for a strict "reject if any single
                    jump is bad" QC gate.
                'max': the single BEST per-pair continuity anywhere in the curve
                    (its smoothest local step); mostly useful together with 'min'
                    to see the full best/worst range, rarely useful alone as a QC
                    gate since one good pair can't rescue a curve with many bad
                    ones.
                'sum': the total per-pair continuity; scales with the number of
                    valid periods, so only meaningful when comparing curves of
                    similar length, and not bounded by 1 the way the others are.
        norm: how each relative jump's denominator (norm_scale, in km/s) is set --
                'auto' [default]: use this curve's own params['vg_step_max'] (the
                    same per-period maximum-jump tolerance _pick_group_velocity()
                    itself enforced while picking), falling back to the
                    aftan()/aftan_pmf() default of 0.5 km/s if that key isn't
                    present in params (e.g. a hand-built or synthetic DispData).
                    This ties "jumpy" directly to the picker's own notion of a
                    suspiciously large per-period jump, which for typical real
                    surface-wave velocities (order 2-5 km/s) is a substantially
                    smaller scale than the curve's own velocity magnitude -- so a
                    jump that looks clearly wrong by eye now actually drives the
                    index down noticeably, instead of being swamped by dividing
                    by an absolute velocity of several km/s.
                'velocity': norm_scale is set, per pair, to the local 
                    mean(|v[i+delta]|, |v[i]|) rather than one fixed scale. On
                    real data this tends to make the index much less sensitive
                    (see 'auto' above), since it divides by the curve's own
                    (typically several km/s) magnitude rather than a tolerance
                    tied to what counts as a bad jump.
                a positive float: an explicit fixed km/s scale to use in place of
                    either of the above, e.g. norm=0.3 to match a non-default
                    vg_step_max the curve was actually picked with, or to make
                    several curves picked with different settings directly
                    comparable to each other.

        ===RETURNS===
        If vtype is 'group' or 'phase': a single float continuity index (NaN if
        fewer than 2 valid, sufficiently-close picks are available to form any
        jump -- or, for 'phase', if phase_velocity was never computed). In (0, 1]
        for every `stat` except 'sum'.
        If vtype is 'both' [default]: a dict {'group': <float>, 'phase': <float>},
        each computed exactly as above (phase's entry is NaN when unavailable).
        """
        valid_stats = ('mean', 'median', 'sum', 'min', 'max')
        if stat not in valid_stats:
            raise ValueError("continuity(): stat must be one of %s, got %r" % (valid_stats, stat))
        if step < 1:
            raise ValueError("continuity(): step must be >= 1, got %r" % (step,))
        if vtype not in ('group', 'phase', 'both'):
            raise ValueError("continuity(): vtype must be 'group', 'phase', or 'both', got %r" % (vtype,))

        base_scale = None  # None => 'velocity' mode: normalize per-pair by local |v|
        if isinstance(norm, str):
            if norm == 'velocity':
                base_scale = None
            elif norm == 'auto':
                base_scale = None
                if isinstance(self.params, dict):
                    base_scale = self.params.get('vg_step_max', None)
                if base_scale is None or not np.isfinite(base_scale) or base_scale <= 0:
                    base_scale = 0.5  # aftan()/aftan_pmf()'s own default vg_step_max
            else:
                raise ValueError("continuity(): norm must be 'auto', 'velocity', or a positive "
                                  "float (km/s), got %r" % (norm,))
        elif isinstance(norm, (int, float)) and not isinstance(norm, bool):
            if norm <= 0:
                raise ValueError("continuity(): norm as a number must be > 0, got %r" % (norm,))
            base_scale = float(norm)
        else:
            raise ValueError("continuity(): norm must be 'auto', 'velocity', or a positive "
                              "float (km/s), got %r" % (norm,))
        velocity_mode = (isinstance(norm, str) and norm == 'velocity')

        def _score(v):
            v = np.asarray(v, dtype=np.float64)
            n = len(v)
            jumps = []
            for delta in range(1, step + 1):
                if n <= delta:
                    continue
                v0, v1 = v[:-delta], v[delta:]
                good = np.isfinite(v0) & np.isfinite(v1)
                if not np.any(good):
                    continue
                if velocity_mode:
                    denom = 0.5 * (np.abs(v0[good]) + np.abs(v1[good]))
                    # guard the (degenerate) case of a near/exactly-zero velocity
                    # pair, which would otherwise blow the relative jump up to inf/nan
                    denom = np.where(denom > 0, denom, np.nan)
                else:
                    denom = base_scale * delta
                jumps.append(np.abs(v1[good] - v0[good]) / denom)
            if not jumps:
                return np.nan
            allj = np.concatenate(jumps)
            allj = allj[np.isfinite(allj)]
            if len(allj) == 0:
                return np.nan
            # per-pair continuity, inverted BEFORE aggregating -- see docstring
            allc = 1.0 / (1.0 + allj)
            if stat == 'mean':
                return float(np.mean(allc))
            elif stat == 'median':
                return float(np.median(allc))
            elif stat == 'sum':
                return float(np.sum(allc))
            elif stat == 'min':
                return float(np.min(allc))
            else:
                return float(np.max(allc))

        if vtype == 'group':
            return _score(self.group_velocity)
        if vtype == 'phase':
            return _score(self.phase_velocity)
        return {'group': _score(self.group_velocity), 'phase': _score(self.phase_velocity)}

    def peak_alignment(self, vtype='group', stat='mean'):
        """
        Complementary QC metric to continuity(): whether the picked curve is
        actually riding the dispersion image's own energy peak at each period, or
        has quietly settled on a smoother-but-weaker secondary branch/sidelobe --
        precisely the failure mode continuity() cannot see, since a curve can be
        perfectly smooth from period to period while consistently sitting on the
        wrong (but locally continuous) energy ridge the entire time. Requires this
        object to have been created with store_image=True (the default, so
        self.envelope is populated), since it is built directly from the stored
        envelope image rather than from the picked curve alone.

        For each period with a valid (non-NaN) pick, computes

            alignment = amplitude_at_the_pick / that_period's_own_peak_amplitude

        where "that period's own peak amplitude" is found the SAME way
        get_image(normalize_mode='peak') normalizes each row -- the largest
        GENUINE interior local maximum in that period's envelope (a sample
        strictly greater than both neighbors), i.e. the strongest candidate
        _pick_group_velocity() itself would have seen at that period, regardless
        of which candidate was actually selected (see get_image()'s docstring for
        why a plain row maximum is not a safe reference for a real ambient-noise
        image). alignment is 1.0 (up to interpolation slack) when the pick sits
        exactly on the period's own strongest peak, and drops toward 0 the more
        amplitude is being left on the table by picking a weaker candidate
        instead -- exactly the "pick is far from the energy peak" symptom
        continuity() alone cannot catch.

        ===PARAMETERS===
        vtype: 'group' [default] or 'phase' -- which picked curve's amplitude to
                score. 'group' uses self.amplitude (already stored at the exact
                picked arrival by the analysis that produced this object).
                'phase' has no independently stored amplitude of its own, so its
                envelope amplitude is recovered by re-interpolating self.envelope
                at dist/phase_velocity[i] for each period.
        stat: how the per-period alignment values are combined -- 'mean'
                [default], 'median', 'min' (the QC gate for "is any single period
                sitting far off its own ridge" -- the direct counterpart of
                continuity(stat='min')), 'max', or 'sum'.

        ===RETURNS===
        A single float, ordinarily in [0, 1] (very rarely a hair above 1, when the
        picked amplitude and the peak-finding grid disagree slightly due to
        interpolation -- harmless for QC purposes), or NaN if this object has no
        stored envelope, or no valid picks/amplitudes are available.
        """
        valid_stats = ('mean', 'median', 'sum', 'min', 'max')
        if stat not in valid_stats:
            raise ValueError("peak_alignment(): stat must be one of %s, got %r" % (valid_stats, stat))
        if vtype not in ('group', 'phase'):
            raise ValueError("peak_alignment(): vtype must be 'group' or 'phase', got %r" % (vtype,))
        if self.envelope is None:
            return np.nan

        npts = self.envelope.shape[1]
        t = np.arange(npts) * self.dt
        nper = len(self.period)

        # this period's own peak amplitude, in the same "genuine interior local
        # maximum" sense get_image(normalize_mode='peak') normalizes by -- see
        # that method's docstring for why the plain row maximum is unsafe here.
        row_peak = np.full(nper, np.nan)
        for i in range(nper):
            row = self.envelope[i]
            finite = np.isfinite(row)
            if finite.sum() < 3:
                row_peak[i] = np.nanmax(row) if np.any(finite) else np.nan
                continue
            inner = np.arange(1, len(row) - 1)
            inner = inner[finite[inner] & finite[inner - 1] & finite[inner + 1]]
            is_local_max = (row[inner] > row[inner - 1]) & (row[inner] > row[inner + 1])
            local_max_vals = row[inner[is_local_max]]
            row_peak[i] = np.max(local_max_vals) if len(local_max_vals) > 0 else np.nanmax(row)

        if vtype == 'group':
            amp_at_pick = np.asarray(self.amplitude, dtype=np.float64)
        else:
            amp_at_pick = np.full(nper, np.nan)
            pv = np.asarray(self.phase_velocity, dtype=np.float64)
            for i in range(nper):
                if not np.isfinite(pv[i]) or pv[i] <= 0:
                    continue
                tt = self.dist / pv[i]
                if t[0] <= tt <= t[-1]:
                    amp_at_pick[i] = np.interp(tt, t, self.envelope[i])

        with np.errstate(invalid='ignore', divide='ignore'):
            align = amp_at_pick / np.where(row_peak > 0, row_peak, np.nan)
        align = align[np.isfinite(align)]
        if len(align) == 0:
            return np.nan
        if stat == 'mean':
            return float(np.mean(align))
        elif stat == 'median':
            return float(np.median(align))
        elif stat == 'sum':
            return float(np.sum(align))
        elif stat == 'min':
            return float(np.min(align))
        else:
            return float(np.max(align))

    def plot(self, ax=None, snr_min=None, show='both', figsize=(6, 4.5), **kwargs):
        """
        Quick-look plot of the dispersion curve(s).

        ===PARAMETERS===
        ax: existing matplotlib Axes to plot into. default None: creates a new figure.
        snr_min: if given, mask (as gaps) periods with snr below this threshold (dB).
        show: 'group','phase', or 'both' [default].
        figsize: figure size when a new figure is created.
        kwargs: passed to ax.plot().

        ===RETURNS===
        ax: the matplotlib Axes used.
        """
        show_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            show_fig = True
        mask = self.snr >= snr_min if snr_min is not None else np.ones(len(self.period), dtype=bool)
        if show.lower() in ('group', 'both'):
            v = np.where(mask, self.group_velocity, np.nan)
            ax.plot(self.period, v, '-o', ms=3, label='group velocity', **kwargs)
        if show.lower() in ('phase', 'both'):
            c = np.where(mask, self.phase_velocity, np.nan)
            if np.any(~np.isnan(c)):
                ax.plot(self.period, c, '-s', ms=3, label='phase velocity', **kwargs)
        ax.set_xlabel('Period (s)')
        ax.set_ylabel('Velocity (km/s)')
        ax.set_title('%s dispersion: dist=%.1f km, side=%s' % (self.method, self.dist, str(self.side)))
        ax.legend()
        if show_fig:
            plt.show()
        return ax

    def get_image(self, vmin=None, vmax=None, dv=0.02, normalize=True, normalize_mode='peak'):
        """
        Resample the stored envelope matrix from (period,time) onto (period,group
        velocity), i.e. build the classical AFTAN dispersion-energy image. Requires
        this object to have been created with store_image=True (the default).

        ===PARAMETERS===
        vmin,vmax: group-velocity axis range (km/s). default None: taken from
                self.params['vmin']/['vmax'] (the search range used when this object
                was created).
        dv: velocity-axis sampling step (km/s). default 0.02.
        normalize: normalize each period's (row's) energy, as is conventional for
                an AFTAN image. default True.
        normalize_mode: 'peak' [default] -- normalize each row by its largest
                GENUINE interior local maximum (a sample strictly greater than
                both neighbors), i.e. exactly the kind of peak
                _pick_group_velocity() itself would consider a real detection;
                falls back to the row's plain maximum only for a row with no
                interior local max at all (purely monotonic across the window).
                'max' -- the original, naive per-row maximum; kept for comparison/
                backward compatibility.

                Why this matters: a real ambient-noise correlation's near-zero-lag
                samples often carry a large, non-physical pulse (direct-wave
                leakage, cross-talk, instrument coupling -- not surface-wave
                energy) whose envelope can still be substantial all the way out
                to the fast (vmax) edge of the plotted window, without ever
                forming an interior peak there (it's still rising/falling *into*
                the edge, not peaking inside [vmin,vmax]). With mode='max' that
                edge value sets the entire row's color scale, making the real,
                smaller-but-genuine picked peak elsewhere in the row look
                artificially dim -- so the picked curve can appear to run
                through a "faint" part of the image even though it's correctly
                sitting on the actual local maximum (this is what
                _pick_group_velocity()'s own boundary-pick-rejection logic
                already excludes when picking, so mode='peak' just makes the
                *displayed* color scale consistent with what the curve was
                actually picked against). Values at a genuinely monotonic edge
                can end up >1 after mode='peak' normalization -- see
                plot_image(), which clips the color scale to [0,1] so such
                samples simply saturate rather than washing out the row's real
                peak.

        ===RETURNS===
        vgrid: the group-velocity axis (km/s).
        image: 2-D array, shape (len(self.period), len(vgrid)).
        """
        if self.envelope is None:
            raise ValueError("This DispData has no stored envelope matrix -- create it with "
                              "aftan()/aftan_pmf()'s default store_image=True.")
        npts = self.envelope.shape[1]
        t = np.arange(npts) * self.dt
        if vmin is None:
            vmin = self.params.get('vmin', 1.0)
        if vmax is None:
            vmax = self.params.get('vmax', 5.0)
        vgrid = np.arange(vmin, vmax + 0.5 * dv, dv)
        tgrid = self.dist / vgrid  # time corresponding to each velocity sample
        image = np.full((len(self.period), len(vgrid)), np.nan)
        for i in range(len(self.period)):
            valid = (tgrid >= t[0]) & (tgrid <= t[-1])
            image[i, valid] = np.interp(tgrid[valid], t, self.envelope[i])
        if normalize:
            if normalize_mode == 'peak':
                row_peak = np.full(image.shape[0], np.nan)
                for i in range(image.shape[0]):
                    row = image[i]
                    finite = np.isfinite(row)
                    if finite.sum() < 3:
                        row_peak[i] = np.nanmax(row) if np.any(finite) else np.nan
                        continue
                    inner = np.arange(1, len(row) - 1)
                    inner = inner[finite[inner] & finite[inner - 1] & finite[inner + 1]]
                    is_local_max = (row[inner] > row[inner - 1]) & (row[inner] > row[inner + 1])
                    local_max_vals = row[inner[is_local_max]]
                    # no genuine interior peak (row is monotonic across the window):
                    # nothing better to normalize by than the plain max.
                    row_peak[i] = np.max(local_max_vals) if len(local_max_vals) > 0 else np.nanmax(row)
                row_peak = row_peak[:, None]
            else:
                row_peak = np.nanmax(image, axis=1, keepdims=True)
            row_peak[~(row_peak > 0)] = 1.0
            image = image / row_peak
        return vgrid, image

    def plot_image(self, ax=None, vmin=None, vmax=None, dv=0.02, cmap='jet',
                    normalize=True, normalize_mode='peak',
                    overlay_curve=True, snr_min=None, figsize=(7, 5)):
        """
        Plot the classical AFTAN dispersion-energy image (period vs. group velocity,
        colored by normalized narrow-band envelope amplitude), optionally overlaid
        with the picked group-velocity (and, if available, phase-velocity) curve(s).
        Requires this object to have been created with store_image=True (the default).

        ===PARAMETERS===
        ax: existing matplotlib Axes to plot into. default None: creates a new figure.
        vmin,vmax,dv: passed to get_image().
        cmap: colormap. default 'jet' (the conventional AFTAN-image colormap).
        normalize,normalize_mode: passed to get_image() -- see that method's
                docstring for why normalize_mode='peak' (the default) makes this
                plot's color scale consistent with what was actually picked,
                rather than dominated by a non-physical near-zero-lag pulse's
                tail bleeding into the vmax edge. When normalize=True, the color
                scale is also explicitly clipped to [0,1] below, so any row's
                edge-of-window sample that still exceeds its own interior peak
                (an unremoved monotonic contamination) simply saturates instead
                of stretching the colorbar and dimming everything else.
        overlay_curve: overlay the picked group_velocity (and phase_velocity, if any
                valid values exist) curve(s) on top of the image. default True.
        snr_min: if given, mask (as gaps) overlaid-curve periods with snr below this
                threshold (dB).
        figsize: figure size when a new figure is created.

        ===RETURNS===
        ax: the matplotlib Axes used.
        """
        vgrid, image = self.get_image(vmin=vmin, vmax=vmax, dv=dv, normalize=normalize,
                                       normalize_mode=normalize_mode)
        show_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            show_fig = True
        color_kw = dict(vmin=0, vmax=1) if normalize else {}
        pc = ax.pcolormesh(self.period, vgrid, image.T, shading='auto', cmap=cmap, **color_kw)
        plt.colorbar(pc, ax=ax, label='normalized amplitude' if normalize else 'amplitude')
        if overlay_curve:
            mask = self.snr >= snr_min if snr_min is not None else np.ones(len(self.period), dtype=bool)
            gv = np.where(mask, self.group_velocity, np.nan)
            ax.plot(self.period, gv, 'w--', lw=1.2, label='group velocity')
            pv = np.where(mask, self.phase_velocity, np.nan)
            if np.any(~np.isnan(pv)):
                ax.plot(self.period, pv, 'k:', lw=1.2, label='phase velocity')
            ax.legend(loc='best')
        ax.set_xlabel('Period (s)')
        ax.set_ylabel('Group velocity (km/s)')
        ax.set_title('%s dispersion image: dist=%.1f km, side=%s' % (self.method, self.dist, str(self.side)))
        if show_fig:
            plt.show()
        return ax

    def save(self, filename, overwrite=True):
        """
        Save this DispData to a self-contained HDF5 file: the picked-curve arrays,
        distance/sampling metadata, the full `params` dict the producing method
        (aftan()/aftan_pmf()/...) was called with, and -- when present -- the full
        envelope/phase_matrix dispersion image. Meant for batch dispersion runs over
        many station pairs, where results are written to disk (one file per pair)
        rather than kept in memory; reload with read_dispdata() (a module-level
        function, not a method of this class -- see its docstring).

        ===PARAMETERS===
        filename: output .h5 path. Its parent directory is created if missing.
        overwrite: if False and `filename` already exists, raise FileExistsError
                instead of silently overwriting it. default True.
        """
        if not overwrite and os.path.exists(filename):
            raise FileExistsError("%s already exists (overwrite=False)." % filename)
        outdir = os.path.dirname(os.path.abspath(filename))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with h5py.File(filename, 'w') as f:
            f.attrs['dist'] = float(self.dist) if self.dist is not None else np.nan
            f.attrs['dt'] = float(self.dt) if self.dt is not None else np.nan
            f.attrs['method'] = self.method if self.method is not None else ''
            # side/params can hold None/nested values h5py attrs can't store directly
            # (and params may carry numpy scalars from upstream code) -- JSON-encode
            # both, converting numpy types to native Python first.
            f.attrs['side_json'] = json.dumps(_json_safe(self.side))
            f.attrs['params_json'] = json.dumps(_json_safe(self.params))
            # station-pair metadata (all optional -- see DispData's docstring); stored
            # as one JSON blob rather than individual attrs since several entries are
            # commonly None (net/sta strings, or lon/lat when unavailable).
            f.attrs['station_json'] = json.dumps(_json_safe({
                'src_net': self.src_net, 'src_sta': self.src_sta,
                'src_lon': self.src_lon, 'src_lat': self.src_lat,
                'rcv_net': self.rcv_net, 'rcv_sta': self.rcv_sta,
                'rcv_lon': self.rcv_lon, 'rcv_lat': self.rcv_lat}))
            for name in ('period', 'group_velocity', 'phase_velocity', 'amplitude', 'snr',
                         'inst_period', 'arrival_time', 'phase_pick'):
                arr = getattr(self, name)
                if arr is None:
                    continue
                f.create_dataset(name, data=np.asarray(arr, dtype=np.float64))
            for name in ('envelope', 'phase_matrix'):
                arr = getattr(self, name)
                if arr is not None:
                    f.create_dataset(name, data=np.asarray(arr, dtype=np.float64),
                                      compression='gzip', compression_opts=4)


class PhaseVelocityMap(object):
    """
    Container for the output of seisgo.imaging.eikonal.eikonal_tomography():
    gridded phase- or group-velocity maps as a function of period, assembled by
    stacking many per-virtual-source eikonal travel-time-gradient maps built from
    an ensemble of DispData station-pair dispersion measurements. See
    eikonal_tomography()'s own docstring for the full method description.

    ===Attributes===
    period: 1-D period vector (s).
    lon_grid,lat_grid: 1-D grid coordinate vectors (deg).
    velocity: 3-D array, shape (len(period), len(lat_grid), len(lon_grid)) --
            the stacked (mean-across-sources) phase or group velocity (km/s) at
            each grid node/period. NaN where fewer than `min_sources_per_node`
            sources contributed a valid value.
    uncertainty: same shape as velocity -- the standard deviation across
            contributing sources at each grid node/period (NaN wherever velocity
            is NaN). Small does not by itself mean "well-resolved" if n_sources
            there is also small (or, worse, 1) -- always check both together.
    n_sources: same shape, int -- the number of independent virtual sources that
            contributed a valid value at each grid node/period; the main
            azimuthal-coverage/reliability diagnostic (see eikonal_tomography()'s
            docstring).
    vtype: 'phase' or 'group'.
    method: name of the function that produced this object (currently always
            'eikonal').
    params: dict of the parameters eikonal_tomography() was called with.
    """
    def __init__(self, period, lon_grid, lat_grid, velocity, uncertainty, n_sources,
                 vtype='phase', method='eikonal', params=None):
        self.type = 'Phase Velocity Map'
        self.period = np.asarray(period, dtype=np.float64)
        self.lon_grid = np.asarray(lon_grid, dtype=np.float64)
        self.lat_grid = np.asarray(lat_grid, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.uncertainty = np.asarray(uncertainty, dtype=np.float64)
        self.n_sources = np.asarray(n_sources, dtype=np.int32)
        self.vtype = vtype
        self.method = method
        self.params = params if params is not None else {}

    def __repr__(self):
        nper = len(self.period)
        nlat, nlon = len(self.lat_grid), len(self.lon_grid)
        valid_frac = (np.mean(np.isfinite(self.velocity)) if self.velocity.size else 0.0)
        lines = []
        lines.append("type          :   " + str(self.type))
        lines.append("method        :   " + str(self.method))
        lines.append("vtype         :   " + str(self.vtype))
        if nper > 0:
            lines.append("period        :   (%d,)  (%.3g to %.3g s)" %
                          (nper, np.nanmin(self.period), np.nanmax(self.period)))
        else:
            lines.append("period        :   none")
        lines.append("grid          :   %d lat x %d lon  (lon %.3f to %.3f, lat %.3f to %.3f)" %
                      (nlat, nlon, np.nanmin(self.lon_grid), np.nanmax(self.lon_grid),
                       np.nanmin(self.lat_grid), np.nanmax(self.lat_grid)))
        lines.append("velocity      :   %s  (%.1f%% of nodes valid)" %
                      (str(self.velocity.shape), 100.0 * valid_frac))
        if np.any(self.n_sources > 0):
            lines.append("n_sources     :   min=%d max=%d (at valid nodes)" %
                          (int(np.min(self.n_sources[self.n_sources > 0])),
                           int(np.max(self.n_sources))))
        else:
            lines.append("n_sources     :   none")
        lines.append("params        :   " + str(self.params))
        return "<PhaseVelocityMap object>\n" + "\n".join(lines)

    __str__ = __repr__

    def period_index(self, period):
        """Index into self.period nearest the requested `period` (s)."""
        return int(np.argmin(np.abs(self.period - period)))

    def plot(self, period, ax=None, cmap='viridis_r', vmin=None, vmax=None,
              show_coverage=True, min_sources=1, figsize=(7, 5.5)):
        """
        Map view of the stacked velocity at the period nearest `period`.

        ===PARAMETERS===
        period: target period (s); the nearest available period is used (see
                period_index()).
        ax: existing matplotlib Axes. default None: creates a new figure.
        cmap,vmin,vmax: passed to pcolormesh.
        show_coverage: overlay a contour at n_sources==min_sources-0.5, marking
                the boundary of the region with at least `min_sources`
                contributing virtual sources. default True.
        min_sources: coverage-contour threshold (see show_coverage). default 1.
        figsize: figure size when a new figure is created.

        ===RETURNS===
        ax: the matplotlib Axes used.
        """
        ip = self.period_index(period)
        show_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            show_fig = True
        pc = ax.pcolormesh(self.lon_grid, self.lat_grid, self.velocity[ip], shading='auto',
                            cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pc, ax=ax, label='%s velocity (km/s)' % self.vtype)
        if show_coverage:
            cov = self.n_sources[ip].astype(np.float64)
            if np.any(cov >= min_sources) and np.any(cov < min_sources):
                ax.contour(self.lon_grid, self.lat_grid, cov, levels=[min_sources - 0.5],
                           colors='k', linewidths=0.8)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Eikonal %s-velocity map: T=%.2f s (nearest to requested %.2f s)' %
                      (self.vtype, self.period[ip], period))
        if show_fig:
            plt.show()
        return ax

    def save(self, filename, overwrite=True):
        """
        Save to a self-contained HDF5 file. Reload with
        seisgo.imaging.eikonal.read_phase_velocity_map().

        ===PARAMETERS===
        filename: output .h5 path. Its parent directory is created if missing.
        overwrite: if False and `filename` already exists, raise FileExistsError.
                default True.
        """
        if not overwrite and os.path.exists(filename):
            raise FileExistsError("%s already exists (overwrite=False)." % filename)
        outdir = os.path.dirname(os.path.abspath(filename))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with h5py.File(filename, 'w') as f:
            f.attrs['vtype'] = self.vtype
            f.attrs['method'] = self.method
            f.attrs['params_json'] = json.dumps(_json_safe(self.params))
            f.create_dataset('period', data=self.period)
            f.create_dataset('lon_grid', data=self.lon_grid)
            f.create_dataset('lat_grid', data=self.lat_grid)
            f.create_dataset('velocity', data=self.velocity, compression='gzip', compression_opts=4)
            f.create_dataset('uncertainty', data=self.uncertainty, compression='gzip',
                              compression_opts=4)
            f.create_dataset('n_sources', data=self.n_sources, compression='gzip',
                              compression_opts=4)

