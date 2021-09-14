#define key classes
import os
import obspy
import pyasdf
from obspy.core import Trace,Stream
import numpy as np
import matplotlib.pyplot as plt
from obspy.io.sac.sactrace import SACTrace
from obspy.signal.filter import bandpass
from scipy.fftpack import fft,ifft,fftfreq,next_fast_len
from seisgo import utils,stacking
######
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

    def __str__(self):
        """
        Display key content of the object.
        """
        print("network      :   "+str(self.net))
        print("station      :   "+str(self.sta))
        print("location     :   "+str(self.loc))
        print("channel      :   "+str(self.chan))
        print("longitude   :   "+str(self.lon))
        print("latitude    :   "+str(self.lat))
        print("elevation   :   "+str(self.ele))

        print("")

        return "<Station object>"

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
            self.net=trace[0].stats.network
            self.sta=trace[0].stats.station
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
            self.smooth=smooth
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
        # cut daily-long data into smaller segments (dataS always in 2D)
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
            Nfft = int(next_fast_len(int(dataS.shape[1])))
            if white.ndim == 1:
                axis = 0
            elif white.ndim == 2:
                axis = 1
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

    def __str__(self):
        """
        Display key content of the object.
        """
        print("id           :   "+str(self.id))
        print("net          :   "+str(self.net))
        print("sta          :   "+str(self.sta))
        print("loc          :   "+str(self.loc))
        print("chan         :   "+str(self.chan))
        print("lon          :   "+str(self.lon))
        print("lat          :   "+str(self.lat))
        print("ele          :   "+str(self.ele))
        print("dt           :   "+str(self.dt))
        print("freqmin      :   "+str(self.freqmin))
        print("freqmax      :   "+str(self.freqmax))
        print("time_norm    :   "+self.time_norm)
        print("freq_norm    :   "+self.freq_norm)
        print("smooth       :   "+str(self.smooth))
        print("win_len      :   "+str(self.win_len))
        print("step         :   "+str(self.step))
        if self.std is not None:
            print("std          :   "+str(self.std.shape))
        else:
            print("std          :   none")
        if self.time is not None:
            print("time         :   "+str(obspy.UTCDateTime(self.time[0]))+" to "+str(obspy.UTCDateTime(self.time[-1])))
        else:
            print("time         :   none")
        print("Nfft         :   "+str(self.Nfft))
        print("misc         :   "+str(self.misc))
        if self.data is not None:
            print("data         :   "+str(self.data.shape))
        else:
            print("data         :   none")
        print("")
        return "<FFTData object>"

    def __add__(f1,f2):
        """
        Merge two FFTData objects with the same id. Only merge [time],[std],[data] attributes.
        """
        if f1.id != f2.id:
            raise ValueError('The object to be merged has a different ID (net.sta.loc.chan). Cannot merge!')

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
        if side.lower() not in ["a","p","n"]:
            raise ValueError("Wrong side attribute value [%s], which has to be one of A, N, P."%(side))
        else:
            self.side=side
        self.substack=substack
        self.misc=misc

    def __str__(self):
        """
        Display key content of the object.
        """
        print("type     :   "+str(self.type))
        print("id       :   "+str(self.id))
        print("net      :   "+str(self.net))
        print("sta      :   "+str(self.sta))
        print("loc      :   "+str(self.loc))
        print("chan     :   "+str(self.chan))
        print("lon      :   "+str(self.lon))
        print("lat      :   "+str(self.lat))
        print("ele      :   "+str(self.ele))
        print("cc_comp  :   "+str(self.cc_comp))
        print("lag      :   "+str(self.lag))
        print("dt       :   "+str(self.dt))
        print("cc_len   :   "+str(self.cc_len))
        print("cc_step  :   "+str(self.cc_step))
        print("dist     :   "+str(self.dist))
        print("side     :   "+str(self.side))
        if self.time is not None:
            if self.substack:
                print("time     :   "+str(obspy.UTCDateTime(self.time[0]))+" to "+str(obspy.UTCDateTime(self.time[-1])))
            else:
                print("time     :   "+str(obspy.UTCDateTime(self.time)))
        else:
            print("time     :   none")
        print("substack :   "+str(self.substack))
        if self.stack_method is not None:
            print("stack_method:"+str(self.stack_method))
        if self.data is not None:
            print("data     :   "+str(self.data.shape))
            print(str(self.data))
        else:
            print("data     :   none")
        print("")

        return "<CorrData object>"

    def __add__(c1,c2):
        """
        Merge with another object for the same station pair. The idea is to merge multiple sets
        of CorrData at different time chunks. Therefore, this function will merge the following
        attributes only: <time>,<data>

        **Note: substack will be set to True after merging, regardless the value in the original object.**
        """
        #sanity check: stop merging and raise error if the two objects have different IDs.
        if c1.id != c2.id:
            raise ValueError('The object to be merged has a different ID (net.sta.loc.chan). Cannot merge!')
        if c1.side != c2.side:
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

    def merge(self,c):
        """
        Merge with another object for the same station pair. The idea is to merge multiple sets
        of CorrData at different time chunks. Therefore, this function will merge the following
        attributes only: <time>,<data>

        **Note: substack will be set to True after merging, regardless the value in the original object.**

        ===PARAMETERS===
        c: the other CorrData object to merge with.
        """
        #sanity check: stop merging and raise error if the two objects have different IDs.
        if self.id != c.id:
            raise ValueError('The object to be merged has a different ID (net.sta.loc.chan). Cannot merge!')
        if not self.substack:
            self.time=np.reshape(self.time,(1))
            self.data=np.reshape(self.data,(1,self.data.shape[0]))

        if not c.substack:
            c.time=np.reshape(c.time,(1))
            if np.ndim(c.data)==1:
                c.data=np.reshape(c.data,(1,c.data.shape[0]))

        self.time=np.concatenate((self.time,c.time))
        self.data=np.concatenate((self.data,c.data),axis=0)

        self.substack=True

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
                        baz=self.baz,time=self.time,substack=self.substack,\
                        side=self.side,misc=self.misc)
        if not dataless:
            cout.data=self.data

        return cout

    def stack(self,win_len=None,method='linear',overwrite=True,ampcut=20,verbose=False):
        '''
        This function stacks the cross correlation data. It will overwrite the
        [data] attribute with the stacked trace, if overwrite is True. Substack will
        be set to False if win_len is None or there is only one trace left.

        PARAMETERS:
        ----------------------
        in_len: windown length in seconds for the substack, over which all the
                corrdata.data subset will be stacked. If None [default],it stacks
                all data into one single trace.
        method: stacking method, could be: linear, robust, pws, acf, or nroot.
        overwrite: if True, it replaces the data attribute in CorrData. Otherwise,
                    it returns the stacked data as a vector. Default: True.
        ampcut: used in QC, only stack traces that satisfy ampmax<ampcut*np.median(ampmax)).
                Default: 20.

        RETURNS:
        -----------------------
        Only returns when overwrite is False.

        ds: stacked data.
        ts: timeflag of the substacks, only returns when win_len is NOT None.
        '''
        if isinstance(method,list):method=method[0]
        if win_len is None:
            if self.substack:
                cc_temp = utils.demean(self.data)
                ampmax = np.max(cc_temp,axis=1)
                tindx  = np.where( (ampmax<ampcut*np.median(ampmax)) & (ampmax>0))[0]
                nstacks=len(tindx)
                if nstacks >0:
                    cc_array = cc_temp[tindx,:]

                    # do stacking
                    ds = np.zeros((self.data.shape[1]),dtype=self.data.dtype)
                    if nstacks==1: ds=cc_array
                    else:
                        if method == 'linear':
                            ds = np.mean(cc_array,axis=0)
                        elif method == 'pws':
                            ds = stacking.pws(cc_array,1.0/self.dt)
                        elif method == 'robust':
                            ds = stacking.robust_stack(cc_array)[0]
                        elif method == 'acf':
                            ds = stacking.adaptive_filter(cc_array,1)
                        elif method == 'nroot':
                            ds = stacking.nroot_stack(cc_array,2)
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
                print('substack is set to: False. No stacking applicable.')
                pass
        else: #### stacking over segments of time windows.
            if np.ndim(self.data)>1:
                if verbose: print('Stacking with given windown len %f'%(win_len))

                win=np.arange(self.time[0],self.time[-1],win_len)  #all time chunks
                ts_temp=[]
                ds=np.ndarray((len(win),self.data.shape[1]),dtype=self.data.dtype)
                ngood=[]
                for i in range(len(win)):
                    widx=np.where((self.time>=win[i]) & (self.time<win[i]+win_len))[0]
                    if len(widx) >0:
                        cc0 = utils.demean(self.data[widx,:])
                        ampmax = np.max(cc0,axis=1)
                        tindx  = np.where( (ampmax<ampcut*np.median(ampmax)) & (ampmax>0))[0]
                        nstacks=len(tindx)
                        dstack = np.zeros((self.data.shape[1]),dtype=self.data.dtype)
                        if nstacks>0:
                            cc_array = cc0[tindx,:]

                            # do stacking
                            if nstacks==1: dstack=cc_array
                            else:
                                if method == 'linear':
                                    dstack = np.mean(cc_array,axis=0)
                                elif method == 'pws':
                                    dstack = stacking.pws(cc_array,1.0/self.dt)
                                elif method == 'robust':
                                    dstack = stacking.robust_stack(cc_array)[0]
                                elif method == 'acf':
                                    dstack = stacking.adaptive_filter(cc_array,1)
                                elif method == 'nroot':
                                    dstack = stacking.nroot_stack(cc_array,2)

                            ds[i,:]=dstack
                            ngood.append(i)
                            ts_temp.append(self.time[widx[0]])

                #
                ts=np.array(ts_temp)
                ds=ds[ngood,:]

                if overwrite:
                    self.data=ds
                    self.time=ts
                    if len(ngood) ==1: self.substack = False
                    else: self.substack=True
                    self.stack_method=method
                else:
                    return ts,ds
            else:
                self.substack = False
                if overwrite: pass
                else:
                    return [],[]

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
        if side.lower()=="A" or side.lower()=="a":
            if verbose: print("Splitting negative and positive sides.")
        else:
            print("side attribute is %s. Only splits when side is A."%(self.side))
            return cout

        dt=self.dt
        #
        #initiate as zeros

        if self.substack:
            nhalfpoint=np.int(self.data.shape[1]/2)

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
            nhalfpoint=np.int(self.data.shape[0]/2)
            d_p=np.zeros((nhalfpoint+1),dtype=self.data.dtype)
            d_n=np.zeros((nhalfpoint+1),dtype=self.data.dtype)

            if taper:
                d_p=utils.taper(self.data[nhalfpoint:],
                                            fraction=taper_frac,maxlen=taper_maxlen)
                d_n=np.flip(utils.taper(self.data[:nhalfpoint+1],
                                            fraction=taper_frac,maxlen=taper_maxlen),axis=1)
            else:
                d_p=self.data[nhalfpoint:]
                d_n=np.flip(self.data[:nhalfpoint+1],axis=1)

        c_n=self.copy()
        c_n.side="N"
        c_n.data=d_n
        cout.append(c_n)

        c_p=self.copy()
        c_p.side="P"
        c_p.data=d_p
        cout.append(c_p)

        return cout

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
                nhalfpoint=np.int(self.data.shape[1]/2)
                #positive side
                egf[:,nhalfpoint:]=utils.taper(-1.0*np.gradient(self.data[:,nhalfpoint:],axis=1)/dt,
                                                fraction=taper_frac,maxlen=taper_maxlen)
                #negative side
                egf[:,:nhalfpoint+1]=np.flip(utils.taper(np.gradient(np.flip(self.data[:,:nhalfpoint+1],axis=1),
                                                axis=1)/dt,fraction=taper_frac,maxlen=taper_maxlen),axis=1)
                egf[:,[0,nhalfpoint,-1]]=0
            else:
                egf=utils.taper(-1.0*np.gradient(self.data,axis=1)/dt,
                                                fraction=taper_frac,maxlen=taper_maxlen)
        else:
            if side.lower()=="a":
                nhalfpoint=np.int(self.data.shape[0]/2)
                #positive side
                egf[nhalfpoint:]=utils.taper(-1.0*np.gradient(self.data[nhalfpoint:])/dt,
                                            fraction=taper_frac,maxlen=taper_maxlen)
                #negative side
                egf[:nhalfpoint+1]=np.flip(utils.taper(np.gradient(np.flip(self.data[:nhalfpoint+1]))/dt,
                                            fraction=taper_frac,maxlen=taper_maxlen))
                egf[[0,nhalfpoint,-1]]=0
            else:
                egf=utils.taper(-1.0*np.gradient(self.data)/dt,
                                            fraction=taper_frac,maxlen=taper_maxlen)
        self.data=egf
        self.type="Empirical Green's Functions"

    def to_asdf(self,file,v=True):
        """
        Save CorrData object too asdf file.
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
        parameters = {'dt':self.dt,
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
            'side':self.side}

        with pyasdf.ASDFDataSet(file,mpi=False) as ccf_ds:
            ccf_ds.add_auxiliary_data(data=self.data, data_type=netsta_pair, path=chan_pair, parameters=parameters)
        if v: print('CorrData saved to: '+file)


    def to_sac(self,outdir='.',file=None,v=True):
        """
        Save CorrData object to sac file.

        ====PARAMETERS====
        outdir: output file directory. default is the current folder.
        file: specify file name, ONLY when there is only one trace. i.e., substack is False.
        v: verbose, default is True.
        """
        try:
            if not os.path.isdir(outdir):os.makedirs(outdir)
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
        if not self.substack:
            corrtime=obspy.UTCDateTime(self.time)
            nzyear=corrtime.year
            nzjday=corrtime.julday
            nzhour=corrtime.hour
            nzmin=corrtime.minute
            nzsec=corrtime.second
            nzmsec=corrtime.microsecond

            if file is None:
                file=str(corrtime).replace(':', '-')+'_'+self.id+'_'+self.cc_comp+'.sac'
            sac = SACTrace(nzyear=nzyear,nzjday=nzjday,nzhour=nzhour,nzmin=nzmin,nzsec=nzsec,nzmsec=nzmsec,
                           b=b,delta=self.dt,stla=rlat,stlo=rlon,stel=sele,evla=slat,evlo=slon,evdp=rele,
                           evel=rele,dist=self.dist,az=self.az,baz=self.baz,data=self.data)

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
                    ofile=str(corrtime).replace(':', '-')+'_'+self.id+'_'+self.cc_comp+'.sac'
                    sacfile  = os.path.join(outdir,ofile)
                else:
                    sacfile  = os.path.join(outdir,file)
                sac = SACTrace(nzyear=nzyear,nzjday=nzjday,nzhour=nzhour,nzmin=nzmin,nzsec=nzsec,nzmsec=nzmsec,
                               b=b,delta=self.dt,stla=rlat,stlo=rlon,stel=sele,evla=slat,evlo=slon,evdp=rele,
                               evel=rele,dist=self.dist,az=self.az,baz=self.baz,data=self.data[i,:])

                sac.write(sacfile,byteorder='big')
                if v: print('saved sac to: '+sacfile)

    def plot(self,freqmin=None,freqmax=None,lag=None,save=False,figdir=None,figsize=(10,8),
            figname=None,format='png',stack_method='linear',get_stack=False):
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
        stack_method: method to get the stack, default is 'linear'
        get_stack: returns the sacked trace if True. Default is False.
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
            timestamp = np.empty(ttime.size,dtype='datetime64[s]')
            # print(data.shape)
            nwin = data.shape[0]
            amax = np.zeros(nwin,dtype=np.float32)
            if nwin==0:
                print('continue! no enough trace to plot!')
                return

            tmarks = []
            data_normalizd=np.zeros(data.shape)

            # load cc for each station-pair
            for ii in range(nwin):
                if freqmin is not None and freqmax is not None:
                    data[ii] = bandpass(data[ii],freqmin,freqmax,1/dt,corners=4, zerophase=True)
                data[ii] = utils.taper(data[ii]-np.mean(data[ii]),maxlen=10)
                amax[ii] = np.max(np.abs(data[ii]))
                data_normalizd[ii] = data[ii]/amax[ii]
                timestamp[ii] = obspy.UTCDateTime(ttime[ii])
                tmarks.append(obspy.UTCDateTime(ttime[ii]).strftime('%Y-%m-%dT%H:%M:%S'))

            if stack_method == 'linear':
                dstack = np.mean(data,axis=0)
            elif stack_method == 'pws':
                dstack = stacking.pws(data,1.0/dt)
            elif stack_method == 'robust':
                dstack = stacking.robust_stack(data)[0]
            elif stack_method == 'acf':
                dstack = stacking.adaptive_filter(data,1)
            elif stack_method == 'nroot':
                dstack = stacking.nroot_stack(data,2)
            del data
    #         dstack_robust=stack.robust_stack(data)[0]

            # plotting
            if nwin>10:
                tick_inc = int(nwin/5)
            else:
                tick_inc = 2

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(6,1,(1,4))
            if side.lower()=="a":
                extent=[-lag0,lag0,nwin,0]
            else:
                extent=[0,lag0,nwin,0]
            ax.matshow(data_normalizd,cmap='seismic',extent=extent,aspect='auto')
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
            tmarks=obspy.UTCDateTime(ttime).strftime('%Y-%m-%dT%H:%M:%S')

            if side.lower()=="a":
                tx=np.arange(-lag0,lag0+0.5*dt,dt)
            else:
                tx=np.arange(0,lag0+0.5*dt,dt)
            if len(tx)>len(data):tx=tx[:-1]
            plt.figure(figsize=figsize)
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
class DVVData(object):
    """
    Object to store dv/v (seismic velocity change) data.
    ======= Attributes ======
    STATION INFORMATION:
    net=['',''],sta=['',''],loc=['',''],chan=['',''],
    lon=[0.0,0.0],lat=[0.0,0.0],ele=[0.0,0.0],cc_comp='',
    dist=0.0,az=0.0,baz=0.0: parameters specifying the stations.
    
    DVV PARAMETERS:
    method=None,window=None,dt=None,time=None,freq=None,misc=dict()

    misc is a dictionary that stores additional parameters.

    DVV DATA:
    data1=None,data2=None: data1 is for dvv measurement using negative side correlation data. data2 is for the positive side.

    ======= Methods ======
    to_asdf(): save to asdf file.
    plot(): simple plotting function to display the cross-correlation data.
    """
    def __init__(self,net=['',''],sta=['',''],loc=['',''],chan=['',''],\
                    lon=[0.0,0.0],lat=[0.0,0.0],ele=[0.0,0.0],cc_comp='',dist=0.0,\
                    method=None,window=None,dt=None,az=0.0,baz=0.0,time=[],freq=None,\
                    data1=None,data2=None,misc=dict()):
        self.type='dv/v Data'
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
        self.freq=freq
        self.dt=dt
        self.dist=dist
        self.az=az
        self.baz=baz
        self.time=time
        self.data1=data1
        self.data2=data2
        self.method=method
        self.window=window
        self.misc=misc

    def __str__(self):
        """
        Display key content of the object.
        """
        print("type     :   "+str(self.type))
        print("id       :   "+str(self.id))
        print("net      :   "+str(self.net))
        print("sta      :   "+str(self.sta))
        print("loc      :   "+str(self.loc))
        print("chan     :   "+str(self.chan))
        print("lon      :   "+str(self.lon))
        print("lat      :   "+str(self.lat))
        print("ele      :   "+str(self.ele))
        print("cc_comp  :   "+str(self.cc_comp))
        print("dt       :   "+str(self.dt))
        print("dist     :   "+str(self.dist))
        if self.time is not None:
            if self.substack:
                print("time     :   "+str(obspy.UTCDateTime(self.time[0]))+" to "+str(obspy.UTCDateTime(self.time[-1])))
            else:
                print("time     :   "+str(obspy.UTCDateTime(self.time)))
        else:
            print("time     :   none")
        print("substack :   "+str(self.substack))
        if self.method is not None:
            print("method:"+str(self.stack_method))
        if self.data1 is not None:
            print("data1 [N]:   "+str(self.data1.shape))
            print(str(self.data1))
        else:
            print("data1 [N]:   none")
        if self.data2 is not None:
            print("data2 [P]:   "+str(self.data2.shape))
            print(str(self.data2))
        else:
            print("data2 [P]:   none")
        print("")

        return "<DVVData object>"

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
