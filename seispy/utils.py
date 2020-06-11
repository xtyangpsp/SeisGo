#!/usr/bin/env python
# coding: utf-8
############################################
##Utility functions used in processing seismic data.
############################################
#import needed packages.
import sys
# from warnings import warn
import time
import scipy
import obspy
import pyasdf
import datetime
import os, glob
import numpy as np
# import pandas as pd
import matplotlib.pyplot  as plt
from collections import OrderedDict
from scipy.signal import tukey
from obspy.clients.fdsn import Client
# from scipy.fftpack import fft,fftfreq,ifft
# from scipy.fftpack import rfft,rfftfreq,irfft

####
#
def getdata(net,sta,starttime,endtime,chan='*',source='IRIS',samp_freq=None,
            rmresp=True,rmresp_output='VEL',pre_filt=None,plot=False,debug=False,
            sacheader=False,getstainv=False):
    """
    This is a wrapper that downloads seismic data and (optionally) removes response
    and downsamples if needed. Most of the arguments have the same meaning as for
    obspy.Client.get_waveforms().

    Parameters
    ----------
    net,sta,chan : string
            network, station, and channel names for the request.
    starttime, endtime : UTCDateTime
            Starting and ending date time for the request.
    source : string
            Client names.
            To get a list of available clients:
            >> from obspy.clients.fdsn.header import URL_MAPPINGS
            >> for key in sorted(URL_MAPPINGS.keys()):
                 print("{0:<11} {1}".format(key,  URL_MAPPINGS[key]))
    samp_freq : float
            Target sampling rate. Skip resampling if None.
    rmresp : bool
            Remove response if true. For the purpose of download OBS data and remove
            tilt and compliance noise, the output is "VEL" for pressure data and "DISP"
            for seismic channels.
    rmresp_output : string
            Output format when removing the response, following the same rule as by OBSPY.
            The default is 'VEL' for velocity output.
    pre_filt : :class: `numpy.ndarray`
            Same as the pre_filt in obspy when removing instrument responses.
    plot : bool
            Plot the traces after preprocessing (sampling, removing responses if specified).
    debug : bool
            Plot raw waveforms before preprocessing.
    sacheader : bool
            Key sacheader information in a dictionary using the SAC header naming convention.
    """
    client = Client(source)
    tr = None
    sac=dict() #place holder to save some sac headers.
    #check arguments
    if rmresp:
        if pre_filt is None:
            raise(Exception("Error getdata() - "
                            + " pre_filt is not specified (needed when removing response)"))

    """
    a. Downloading
    """
    if sacheader or getstainv:
        inv = client.get_stations(network=net,station=sta,
                        channel=chan,location="*",starttime=starttime,endtime=endtime,
                        level='response')
        if sacheader:
            tempnet,tempsta,stlo, stla,stel,temploc=sta_info_from_inv(inv)
            sac['knetwk']=tempnet
            sac['kstnm']=tempsta
            sac['stlo']=stlo
            sac['stla']=stla
            sac['stel']=stel
            sac['kcmpnm']=chan
            sac['khole']=temploc

    # pressure channel
    tr=client.get_waveforms(network=net,station=sta,
                    channel=chan,location="*",starttime=starttime,endtime=endtime,attach_response=True)
#     trP[0].detrend()
    tr=tr[0]
    tr.stats['sac']=sac

    print("station "+net+"."+sta+" --> seismic channel: "+chan)

    if plot or debug:
        year = tr.stats.starttime.year
        julday = tr.stats.starttime.julday
        hour = tr.stats.starttime.hour
        mnt = tr.stats.starttime.minute
        sec = tr.stats.starttime.second
        tstamp = str(year) + '.' + str(julday)+'T'+str(hour)+'-'+str(mnt)+'-'+str(sec)
        trlabels=[net+"."+sta+"."+tr.stats.channel]
    """
    b. Resampling
    """
    if samp_freq is not None:
        sps=int(tr.stats.sampling_rate)
        delta = tr.stats.delta
        #assume pressure and vertical channels have the same sampling rat
        # make downsampling if needed
        if sps > samp_freq:
            print("  downsamping from "+str(sps)+" to "+str(samp_freq))
            if np.sum(np.isnan(tr.data))>0:
                raise(Exception('NaN found in trace'))
            else:
                tr.interpolate(samp_freq,method='weighted_average_slopes')
                # when starttimes are between sampling points
                fric = tr.stats.starttime.microsecond%(delta*1E6)
                if fric>1E-4:
                    tr.data = segment_interpolate(np.float32(tr.data),float(fric/(delta*1E6)))
                    #--reset the time to remove the discrepancy---
                    tr.stats.starttime-=(fric*1E-6)
                # print('new sampling rate:'+str(tr.stats.sampling_rate))

    """
    c. Plot raw data before removing responses.
    """
    if plot and debug:
        plot_trace([tr],size=(12,3),title=trlabels,freq=[0.005,0.1],ylabels=["raw"],
                        outfile=net+"."+sta+"_"+tstamp+"_raw.png")

    """
    d. Remove responses
    """
    if rmresp:
        if np.sum(np.isnan(tr.data))>0:
            raise(Exception('NaN found in trace'))
        else:
            try:
                print('  removing response using inv for '+net+"."+sta+"."+tr.stats.channel)
                tr.remove_response(output=rmresp_output,pre_filt=pre_filt,
                                          water_level=60,zero_mean=True,plot=False)

                # Detrend, filter
                tr.detrend('demean')
                tr.detrend('linear')
                tr.filter('lowpass', freq=0.49*samp_freq,
                           corners=2, zerophase=True)
            except Exception as e:
                print(e)
                tr = []

    """
    e. Plot raw data after removing responses.
    """
    if plot:
        plot_trace([trP],size=(12,3),title=trlabels,freq=[0.005,0.1],ylabels=[rmresp_output],
                   outfile=net+"."+sta+"_"+tstamp+"_raw_rmresp.png")

    #
    if getstainv:return tr,inv
    else: return tr

# ##################### qml_to_event_list #####################################
# modified from obspyDMT.utils.event_handler.py
def qml_to_event_list(events_QML):
    """
    convert QML to event list
    :param events_QML:
    :return:
    """
    events = []
    for i in range(len(events_QML)):
        try:
            event_time = events_QML.events[i].preferred_origin().time or \
                         events_QML.events[i].origins[0].time
            event_time_month = '%02i' % int(event_time.month)
            event_time_day = '%02i' % int(event_time.day)
            event_time_hour = '%02i' % int(event_time.hour)
            event_time_minute = '%02i' % int(event_time.minute)
            event_time_second = '%02i' % int(event_time.second)

            if not hasattr(events_QML.events[i], 'preferred_mag'):
                events_QML.events[i].preferred_mag = \
                    events_QML.events[i].magnitudes[0].mag
                events_QML.events[i].preferred_mag_type = \
                    events_QML.events[i].magnitudes[0].magnitude_type
                events_QML.events[i].preferred_author = 'None'
            else:
                if not hasattr(events_QML.events[i], 'preferred_author'):
                    if events_QML.events[i].preferred_magnitude().creation_info:
                        events_QML.events[i].preferred_author = \
                            events_QML.events[i].preferred_magnitude().creation_info.author
                    elif events_QML.events[i].magnitudes[0].creation_info:
                        events_QML.events[i].preferred_author = \
                            events_QML.events[i].magnitudes[0].creation_info.author
        except Exception as error:
            print(error)
            continue
        try:
            if not events_QML.events[i].focal_mechanisms == []:
                if events_QML.events[i].preferred_focal_mechanism()['moment_tensor']['tensor']:
                    focal_mechanism = [
                        events_QML.events[i].preferred_focal_mechanism()
                        ['moment_tensor']['tensor']['m_rr'],
                        events_QML.events[i].preferred_focal_mechanism()
                        ['moment_tensor']['tensor']['m_tt'],
                        events_QML.events[i].preferred_focal_mechanism()
                        ['moment_tensor']['tensor']['m_pp'],
                        events_QML.events[i].preferred_focal_mechanism()
                        ['moment_tensor']['tensor']['m_rt'],
                        events_QML.events[i].preferred_focal_mechanism()
                        ['moment_tensor']['tensor']['m_rp'],
                        events_QML.events[i].preferred_focal_mechanism()
                        ['moment_tensor']['tensor']['m_tp']]
                else:
                    found_foc_mech = False
                    for foc_mech_qml in events_QML.events[i].focal_mechanisms:
                        if foc_mech_qml['moment_tensor']['tensor']:
                            focal_mechanism = [
                                foc_mech_qml['moment_tensor']['tensor']['m_rr'],
                                foc_mech_qml['moment_tensor']['tensor']['m_tt'],
                                foc_mech_qml['moment_tensor']['tensor']['m_pp'],
                                foc_mech_qml['moment_tensor']['tensor']['m_rt'],
                                foc_mech_qml['moment_tensor']['tensor']['m_rp'],
                                foc_mech_qml['moment_tensor']['tensor']['m_tp']
                            ]
                            found_foc_mech = True
                            break
                    if not found_foc_mech:
                        focal_mechanism = False
            else:
                focal_mechanism = False
        except AttributeError:
            print("[WARNING] focal_mechanism does not exist for " \
                  "event: %s -- set to False" % (i+1))
            focal_mechanism = False
        except TypeError:
            focal_mechanism = False
        except Exception as error:
            print(error)
            focal_mechanism = False

        try:
            if not events_QML.events[i].focal_mechanisms == []:
                source_duration = [
                    events_QML.events[i].preferred_focal_mechanism()
                    ['moment_tensor']['source_time_function']['type'],
                    events_QML.events[i].preferred_focal_mechanism()
                    ['moment_tensor']['source_time_function']
                    ['duration']]
                if not source_duration[1]:
                    source_duration = mag_duration(
                        mag=events_QML.events[i].preferred_mag)
            else:
                source_duration = mag_duration(
                    mag=events_QML.events[i].preferred_mag)
        except AttributeError:
            print("[WARNING] source duration does not exist for " \
                  "event: %s -- set to False" % (i+1))
            source_duration = False
        except TypeError:
            source_duration = False
        except Exception as error:
            print(error)
            source_duration = False

        try:
            events.append(OrderedDict(
                [('number', i+1),
                 ('latitude',
                  events_QML.events[i].preferred_origin().latitude or
                  events_QML.events[i].origins[0].latitude),
                 ('longitude',
                  events_QML.events[i].preferred_origin().longitude or
                  events_QML.events[i].origins[0].longitude),
                 ('depth',
                  events_QML.events[i].preferred_origin().depth/1000. or
                  events_QML.events[i].origins[0].depth/1000.),
                 ('datetime', event_time),
                 ('magnitude',
                  events_QML.events[i].preferred_mag),
                 ('magnitude_type',
                  events_QML.events[i].preferred_mag_type),
                 ('author',
                  events_QML.events[i].preferred_author),
                 ('event_id', str(event_time.year) +
                  event_time_month + event_time_day + '_' +
                  event_time_hour + event_time_minute +
                  event_time_second + '.a'),
                 ('origin_id', events_QML.events[i].preferred_origin_id or
                  events_QML.events[i].origins[0].resource_id.resource_id),
                 ('focal_mechanism', focal_mechanism),
                 ('source_duration', source_duration),
                 ('flynn_region', 'NAN'),
                 ]))
        except Exception as error:
            print(error)
            continue
    return events

# sta_info_from_inv(inv) is modified from noise_module (with the same name)
#Check NoisePy: https://github.com/mdenolle/NoisePy
# added functionality to process an array of inventory
def sta_info_from_inv(inv):
    '''
    this function outputs station info from the obspy inventory object.
    PARAMETERS:
    ----------------------
    inv: obspy inventory object
    RETURNS:
    ----------------------
    sta: station name
    net: netowrk name
    lon: longitude of the station
    lat: latitude of the station
    elv: elevation of the station
    location: location code of the station
    '''
    # load from station inventory
    sta=[]
    net=[]
    lon=[]
    lat=[]
    elv=[]
    location=[]

    for i in range(len(inv[0])):
        sta.append(inv[0][i].code)
        net.append(inv[0].code)
        lon.append(inv[0][i].longitude)
        lat.append(inv[0][i].latitude)
        if inv[0][i].elevation:
            elv.append(inv[0][i].elevation)
        else:
            elv.append(0.)

        if len(inv[0][i])>0:
            location.append(inv[0][i][0].location_code)
        else:
            location.append('00')

    if len(inv[0])==1:
        sta=sta[0]
        net=net[0]
        lon=lon[0]
        lat=lat[0]
        elv=elv[0]
        location=location[0]

    return sta,net,lon,lat,elv,location

# split_datetimestr(inv) is modified from NoisePy.noise_module.get_event_list()
#Check NoisePy: https://github.com/mdenolle/NoisePy
def split_datetimestr(dtstr1,dtstr2,inc_hours):
    '''
    this function calculates the datetime list between datetime1 and datetime2 by
    increment of inc_hours in the formate of %Y_%m_%d_%H_%M_%S

    PARAMETERS:
    ----------------
    dtstr1: string of the starting time -> 2010_01_01_0_0
    dtstr2: string of the ending time -> 2010_10_11_0_0
    inc_hours: integer of incremental hours
    RETURNS:
    ----------------
    dtlist: a numpy character list
    '''
    date1=dtstr1.split('_')
    date2=dtstr2.split('_')
    y1=int(date1[0]);m1=int(date1[1]);d1=int(date1[2])
    h1=int(date1[3]);mm1=int(date1[4]);mn1=int(date1[5])
    y2=int(date2[0]);m2=int(date2[1]);d2=int(date2[2])
    h2=int(date2[3]);mm2=int(date2[4]);mn2=int(date2[5])

    d1=datetime.datetime(y1,m1,d1,h1,mm1,mn1)
    d2=datetime.datetime(y2,m2,d2,h2,mm2,mn2)
    dt=datetime.timedelta(hours=inc_hours)

    dtlist = []
    while(d1<d2):
        dtlist.append(d1.strftime('%Y_%m_%d_%H_%M_%S'))
        d1+=dt
    dtlist.append(d2.strftime('%Y_%m_%d_%H_%M_%S'))

    return dtlist

#Stolen from NoisePy
def segment_interpolate(sig1,nfric):
    '''
    this function interpolates the data to ensure all points located on interger times of the
    sampling rate (e.g., starttime = 00:00:00.015, delta = 0.05.)
    PARAMETERS:
    ----------------------
    sig1:  seismic recordings in a 1D array
    nfric: the amount of time difference between the point and the adjacent assumed samples
    RETURNS:
    ----------------------
    sig2:  interpolated seismic recordings on the sampling points
    '''
    npts = len(sig1)
    sig2 = np.zeros(npts,dtype=np.float32)

    #----instead of shifting, do a interpolation------
    for ii in range(npts):

        #----deal with edges-----
        if ii==0 or ii==npts-1:
            sig2[ii]=sig1[ii]
        else:
            #------interpolate using a hat function------
            sig2[ii]=(1-nfric)*sig1[ii+1]+nfric*sig1[ii]

    return sig2

# Modified from Zhitu Ma. Modified by Xiaotao to get filter frequencies from the arguments
# 1. added titles for multiple plots
# 2. determine freqmax as the Nyquist frequency, if not specified
# 3. Added mode with option to plot overlapping figures.
def plot_trace(tr_list,freq=[],size=(10,9),ylabels=[],datalabels=[],\
               title=[],outfile='test.ps',xlimit=[],subplotpar=[],              \
               mode="subplot",spacing=2.0,colors=[]):
    """
    mode: subplot OR overlap
    """
    plt.figure(figsize=size)
    ntr=len(tr_list)
    if len(subplotpar)==0 and mode=="subplot":
        subplotpar=(ntr,1)

    myymin=[]
    myymax=[]
    for itr,tr in enumerate(tr_list,1):
        tt=tr.times()
        if len(xlimit)==0:
            xlimit=[np.min(tt),np.max(tt)]

        imin = np.searchsorted(tt,xlimit[0],side="left")
        imax = np.searchsorted(tt,xlimit[1],side="left")

        tc=tr.copy()

#         if freqmax==[]:
#             freqmax=0.4999*tr.stats.sampling_rate #slightly lower than the Nyquist frequency


        if len(freq)>0:
            print("station %s.%s, filtered at [%6.3f, %6.3f]" % (tc.stats.network,                                                             tr.stats.station,freq[0],freq[1]))
            tc.filter('bandpass',freqmin=freq[0],freqmax=freq[1],zerophase=True)
        else:
            print("station %s.%s" % (tc.stats.network,tc.stats.station))

        if mode=="subplot":
            ax=plt.subplot(subplotpar[0],subplotpar[1],itr)
            plt.tight_layout(pad=spacing)
            if len(colors)==0:
                plt.plot(tt,tc.data)
            elif len(colors)==1:
                plt.plot(tt,tc.data,colors[0])
            else:
                plt.plot(tt,tc.data,colors[itr-1])
            plt.xlabel("time (s)")
            ax.ticklabel_format(axis='x',style='plain')
            if np.max(np.abs(tc.data[imin:imax])) >= 1e+4 or np.max(np.abs(tc.data[imin:imax])) <= 1e-4:
                ax.ticklabel_format(axis='both',style='sci')
            if len(ylabels)>0:
                plt.ylabel(ylabels[itr-1])
            if len(title)>0:
                plt.title(title[itr-1])
            if len(xlimit)>0:
                plt.xlim(xlimit)
            plt.ylim(0.9*np.min(tc.data[imin:imax]),1.1*np.max(tc.data[imin:imax]))
            if len(freq)>0:
                plt.text(np.mean(xlimit),0.9*np.max(tc.data[imin:imax]),"["+str(freq[0])+", "+str(freq[1])+"] Hz", \
                         horizontalalignment='center',verticalalignment='center',fontsize=12)
        else:
            if itr==1:ax=plt.subplot(1,1,1)
            if len(colors)==0:
                plt.plot(tt,tc.data)
            elif len(colors)==1:
                plt.plot(tt,tc.data,colors[0])
            else:
                plt.plot(tt,tc.data,colors[itr-1])
            plt.xlabel("time (s)")
            myymin.append(0.9*np.min(tc.data[imin:imax]))
            myymax.append(1.1*np.max(tc.data[imin:imax]))
            if itr==ntr:
                ax.ticklabel_format(axis='x',style='plain')
                ax.legend(datalabels)
                if len(ylabels)>0:
                    plt.ylabel(ylabels[0])
                if len(title)>0:
                    plt.title(title)
                if len(xlimit)>0:
                    plt.xlim(xlimit)
                plt.ylim(np.min(myymin),np.max(myymax))
                if len(freq)>0:
                    plt.text(np.mean(xlimit),0.85*np.max(myymax),"["+str(freq[0])+", "+str(freq[1])+"] Hz",\
                             horizontalalignment='center',verticalalignment='center',fontsize=14)

    plt.savefig(outfile,orientation='landscape')
    plt.show()
    plt.close()

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
# Modified by Xiaotao to return window starting indices and the option of forcing to slide through full length.
def sliding_window(a, ws, ss=None, wind=None, getindex=False,full_length=False,verbose=False):
    """
    Function to split a data array into overlapping, possibly tapered sub-windows.

    Parameters
    ----------
    a : :class:`~numpy.ndarray`
        1D array of data to split
    ws : int
        Window size in samples
    ss : int
        Step size in samples. If not provided, window and step size
         are equal.
    wind : :class:`~numpy.ndarray`
        1d array to specify the window used to apply taper or None (default).
    getindex : bool
        Save/return the start index for each window if True.
    full_length : bool
        Add an extra window to include the leftover samples to make sure sliding
        through the entire trace with full-length. This is done by measuring one
        window starting from the end backward. When False, this function skips the
        tailing samples if less than the window size.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        1D array of windowed data
    nd : int
        Number of windows
    idx : :class:`~numpy.ndarray`
        (Optional) The starting indices of the windows, with the size of [nd,1]

    """

    if full_length and verbose:
        print("WARNING: Force slide to the full length, the last window measures backward from the end.")
    if ws > len(a):
        raise(Exception("Error slicing() - window size is bigger than data length."))

    if ss is None:
        # no step size was provided. Return non-overlapping windows
        ss = ws

    # Calculate the number of windows to return, ignoring leftover samples, and
    # allocate memory to contain the samples
    nd = len(a) // ss

    tailcare=False

    if (nd-1)*ss + ws > len(a):
        if full_length:
            tailcare = True
        else:
            nd = nd - 1
    elif (nd-1)*ss + ws < len(a) and full_length:
        tailcare = True
        nd = nd + 1

    out = np.ndarray((nd, ws), dtype=a.dtype)
    idx = np.ndarray((nd,),dtype=int)

    if nd==0:
        if wind is not None:
            out = a * wind
        else:
            out = a
        idx=0
    else:
        for i in range(nd):
            # "slide" the window along the samples
            start = i * ss
            stop = start + ws
            # print(i,start,stop,len(a))
            # print(i,nd)
            if stop > len(a) and tailcare:
                    stop = len(a)
                    start = stop - ws
                    # print(i,start,stop)

            if stop <= len(a):
                if wind is not None:
                    out[i] = a[start: stop] * wind
                else:
                    out[i] = a[start: stop]

                idx[i] = start
            # idx[i][1] = stop

    if getindex:
        return out,nd,idx
    else:
        return out, nd

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def calculate_windowed_fft(a, fs, ws, ss=None, wind=None,getindex=False,full_length=False):
    """
    Calculates windowed Fourier transform

    Parameters
    ----------
    a : :class:`~numpy.ndarray`
        1d array
    fs : int
        sampling rate (samples per second)
    ws : int
        Window size, in number of samples
    ss : int
        Step size, or number of samples until next window
    wind : :class:`~numpy.ndarray`
        1d array to specify the window used to apply taper or None (default).
    getindex : bool
        Save/return the start index for each window if True.
    full_length : bool
        Add an extra window to include the leftover samples to make sure sliding
        through the entire trace with full-length. This is done by measuring one
        window starting from the end backward. When False, this function skips the
        tailing samples if less than the window size.

    Returns
    -------
    ft : :class:`~numpy.ndarray`
        Fourier transform of trace
    f : :class:`~numpy.ndarray`
        Frequency axis in Hz
    idx : :class:`~numpy.ndarray`
        (Optional) The starting indices of the windows, with the size of [nd,1]
    """

    n2 = _npow2(ws)
    f = np.fft.rfftfreq(n2,1/fs)

    # Extract sliding windows
    if getindex:
        tr, nd,idx = sliding_window(a, ws, ss, wind=wind,getindex=True,
                                    full_length=full_length)
    else:
        tr, nd = sliding_window(a, ws, ss,wind=wind,
                                    full_length=full_length)

    # Fourier transform
    ft = np.fft.fft(tr, n=n2)
    if getindex:
        return ft,f,idx
    else:
        return ft, f


def plot_slidingwindows(duration=3600*6,fs=20,window=7200,
                        overlaps=[None,0.1,0.1,0.2,0.2,0.3],
                        tapers=[None,None,0.05,0.05,0.1,0.1],
                        full_length=True,size=(12,12),save=False,
                        format='png'):
    """
    This function plots tapered sliding windows for illustration purpose.

    Parameters
    ----------
    duration: length of the demo data in seconds.
    fs: sampling rate of the data, used to get time information
    window: the window length you want to test.
    overlaps: an array specifying the series of window overlaps (0.0-1.0) for test.
    tapers: window ends will be tapered.
    """
    data=np.zeros((duration*fs,))
    t=np.arange(len(data))/fs
    ws=int(window*fs)
    plt.figure(figsize=size)
    print("connecting locations")
    print("start  end")
    colorlist=['k','b','g','c','y','r','m']
    for i in range(len(overlaps)):
        if overlaps[i] is None:
            ol=0
        else:
            ol=overlaps[i]

        if tapers[i] is None:
            tp=0
        else:
            tp=tapers[i]

        tps = int(0.5*window*ol*fs) #location where the two windows connect.
        step = int(window*(1-ol)*fs)
        wind=tukey(ws,2*tp)
        print(tps/fs,window - tps/fs)

        dout,nd,idx=sliding_window(data,ws,ss=step,getindex=True,
                                        full_length=full_length,verbose=False)
        ax=plt.subplot(len(tapers),1,i+1)
        if len(idx) > len(colorlist):
            windcolor=colorlist*(len(idx)//len(colorlist) + 1)
        else:
            windcolor=colorlist

        for j in range(len(idx)):
            plt.tight_layout(pad=1)
            plt.plot(t[np.arange(ws)+idx[j]],wind,windcolor[j])

            if j >0 and j+1 < len(idx):
                plt.plot(t[tps+j*step],1,'og')
                plt.plot(t[int(ol*window*fs)+j*step],1,'^r')
            plt.title("overlap: "+str(ol)+", one-side taper: "+str(tp))

        plt.xlim((np.min(t),np.max(t)))
        ax.legend(['tukey','tukey','connection','overlap'])

    if save:
        plt.savefig("slidingwindows_illustration."+format)
    else:
        plt.show()

# def smooth(data, np, poly=0, axis=0):
#     return savgol_filter(data, np, poly, axis=axis, mode='wrap')

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def smooth(data, nd, axis=0):
    """
    Function to smooth power spectral density functions from the convolution
    of a boxcar function with the PSD

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Real-valued array to smooth (PSD)
    nd : int
        Number of samples over which to smooth
    axis : int
        axis over which to perform the smoothing

    Returns
    -------
    filt : :class:`~numpy.ndarray`, optional
        Filtered data

    """
    if np.any(data):
        if data.ndim > 1:
            filt = np.zeros(data.shape)
            for i in range(data.shape[::-1][axis]):
                if axis == 0:
                    filt[:, i] = np.convolve(
                        data[:, i], np.ones((nd,))/nd, mode='same')
                elif axis == 1:
                    filt[i, :] = np.convolve(
                        data[i, :], np.ones((nd,))/nd, mode='same')
        else:
            filt = np.convolve(data, np.ones((nd,))/nd, mode='same')
        return filt
    else:
        return None

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def admittance(Gxy, Gxx):
    """
    Calculates admittance between two components

    Parameters
    ---------
    Gxy : :class:`~numpy.ndarray`
        Cross spectral density function of `x` and `y`
    Gxx : :class:`~numpy.ndarray`
        Power spectral density function of `x`

    Returns
    -------
    : :class:`~numpy.ndarray`, optional
        Admittance between `x` and `y`

    """

    if np.any(Gxy) and np.any(Gxx):
        return np.abs(Gxy)/Gxx
    else:
        return None

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def coherence(Gxy, Gxx, Gyy):
    """
    Calculates coherence between two components

    Parameters
    ---------
    Gxy : :class:`~numpy.ndarray`
        Cross spectral density function of `x` and `y`
    Gxx : :class:`~numpy.ndarray`
        Power spectral density function of `x`
    Gyy : :class:`~numpy.ndarray`
        Power spectral density function of `y`

    Returns
    -------
    : :class:`~numpy.ndarray`, optional
        Coherence between `x` and `y`

    """

    if np.any(Gxy) and np.any(Gxx) and np.any(Gxx):
        return np.abs(Gxy)**2/(Gxx*Gyy)
    else:
        return None

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def phase(Gxy):
    """
    Calculates phase angle between two components

    Parameters
    ---------
    Gxy : :class:`~numpy.ndarray`
        Cross spectral density function of `x` and `y`

    Returns
    -------
    : :class:`~numpy.ndarray`, optional
        Phase angle between `x` and `y`

    """

    if np.any(Gxy):
        return np.angle(Gxy)
    else:
        return None

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def rotate_dir(tr1, tr2, direc):

    d = -direc*np.pi/180.+np.pi/2.
    rot_mat = np.array([[np.cos(d), -np.sin(d)],
                        [np.sin(d), np.cos(d)]])

    v12 = np.array([tr2, tr1])
    vxy = np.tensordot(rot_mat, v12, axes=1)
    tr_2 = vxy[0, :]
    tr_1 = vxy[1, :]

    return tr_1

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def ftest(res1, pars1, res2, pars2):

    from scipy.stats import f as f_dist

    N1 = len(res1)
    N2 = len(res2)

    dof1 = N1 - pars1
    dof2 = N2 - pars2

    Ea_1 = np.sum(res1**2)
    Ea_2 = np.sum(res2**2)

    Fobs = (Ea_1/dof1)/(Ea_2/dof2)

    P = 1. - (f_dist.cdf(Fobs, dof1, dof2) - f_dist.cdf(1./Fobs, dof1, dof2))

    return P

def _npow2(x):
    return 1 if x == 0 else 2**(x-1).bit_length()

#save trace to files.
def save2asdf(fname,data,tag,sta_inv=None,group='waveforms',para=None):
    """
    A wrapper to save obspy stream to asdf file.

    Parameters
    ----------
    fname : string
        Output ASDF file name, with *.h5 extension.
    data :: class `~obspy.core.Stream` or class `~numpy.ndarray`
        Obspy Stream or numpy.ndarray object. For stream, all traces should belong to one single station,
        particularly when sta_inv is provided.
    tag :: string list
        List of tags for each trace in the `data` object.
    sta_inv : station inventory
        Staion xml (obspy station inventory).
    group : string
        Group to save the data. Available options include 'waveforms', 'auxiliary'
    para : dictionary
        A dictionary to store saving parameters.
    """
    if group == 'waveforms':
        if len(data) != len(tag):
            raise(Exception('save2asdf: the stream and tag list should have the same length.'))

    if not os.path.isfile(fname):
        ds=pyasdf.ASDFDataSet(fname,mpi=False,compression="gzip-3",mode='w')
    else:
        ds=pyasdf.ASDFDataSet(fname,mpi=False,compression="gzip-3",mode='a')

    #save
    if sta_inv is not None:
        ds.add_stationxml(sta_inv)

    if group == 'waveforms':
        for i in range(len(data)):
            ds.add_waveforms(data[i],tag=tag[i])
    elif group == 'auxiliary':
        try:
            data_type=para['data_type']
            data_path=para['data_path']
            parameters = para['parameters']
        except Exception as e:
            raise(Exception('save2adsf: '+e))

        try:
            provenance_id=para['provenance_id']
        except Exception as e:
            provenance_id=None

        ds.add_auxiliary_data(data,data_type,data_path,parameters=parameters,
                            provenance_id=provenance_id)
