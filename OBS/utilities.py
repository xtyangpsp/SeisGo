#!/usr/bin/env python
# coding: utf-8
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
from collections import OrderedDict
# from scipy.fftpack import fft,fftfreq,ifft
# from scipy.fftpack import rfft,rfftfreq,irfft

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
    this function outputs station info from the obspy inventory object
    (used in S0B)
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
        else: elv.append(0.)

        if inv[0][i][0].location_code:
            location.append(inv[0][i][0].location_code)
        else: location.append('00')

    if len(inv[0])==1:
        sta=sta[0]
        net=net[0]
        lon=lon[0]
        lat=lat[0]
        elv=elv[0]
        location=location[0]
        
    return sta,net,lon,lat,elv,location

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
def plot_trace(tr_list,freqmin=0.02,freqmax=[],size=(10,9),ylabels=[],datalabels=[],               title=[],outfile='test.ps',xlimit=[],subplotpar=[],              mode="subplot",spacing=2.0,colors=[]):
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

        if freqmax==[]:
            freqmax=0.4999*tr.stats.sampling_rate #slightly lower than the Nyquist frequency

        print("station %s.%s, filtered at [%6.3f, %6.3f]" % (tr.stats.network,                                                             tr.stats.station,freqmin,freqmax))
        tc.filter('bandpass',freqmin=freqmin,freqmax=freqmax)

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
            plt.text(np.mean(xlimit),0.9*np.max(tc.data[imin:imax]),"["+str(freqmin)+", "+str(freqmax)+"] Hz",                     horizontalalignment='center',verticalalignment='center',fontsize=12)
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
                plt.text(np.mean(xlimit),0.85*np.max(myymax),"["+str(freqmin)+", "+str(freqmax)+"] Hz",                         horizontalalignment='center',verticalalignment='center',fontsize=14)

    plt.savefig(outfile,orientation='landscape')
    plt.show()
    plt.close()
