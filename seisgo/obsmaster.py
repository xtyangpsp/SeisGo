#!/usr/bin/env python
# coding: utf-8
# import needed packages and functions
from seisgo import utils,downloaders
from seisgo.types import Power,Cross,Rotation
# from warnings import warn
from scipy.signal import spectrogram, detrend,tukey
from scipy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time,sys
import os, glob
from obspy.clients.fdsn import Client
from obspy.core import Stream, Trace, read

####
def getobsdata(net,sta,starttime,endtime,pchan='HDH',source='IRIS',samp_freq=None,
            rmresp=True,pre_filt=None,plot=False,debug=False,
            sacheader=False,getstainv=False):
    """
    Function to download 4 component OBS data and (optionally) remove response and downsample if needed.
    Most of the arguments have the same meaning as for obspy.Client.get_waveforms().

    Parameters
    ----------
    net,sta : string
            network and station names for the request.
    starttime, endtime : UTCDateTime
            Starting and ending date time for the request.
    pchan: pressure channel name, default is "HDH".
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
    pre_filt : :class: `numpy.ndarray`
            Same as the pre_filt in obspy when removing instrument responses.
    plot : bool
            Plot the traces after preprocessing (sampling, removing responses if specified).
    debug : bool
            Plot raw waveforms before preprocessing.
    sacheader : bool
            Key sacheader information in a dictionary using the SAC header naming convention.
    """

    raise ValueError("This function is deprecated. Please use seisgo.downloaders.download()."+\
                    "You can first use seisgo.downloaders.get_sta_list() to get a station list that "+\
                    "can be used by download(). Most options should be the same, though check the "+\
                    "funciton download() for updates.")
    client = Client(source)
    tr1, tr2, trZ, trP=[None for _ in range(4)]

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
                        channel="*",location="*",starttime=starttime,endtime=endtime,
                        level='response')
        # print(inv[0])
        ssta,snet,stlo,stla,stel,location=utils.sta_info_from_inv(inv)
        # print(ssta,snet,stlo,stla,stel,location)
        # stlo, stla,stel=utils.sta_info_from_inv(inv[0])[2:4]
        sac['stlo']=stlo
        sac['stla']=stla
        sac['stel']=stel

    # pressure channel
    trP=client.get_waveforms(network=net,station=sta,
                    channel=pchan,location="*",starttime=starttime,endtime=endtime,attach_response=True)
#     trP[0].detrend()
    trP=trP[0]
    trP.stats['sac']=sac

    pchan0=trP.stats.channel
    print("station "+net+"."+sta+" --> pressure channel: "+pchan0)

    #other seismic channels
    schan=pchan0[0]+"H?"
    tr1,tr2, trZ=client.get_waveforms(network=net,station=sta,
                    channel=schan,location="*",starttime=starttime,endtime=endtime,attach_response=True)
    hchan1=tr1.stats.channel
    hchan2=tr2.stats.channel
    zchan=trZ.stats.channel
    tr1.stats['sac']=sac
    tr2.stats['sac']=sac
    trZ.stats['sac']=sac

    print("station "+net+"."+sta+" --> seismic channels: "+hchan1+", "+hchan2+", "+zchan)

    if plot or debug:
        year = trZ.stats.starttime.year
        julday = trZ.stats.starttime.julday
        hour = trZ.stats.starttime.hour
        mnt = trZ.stats.starttime.minute
        sec = trZ.stats.starttime.second
        tstamp = str(year) + '.' + str(julday)+'T'+str(hour)+'-'+str(mnt)+'-'+str(sec)
        trlabels=[net+"."+sta+"."+tr1.stats.channel,
                 net+"."+sta+"."+tr2.stats.channel,
                 net+"."+sta+"."+trZ.stats.channel,
                 net+"."+sta+"."+trP.stats.channel]
    """
    b. Resampling
    """
    if samp_freq is not None:
        sps=int(trZ.stats.sampling_rate)
        delta = trZ.stats.delta
        #assume pressure and vertical channels have the same sampling rat
        # make downsampling if needed
        if sps > samp_freq:
            print("  downsamping from "+str(sps)+" to "+str(samp_freq))
            for tr in [tr1,tr2,trZ,trP]:
                if np.sum(np.isnan(tr.data))>0:
                    raise(Exception('NaN found in trace'))
                else:
                    tr.interpolate(samp_freq,method='weighted_average_slopes')
                    # when starttimes are between sampling points
                    fric = tr.stats.starttime.microsecond%(delta*1E6)
                    if fric>1E-4:
                        tr.data = utils.segment_interpolate(np.float32(tr.data),float(fric/(delta*1E6)))
                        #--reset the time to remove the discrepancy---
                        tr.stats.starttime-=(fric*1E-6)
                    # print('new sampling rate:'+str(tr.stats.sampling_rate))

    """
    c. Plot raw data before removing responses.
    """
    if plot and debug:
        utils.plot_trace([tr1,tr2,trZ,trP],size=(12,13),title=trlabels,freq=[0.005,0.1],
                   subplotpar=(4,1),ylabels=["raw","raw",
                                             "raw","raw"],
                   outfile=net+"."+sta+"_"+tstamp+"_raw.png",spacing=1,colors=['r','b','g','k'])

    """
    d. Remove responses
    """
    if rmresp:
        for tr in [tr1,tr2,trZ,trP]:
            if np.sum(np.isnan(tr.data))>0:
                raise(Exception('NaN found in trace'))
            else:
                try:
                    print('  removing response using inv for '+net+"."+sta+"."+tr.stats.channel)
                    if tr.stats.channel == pchan0:
                        tr.remove_response(output='VEL',pre_filt=pre_filt,
                                                  water_level=60,zero_mean=True,plot=False)
                    else:
                        tr.remove_response(output='DISP',pre_filt=pre_filt,
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
        utils.plot_trace([tr1,tr2,trZ,trP],size=(12,13),title=trlabels,freq=[0.005,0.1],
                   subplotpar=(4,1),ylabels=["displacement (m)","displacement (m)",
                                             "displacement (m)","pressure (pa)"],
                   outfile=net+"."+sta+"_"+tstamp+"_raw_rmresp.png",spacing=1,
                         colors=['r','b','g','k'])

    #
    if getstainv:return tr1,tr2,trZ,trP,inv
    else: return tr1,tr2,trZ,trP

#
def get_orientations(infile,help=False):
    """
    Assemble orientations for horizontal components from CSV file.
    Columns in the CSV file:
    net,station,orientation_ch1,orientation_ch2,error[optional]
    """
    orient_data=dict()
    if help:
        print('The CSV file that contains the orientation information needs to have the following columns. Note: headers are required!')
        print('net,station,orientation_ch1,orientation_ch2,error[optional]')
        print('7D,J37A,264,354,9')
        print('...')
        print('**Additional columns are fine but will be ignored!**')
        return orient_data

    #read in station list.
    if not os.path.isfile(infile):
        raise IOError('file %s not exist! double check!' % infile)
        sys.exit()

    # read station info from list
    orient=pd.read_csv(infile)
    onet  = list(orient.iloc[:]['net'])
    osta  = list(orient.iloc[:]['station'])
    oh1  = list(orient.iloc[:]['orientation_ch1'])
    oh2  = list(orient.iloc[:]['orientation_ch2'])
    if 'error' in list(orient.columns):
        oerror  = list(orient.iloc[:]['error'])
    else:
        oerror = np.empty(len(osta))
        oerror[:]=np.nan

    for i in range(len(onet)):
        onetsta=onet[i]+'.'+osta[i]
        orient_data[onetsta]=[oh1[i],oh2[i],oerror[i]]

    return orient_data
#
def correct_orientations(tr1,tr2,orient):
    """
    Correct horizontal orientations with given orientation data. The output traces
    are corrected and renamed to *E and *N convention.

    Parameters
    ----------
    tr1,tr2: :class:`~obspy.core.Trace`
        Seismic traces for horizontals.
    orient:: Dictionary
        Dictionary containing the orientation information for the horizonal
        components for each station in the format of [orient_h1,orient_h2,orient_error].
        This information can be assembed by calling get_orientations().
    """
    # Check that all traces are valid Trace objects
    for tr in [tr1, tr2]:
        if not isinstance(tr, Trace):
            raise(Exception("Error correct_orientations() - "
                            + str(tr)+" is not a Trace object"))

    #traces after orientation corrections.
    trE=[]
    trN=[]

    #get net and station name for the trace data
    netsta=tr1.stats.network+'.'+tr1.stats.station
    if netsta not in orient.keys():
        print("Error correct_orientations() - "
                    + netsta+" is not in the orientation list.")
        return trE,trN

    oh1,oh2,oerror=orient[netsta]

    chan1=tr1.stats.channel
    chan2=tr2.stats.channel
    data1=tr1.data
    data2=tr2.data

    trE=tr2.copy()
    trE.stats.channel=chan2[0:2]+'E'
    trN=tr1.copy()
    trN.stats.channel=chan2[0:2]+'N'

    angle=360 - oh1 #rotation angle to rotate tr1 to trN
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    v12 = np.array([data2, data1])
    vEN = np.tensordot(rot_mat, v12, axes=1)
    trE.data = vEN[0, :]
    trN.data = vEN[1, :]


    #match and do the rotation.
    # if oh1==0: #due north
    #     trE=tr2.copy()
    #     trE.stats.channel=chan2[0:2]+'E'
    #     trN=tr1.copy()
    #     trN.stats.channel=chan1[0:2]+'N'
    # elif oh1==90:
    #     trE=tr1.copy()
    #     trE.stats.channel=chan1[0:2]+'E'
    #     trN=tr2.copy()
    #     trN.stats.channel=chan2[0:2]+'N'
    #     trN.data=-1.0*data2
    # elif oh1==180:
    #     trE=tr2.copy()
    #     trE.data=-1.0*data2
    #     trE.stats.channel=chan2[0:2]+'E'
    #     trN=tr1.copy()
    #     trN.stats.channel=chan1[0:2]+'N'
    #     trN.data=-1.0*data1
    # elif oh1==270:
    #     trE=tr1.copy()
    #     trE.data=-1.0*data1
    #     trE.stats.channel=chan1[0:2]+'E'
    #     trN=tr2.copy()
    #     trN.stats.channel=chan2[0:2]+'N'

    return trE,trN

#
def maxcompfreq(d,iplot=False,figname="waterdepth_maxcompfreq.png"):
    """
    computes the maximum compliance frequency based on eq-7 of Tian and Ritzwoller, GJI, 2017
    """
#     d=np.arange(1,5051,50) #water depth
    f=np.sqrt(9.8/1.6/np.pi/d)
    if iplot:
        plt.figure(figsize=(10,5))
        plt.plot(d,f)
        plt.yscale("log")
        plt.grid(which="both")
        plt.xlabel("water depth (m)")
        plt.ylabel("frequency (Hz)")
        plt.text(1.2*np.mean(d),0.5*np.max(f),r'$\sqrt{(\frac{g}{1.6 \pi d})}$',fontsize=20)
        plt.savefig(figname,orientation='landscape')
        plt.show()
        plt.close()

    return f

#
def gettflist(help=False,correctlist=None):
    """
    Get a full list of transfer function components. This is usefull to get a default
    list and modify based on cases.

    Parameters
    ----------
    help : bool
        Print out explanation of each transfer function component. Default: False.
    """

    #print exaplanations of each elements in the output tflist.
    if help:
        print('------------------------------------------------------------------')
        print('| Key    |'+' Default  |'+' Note                                       |')
        print('------------------------------------------------------------------')
        print('| ZP     |'+' True     |'+' Vertical and pressure                      |')
        print('| Z1     |'+' True     |'+' Vertical and horizontal-1                  |')
        print('| Z2-1   |'+' True     |'+' Vertical and horizontals (1 and 2)         |')
        print('| ZP-21  |'+' True     |'+' Vertical, pressure, and two horizontals    |')
        print('| ZH     |'+' True     |'+' Vertical and rotated horizontal            |')
        print('| ZP-H   |'+' True     |'+' Vertical, pressure, and rotated horizontal |')
        print('------------------------------------------------------------------')

    tflist={
        'ZP':True,
        'Z1':True,
        'Z2-1':True,
        'ZP-21':True,
        'ZH':True,
        'ZP-H':True
    }

    #find the tflist that matches the correctlist.
    if correctlist is not None:
        tflist={
            'ZP':False,
            'Z1':False,
            'Z2-1':False,
            'ZP-21':False,
            'ZH':False,
            'ZP-H':False
        }
        for key in correctlist:
            tflist[key]=True
            if key == 'Z2-1': tflist['Z1']=True
            elif key == 'ZP-H': tflist['ZH']=True

    return tflist

#
def getcorrectlist(help=False):
    """
    Get a full list of corrections. This is usefull to get a default
    list and modify based on cases.

    Parameters
    ----------
    help : bool
        Print out explanation of each correction component. Default: False.
    """

    #print exaplanations of each elements in the output tflist.
    if help:
        print('--------------------------------------------------------------------------------------')
        print('| Key    |'+' Note                                                                      |')
        print('--------------------------------------------------------------------------------------')
        print('| ZP     |'+' Vertical corrected for compliance noise                                   |')
        print('| Z1     |'+' Vertical corrected for tilt leakage from horizontal-1                     |')
        print('| Z2-1   |'+' Vertical corrected for tilt leakage from horizontals (1 and 2)            |')
        print('| ZP-21  |'+' Vertical corrected for compliance and tilt noise                          |')
        print('| ZH     |'+' Vertical corrected for tilt leakage from rotated horizontal               |')
        print('| ZP-H   |'+' Vertical corrected for compliance and tilt noise from rotated horizontals |')
        print('--------------------------------------------------------------------------------------')

    tflist=gettflist()
    clist=tflist.keys()

    return clist
# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def calculate_tilt(ft1, ft2, ftZ, ftP, f, tiltfreq=[0.005, 0.035]):
    """
    Determines tilt direction from maximum coherence between rotated H1 and Z.

    Parameters
    ----------
    ft1, ft2, ftZ, ftP : :class:`~numpy.ndarray`
        Fourier transform of corresponding H1, H2, HZ and HP components
    f : :class:`~numpy.ndarray`
        Frequency axis in Hz
    tiltfreq : list
        Two floats representing the frequency band at which the tilt is calculated

    Returns
    -------
    cHH, cHZ, cHP : :class:`~numpy.ndarray`
        Arrays of power and cross-spectral density functions of components HH (rotated H1
        in direction of maximum tilt), HZ, and HP
    coh : :class:`~numpy.ndarray`
        Coherence value between rotated H and Z components, as a function of directions (azimuths)
    ph : :class:`~numpy.ndarray`
        Phase value between rotated H and Z components, as a function of directions (azimuths)
    angle : :class:`~numpy.ndarray`
        Tilt angle
    tilt : float
        Direction (azimuth) of maximum coherence between rotated H1 and Z
    coh_value : float
        Coherence value at tilt direction
    phase_value : float
        Phase value at tilt direction

    """

    direc = np.arange(0., 360., 10.)
    coh = np.zeros(len(direc))
    ph = np.zeros(len(direc))
    cZZ = np.abs(np.mean(ftZ *
                         np.conj(ftZ), axis=0))[0:len(f)]

    for i, d in enumerate(direc):

        # Rotate horizontals
        ftH = utils.rotate_dir(ft1, ft2, d)

        # Get transfer functions
        cHH = np.abs(np.mean(ftH *
                             np.conj(ftH), axis=0))[0:len(f)]
        cHZ = np.mean(ftH *
                      np.conj(ftZ), axis=0)[0:len(f)]

        Co = utils.coherence(cHZ, cHH, cZZ)
        Ph = utils.phase(cHZ)

        # Calculate coherence over frequency band
        coh[i] = np.mean(Co[(f > tiltfreq[0]) & (f < tiltfreq[1])])
        ph[i] = np.pi/2. - np.mean(Ph[(f > tiltfreq[0]) & (f < tiltfreq[1])])

    # Index where coherence is max
    ind = np.argwhere(coh == coh.max())

    # Phase and direction at maximum coherence
    phase_value = ph[ind[0]][0]
    coh_value = coh[ind[0]][0]
    tilt = direc[ind[0]][0]

    # Refine search
    rdirec = np.arange(direc[ind[0]][0]-10., direc[ind[0]][0]+10., 1.)
    rcoh = np.zeros(len(direc))
    rph = np.zeros(len(direc))

    for i, d in enumerate(rdirec):

        # Rotate horizontals
        ftH = utils.rotate_dir(ft1, ft2, d)

        # Get transfer functions
        cHH = np.abs(np.mean(ftH *
                             np.conj(ftH), axis=0))[0:len(f)]
        cHZ = np.mean(ftH *
                      np.conj(ftZ), axis=0)[0:len(f)]

        Co = utils.coherence(cHZ, cHH, cZZ)
        Ph = utils.phase(cHZ)

        # Calculate coherence over frequency band
        rcoh[i] = np.mean(Co[(f > tiltfreq[0]) & (f < tiltfreq[1])])
        rph[i] = np.pi/2. - np.mean(Ph[(f > tiltfreq[0]) & (f < tiltfreq[1])])

    # Index where coherence is max
    ind = np.argwhere(rcoh == rcoh.max())

    # Phase and direction at maximum coherence
    phase_value = rph[ind[0]][0]
    coh_value = rcoh[ind[0]][0]
    tilt = rdirec[ind[0]][0]

    # Phase has to be close to zero - otherwise add pi
    if phase_value > 0.5*np.pi:
        tilt += 180.
    if tilt > 360.:
        tilt -= 360.

    # print('Maximum coherence for tilt = ', tilt)

    # Now calculate spectra at tilt direction
    ftH = utils.rotate_dir(ft1, ft2, tilt)

    # Get transfer functions
    cHH = np.abs(np.mean(ftH *
                         np.conj(ftH), axis=0))[0:len(f)]
    cHZ = np.mean(ftH*np.conj(ftZ), axis=0)[0:len(f)]
    admt_value = utils.admittance(cHZ,cHH)
    if np.any(ftP):
        cHP = np.mean(ftH *
                      np.conj(ftP), axis=0)[0:len(f)]
    else:
        cHP = None

    return cHH, cHZ, cHP, coh, ph, direc,tilt, admt_value,coh_value, phase_value

#modified from QC_daily_spectra() method in DayNoise class from OBStools
#https://github.com/nfsi-canada/OBStools
def getspectra(tr1,tr2,trZ,trP,window=7200,overlap=0.3,pd=[0.004, 0.2], tol=1.5, alpha=0.05,
                     smooth=True,QC=True,fig=False, debug=False, save=False,
                     format='png'):
    """
    Compute the cross-spectra between multiple components by averaging through good windows.
    The function performs QC by default to determine daily time windows for which the spectra are
    anomalous and should be discarded in the calculation of the transfer functions.

    Parameters
    ----------
    tr1,tr2,trZ,trP : :class:`~obspy.core.Trace`
        Seismic traces for horizontals, vertical, and pressure. Use None if missing.
    window : float
        Length of time window in seconds
    overlap : float
        Fraction of overlap between adjacent windows (0 - 1)
    pd : list
        Frequency corners of passband for calculating the spectra
    tol : float
        Tolerance threshold. If spectrum > std*tol, window is flagged as
        bad
    alpha : float
        Confidence interval for f-test
    smooth : boolean
        Determines if the smoothed (True) or raw (False) spectra are used
    fig : boolean
        Whether or not to produce a figure showing the results of the
        quality control
    debug : boolean
        Whether or not to plot intermediate steps in the QC procedure
        for debugging

    Returns
    ----------
    outdict : Dictionary
        The outputs are wrapped in a dictionary that contains spectra
        computing parameters and cross-spectra for transfer functions.
    """
    # Check that all traces are valid Trace objects
    for tr in [tr1, tr2, trZ, trP]:
        if not isinstance(tr, Trace):
            raise(Exception("Error getspectra() - "
                            + str(tr)+" is not a Trace object"))
    #get fs,dt,ncomp based on the input traces.
    dt = trZ.stats.delta
    npts = trZ.stats.npts
    fs = trZ.stats.sampling_rate
    st = trZ.stats.network+trZ.stats.station
    year = trZ.stats.starttime.year
    julday = trZ.stats.starttime.julday
    hour = trZ.stats.starttime.hour
    mnt = trZ.stats.starttime.minute
    sec = trZ.stats.starttime.second
    tstamp = str(year) + '.' + str(julday)+'T'+str(hour)+'-'+str(mnt)+'-'+str(sec)
    # Get number of components for the available, non-empty traces
    ncomp = np.sum(1 for tr in
                   Stream(traces=[tr1, tr2, trZ, trP]) if np.any(tr.data))

    # Points in window
    ws = int(window/dt)

    # Number of points to overlap
    ss = int(window*overlap/dt)

    # hanning window
    hanning = np.hanning(2*ss)
    wind = np.ones(ws)
    wind[0:ss] = hanning[0:ss]
    wind[-ss:ws] = hanning[ss:2*ss]
    t00=time.time()
    # Get spectrograms for single day-long keys
    f, t, psd1_temp = spectrogram(
        tr1.data, fs, window=wind, nperseg=ws, noverlap=ss)
    f, t, psd2_temp = spectrogram(
        tr2.data, fs, window=wind, nperseg=ws, noverlap=ss)
    f, t, psdZ_temp = spectrogram(
        trZ.data, fs, window=wind, nperseg=ws, noverlap=ss)
    f, t, psdP_temp = spectrogram(
        trP.data, fs, window=wind, nperseg=ws, noverlap=ss)

    #convert to log scale
    psd1=np.log(psd1_temp)
    psd2=np.log(psd2_temp)
    psdZ=np.log(psdZ_temp)
    psdP=np.log(psdP_temp)

    if fig:
        plt.figure(1,figsize=[10,8])
        plt.subplot(4, 1, 1)
        plt.pcolormesh(t, f, psd1)
        plt.title(st+"."+tr1.stats.channel, fontdict={'fontsize': 10})
        plt.ylabel("Hz")
        plt.subplot(4, 1, 2)
        plt.pcolormesh(t, f, psd2)
        plt.title(st+"."+tr2.stats.channel, fontdict={'fontsize': 10})
        plt.ylabel("Hz")
        plt.subplot(4, 1, 3)
        plt.pcolormesh(t, f, psdZ)
        plt.title(st+"."+trZ.stats.channel, fontdict={'fontsize': 10})
        plt.ylabel("Hz")
        plt.subplot(4, 1, 4)
        plt.pcolormesh(t, f, psdP)
        plt.title(st+"."+trP.stats.channel, fontdict={'fontsize': 10})
        plt.xlabel('Seconds')
        plt.tight_layout()

        if save:
            figname=st+"_"+tstamp+"_spectrogram."+format
            plt.savefig(figname,dpi=300, bbox_inches='tight', format=format)
        else:
            plt.show()

    # Select bandpass frequencies
    indf = (f >= pd[0]) & (f <= pd[1])
    # ff=f[indf]
    if smooth:
        # Smooth out the log of the PSDs
        sl_psdZ = utils.smooth(psdZ, 50, axis=0)
        sl_psdP = utils.smooth(psdP, 50, axis=0)
        sl_psd1 = utils.smooth(psd1, 50, axis=0)
        sl_psd2 = utils.smooth(psd2, 50, axis=0)
    else:
        # Take the log of the PSDs
        sl_psdZ = psdZ
        sl_psdP = psdP
        sl_psd1 = psd1
        sl_psd2 = psd2

    # Remove mean of the log PSDs
    dsl_psdZ = sl_psdZ[indf, :] - np.mean(sl_psdZ[indf, :], axis=0)
    dsl_psd1 = sl_psd1[indf, :] - np.mean(sl_psd1[indf, :], axis=0)
    dsl_psd2 = sl_psd2[indf, :] - np.mean(sl_psd2[indf, :], axis=0)
    dsl_psdP = sl_psdP[indf, :] - np.mean(sl_psdP[indf, :], axis=0)
    dsls = [dsl_psd1, dsl_psd2, dsl_psdZ, dsl_psdP]

    t_psd=time.time() - t00
    if debug: print("time on getting PSD: "+str(t_psd))
    if fig:
        plt.figure(2,figsize=[10,8])
        plt.subplot(4, 1, 1)
        plt.semilogx(f[indf], dsl_psd1, 'r', lw=0.5)
        plt.ylabel("PSD")
        plt.title(st+"."+tr1.stats.channel, fontdict={'fontsize': 10})
        plt.subplot(4, 1, 2)
        plt.semilogx(f[indf], dsl_psd2, 'b', lw=0.5)
        plt.ylabel("PSD")
        plt.title(st+"."+tr2.stats.channel, fontdict={'fontsize': 10})
        plt.subplot(4, 1, 3)
        plt.semilogx(f[indf], dsl_psdZ, 'g', lw=0.5)
        plt.ylabel("PSD")
        plt.title(st+"."+trZ.stats.channel, fontdict={'fontsize': 10})
        plt.subplot(4, 1, 4)
        plt.semilogx(f[indf], dsl_psdP, 'k', lw=0.5)
        plt.ylabel("PSD")
        plt.title(st+"."+trP.stats.channel, fontdict={'fontsize': 10})
        plt.tight_layout()

        if save:
            figname=st+"_"+tstamp+"_PSDsmooth."+format
            plt.savefig(figname,dpi=300, bbox_inches='tight', format=format)
        else:
            plt.show()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # start QC
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Cycle through to kill high-std-norm windows
    goodwins = np.repeat([True], len(t))
    if QC:
        #%%%%%%%%%%%%%%%%%%
        # QC by setting googwins to False for windows
        # with anomalous spectra
        #%%%%%%%%%%%%%%%%%%
        moveon = False
        indwin = np.argwhere(goodwins == True)
        while moveon == False:
            ubernorm = np.empty((ncomp, np.sum(goodwins)))
            for ind_u, dsl in enumerate(dsls):
                normvar = np.zeros(np.sum(goodwins))
                for ii, tmp in enumerate(indwin):
                    ind = np.copy(indwin)
                    ind = np.delete(ind, ii)
                    normvar[ii] = norm(np.std(dsl[:, ind], axis=1), ord=2)
                ubernorm[ind_u, :] = np.median(normvar) - normvar

            penalty = np.sum(ubernorm, axis=0)

            if debug:
                plt.figure()
                for i in range(ncomp):
                    plt.plot(range(0, np.sum(goodwins)), detrend(
                        ubernorm, type='constant')[i], 'o-')
                plt.ylabel("std norm of PSD")
                plt.show()

                plt.figure()
                plt.plot(range(0, np.sum(goodwins)),
                         np.sum(ubernorm, axis=0), 'o-')
                plt.ylabel("sum of std norm")
                plt.show()

            kill = penalty > tol*np.std(penalty)
            if np.sum(kill) == 0:
                moveon = True

            trypenalty = penalty[np.argwhere(kill == False)].T[0]

            if utils.ftest(penalty, 1, trypenalty, 1) < alpha:
                goodwins[indwin[kill == True]] = False
                indwin = np.argwhere(goodwins == True)
                moveon = False
            else:
                moveon = True

        #%%%%%%%%%%%%%%%%%%
        # QC by excluding windows with transient signals. TODO.
        #%%%%%%%%%%%%%%%%%%
    t_qc=time.time() - t00 - t_psd
    if debug: print("time on getting qc: "+str(t_qc))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #end of QC
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # get average spectra of all good windows.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rotation = None
    auto = None
    cross = None
    bad = None
    getbad=False

    step = int(window*(1-overlap)/dt)
    out,nd,window_index = utils.sliding_window(trZ.data,ws,step,getindex=True)
    ftZ, f = utils.calculate_windowed_fft(trZ.data,trZ.stats.sampling_rate, ws, ss=step,wind=np.hanning(ws))
    ftP, f = utils.calculate_windowed_fft(trP.data,trP.stats.sampling_rate, ws, ss=step,wind=np.hanning(ws))
    ft1, f = utils.calculate_windowed_fft(tr1.data,tr1.stats.sampling_rate, ws, ss=step,wind=np.hanning(ws))
    ft2, f = utils.calculate_windowed_fft(tr2.data,tr2.stats.sampling_rate, ws, ss=step,wind=np.hanning(ws))

    # Extract good windows
    cZZ = np.abs(np.mean(ftZ[goodwins, :]*np.conj(ftZ[goodwins, :]),
                axis=0))[0:len(f)]
    cPP = np.abs(np.mean(ftP[goodwins, :]*np.conj(ftP[goodwins, :]),
                axis=0))[0:len(f)]
    c11 = np.abs(np.mean(ft1[goodwins, :]*np.conj(ft1[goodwins, :]),
                axis=0))[0:len(f)]
    c22 = np.abs(np.mean(ft2[goodwins, :]*np.conj(ft2[goodwins, :]),
                axis=0))[0:len(f)]

    # Extract bad windows
    if getbad:
        if np.sum(~goodwins) > 0:
            bcZZ = np.abs(np.mean(ftZ[~goodwins, :]*np.conj(ftZ[~goodwins, :]),
                axis=0))[0:len(f)]
            bcPP = np.abs(np.mean(ftP[~goodwins, :]*np.conj(ftP[~goodwins, :]),
                axis=0))[0:len(f)]
            bc11 = np.abs(np.mean(ft1[~goodwins, :]*np.conj(ft1[~goodwins, :]),
                axis=0))[0:len(f)]
            bc22 = np.abs(np.mean(ft2[~goodwins, :]*np.conj(ft2[~goodwins, :]),
                axis=0))[0:len(f)]
        else:
            bc11 = None
            bc22 = None
            bcZZ = None
            bcPP = None
    # Calculate mean of all good windows if component combinations exist
    c12 = np.mean(ft1[goodwins, :] *
                  np.conj(ft2[goodwins, :]), axis=0)[0:len(f)]
    c1Z = np.mean(ft1[goodwins, :] *
                  np.conj(ftZ[goodwins, :]), axis=0)[0:len(f)]
    c2Z = np.mean(ft2[goodwins, :] *
                  np.conj(ftZ[goodwins, :]), axis=0)[0:len(f)]
    c1P = np.mean(ft1[goodwins, :] *
                  np.conj(ftP[goodwins, :]), axis=0)[0:len(f)]
    c2P = np.mean(ft2[goodwins, :] *
                  np.conj(ftP[goodwins, :]), axis=0)[0:len(f)]
    cZP = np.mean(ftZ[goodwins, :] *
                  np.conj(ftP[goodwins, :]), axis=0)[0:len(f)]

    cHH, cHZ, cHP, coh, ph, direc,tilt,admt_value,coh_value, phase_value = \
            calculate_tilt(ft1[goodwins, :], ft2[goodwins, :], ftZ[goodwins, :],
                            ftP[goodwins, :], f)

    # Store as attribute containers
    rotation = Rotation(cHH, cHZ, cHP, coh, ph, direc,tilt, admt_value,
                        coh_value, phase_value, window,overlap,f)
    auto = Power(c11, c22, cZZ, cPP,window,overlap,f)
    cross = Cross(c12, c1Z, c1P, c2Z, c2P, cZP,window,overlap,f)
    # bad = Power(bc11, bc22, bcZZ, bcPP,window,overlap,f)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # end of getting average spectra.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_avg=time.time() - t00 - t_psd - t_qc
    if debug: print("time on getting avg: "+str(t_avg))

    tflist=gettflist()
    if rotation is None:
        tflist['ZH'] = False
        tflist['ZP-H'] = False

    #wrap all returns in a dictionary
    #"psd1":dsl_psd1,"psd2":dsl_psd2,"psdZ":dsl_psdZ,"psdP":dsl_psdP,
    outdict={"freq":f,"goodwins":goodwins,"window_index":window_index,"window":window,
            "overlap":overlap,"dt":dt,"st":st,"tstamp":tstamp,"ncomp":ncomp,
            "tflist":tflist,"auto":auto,"cross":cross,"rotation":rotation,"bad":bad} #

    return outdict

def gettransferfunc(auto,cross,rotation,tflist=gettflist()):
    """
    Calculates transfer functions between multiple
    components (and component combinations) from the averaged
    noise spectra.

    Parameters
    ----------
    auto : :class: `obsmaster.Power`
            The power of spectra for the same component.
    cross ::class: `obsmaster.Cross`
            The cross spectra for multiple components.
    rotation ::class: `obsmaster.Rotation`
            The cross spectra for multiple components after rotating horizontal components.
    Returns
    ----------
    transfunc : Dictionary
            Container Dictionary for all possible transfer functions
    *** TO-DO: add plotting option to plot transfer functions.
    """
    #
    # tflistdefault=gettflist()
    # if tflist is None:
    #     tflist = tflistdefault

    transfunc = dict()
    transfunc['window']=auto.window
    transfunc['overlap']=auto.overlap
    transfunc['freq']=auto.freq

    for tfkey, value in tflist.items():
        if tfkey == 'ZP':
            if value:
                tf_ZP = {'TF_ZP': cross.cZP/auto.cPP}
                transfunc['ZP']= tf_ZP
        elif tfkey == 'Z1':
            if value:
                tf_Z1 = {'TF_Z1': np.conj(cross.c1Z)/auto.c11}
                transfunc['Z1'] = tf_Z1
        elif tfkey == 'Z2-1':
            if value:
                lc1c2 = np.conj(cross.c12)/auto.c11
                coh_12 = utils.coherence(cross.c12, auto.c11, auto.c22)
                gc2c2_c1 = auto.c22*(1. - coh_12)
                gc2cZ_c1 = np.conj(cross.c2Z) - np.conj(lc1c2*cross.c1Z)
                lc2cZ_c1 = gc2cZ_c1/gc2c2_c1
                tf_Z2_1 = {'TF_21': lc1c2, 'TF_Z2-1': lc2cZ_c1}
                transfunc['Z2-1'] = tf_Z2_1
        elif tfkey == 'ZP-21':
            if value:
                lc1cZ = np.conj(cross.c1Z)/auto.c11
                lc1c2 = np.conj(cross.c12)/auto.c11
                lc1cP = np.conj(cross.c1P)/auto.c11

                coh_12 = utils.coherence(cross.c12, auto.c11, auto.c22)
                coh_1P = utils.coherence(cross.c1P, auto.c11, auto.cPP)

                gc2c2_c1 = auto.c22*(1. - coh_12)
                gcPcP_c1 = auto.cPP*(1. - coh_1P)

                gc2cZ_c1 = np.conj(cross.c2Z) - np.conj(lc1c2*cross.c1Z)
                gcPcZ_c1 = cross.cZP - np.conj(lc1cP*cross.c1Z)

                gc2cP_c1 = np.conj(cross.c2P) - np.conj(lc1c2*cross.c1P)

                lc2cP_c1 = gc2cP_c1/gc2c2_c1
                lc2cZ_c1 = gc2cZ_c1/gc2c2_c1

                coh_c2cP_c1 = utils.coherence(gc2cP_c1, gc2c2_c1,
                                              gcPcP_c1)

                gcPcP_c1c2 = gcPcP_c1*(1. - coh_c2cP_c1)
                gcPcZ_c1c2 = gcPcZ_c1 - np.conj(lc2cP_c1)*gc2cZ_c1

                lcPcZ_c2c1 = gcPcZ_c1c2/gcPcP_c1c2

                tf_ZP_21 = {'TF_Z1': lc1cZ, 'TF_21': lc1c2,
                            'TF_P1': lc1cP, 'TF_P2-1': lc2cP_c1,
                            'TF_Z2-1': lc2cZ_c1, 'TF_ZP-21': lcPcZ_c2c1}
                transfunc['ZP-21'] = tf_ZP_21
        elif tfkey == 'ZH':
            if value:
                tf_ZH = {'TF_ZH': np.conj(rotation.cHZ)/rotation.cHH}
                transfunc['ZH'] = tf_ZH
                transfunc['tilt'] = rotation.tilt
        elif tfkey == 'ZP-H':
            if value:
                lcHcP = np.conj(rotation.cHP)/rotation.cHH
                coh_HP = utils.coherence(rotation.cHP, rotation.cHH, auto.cPP)
                gcPcP_cH = auto.cPP*(1. - coh_HP)
                gcPcZ_cH = cross.cZP - np.conj(lcHcP*rotation.cHZ)
                lcPcZ_cH = gcPcZ_cH/gcPcP_cH
                tf_ZP_H = {'TF_PH': lcHcP, 'TF_ZP-H': lcPcZ_cH}
                transfunc['ZP-H'] = tf_ZP_H
        else:
            raise(Exception('Incorrect tfkey'))

    return transfunc

###########################################################################
def docorrection(tr1,tr2,trZ,trP,tf,correctlist=getcorrectlist(),overlap=0.1,
                taper=None,full_length=True,verbose=False):
    """
    Applies transfer functions between multiple components (and
    component combinations) to produce corrected/cleaned vertical
    components.

    Parameters
    ----------
    tr1,tr2,trZ,trP : :class:`~obspy.core.Trace`
        Seismic traces for horizontals, vertical, and pressure. Use None if missing.
    tf : Dictionary
        Transfer functions computed using gettransferfunc().
    correctlist : string list
        Correction components.
    overlap : float
        Fraction of overlap when sliding windows to remove the noise. The window length is the same as
        used when computing the spectra and the transfer functions and is passed from `tf`.
    taper : float
        Taper fraction when overlapping windows. It has to be <= 0.5*overlap (force to this value if larger).
        In utils.plot_slidingwindows() can be used to demonstrate how the corrections are done.
    full_length : bool
        Force the output after correction to be the same length as the input traces. If False,
        the tailing section will be filled with zeros if shorter than one window length.

    Returns
    ----------
    correct : dictionary
        The corrected vertical data after applying multiple transfer functions.
    """
    # tflistdefault=gettflist()
    # if tflist is None:
    #     tflist = tflistdefault

    correct = dict()

    # Extract list and transfer functions available
    # Points in window
    window=tf['window']
    freq=tf['freq']
    ws = int(window*trZ.stats.sampling_rate)
    tps = int(0)
    step=None
    wind=None
    tflist=gettflist()

    if overlap is not None:
        step = int(window*(1-overlap)*trZ.stats.sampling_rate)
        tps = int(0.5*window*overlap*trZ.stats.sampling_rate) #taper size: falf of the overlap size
        if taper is not None:
            if 2*taper > overlap:
                print('WARNING: one-side taper can not be larger than half of overlap. Use 0.5*overlap fraction as taper!')
            wind = tukey(ws,2*taper)

    # Fourier transform
    if trZ is None:
        raise(Exception("Error docorrection() - vertical trace can't be None"))
    else:
        ftZ, f, idx = utils.calculate_windowed_fft(trZ.data,trZ.stats.sampling_rate, ws, ss=step,
                                                wind=wind,getindex=True,full_length=full_length)

    if tr1 is None and tr2 is None and trP is None:
        raise(Exception("Error docorrection() - horizontals and pressure channel can't all be None"))

    if tr1 is None:
        tflist['Z1']=False
        tflist['Z2-1']=False
        tflist['ZP-21']=False
        tflist['ZH']=False
        tflist['ZP-H']=False
        print('WARNING: Horizontal-1 is None.')
    else:
        ft1, f = utils.calculate_windowed_fft(tr1.data,tr1.stats.sampling_rate,
                                            ws, ss=step,wind=wind,full_length=full_length)

    if tr2 is None:
        tflist['Z2-1']=False
        tflist['ZP-21']=False
        tflist['ZH']=False
        tflist['ZP-H']=False
        print('WARNING: Horizontal-2 is None.')
    else:
        ft2, f = utils.calculate_windowed_fft(tr2.data,tr2.stats.sampling_rate,
                                            ws,ss=step, wind=wind,full_length=full_length)

    if trP is None:
        tflist['ZP']=False
        tflist['ZP-21']=False
        print('WARNING: Pressure data is None.')
    else:
        ftP, f = utils.calculate_windowed_fft(trP.data,trP.stats.sampling_rate,
                                            ws,ss=step, wind=wind,full_length=full_length)

    if not np.allclose(f, freq):
        raise(Exception('Frequency axes (data to correct & TF info) are different: ', f, freq))

    # Important step below: merge multiple windows after correction.
    # The windows will be put back to the exact location as documented in idx.
    if len(idx) > 1 and verbose:
        print('Merging multiple corrected segments.')
        print("windows connect at the following times:")
        print((idx+tps)/trZ.stats.sampling_rate)

    for key in correctlist:
        if key == 'ZP' and tflist[key]:
            dout = np.zeros((trZ.stats.npts,), dtype=trZ.data.dtype)
            TF_ZP = tf[key]['TF_ZP']
            fTF_ZP = np.hstack(
                (TF_ZP, np.conj(TF_ZP[::-1][1:len(f)-1])))
            for j in range(len(idx)):
                ftZ_temp = ftZ[j]
                ftP_temp = ftP[j]
                corrspec = ftZ_temp - fTF_ZP*ftP_temp
                corrtime = np.real(np.fft.ifft(corrspec))[0:ws]
                if j==0:
                    dout[idx[j]:idx[j]+ws-tps] = corrtime[0:ws-tps]
                elif j+1 == len(idx):
                    dout[idx[j]+tps:idx[j]+ws] = corrtime[tps:ws]
                else:
                    dout[idx[j]+tps:idx[j]+ws-tps] = corrtime[tps:ws-tps]

            correct['ZP'] = dout

        if key == 'Z1' and tflist[key]:
            dout = np.zeros((trZ.stats.npts,), dtype=trZ.data.dtype)
            TF_Z1 = tf[key]['TF_Z1']
            fTF_Z1 = np.hstack(
                (TF_Z1, np.conj(TF_Z1[::-1][1:len(f)-1])))
            for j in range(len(idx)):
                ftZ_temp = ftZ[j]
                ft1_temp = ft1[j]
                corrspec = ftZ_temp - fTF_Z1*ft1_temp
                corrtime = np.real(np.fft.ifft(corrspec))[0:ws]
                if j==0:
                    dout[idx[j]:idx[j]+ws-tps] = corrtime[0:ws-tps]
                elif j+1 == len(idx):
                    dout[idx[j]+tps:idx[j]+ws] = corrtime[tps:ws]
                else:
                    dout[idx[j]+tps:idx[j]+ws-tps] = corrtime[tps:ws-tps]

            correct['Z1'] = dout

        if key == 'Z2-1' and tflist[key]:
            dout = np.zeros((trZ.stats.npts,), dtype=trZ.data.dtype)
            TF_Z1 = tf['Z1']['TF_Z1']
            fTF_Z1 = np.hstack(
                (TF_Z1, np.conj(TF_Z1[::-1][1:len(f)-1])))
            TF_21 = tf[key]['TF_21']
            fTF_21 = np.hstack(
                (TF_21, np.conj(TF_21[::-1][1:len(f)-1])))
            TF_Z2_1 = tf[key]['TF_Z2-1']
            fTF_Z2_1 = np.hstack(
                (TF_Z2_1, np.conj(TF_Z2_1[::-1][1:len(f)-1])))

            for j in range(len(idx)):
                ftZ_temp = ftZ[j]
                ft1_temp = ft1[j]
                ft2_temp = ft2[j]
                corrspec = ftZ_temp - fTF_Z1*ft1_temp - (ft2_temp - ft1_temp*fTF_21)*fTF_Z2_1
                corrtime = np.real(np.fft.ifft(corrspec))[0:ws]
                if j==0:
                    dout[idx[j]:idx[j]+ws-tps] = corrtime[0:ws-tps]
                elif j+1 == len(idx):
                    dout[idx[j]+tps:idx[j]+ws] = corrtime[tps:ws]
                else:
                    dout[idx[j]+tps:idx[j]+ws-tps] = corrtime[tps:ws-tps]

            correct['Z2-1'] = dout

        if key == 'ZP-21' and tflist[key]:
            dout = np.zeros((trZ.stats.npts,), dtype=trZ.data.dtype)
            TF_Z1 = tf[key]['TF_Z1']
            fTF_Z1 = np.hstack(
                (TF_Z1, np.conj(TF_Z1[::-1][1:len(f)-1])))
            TF_21 = tf[key]['TF_21']
            fTF_21 = np.hstack(
                (TF_21, np.conj(TF_21[::-1][1:len(f)-1])))
            TF_Z2_1 = tf[key]['TF_Z2-1']
            fTF_Z2_1 = np.hstack(
                (TF_Z2_1, np.conj(TF_Z2_1[::-1][1:len(f)-1])))
            TF_P1 = tf[key]['TF_P1']
            fTF_P1 = np.hstack(
                (TF_P1, np.conj(TF_P1[::-1][1:len(f)-1])))
            TF_P2_1 = tf[key]['TF_P2-1']
            fTF_P2_1 = np.hstack(
                (TF_P2_1, np.conj(TF_P2_1[::-1][1:len(f)-1])))
            TF_ZP_21 = tf[key]['TF_ZP-21']
            fTF_ZP_21 = np.hstack(
                (TF_ZP_21, np.conj(TF_ZP_21[::-1][1:len(f)-1])))

            for j in range(len(idx)):
                ftZ_temp = ftZ[j]
                ftP_temp = ftP[j]
                ft1_temp = ft1[j]
                ft2_temp = ft2[j]
                corrspec = ftZ_temp - fTF_Z1*ft1_temp - \
                    (ft2_temp - ft1_temp*fTF_21)*fTF_Z2_1 - \
                    (ftP_temp - ft1_temp*fTF_P1 -
                     (ft2_temp - ft1_temp*fTF_21)*fTF_P2_1)*fTF_ZP_21
                corrtime = np.real(np.fft.ifft(corrspec))[0:ws]
                if j==0:
                    dout[idx[j]:idx[j]+ws-tps] = corrtime[0:ws-tps]
                elif j+1 == len(idx):
                    dout[idx[j]+tps:idx[j]+ws] = corrtime[tps:ws]
                else:
                    dout[idx[j]+tps:idx[j]+ws-tps] = corrtime[tps:ws-tps]

            correct['ZP-21'] = dout

        if key == 'ZH' and tflist[key]:
            dout = np.zeros((trZ.stats.npts,), dtype=trZ.data.dtype)
            TF_ZH = tf[key]['TF_ZH']
            fTF_ZH = np.hstack(
                (TF_ZH, np.conj(TF_ZH[::-1][1:len(f)-1])))

            for j in range(len(idx)):
                ftZ_temp = ftZ[j]
                ft1_temp = ft1[j]
                ft2_temp = ft2[j]
                # Rotate horizontals
                ftH = utils.rotate_dir(ft1_temp, ft2_temp, tf['tilt'])

                corrspec = ftZ_temp - fTF_ZH*ftH
                corrtime = np.real(np.fft.ifft(corrspec))[0:ws]
                if j==0:
                    dout[idx[j]:idx[j]+ws-tps] = corrtime[0:ws-tps]
                elif j+1 == len(idx):
                    dout[idx[j]+tps:idx[j]+ws] = corrtime[tps:ws]
                else:
                    dout[idx[j]+tps:idx[j]+ws-tps] = corrtime[tps:ws-tps]

            correct['ZH'] = dout

        if key == 'ZP-H' and tflist[key]:
            dout = np.zeros((trZ.stats.npts,), dtype=trZ.data.dtype)
            TF_ZH = tf['ZH']['TF_ZH']
            fTF_ZH = np.hstack(
                (TF_ZH, np.conj(TF_ZH[::-1][1:len(f)-1])))
            TF_PH = tf[key]['TF_PH']
            fTF_PH = np.hstack(
                (TF_PH, np.conj(TF_PH[::-1][1:len(f)-1])))
            TF_ZP_H = tf[key]['TF_ZP-H']
            fTF_ZP_H = np.hstack(
                (TF_ZP_H, np.conj(TF_ZP_H[::-1][1:len(f)-1])))

            for j in range(len(idx)):
                ftZ_temp = ftZ[j]
                ftP_temp = ftP[j]
                ft1_temp = ft1[j]
                ft2_temp = ft2[j]
                # Rotate horizontals
                ftH = utils.rotate_dir(ft1_temp, ft2_temp, tf['tilt'])

                corrspec = ftZ_temp - fTF_ZH*ftH - (ftP_temp - ftH*fTF_PH)*fTF_ZP_H
                corrtime = np.real(np.fft.ifft(corrspec))[0:ws]
                if j==0:
                    dout[idx[j]:idx[j]+ws-tps] = corrtime[0:ws-tps]
                elif j+1 == len(idx):
                    dout[idx[j]+tps:idx[j]+ws] = corrtime[tps:ws]
                else:
                    dout[idx[j]+tps:idx[j]+ws-tps] = corrtime[tps:ws-tps]

            correct['ZP-H'] = dout

    return correct

def plotcorrection(trIN, correctdict, freq=None,size=None,normalize=False,
            xlimit=None,save=False, fname=None, form='png'):
    """
    Function to plot the corrected vertical component seismograms.

    Parameters
    ----------
    trIN : :class:`~obspy.core.Trace`
            Original vertical component.
    correctdict : dictionary
            Corrected vertical records in a dictionary. See obsmaster.docorrection() for details.
    freq : :class:`~numpy.ndarray`
            Two element array specifying the frequency range for plotting. Default: [0.001, 0.49*samping_rate]
    normalize : bool
            If True, the traces will be normalized by the maximum absolute amplitudes before plotting.
    """
    sr = trIN.stats.sampling_rate
    taxis = trIN.times()

    if freq is None:
        freq = [0.001,0.49*sr]

    freqmin = freq[0]
    freqmax = freq[1]

    if xlimit is None:
        xlimit=[np.min(taxis),np.max(taxis)]

    if size is None:
        size = (10,10)
    imin = np.searchsorted(taxis,xlimit[0],side="left")
    imax = np.searchsorted(taxis,xlimit[1],side="left")

    st = trIN.stats.network+"."+trIN.stats.station
    year = trIN.stats.starttime.year
    julday = trIN.stats.starttime.julday
    hour = trIN.stats.starttime.hour
    mnt = trIN.stats.starttime.minute
    sec = trIN.stats.starttime.second
    tstamp = str(year) + '.' + str(julday)+'T'+str(hour)+'-'+str(mnt)+'-'+str(sec)

    clist=correctdict.keys()

    tr=trIN.copy()
    tr.filter('bandpass', freqmin=freqmin,
                        freqmax=freqmax, corners=2, zerophase=True)
    if normalize:
        rawdata=trIN.data/np.max(np.abs(trIN.data[imin:imax]))
    else:
        rawdata=trIN.data
    plt.figure(figsize=size)

    for ickey,ckey in enumerate(clist,1):
        plt.subplot(len(clist),1,ickey)
        plt.plot(taxis, rawdata, 'lightgray', lw=0.5)

        tr.data=np.squeeze(correctdict[ckey])
        tr.filter('bandpass', freqmin=freqmin,
                            freqmax=freqmax, corners=2, zerophase=True)
        if normalize:
            tr.data=tr.data/np.max(np.abs(tr.data[imin:imax]))
        plt.plot(taxis, tr.data, 'k', lw=0.5)

        if normalize:
            plt.ylim(-1.0,1.0)
        else:
            plt.ylim(0.9*np.min([np.min(tr.data[imin:imax]),np.min(rawdata[imin:imax])]),
                    1.1*np.max([np.max(tr.data[imin:imax]),np.max(rawdata[imin:imax])]))

        plt.title(st+':'+tstamp +
                  ': '+ckey, fontdict={'fontsize': 8})
        plt.gca().ticklabel_format(axis='y', style='sci', useOffset=True,
                                   scilimits=(-3, 3))
        plt.xlim(xlimit)

    plt.xlabel('Time (sec)')
    plt.tight_layout()

    # Save or show figure
    if save:
        if fname is None:
            if normalize:
                fname = st+"_"+tstamp+"_corrections_normalized.png"
            else:
                fname = st+"_"+tstamp+"_corrections_normalized.png"
        plt.savefig(fname,dpi=300, bbox_inches='tight', format=form)
    else:
        plt.show()

# wrappers for key functionalities
def TCremoval_wrapper(tr1,tr2,trZ,trP,window=7200,overlap=0.3,merge_taper=0.1,
                    qc_freq=[0.004, 0.2],qc_spectra=True,fig_spectra=False,
                    save_spectrafig=False,fig_transfunc=False,correctlist=getcorrectlist(),
                    targettracelist=None):
    """
    This is a wrapper to remove tilt and compliance noises.
    """
    """
    1. compute spectra for all traces.
    """
    spectra=getspectra(tr1,tr2,trZ,trP,pd=[0.004, 0.2],window=window,overlap=overlap,
                          QC=True,fig=False,save=True,debug=False)

    """
    2. Compute transfer functions
    """
    #compute transfer functions for all possible combinations
    transferfunc=gettransferfunc(spectra['auto'],spectra['cross'],spectra['rotation'],
                                tflist=gettflist(correctlist=correctlist))

    """
    3. Do corrections and plot the comparison
    """
    if targettracelist is None:
        correct = docorrection(tr1,tr2,trZ,trP,transferfunc,correctlist,
                                overlap=overlap,taper=merge_taper)
    else:
        correct = docorrection(targettracelist[0],targettracelist[1],targettracelist[2],
                                targettracelist[3],transferfunc,correctlist,
                                overlap=overlap,taper=merge_taper)

    return spectra,transferfunc,correct

# convert the correction dictionary to obspy stream
def correctdict2stream(trIN,correctdict,subset=None):
    """
    Convert corrected vertical trace (numpy.ndarray) to obspy.core.stream Stream object.

    Parameters
    ----------
    trIN :: class `~obspy.core.Trace`
            Trace before correction that provides metadata for the corrected traces.
    correctdict : dictionary
            Dictionary containing the tilt and compliance correction results.
    subset : dictionary
            Subset to only save selected corrections.

    Returns
    ----------
    outstream :: class `~obspy.core.Stream`
            The output stream containing the corrected vertical data, with metadata
            inherited from trIN
    tags :: string list
            List of string that label the corrected data, e.g., c_zp_h means corrected
            data component ZP-H. These tags follow the waveform tag requirement by ASDF.
    """

    clist=correctdict.keys()
    if subset is None:
        subset=clist

    # This is to hold the metadata.
    tr=trIN.copy()
    tr.data=np.ndarray((len(trIN.data),),dtype=trIN.data.dtype)
    trall=[]
    tags=[]

    for ckey in subset:
        if ckey in clist:
            tr.data=np.squeeze(correctdict[ckey])
            tag = 'c_'+ckey.replace('-','_')
            trall.append(tr)
            tags.append(tag.lower())
        else:
            print('Correction key ['+ckey+'] is NOT found in the correctdict. Skipped!')

    outstream=Stream(traces=trall)

    return outstream,tags

#save corrected traces
def savecorrection(trIN,correctdict,fname,subset=None,sta_inv=None,format='asdf',
                    debug=False,saveoriginal=False):
    """
    Save the corrected vertical trace to file with specified format (default is ASDF format).

    Parameters
    ----------
    trIN :: class `~obspy.core.Trace`
            Trace before correction that provides metadata for the corrected traces.
    correctdict : dictionary
            Dictionary containing the tilt and compliance correction results.
    fname : string
            File name.
    subset : dictionary
            Subset to only save selected corrections.
    sta_inv :: class `~obspy.core.inventory`
            Station inventory to handle station metadata.
    format : string
            Saving format, default is 'asdf'
    saveoriginal : bool
            If True, the original vertical trace will also be saved. Default is False.
    """

    clist=correctdict.keys()
    if subset is None:
        subset=clist

    if fname is None:
        year = trIN.stats.starttime.year
        julday = trIN.stats.starttime.julday
        hour = trIN.stats.starttime.hour
        mnt = trIN.stats.starttime.minute
        sec = trIN.stats.starttime.second
        tstamp = str(year) + '.' + str(julday)+'T'+str(hour)+'-'+str(mnt)+'-'+str(sec)

    streams,tags = correctdict2stream(trIN,correctdict,subset)

    if saveoriginal:
        streams.append(trIN)
        if len(trIN.stats.location) == 0:
            tlocation='00'
        else:
            tlocation=trIN.stats.location

        tags.append(trIN.stats.channel.lower()+'_'+tlocation.lower())

    if debug: print(streams)
    if format.lower() == 'asdf':
        if fname is None:
            fname = trIN.stats.network+'.'+trIN.stats.station+'_'+tstamp+'_LEN'+\
            str(trIN.stats.endtime-trIN.stats.starttime)+'s_corrected.h5'
        utils.save2asdf(fname,streams,tags,sta_inv)
    else:
        print('Saving to format other than ASDF is currently under develpment.')
