import os
import sys
import glob
import obspy
import scipy
import pyasdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from obspy.signal.filter import bandpass

'''
Inherited and modified from the plotting functions in the plotting_module of NoisePy (https://github.com/mdenolle/NoisePy).
Credits should be given to the development team for NoisePy (Chengxin Jiang and Marine Denolle).
'''

#############################################################################
############### PLOTTING RAW SEISMIC WAVEFORMS ##########################
#############################################################################
def plot_waveform(sfile,net,sta,freqmin,freqmax,save=False,figdir=None):
    '''
    display the downloaded waveform for station A
    PARAMETERS:
    -----------------------
    sfile: containing all wavefrom data for a time-chunck in ASDF format
    net,sta,comp: network, station name and component
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    USAGE:
    -----------------------
    plot_waveform('temp.h5','CI','BLC',0.01,0.5)
    '''
    # open pyasdf file to read
    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        sta_list = ds.waveforms.list()
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    # check whether station exists
    tsta = net+'.'+sta
    if tsta not in sta_list:
        raise ValueError('no data for %s in %s'%(tsta,sfile))

    tcomp = ds.waveforms[tsta].get_waveform_tags()
    ncomp = len(tcomp)
    if ncomp == 1:
        tr   = ds.waveforms[tsta][tcomp[0]]
        dt   = tr[0].stats.delta
        npts = tr[0].stats.npts
        tt   = np.arange(0,npts)*dt
        data = tr[0].data
        data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
        plt.figure(figsize=(9,3))
        plt.plot(tt,data,'k-',linewidth=1)
        plt.title('T\u2080:%s   %s.%s.%s   @%5.3f-%5.2f Hz' % (tr[0].stats.starttime,net,sta,tcomp[0].split('_')[0].upper(),freqmin,freqmax))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
    elif ncomp == 3:
        tr   = ds.waveforms[tsta][tcomp[0]]
        dt   = tr[0].stats.delta
        npts = tr[0].stats.npts
        tt   = np.arange(0,npts)*dt
        data = np.zeros(shape=(ncomp,npts),dtype=np.float32)
        for ii in range(ncomp):
            data[ii] = ds.waveforms[tsta][tcomp[ii]][0].data
            data[ii] = bandpass(data[ii],freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
        plt.figure(figsize=(9,6))
        plt.subplot(311)
        plt.plot(tt,data[0],'k-',linewidth=1)
        plt.title('T\u2080:%s   %s.%s   @%5.3f-%5.2f Hz' % (tr[0].stats.starttime,net,sta,freqmin,freqmax))
        plt.legend([tcomp[0].split('_')[0].upper()],loc='upper left')
        plt.subplot(312)
        plt.plot(tt,data[1],'k-',linewidth=1)
        plt.legend([tcomp[1].split('_')[0].upper()],loc='upper left')
        plt.subplot(313)
        plt.plot(tt,data[2],'k-',linewidth=1)
        plt.legend([tcomp[2].split('_')[0].upper()],loc='upper left')
        plt.xlabel('Time [s]')
        plt.tight_layout()

        if save:
            if not os.path.ifigdir(figdir):os.mkdir(figdir)
            outfname = figdir+'/{0:s}_{1:s}.{2:s}.pdf'.format(sfile.split('.')[0],net,sta)
            plt.savefig(outfname, format='pdf', dpi=400)
            plt.close()
        else:
            plt.show()


#############################################################################
###############PLOTTING XCORR RESULTS AS THE OUTPUT OF NoisePy S1 STEP ##########################
#############################################################################

def plot_substack_cc(sfile,freqmin,freqmax,lag=None,save=True,figdir='./'):
    '''
    display the 2D matrix of the cross-correlation functions for a certain time-chunck.
    PARAMETERS:
    --------------------------
    sfile: cross-correlation functions outputed by S1 of NoisePy workflow
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    lag: time ranges for display
    USAGE:
    --------------------------
    plot_substack_cc('temp.h5',0.1,1,100,True,'./')
    Note: IMPORTANT!!!! this script only works for cross-correlation with sub-stacks being set to True in S1.
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        # extract common variables
        spairs = ds.auxiliary_data.list()
        path_lists = ds.auxiliary_data[spairs[0]].list()
        flag   = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['substack']
        dt     = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['dt']
        maxlag = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    # only works for cross-correlation with substacks generated
    if not flag:
        raise ValueError('seems no substacks have been done! not suitable for this plotting function')

    # lags for display
    if not lag:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')

    # t is the time labels for plotting
    t = np.arange(-int(lag),int(lag)+dt,step=int(2*int(lag)/4))
    # windowing the data
    indx1 = int((maxlag-lag)/dt)
    indx2 = indx1+2*int(lag/dt)+1

    for spair in spairs:
        ttr = spair.split('_')
        net1,sta1 = ttr[0].split('.')
        net2,sta2 = ttr[1].split('.')
        for ipath in path_lists:
            chan1,chan2 = ipath.split('_')
            try:
                dist = ds.auxiliary_data[spair][ipath].parameters['dist']
                ngood= ds.auxiliary_data[spair][ipath].parameters['ngood']
                ttime= ds.auxiliary_data[spair][ipath].parameters['time']
                timestamp = np.empty(ttime.size,dtype='datetime64[s]')
            except Exception:
                print('continue! something wrong with %s %s'%(spair,ipath))
                continue

            # cc matrix
            data = ds.auxiliary_data[spair][ipath].data[:,indx1:indx2]
            nwin = data.shape[0]
            amax = np.zeros(nwin,dtype=np.float32)
            if nwin==0 or len(ngood)==1: print('continue! no enough substacks!');continue

            tmarks = []
            # load cc for each station-pair
            for ii in range(nwin):
                data[ii] = bandpass(data[ii],freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                amax[ii] = max(data[ii])
                data[ii] /= amax[ii]
                timestamp[ii] = obspy.UTCDateTime(ttime[ii])
                tmarks.append(obspy.UTCDateTime(ttime[ii]).strftime('%H:%M:%S'))

            # plotting
            if nwin>10:
                tick_inc = int(nwin/5)
            else:
                tick_inc = 2
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(211)
            ax.matshow(data,cmap='seismic',extent=[-lag,lag,nwin,0],aspect='auto')
            ax.set_title('%s.%s.%s  %s.%s.%s  dist:%5.2fkm' % (net1,sta1,chan1,net2,sta2,chan2,dist))
            ax.set_xlabel('time [s]')
            ax.set_xticks(t)
            ax.set_yticks(np.arange(0,nwin,step=tick_inc))
            ax.set_yticklabels(timestamp[0:-1:tick_inc])
            ax.xaxis.set_ticks_position('bottom')
            ax1 = fig.add_subplot(413)
            ax1.set_title('stacked and filtered at %4.2f-%4.2f Hz'%(freqmin,freqmax))
            ax1.plot(np.arange(-lag,lag+dt,dt),np.mean(data,axis=0),'k-',linewidth=1)
            ax1.set_xticks(t)
            ax2 = fig.add_subplot(414)
            ax2.plot(amax/min(amax),'r-')
            ax2.plot(ngood,'b-')
            ax2.set_xlabel('waveform number')
            ax2.set_xticks(np.arange(0,nwin,step=tick_inc))
            ax2.set_xticklabels(tmarks[0:nwin:tick_inc])
            #for tick in ax[2].get_xticklabels():
            #    tick.set_rotation(30)
            ax2.legend(['relative amp','ngood'],loc='upper right')
            fig.tight_layout()

            # save figure or just show
            if save:
                if figdir==None:figdir = sfile.split('.')[0]
                if not os.path.ifigdir(figdir):os.mkdir(figdir)
                outfname = figdir+'/{0:s}.{1:s}.{2:s}_{3:s}.{4:s}.{5:s}.pdf'.format(net1,sta1,chan1,net2,sta2,chan2)
                fig.savefig(outfname, format='pdf', dpi=400)
                plt.close()
            else:
                fig.show()


def plot_substack_cc_spect(sfile,freqmin,freqmax,lag=None,save=True,figdir='./'):
    '''
    display the amplitude spectrum of the cross-correlation functions for a time-chunck.
    PARAMETERS:
    -----------------------
    sfile: cross-correlation functions outputed by S1
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    lag: time ranges for display
    USAGE:
    -----------------------
    plot_substack_cc('temp.h5',0.1,1,200,True,'./')
    Note: IMPORTANT!!!! this script only works for the cross-correlation with sub-stacks in S1.
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        # extract common variables
        spairs = ds.auxiliary_data.list()
        path_lists = ds.auxiliary_data[spairs[0]].list()
        flag   = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['substack']
        dt     = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['dt']
        maxlag = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    # only works for cross-correlation with substacks generated
    if not flag:
        raise ValueError('seems no substacks have been done! not suitable for this plotting function')

    # lags for display
    if not lag:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(lag),int(lag)+dt,step=int(2*int(lag)/4))
    indx1 = int((maxlag-lag)/dt)
    indx2 = indx1+2*int(lag/dt)+1
    nfft  = int(next_fast_len(indx2-indx1))
    freq  = scipy.fftpack.fftfreq(nfft,d=dt)[:nfft//2]

    for spair in spairs:
        ttr = spair.split('_')
        net1,sta1 = ttr[0].split('.')
        net2,sta2 = ttr[1].split('.')
        for ipath in path_lists:
            chan1,chan2 = ipath.split('_')
            try:
                dist = ds.auxiliary_data[spair][ipath].parameters['dist']
                ngood= ds.auxiliary_data[spair][ipath].parameters['ngood']
                ttime= ds.auxiliary_data[spair][ipath].parameters['time']
                timestamp = np.empty(ttime.size,dtype='datetime64[s]')
            except Exception:
                print('continue! something wrong with %s %s'%(spair,ipath))
                continue

            # cc matrix
            data = ds.auxiliary_data[spair][ipath].data[:,indx1:indx2]
            nwin = data.shape[0]
            amax = np.zeros(nwin,dtype=np.float32)
            spec = np.zeros(shape=(nwin,nfft//2),dtype=np.complex64)
            if nwin==0 or len(ngood)==1: print('continue! no enough substacks!');continue

            # load cc for each station-pair
            for ii in range(nwin):
                spec[ii] = scipy.fftpack.fft(data[ii],nfft,axis=0)[:nfft//2]
                spec[ii] /= np.max(np.abs(spec[ii]),axis=0)
                data[ii] = bandpass(data[ii],freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                amax[ii] = max(data[ii])
                data[ii] /= amax[ii]
                timestamp[ii] = obspy.UTCDateTime(ttime[ii])

            # plotting
            if nwin>10:
                tick_inc = int(nwin/5)
            else:
                tick_inc = 2
            fig,ax = plt.subplots(3,sharex=False)
            ax[0].matshow(data,cmap='seismic',extent=[-lag,lag,nwin,0],aspect='auto')
            ax[0].set_title('%s.%s.%s  %s.%s.%s  dist:%5.2f km' % (net1,sta1,chan1,net2,sta2,chan2,dist))
            ax[0].set_xlabel('time [s]')
            ax[0].set_xticks(t)
            ax[0].set_yticks(np.arange(0,nwin,step=tick_inc))
            ax[0].set_yticklabels(timestamp[0:-1:tick_inc])
            ax[0].xaxis.set_ticks_position('bottom')
            ax[1].matshow(np.abs(spec),cmap='seismic',extent=[freq[0],freq[-1],nwin,0],aspect='auto')
            ax[1].set_xlabel('freq [Hz]')
            ax[1].set_ylabel('amplitudes')
            ax[1].set_yticks(np.arange(0,nwin,step=tick_inc))
            ax[1].xaxis.set_ticks_position('bottom')
            ax[2].plot(amax/min(amax),'r-')
            ax[2].plot(ngood,'b-')
            ax[2].set_xlabel('waveform number')
            #ax[1].set_xticks(np.arange(0,nwin,int(nwin/5)))
            ax[2].legend(['relative amp','ngood'],loc='upper right')
            fig.tight_layout()

            # save figure or just show
            if save:
                if figdir==None:figdir = sfile.split('.')[0]
                if not os.path.ifigdir(figdir):os.mkdir(figdir)
                outfname = figdir+'/{0:s}.{1:s}.{2:s}_{3:s}.{4:s}.{5:s}.pdf'.format(net1,sta1,chan1,net2,sta2,chan2)
                fig.savefig(outfname, format='pdf', dpi=400)
                plt.close()
            else:
                fig.show()


#############################################################################
###############PLOTTING THE POST-STACKING XCORR FUNCTIONS AS OUTPUT OF S2 STEP IN NOISEPY ##########################
#############################################################################

def plot_substack_all(sfile,freqmin,freqmax,comp,lag=None,save=False,figdir=None):
    '''
    display the 2D matrix of the cross-correlation functions stacked for all time windows.
    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    lag: time ranges for display
    comp: cross component of the targeted cc functions
    USAGE:
    ----------------------
    plot_substack_all('temp.h5',0.1,1,'ZZ',50,True,'./')
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    paths = comp
    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        # extract common variables
        dtype_lists = ds.auxiliary_data.list()
        dt     = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dt']
        dist   = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dist']
        maxlag = ds.auxiliary_data[dtype_lists[0]][paths].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    if len(dtype_lists)==1:
        raise ValueError('Abort! seems no substacks have been done')

    # lags for display
    if not lag:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(lag),int(lag)+dt,step=int(2*int(lag)/4))
    indx1 = int((maxlag-lag)/dt)
    indx2 = indx1+2*int(lag/dt)+1

    # other parameters to keep
    nwin = len(dtype_lists)-1
    data = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
    ngood= np.zeros(nwin,dtype=np.int16)
    ttime= np.zeros(nwin,dtype=np.int)
    timestamp = np.empty(ttime.size,dtype='datetime64[s]')
    amax = np.zeros(nwin,dtype=np.float32)

    for ii,itype in enumerate(dtype_lists[2:]):
        timestamp[ii] = obspy.UTCDateTime(np.float(itype[1:]))
        try:
            ngood[ii] = ds.auxiliary_data[itype][paths].parameters['ngood']
            ttime[ii] = ds.auxiliary_data[itype][paths].parameters['time']
            #timestamp[ii] = obspy.UTCDateTime(ttime[ii])
            # cc matrix
            data[ii] = ds.auxiliary_data[itype][paths].data[indx1:indx2]
            data[ii] = bandpass(data[ii],freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            amax[ii] = np.max(data[ii])
            data[ii] /= amax[ii]
        except Exception as e:
            print(e);continue

        if len(ngood)==1:
            raise ValueError('seems no substacks have been done! not suitable for this plotting function')

    # plotting
    if nwin>100:
        tick_inc = int(nwin/10)
    elif nwin>10:
        tick_inc = int(nwin/5)
    else:
        tick_inc = 2
    fig,ax = plt.subplots(2,sharex=False)
    ax[0].matshow(data,cmap='seismic',extent=[-lag,lag,nwin,0],aspect='auto')
    ax[0].set_title('%s dist:%5.2f km filtered at %4.2f-%4.2fHz' % (sfile.split('/')[-1],dist,freqmin,freqmax))
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('wavefroms')
    ax[0].set_xticks(t)
    ax[0].set_yticks(np.arange(0,nwin,step=tick_inc))
    ax[0].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[0].xaxis.set_ticks_position('bottom')
    ax[1].plot(amax/max(amax),'r-')
    ax[1].plot(ngood,'b-')
    ax[1].set_xlabel('waveform number')
    ax[1].set_xticks(np.arange(0,nwin,nwin//5))
    ax[1].legend(['relative amp','ngood'],loc='upper right')
    # save figure or just show
    if save:
        if figdir==None:figdir = sfile.split('.')[0]
        if not os.path.ifigdir(figdir):os.mkdir(figdir)
        outfname = figdir+'/{0:s}_{1:4.2f}_{2:4.2f}Hz.pdf'.format(sfile.split('/')[-1],freqmin,freqmax)
        fig.savefig(outfname, format='pdf', dpi=400)
        plt.close()
    else:
        fig.show()


def plot_substack_all_spect(sfile,freqmin,freqmax,comp,lag=None,save=False,figdir=None):
    '''
    display the amplitude spectrum of the cross-correlation functions stacked for all time windows.
    PARAMETERS:
    -----------------------
    sfile: cross-correlation functions outputed by S2
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    lag: time ranges for display
    comp: cross component of the targeted cc functions
    USAGE:
    -----------------------
    plot_substack_all('temp.h5',0.1,1,'ZZ',50,True,'./')
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    paths = comp
    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        # extract common variables
        dtype_lists = ds.auxiliary_data.list()
        dt     = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dt']
        dist   = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dist']
        maxlag = ds.auxiliary_data[dtype_lists[0]][paths].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    if len(dtype_lists)==1:
        raise ValueError('Abort! seems no substacks have been done')

    # lags for display
    if not lag:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(lag),int(lag)+dt,step=int(2*int(lag)/4))
    indx1 = int((maxlag-lag)/dt)
    indx2 = indx1+2*int(lag/dt)+1
    nfft  = int(next_fast_len(indx2-indx1))
    freq  = scipy.fftpack.fftfreq(nfft,d=dt)[:nfft//2]

    # other parameters to keep
    nwin = len(dtype_lists)-1
    data = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
    spec = np.zeros(shape=(nwin,nfft//2),dtype=np.complex64)
    ngood= np.zeros(nwin,dtype=np.int16)
    ttime= np.zeros(nwin,dtype=np.int)
    timestamp = np.empty(ttime.size,dtype='datetime64[s]')
    amax = np.zeros(nwin,dtype=np.float32)

    for ii,itype in enumerate(dtype_lists[1:]):
        timestamp[ii] = obspy.UTCDateTime(np.float(itype[1:]))
        try:
            ngood[ii] = ds.auxiliary_data[itype][paths].parameters['ngood']
            ttime[ii] = ds.auxiliary_data[itype][paths].parameters['time']
            #timestamp[ii] = obspy.UTCDateTime(ttime[ii])
            # cc matrix
            tdata = ds.auxiliary_data[itype][paths].data[indx1:indx2]
            spec[ii] = scipy.fftpack.fft(tdata,nfft,axis=0)[:nfft//2]
            spec[ii] /= np.max(np.abs(spec[ii]))
            data[ii] = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            amax[ii] = np.max(data[ii])
            data[ii] /= amax[ii]
        except Exception as e:
            print(e);continue

        if len(ngood)==1:
            raise ValueError('seems no substacks have been done! not suitable for this plotting function')

    # plotting
    tick_inc = 50
    fig,ax = plt.subplots(3,sharex=False)
    ax[0].matshow(data,cmap='seismic',extent=[-lag,lag,nwin,0],aspect='auto')
    ax[0].set_title('%s dist:%5.2f km' % (sfile.split('/')[-1],dist))
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('wavefroms')
    ax[0].set_xticks(t)
    ax[0].set_yticks(np.arange(0,nwin,step=tick_inc))
    ax[0].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[0].xaxis.set_ticks_position('bottom')
    ax[1].matshow(np.abs(spec),cmap='seismic',extent=[freq[0],freq[-1],nwin,0],aspect='auto')
    ax[1].set_xlabel('freq [Hz]')
    ax[1].set_ylabel('amplitudes')
    ax[1].set_yticks(np.arange(0,nwin,step=tick_inc))
    ax[1].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[1].xaxis.set_ticks_position('bottom')
    ax[2].plot(amax/max(amax),'r-')
    ax[2].plot(ngood,'b-')
    ax[2].set_xlabel('waveform number')
    ax[2].set_xticks(np.arange(0,nwin,nwin//15))
    ax[2].legend(['relative amp','ngood'],loc='upper right')
    # save figure or just show
    if save:
        if figdir==None:figdir = sfile.split('.')[0]
        if not os.path.ifigdir(figdir):os.mkdir(figdir)
        outfname = figdir+'/{0:s}.pdf'.format(sfile.split('/')[-1])
        fig.savefig(outfname, format='pdf', dpi=400)
        plt.close()
    else:
        fig.show()


def plot_moveout_heatmap(sfiles,sta,dtype,freq,comp,dist_inc,lag=None,save=False,figdir=None):
    '''
    display the moveout (2D matrix) of the cross-correlation functions stacked for all time chuncks.
    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: station name as the virtual source.
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    comp:   cross component
    dist_inc: distance bins to stack over
    lag: lag times for displaying
    save: set True to save the figures (in pdf format)
    figdir: diresied directory to save the figure (if not provided, save to default dir)
    USAGE:
    ----------------------
    plot_moveout('temp.h5','sta','Allstack_pws',0.1,0.2,1,'ZZ',200,True,'./temp')
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    path  = comp
    freqmin=freq[0]
    freqmax=freq[1]
    receiver = sta+'.h5'
    # extract common variables
    try:
        ds    = pyasdf.ASDFDataSet(sfiles[0],mode='r')
        dt    = ds.auxiliary_data[dtype][path].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][path].parameters['maxlag']
        stack_method = dtype.split('0')[-1]
    except Exception:
        print("exit! cannot open %s to read"%sfiles[0]);sys.exit()

    # lags for display
    if lag is None:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(lag),int(lag)+dt,step=(int(2*int(lag)/4)))
    indx1 = int((maxlag-lag)/dt)
    indx2 = indx1+2*int(lag/dt)+1

    # cc matrix
    nwin = len(sfiles)
    data = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
    dist = np.zeros(nwin,dtype=np.float32)
    ngood= np.zeros(nwin,dtype=np.int16)

    # load cc and parameter matrix
    for ii in range(len(sfiles)):
        sfile = sfiles[ii]
        treceiver = sfile.split('_')[-1]

        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        try:
            # load data to variables
            dist[ii] = ds.auxiliary_data[dtype][path].parameters['dist']
            ngood[ii]= ds.auxiliary_data[dtype][path].parameters['ngood']
            tdata    = ds.auxiliary_data[dtype][path].data[indx1:indx2]
            if treceiver == receiver: tdata=np.flip(tdata,axis=0)
        except Exception:
            print("continue! cannot read %s "%sfile);continue

        data[ii] = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)

    # average cc
    ntrace = int(np.round(np.max(dist)+0.51)/dist_inc)
    ndata  = np.zeros(shape=(ntrace,indx2-indx1),dtype=np.float32)
    ndist  = np.zeros(ntrace,dtype=np.float32)
    for td in range(0,ntrace-1):
        tindx = np.where((dist>=td*dist_inc)&(dist<(td+1)*dist_inc))[0]
        if len(tindx):
            ndata[td] = np.mean(data[tindx],axis=0)
            ndist[td] = (td+0.5)*dist_inc

    # normalize waveforms
    indx  = np.where(ndist>0)[0]
    ndata = ndata[indx]
    ndist = ndist[indx]
    for ii in range(ndata.shape[0]):
        # print(ii,np.max(np.abs(ndata[ii])))
        ndata[ii] /= np.max(np.abs(ndata[ii]))

    # plotting figures
    fig,ax = plt.subplots()
    ax.matshow(ndata,cmap='seismic',extent=[-lag,lag,ndist[-1],ndist[0]],aspect='auto')
    ax.set_title('%s allstack %s @%5.3f-%5.2f Hz'%(sta,stack_method,freqmin,freqmax))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('distance [km]')
    ax.set_xticks(t)
    ax.xaxis.set_ticks_position('bottom')
    #ax.text(np.ones(len(ndist))*(lag-5),dist[ndist],ngood[ndist],fontsize=8)

    # save figure or show
    if save:
        outfname = figdir+'/moveout_'+sta+'_heatmap_'+str(stack_method)+'_'+str(freqmin)+'_'+str(freqmax)+'Hz_'+str(dist_inc)+'kmbin_'+comp+'.png'
        fig.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        fig.show()


def plot_moveout_wiggle(sfiles,sta,dtype,freq,ccomp=['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ'],
                              scale=1.0,lag=None,ylim=None,save=False,figdir=None):
    '''
    display the moveout waveforms of the cross-correlation functions stacked for all time chuncks.
    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: source station name
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    lag: lag times for displaying
    save: set True to save the figures (in pdf format)
    figdir: diresied directory to save the figure (if not provided, save to default dir)
    USAGE:
    ----------------------
    plot_substack_moveout('temp.h5','Allstack0pws',0.1,0.2,'ZZ',200,True,'./temp')
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    freqmin=freq[0]
    freqmax=freq[1]
    receiver = sta+'.h5'
    stack_method = dtype.split('_')[-1]
    typeofcomp=str(type(ccomp)).split("'")[1]
    ccomptemp=[]
    if typeofcomp=='str':
        ccomptemp.append(ccomp)
        ccomp=ccomptemp
    print(ccomp)

    #determine subplot parameters if not specified.
    if len(ccomp)>9:
        raise ValueError('ccomp includes more than 9 (maximum allowed) elements!')
    elif len(ccomp)==9:
        subplot=[3,3]
        figsize=[14,10.5]
    elif len(ccomp) >=7 and len(ccomp) <=8:
        subplot=[2,4]
        figsize=[18,7.5]
    elif len(ccomp) >=5 and len(ccomp) <=6:
        subplot=[2,3]
        figsize=[14,7.5]
    elif len(ccomp) ==4:
        subplot=[2,2]
        figsize=[10,7.5]
    else:
        subplot=[1,len(ccomp)]
        if len(ccomp)==3:
            figsize=[13,3]
        elif len(ccomp)==2:
            figsize=[8,3]
        else:
            figsize=[4,3]

#     ccomp = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']

    # extract common variables
    try:
        ds    = pyasdf.ASDFDataSet(sfiles[0],mode='r')
        dt    = ds.auxiliary_data[dtype][ccomp[0]].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][ccomp[0]].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read"%sfiles[0]);sys.exit()

    # lags for display
    if lag is None:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')
    tt = np.arange(-int(lag),int(lag)+dt,dt)
    indx1 = int((maxlag-lag)/dt)
    indx2 = indx1+2*int(lag/dt)+1

    # load cc and parameter matrix
    plt.figure(figsize=figsize)
    for ic in range(len(ccomp)):
        comp = ccomp[ic]
#         tmp  = '33'+str(ic+1)
        plt.subplot(subplot[0],subplot[1],ic+1)
        mdist=0
        for ii in range(len(sfiles)):
            sfile = sfiles[ii]
            iflip = 0
            treceiver = sfile.split('_')[-1]
            if treceiver == receiver:
                iflip = 1

            ds = pyasdf.ASDFDataSet(sfile,mode='r')
            try:
                # load data to variables
                dist = ds.auxiliary_data[dtype][comp].parameters['dist']
                ngood= ds.auxiliary_data[dtype][comp].parameters['ngood']
                tdata  = ds.auxiliary_data[dtype][comp].data[indx1:indx2]

            except Exception:
                print("continue! cannot read %s "%sfile);continue


            if ylim is not None:
                if dist>ylim[1] or dist<ylim[0]:
                    continue
            elif dist>mdist:
                mdist=dist

            tdata = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            tdata /= np.max(tdata,axis=0)

            if iflip:
                plt.plot(tt,scale*np.flip(tdata,axis=0)+dist,'k',linewidth=0.8)
            else:
                plt.plot(tt,scale*tdata+dist,'k',linewidth=0.8)
            plt.title('%s filtered @%5.3f-%5.3f Hz' % (sta,freqmin,freqmax))
            plt.xlabel('time (s)')
            plt.ylabel('offset (km)')

        plt.xlim([-1.0*lag,lag])
        if ylim is None:
            ylim=[0.0,mdist]
        plt.plot([0,0],ylim,'b--',linewidth=1)

        plt.ylim(ylim)
        font = {'family': 'serif', 'color':  'red', 'weight': 'bold','size': 14}
        plt.text(lag*0.75,ylim[0]+0.07*(ylim[1]-ylim[0]),comp,fontdict=font,
                 bbox=dict(facecolor='white',edgecolor='none',alpha=0.85))
    plt.tight_layout()

    # save figure or show
    if save:
        outfname = figdir+'/moveout_'+sta+'_wiggle_'+str(stack_method)+'_'+str(freqmin)+'_'+str(freqmax)+'Hz_'+str(len(ccomp))+'ccomp.png'
        plt.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        plt.show()
