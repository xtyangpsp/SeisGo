import os,sys,glob,time
import obspy
import scipy
import pycwt
import pyasdf
import datetime
import numpy as np
import pandas as pd
from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression
from scipy.fftpack import fft,ifft,next_fast_len
from seisgo import stacking as stack
from seisgo.types import CorrData, FFTData
from seisgo import utils

#####
########################################################
################ CROSS-CORRELATE FUNCTIONS ##################
########################################################
def cc_memory(inc_hours,sps,nsta,ncomp,cc_len,cc_step):
    """
    Estimates the memory usage with given correlation parameters, assuming float 32.
    """
    nseg_chunk = int(np.floor((3600*inc_hours-cc_len)/cc_step))+1
    npts_chunk = int(nseg_chunk*cc_len*sps)
    memory_size = nsta*npts_chunk*4/1024/1024/1024**ncomp

    return memory_size

def compute_fft(trace,win_len,step,stainv=None,
                 freqmin=None,freqmax=None,time_norm='no',freq_norm='no',
                 smooth=20,smooth_spec=None,misc=dict(),taper_frac=0.05,df=None):
    """
    Call FFTData to build the object. This is an alternative of directly call FFTData().
    The motivation of this function is to provide an user interface to build FFTData object.
    """
    return FFTData(trace=trace,win_len=win_len,step=step,
                    stainv=stainv,freqmin=freqmin,freqmax=freqmax,time_norm=time_norm,
                    freq_norm=freq_norm,smooth=smooth,smooth_spec=smooth_spec,misc=misc,
                    taper_frac=taper_frac,df=df)
#assemble FFT with given asdf file name
def assemble_fft(sfile,win_len,step,freqmin=None,freqmax=None,
                    time_norm='no',freq_norm='no',smooth=20,smooth_spec=20,
                    taper_frac=0.05,df=None,exclude_chan=[None],v=True):
    #only deal with ASDF format for now.

    # retrive station information
    ds=pyasdf.ASDFDataSet(sfile,mpi=False,mode='r')
    sta_list = ds.waveforms.list()
    nsta=len(sta_list)
    print('found %d stations in total'%nsta)

    fftdata_all=[]
    if nsta==0:
        print('no data in %s'%sfile);
        return fftdata_all

    # loop through all stations
    print('working on file: '+sfile.split('/')[-1])

    for ista in sta_list:
        # get station and inventory
        try:
            inv1 = ds.waveforms[ista]['StationXML']
        except Exception as e:
            print('abort! no stationxml for %s in file %s'%(ista,sfile))
            continue

        # get days information: works better than just list the tags
        all_tags = ds.waveforms[ista].get_waveform_tags()
        if len(all_tags)==0:continue

        #----loop through each stream----
        for itag in all_tags:
            if v:print("FFT for station %s and trace %s" % (ista,itag))

            # read waveform data
            source = ds.waveforms[ista][itag]
            if len(source)==0:continue

            # channel info
            comp = source[0].stats.channel
            if comp[-1] =='U': comp.replace('U','Z')

            #exclude some channels in the exclude_chan list.
            if comp in exclude_chan:
                print(comp+" is in the exclude_chan list. Skip it!")
                continue

            fftdata=FFTData(source,win_len,step,stainv=inv1,
                            time_norm=time_norm,freq_norm=freq_norm,
                            smooth=smooth,freqmin=freqmin,freqmax=freqmax,
                            smooth_spec=smooth_spec,taper_frac=taper_frac,df=df)
            if fftdata.data is not None:
                fftdata_all.append(fftdata)
    ####
    return fftdata_all

def smooth_source_spect(fft1,cc_method,sn):
    '''
    this function smoothes amplitude spectrum of the 2D spectral matrix. (used in S1)
    PARAMETERS:
    ---------------------
    cc_para: dictionary containing useful cc parameters
    fft1:    source spectrum matrix

    RETURNS:
    ---------------------
    sfft1: complex numpy array with normalized spectrum
    '''
    smoothspect_N = sn #cc_para['smoothspect_N']

    N=fft1.shape[0]
    Nfft2=fft1.shape[1]
    fft1=fft1.reshape(fft1.size)
    if cc_method == 'deconv':

        #-----normalize single-station cc to z component-----
        temp = utils.moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = fft1/temp**2
        except Exception:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'coherency':
        temp = utils.moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = fft1/temp
        except Exception:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'xcorr':
        sfft1 = fft1

    else:
        raise ValueError('no correction correlation method is selected at L59')

    return sfft1.reshape(N,Nfft2)
#
def do_correlation(sfile,ncomp,win_len,step,maxlag,cc_method='xcorr',
                    acorr_only=False,xcorr_only=True,substack=False,substack_len=None,
                    smoothspect_N=20,maxstd=10,freqmin=None,freqmax=None,time_norm='no',
                    freq_norm='no',smooth_N=20,exclude_chan=[None],outdir='.',v=True):
    """
    Wrapper for computing correlation functions. It includes two key steps: 1) compute and assemble
    the FFT of all data in the sfile, into a list of FFTData objects; 2) loop through the FFTData object
    list and do correlation (auto or xcorr) for each source-receiver pair.

    ====RETURNS====
    ndata: the number of station-component pairs in the sfile, that have been processed.
    """
    if acorr_only and xcorr_only:
        raise ValueError('acorr_only and xcorr_only CAN NOT all be True.')

    tname = sfile.split('/')[-1]
    tmpfile = os.path.join(outdir,tname.split('.')[0]+'.tmp')

    #file to store CC results.
    outfile=os.path.join(outdir,tname)
    # check whether time chunk been processed or not
    if os.path.isfile(tmpfile):
        ftemp = open(tmpfile,'r')
        alines = ftemp.readlines()
        if len(alines) and alines[-1] == 'done':
            return 0
        else:
            ftemp.close()
            os.remove(tmpfile)
            if os.path.isfile(outfile): os.remove(outfile)

    ftmp = open(tmpfile,'w')

    ##############compute FFT#############
    fftdata=assemble_fft(sfile,win_len,step,freqmin=freqmin,freqmax=freqmax,
                    time_norm=time_norm,freq_norm=freq_norm,smooth=smooth_N,exclude_chan=exclude_chan)
    ndata=len(fftdata)

    #############PERFORM CROSS-CORRELATION##################
    if v: print(tname)
    for iiS in range(ndata):
        # get index right for auto/cross correlation
        istart=iiS;iend=ndata
        if acorr_only:iend=np.minimum(iiS+ncomp,ndata)
        if xcorr_only:istart=np.minimum(iiS+ncomp,ndata)
        #-----------now loop III for each receiver B----------
        for iiR in range(istart,iend):
            if v:print('receiver: %s %s' % (fftdata[iiR].net,fftdata[iiR].sta))
            if fftdata[iiS].data is not None and fftdata[iiR].data is not None:
                corrdata=correlate(fftdata[iiS],fftdata[iiR],maxlag,method=cc_method,substack=substack,
                                    smoothspect_N=smoothspect_N,substack_len=substack_len,
                                    maxstd=maxstd)

                if corrdata.data is not None: corrdata.to_asdf(file=outfile)

    # create a stamp to show time chunk being done
    ftmp.write('done')
    ftmp.close()

    return ndata

def correlate(fftdata1,fftdata2,maxlag,method='xcorr',substack=False,
                substack_len=None,smoothspect_N=20,maxstd=10,terror=0.01):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft.

    PARAMETERS:
    ---------------------
    fftdata1: FFTData for the source station
    fftdata2: FFTData of the receiver station
    maxlag:  maximum lags to keep in the cross correlation
    method:  cross-correlation methods selected by the user
    terror: 0-1 fraction of timing error in searching for overlapping. The timing error =
                    terror*dt
    RETURNS:
    ---------------------
    corrdata: CorrData object of cross-correlation functions in time domain
    '''
    corrdata=CorrData()

    #check overlapping timestamps before any other processing
    #this step is required when there are gaps in the data.
    ind1,ind2=utils.check_overlap(fftdata1.time,fftdata2.time,error=terror*fftdata1.dt)
    if not len(ind1):
        print('no overlapped timestamps in the data.')
        return corrdata

    #---------- check the existence of earthquakes by std of the data.----------
    source_std = fftdata1.std[ind1]
    sou_ind = np.where((source_std<maxstd)&(source_std>0)&(np.isnan(source_std)==0))[0]
    if not len(sou_ind): return corrdata

    receiver_std = fftdata2.std[ind2]
    rec_ind = np.where((receiver_std<maxstd)&(receiver_std>0)&(np.isnan(receiver_std)==0))[0]
    bb=np.intersect1d(sou_ind,rec_ind)
    if len(bb)==0:return corrdata

    bb_data1=[ind1[i] for i in bb]
    bb_data2=[ind2[i] for i in bb]

    #----load paramters----
    dt      = fftdata1.dt
    cc_len  = fftdata1.win_len
    cc_step = fftdata1.step
    if substack_len is None: substack_len=cc_len

    Nfft = fftdata1.Nfft
    Nfft2 = Nfft//2

    fft1=fftdata1.data[bb_data1,:Nfft2]
    fft1=np.conj(fft1) #get the conjugate of fft1
    nwin  = fft1.shape[0]
    fft2=fftdata2.data[bb_data2,:Nfft2]

    timestamp=fftdata1.time[bb_data1]

    if method != "xcorr":
        fft1 = smooth_source_spect(fft1,method,smoothspect_N)
    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    corr = fft1.reshape(fft1.size,)*fft2.reshape(fft2.size,)

    if method == "coherency":
        temp = utils.moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    if substack:
        if substack_len == cc_len:
            # choose to keep all fft data for a day
            s_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)   # stacked correlation
            ampmax = np.zeros(nwin,dtype=np.float32)
            n_corr = np.zeros(nwin,dtype=np.int16)                  # number of correlations for each substack
            t_corr = timestamp                                        # timestamp
            crap   = np.zeros(Nfft,dtype=np.complex64)
            for i in range(nwin):
                n_corr[i]= 1
                crap[:Nfft2] = corr[i,:]
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = timestamp[-1]-timestamp[0]             # total duration of what we have now
            tstart = timestamp[0]

            nstack = int(np.round(Ttotal/substack_len))
            ampmax = np.zeros(nstack,dtype=np.float32)
            s_corr = np.zeros(shape=(nstack,Nfft),dtype=np.float32)
            n_corr = np.zeros(nstack,dtype=np.int)
            t_corr = np.zeros(nstack,dtype=np.float)
            crap   = np.zeros(Nfft,dtype=np.complex64)

            for istack in range(nstack):
                # find the indexes of all of the windows that start or end within
                itime = np.where( (timestamp >= tstart) & (timestamp < tstart+substack_len) )[0]
                if len(itime)==0:tstart+=substack_len;continue

                crap[:Nfft2] = np.mean(corr[itime,:],axis=0)   # linear average of the correlation
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[istack,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(itime)               # number of windows stacks
                t_corr[istack] = tstart                   # save the time stamps
                tstart += substack_len
                #print('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        ampmax = np.max(corr,axis=1)
        tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
        n_corr = nwin
        s_corr = np.zeros(Nfft,dtype=np.float32)
        t_corr = timestamp[0]
        crap   = np.zeros(Nfft,dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr[tindx],axis=0)
        crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2],axis=0)
        crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
        s_corr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2+1, Nfft2)*dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim==1:
        s_corr = s_corr[ind]
    elif s_corr.ndim==2:
        s_corr = s_corr[:,ind]

    ### call CorrData to build the object
    cc_comp= fftdata1.chan[-1]+fftdata2.chan[-1]
    dist,azi,baz = obspy.geodetics.base.gps2dist_azimuth(fftdata1.lat,fftdata1.lon,fftdata2.lat,fftdata2.lon)

    corrdata=CorrData(net=[fftdata1.net,fftdata2.net],sta=[fftdata1.sta,fftdata2.sta],\
                    loc=[fftdata1.loc,fftdata2.loc],chan=[fftdata1.chan,fftdata2.chan],\
                    lon=[fftdata1.lon,fftdata2.lon],lat=[fftdata1.lat,fftdata2.lat],\
                    ele=[fftdata1.ele,fftdata2.ele],cc_comp=cc_comp,lag=maxlag,\
                    dt=fftdata1.dt,cc_len=cc_len,cc_step=cc_step,dist=dist/1000,az=azi,\
                    baz=baz,ngood=n_corr,time=t_corr,data=s_corr,substack=substack,\
                    misc={"cc_method":method,"dist_unit":"km"})
    return corrdata

def correlate_nonlinear_stack(fft1_smoothed_abs,fft2,D,Nfft,dataS_t):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)
    PARAMETERS:
    ---------------------
    fft1_smoothed_abs: smoothed power spectral density of the FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    dataS_t: matrix of datetime object.
    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function
    '''
    #----load paramters----
    dt      = D['dt']
    maxlag  = D['maxlag']
    method  = D['cc_method']
    cc_len  = D['cc_len']
    substack= D['substack']
    stack_method  = D['stack_method']
    substack_len  = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    nwin  = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size,)*fft2.reshape(fft2.size,)

    # normalize by receiver spectral for coherency
    if method == "coherency":
        temp = utils.moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    # transform back to time domain waveforms
    s_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)   # stacked correlation
    ampmax = np.zeros(nwin,dtype=np.float32)
    n_corr = np.zeros(nwin,dtype=np.int16)                  # number of correlations for each substack
    t_corr = dataS_t                                        # timestamp
    crap   = np.zeros(Nfft,dtype=np.complex64)
    for i in range(nwin):
        n_corr[i]= 1
        crap[:Nfft2] = corr[i,:]
        crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
        crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
        crap[0]=complex(0,0)
        s_corr[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    ns_corr = s_corr
    for iii in range(ns_corr.shape[0]):
        ns_corr[iii] /= np.max(np.abs(ns_corr[iii]))

    if substack:
        if substack_len == cc_len:

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = dataS_t[-1]-dataS_t[0]             # total duration of what we have now
            tstart = dataS_t[0]

            nstack = int(np.round(Ttotal/substack_len))
            ampmax = np.zeros(nstack,dtype=np.float32)
            s_corr = np.zeros(shape=(nstack,Nfft),dtype=np.float32)
            n_corr = np.zeros(nstack,dtype=np.int)
            t_corr = np.zeros(nstack,dtype=np.float)
            crap   = np.zeros(Nfft,dtype=np.complex64)

            for istack in range(nstack):
                # find the indexes of all of the windows that start or end within
                itime = np.where( (dataS_t >= tstart) & (dataS_t < tstart+substack_len) )[0]
                if len(itime)==0:tstart+=substack_len;continue

                crap[:Nfft2] = np.mean(corr[itime,:],axis=0)   # linear average of the correlation
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[istack,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(itime)               # number of windows stacks
                t_corr[istack] = tstart                   # save the time stamps
                tstart += substack_len
                #print('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        if stack_method == 'linear':
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = np.mean(s_corr[tindx],axis=0)
            t_corr = dataS_t[0]
            n_corr = len(tindx)
        elif stack_method == 'robust':
            print('do robust substacking')
            s_corr = stack.robust_stack(s_corr,0.001)
            t_corr = dataS_t[0]
            n_corr = nwin
      #  elif stack_method == 'selective':
      #      print('do selective substacking')
      #      s_corr = selective_stack(s_corr,0.001)
      #      t_corr = dataS_t[0]
      #      n_corr = nwin

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2+1, Nfft2)*dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim==1:
        s_corr = s_corr[ind]
    elif s_corr.ndim==2:
        s_corr = s_corr[:,ind]
    return s_corr,t_corr,n_corr,ns_corr[:,ind]

def cc_parameters(cc_para,coor,tcorr,ncorr,comp):
    '''
    this function assembles the parameters for the cc function, which is used
    when writing them into ASDF files
    PARAMETERS:
    ---------------------
    cc_para: dict containing parameters used in the fft_cc step
    coor:    dict containing coordinates info of the source and receiver stations
    tcorr:   timestamp matrix
    ncorr:   matrix of number of good segments for each sub-stack/final stack
    comp:    2 character strings for the cross correlation component
    RETURNS:
    ------------------
    parameters: dict containing above info used for later stacking/plotting
    '''
    latS = coor['latS']
    lonS = coor['lonS']
    eles = 0.0
    if 'eleS' in list(coor.keys()):eleS=coor['eleS']
    latR = coor['latR']
    lonR = coor['lonR']
    eleR = 0.0
    if 'eleR' in list(coor.keys()):eleS=coor['eleR']
    dt        = cc_para['dt']
    maxlag    = cc_para['maxlag']
    substack  = cc_para['substack']
    cc_method = cc_para['cc_method']

    dist,azi,baz = obspy.geodetics.base.gps2dist_azimuth(latS,lonS,latR,lonR)
    parameters = {'dt':dt,
        'maxlag':int(maxlag),
        'dist':np.float32(dist/1000),
        'azi':np.float32(azi),
        'baz':np.float32(baz),
        'lonS':np.float32(lonS),
        'latS':np.float32(latS),
        'eleS':np.float32(eleS),
        'lonR':np.float32(lonR),
        'latR':np.float32(latR),
        'eleR':np.float32(eleR),
        'ngood':ncorr,
        'cc_method':cc_method,
        'time':tcorr,
        'substack':substack,
        'comp':comp}
    return parameters

def do_stacking(ccfiles,pairlist=None,outdir='./STACK',method=['linear'],
                rotation=False,correctionfile=None,flag=False,keep_substack=False,
                to_egf=False):
    # source folder
    if pairlist is None:
        pairlist,netsta_all=noise.get_stationpairs(ccfiles,False)
        if len(ccfiles)==0:
            raise IOError('Abort! no available CCF data for stacking')
        for s in netsta_all:
            tmp = os.path.join(outdir,s)
            if not os.path.isdir(tmp):os.mkdir(tmp)
    if isinstance(pairlist,str):pairlist=[pairlist]

    if not os.path.isdir(outdir):os.makedirs(outdir)
    if rotation:
        enz_system = ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']
        rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    for pair in pairlist:
        ttr   = pair.split('_')
        snet,ssta = ttr[0].split('.')
        rnet,rsta = ttr[1].split('.')
        idir  = ttr[0]

        # continue when file is done
        toutfn = os.path.join(outdir,idir+'/'+pair+'.tmp')
        if os.path.isfile(toutfn):continue
        if flag:print('assembling all corrdata ...')
        t0=time.time()
        corrdict_all=dict() #all components for the single station pair
        txtract=np.zeros(len(ccfiles),dtype=np.float32)
        tmerge=np.zeros(len(ccfiles),dtype=np.float32)
        tparameters=None
        for i,ifile in enumerate(ccfiles):
            # tt00=time.time()
            corrdict=extract_corrdata(ifile,pair=pair)
            # txtract[i]=time.time()-tt00
            if len(list(corrdict.keys()))>0:
                comp_list=list(corrdict[pair].keys())

                if len(comp_list)==0:
                    continue
                elif len(comp_list) >9:
                    print(comp_list)
                    raise ValueError('more than 9 cross-component exists for %s %s! please double check'%(ifile,pair))

                ### merge same component corrdata.
                # tt11=time.time()
                for c in comp_list:
                    #convert corrdata to empirical Green's functions by
                    #taking the negative time derivative. See types.CorrData.to_egf() for details.
                    if to_egf:
                        corrdict[pair][c].to_egf()

                    if tparameters is None:tparameters=corrdict[pair][c].misc
                    if c in list(corrdict_all.keys()):
                        corrdict_all[c].merge(corrdict[pair][c])
                    else:corrdict_all[c]=corrdict[pair][c]
                # tmerge[i]=time.time()-tt11
        #
        # if flag:print('extract time:'+str(np.sum(txtract)))
        # if flag:print('merge time:'+str(np.sum(tmerge)))
        t1=time.time()
        if flag:print('finished assembling in %6.2fs ...'%(t1-t0))
        #get length info from anyone of the corrdata, assuming all corrdata having the same length.
        cc_comp=list(corrdict_all.keys()) #final check on number of keys after merging all data.
        if len(cc_comp)==0:
            if flag:print('continue! no cross components for %s'%(pair))
            continue
        elif len(cc_comp)<9 and rotation:
            if flag:print('continue! not enough cross components for %s to do rotation'%(pair))
            continue
        elif len(cc_comp) >9:
            print(cc_comp)
            raise ValueError('more than 9 cross-component exists for %s! please double check'%(pair))

        #save data.
        outfn = pair+'.h5'
        if flag:print('ready to output to %s'%(outfn))

        t2=time.time()
        # loop through cross-component for stacking
        if isinstance(method,str):method=[method]
        tparameters['station_source']=ssta
        tparameters['station_receiver']=rsta
        if rotation: #need to order the components according to enz_system list.
            if corrdict_all[cc_comp[0]].substack:
                npts_segmt  = corrdict_all[cc_comp[0]].data.shape[1]
            else:
                npts_segmt  = corrdict_all[cc_comp[0]].data.shape[0]
            bigstack=np.zeros(shape=(9,npts_segmt),dtype=np.float32)
            if flag:print('applying stacking and rotation ...')
            stack_h5 = os.path.join(outdir,idir+'/'+outfn)
            ds=pyasdf.ASDFDataSet(stack_h5,mpi=False)
            #codes for ratation option.
            for m in method:
                data_type = 'Allstack_'+m
                bigstack=np.zeros(shape=(9,npts_segmt),dtype=np.float32)
                for icomp in range(9):
                    comp = enz_system[icomp]
                    indx = np.where(cc_comp==comp)[0]
                    # jump if there are not enough data
                    dstack,stamps_final=stacking(corrdict_all[cc_comp[indx[0]]],method=m)
                    bigstack[icomp]=dstack
                    tparameters['time']  = stamps_final[0]
                    tparameters['ngood'] = len(stamps_final)
                    ds.add_auxiliary_data(data=dstack, data_type=data_type, path=comp,
                                            parameters=tparameters)
                # start rotation
                if np.all(bigstack==0):continue

                bigstack_rotated = rotation(bigstack,tparameters,correctionfile,flag)

                # write to file
                data_type = 'Allstack_'+m
                for icomp2 in range(9):
                    rcomp  = rtz_components[icomp2]
                    if rcomp != 'ZZ':
                        ds.add_auxiliary_data(data=bigstack_rotated[icomp2], data_type=data_type,
                                                path=rcomp, parameters=tparameters)
            if keep_substack:
                for ic in cc_comp:
                    for ii in range(corrdict_all[ic].data.shape[0]):
                        tparameters2=tparameters
                        tparameters2['time']  = corrdict_all[ic].time[ii]
                        tparameters2['ngood'] = corrdict_all[ic].ngood[ii]
                        data_type = 'T'+str(int(corrdict_all[ic].time[ii]))
                        ds.add_auxiliary_data(data=corrdict_all[ic].data[ii], data_type=data_type,
                                            path=ic, parameters=tparameters2)

        else: #no need to care about the order of components.
            stack_h5 = os.path.join(outdir,idir+'/'+outfn)
            ds=pyasdf.ASDFDataSet(stack_h5,mpi=False)
            if flag:print('applying stacking ...')
            for ic in cc_comp:
                # write stacked data into ASDF file
                dstack,stamps_final=stacking(corrdict_all[ic],method=method)
                tparameters['time']  = stamps_final[0]
                tparameters['ngood'] = len(stamps_final)
                for i in range(len(method)):
                    m=method[i]
                    ds.add_auxiliary_data(data=dstack[i,:], data_type='Allstack_'+m, path=ic,
                                            parameters=tparameters)

                if keep_substack:
                    for ii in range(corrdict_all[ic].data.shape[0]):
                        tparameters2=tparameters
                        tparameters2['time']  = corrdict_all[ic].time[ii]
                        tparameters2['ngood'] = corrdict_all[ic].ngood[ii]
                        data_type = 'T'+str(int(corrdict_all[ic].time[ii]))
                        ds.add_auxiliary_data(data=corrdict_all[ic].data[ii], data_type=data_type,
                                            path=ic, parameters=tparameters2)
        #
        if flag: print('stacking and saving took %6.2fs'%(time.time()-t2))
        # write file stamps
        ftmp = open(toutfn,'w');ftmp.write('done');ftmp.close()

        del corrdict_all

####
def stacking(corrdata,method='linear'):
    '''
    this function stacks the cross correlation data

    PARAMETERS:
    ----------------------
    corrdata: CorrData object.
    method: stacking method, could be: linear, robust, pws, acf, or nroot.

    RETURNS:
    ----------------------
    dstack: 1D matrix of stacked cross-correlation functions over all the segments
    cc_time: timestamps of the traces for the stack
    '''
    if isinstance(method,str):method=[method]
    # remove abnormal data
    if corrdata.data.ndim==1:
        cc_time  = [corrdata.time]

        # do stacking
        dstack = np.zeros((len(method),corrdata.data.shape[0]),dtype=np.float32)
        for i in range(len(method)):
            m =method[i]
            dstack[i,:]=corrdata.data[:]
    else:
        ampmax = np.max(corrdata.data,axis=1)
        tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
        nstacks=len(tindx)
        dstack=[]
        cc_time=[]
        if nstacks >0:
            # remove ones with bad amplitude
            cc_array = corrdata.data[tindx,:]
            cc_time  = corrdata.time[tindx]

            # do stacking
            dstack = np.zeros((len(method),corrdata.data.shape[1]),dtype=np.float32)
            for i in range(len(method)):
                m =method[i]
                if nstacks==1: dstack[i,:]=cc_array
                else:
                    if m == 'linear':
                        dstack[i,:] = np.mean(cc_array,axis=0)
                    elif m == 'pws':
                        dstack[i,:] = stack.pws(cc_array,1.0/corrdata.dt)
                    elif m == 'robust':
                        dstack[i,:] = stack.robust_stack(cc_array)[0]
                    elif m == 'acf':
                        dstack[i,:] = stack.adaptive_filter(cc_array,1)
                    elif m == 'nroot':
                        dstack[i,:] = stack.nroot_stack(cc_array,2)

    # good to return
    return dstack,cc_time


def stacking_rma(cc_array,cc_time,cc_ngood,stack_para):
    '''
    this function stacks the cross correlation data according to the user-defined substack_len parameter
    PARAMETERS:
    ----------------------
    cc_array: 2D numpy float32 matrix containing all segmented cross-correlation data
    cc_time:  1D numpy array of timestamps for each segment of cc_array
    cc_ngood: 1D numpy int16 matrix showing the number of segments for each sub-stack and/or full stack
    stack_para: a dict containing all stacking parameters
    RETURNS:
    ----------------------
    cc_array, cc_ngood, cc_time: same to the input parameters but with abnormal cross-correaltions removed
    allstacks1: 1D matrix of stacked cross-correlation functions over all the segments
    nstacks:    number of overall segments for the final stacks
    '''
    # load useful parameters from dict
    samp_freq = stack_para['samp_freq']
    smethod   = stack_para['stack_method']
    rma_substack = stack_para['rma_substack']
    rma_step     = stack_para['rma_step']
    start_date   = stack_para['start_date']
    end_date     = stack_para['end_date']
    npts = cc_array.shape[1]

    # remove abnormal data
    ampmax = np.max(cc_array,axis=1)
    tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
    if not len(tindx):
        allstacks1=[];allstacks2=[];nstacks=0
        cc_array=[];cc_ngood=[];cc_time=[]
        return cc_array,cc_ngood,cc_time,allstacks1,allstacks2,nstacks
    else:

        # remove ones with bad amplitude
        cc_array = cc_array[tindx,:]
        cc_time  = cc_time[tindx]
        cc_ngood = cc_ngood[tindx]

        # do substacks
        if rma_substack:
            tstart = obspy.UTCDateTime(start_date)-obspy.UTCDateTime(1970,1,1)
            tend   = obspy.UTCDateTime(end_date)-obspy.UTCDateTime(1970,1,1)
            ttime  = tstart
            nstack = int(np.round((tend-tstart)/(rma_step*3600)))
            ncc_array = np.zeros(shape=(nstack,npts),dtype=np.float32)
            ncc_time  = np.zeros(nstack,dtype=np.float)
            ncc_ngood = np.zeros(nstack,dtype=np.int)

            # loop through each time
            for ii in range(nstack):
                sindx = np.where((cc_time>=ttime) & (cc_time<ttime+rma_substack*3600))[0]

                # when there are data in the time window
                if len(sindx):
                    ncc_array[ii] = np.mean(cc_array[sindx],axis=0)
                    ncc_time[ii]  = ttime
                    ncc_ngood[ii] = np.sum(cc_ngood[sindx],axis=0)
                ttime += rma_step*3600

            # remove bad ones
            tindx = np.where(ncc_ngood>0)[0]
            ncc_array = ncc_array[tindx]
            ncc_time  = ncc_time[tindx]
            ncc_ngood  = ncc_ngood[tindx]

        # do stacking
        allstacks1 = np.zeros(npts,dtype=np.float32)
        allstacks2 = np.zeros(npts,dtype=np.float32)
        allstacks3 = np.zeros(npts,dtype=np.float32)
        allstacks4 = np.zeros(npts,dtype=np.float32)

        if smethod == 'linear':
            allstacks1 = np.mean(cc_array,axis=0)
        elif smethod == 'pws':
            allstacks1 = stack.pws(cc_array,samp_freq)
        elif smethod == 'robust':
            allstacks1,w, = stack.robust_stack(cc_array,0.001)
        #elif smethod == 'selective':
        #    allstacks1 = selective_stack(cc_array,0.001)
        elif smethod == 'all':
            allstacks1 = np.mean(cc_array,axis=0)
            allstacks2 = stack.pws(cc_array,samp_freq)
            allstacks3 = stack.robust_stack(cc_array,0.001)
            allstacks4 = stack.selective_stack(cc_array,0.001)
        nstacks = np.sum(cc_ngood)

    # replace the array for substacks
    if rma_substack:
        cc_array = ncc_array
        cc_time  = ncc_time
        cc_ngood = ncc_ngood

    # good to return
    return cc_array,cc_ngood,cc_time,allstacks1,allstacks2,allstacks3,allstacks4,nstacks

def rotation(bigstack,parameters,locs,flag):
    '''
    this function transfers the Green's tensor from a E-N-Z system into a R-T-Z one

    PARAMETERS:
    -------------------
    bigstack:   9 component Green's tensor in E-N-Z system
    parameters: dict containing all parameters saved in ASDF file
    locs:       dict containing station angle info for correction purpose
    RETURNS:
    -------------------
    tcorr: 9 component Green's tensor in R-T-Z system
    '''
    # load parameter dic
    pi = np.pi
    azi = parameters['azi']
    baz = parameters['baz']
    ncomp,npts = bigstack.shape
    if ncomp<9:
        print('crap did not get enough components')
        tcorr=[]
        return tcorr
    staS  = parameters['station_source']
    staR  = parameters['station_receiver']

    if locs is not None:
        sta_list = list(locs['station'])
        angles   = list(locs['angle'])
        # get station info from the name of ASDF file
        ind   = sta_list.index(staS)
        acorr = angles[ind]
        ind   = sta_list.index(staR)
        bcorr = angles[ind]

        #---angles to be corrected----
        cosa = np.cos((azi+acorr)*pi/180)
        sina = np.sin((azi+acorr)*pi/180)
        cosb = np.cos((baz+bcorr)*pi/180)
        sinb = np.sin((baz+bcorr)*pi/180)
    else:
        cosa = np.cos(azi*pi/180)
        sina = np.sin(azi*pi/180)
        cosb = np.cos(baz*pi/180)
        sinb = np.sin(baz*pi/180)

    # rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    tcorr = np.zeros(shape=(9,npts),dtype=np.float32)
    tcorr[0] = -cosb*bigstack[7]-sinb*bigstack[6]
    tcorr[1] = sinb*bigstack[7]-cosb*bigstack[6]
    tcorr[2] = bigstack[8]
    tcorr[3] = -cosa*cosb*bigstack[4]-cosa*sinb*bigstack[3]-sina*cosb*bigstack[1]-sina*sinb*bigstack[0]
    tcorr[4] = cosa*sinb*bigstack[4]-cosa*cosb*bigstack[3]+sina*sinb*bigstack[1]-sina*cosb*bigstack[0]
    tcorr[5] = cosa*bigstack[5]+sina*bigstack[2]
    tcorr[6] = sina*cosb*bigstack[4]+sina*sinb*bigstack[3]-cosa*cosb*bigstack[1]-cosa*sinb*bigstack[0]
    tcorr[7] = -sina*sinb*bigstack[4]+sina*cosb*bigstack[3]+cosa*sinb*bigstack[1]-cosa*cosb*bigstack[0]
    tcorr[8] = -sina*bigstack[5]+cosa*bigstack[2]

    return tcorr

####
def merging(ccfiles,pairlist=None,outdir='./Merged',flag=False,to_egf=False):
    # source folder
    if pairlist is None:
        pairlist,netsta_all=noise.get_stationpairs(ccfiles,False)
        if len(ccfiles)==0:
            raise IOError('Abort! no available CCF data for stacking')
        for s in netsta_all:
            tmp = os.path.join(outdir,s)
            if not os.path.isdir(tmp):os.mkdir(tmp)
    if isinstance(pairlist,str):pairlist=[pairlist]

    if not os.path.isdir(outdir):os.makedirs(outdir)

    for pair in pairlist:
        ttr   = pair.split('_')
        snet,ssta = ttr[0].split('.')
        rnet,rsta = ttr[1].split('.')
        idir  = ttr[0]

        # continue when file is done
        ioutdir=os.path.join(outdir,idir)
        if not os.path.isdir(ioutdir):os.makedirs(ioutdir)
        toutfn = os.path.join(ioutdir,pair+'.tmp')
        if os.path.isfile(toutfn):continue
        if flag:print('assembling all corrdata ...')
        t0=time.time()
        corrdict_all=dict() #all components for the single station pair
        # txtract=np.zeros(len(ccfiles),dtype=np.float32)
        # tmerge=np.zeros(len(ccfiles),dtype=np.float32)
        tparameters=None
        for i,ifile in enumerate(ccfiles):
            # tt00=time.time()
            corrdict=extract_corrdata(ifile,pair=pair)
            # txtract[i]=time.time()-tt00
            if len(list(corrdict.keys()))>0:
                comp_list=list(corrdict[pair].keys())

                if len(comp_list)==0:
                    continue
                ### merge same component corrdata.
                # tt11=time.time()
                for c in comp_list:
                    if c in list(corrdict_all.keys()):
                        corrdict_all[c].merge(corrdict[pair][c])
                    else:corrdict_all[c]=corrdict[pair][c]
                # tmerge[i]=time.time()-tt11
        #
        # if flag:print('extract time:'+str(np.sum(txtract)))
        # if flag:print('merge time:'+str(np.sum(tmerge)))
        t1=time.time()
        if flag:print('finished assembling in %6.2fs ...'%(t1-t0))
        #get length info from anyone of the corrdata, assuming all corrdata having the same length.
        cc_comp=list(corrdict_all.keys()) #final check on number of keys after merging all data.
        if len(cc_comp)==0:
            if flag:print('continue! no cross components for %s'%(pair))
            continue

        #save data.
        outfn = pair+'.h5'
        if flag:print('save to %s'%(outfn))
        merged_h5 = os.path.join(ioutdir,outfn)
        for ic in cc_comp:
            #save components.
            #convert corrdata to empirical Green's functions by
            #taking the negative time derivative. See types.CorrData.to_egf() for details.
            if to_egf:
                corrdict_all[ic].to_egf()
            corrdict_all[ic].to_asdf(file=merged_h5)

        # write file stamps
        ftmp = open(toutfn,'w');ftmp.write('done');ftmp.close()

        del corrdict_all
########################################################
################ XCORR ANALYSIS FUNCTIONS ##################
########################################################
def save_xcorr_amplitudes(dict_in,filenamebase=None):
    """
    This function saves the amplitude data for both negative and positive lags,
    for each xcorr component pair, to csv files.

    PARAMETERS:
    ----------------------------
    dict_in: dictionary containing peak amplitude information from one virtual source to all other receivers.
            This can be the output of get_xcorr_peakamplitudes().
    filenamebase: file name base of the csv file, default is source_component_peakamp.txt in the current dir.
    """
    source=dict_in['source']['name']
    lonS0,latS0,eleS0=dict_in['source']['location']

    #
    if filenamebase is None:
        filenamebase = source

    cc_comp=list(dict_in['cc_comp'].keys())

    for ic in range(len(cc_comp)):
        comp = cc_comp[ic]
        receivers=list(dict_in['cc_comp'][comp].keys())
        lonS=lonS0*np.ones((len(receivers),))
        latS=latS0*np.ones((len(receivers),))
        eleS=eleS0*np.ones((len(receivers),))
        comp_out=len(receivers)*[comp]
        source_out=len(receivers)*[source]

        lonR=[]
        latR=[]
        eleR=[]
        dist=[]
        peakamp_neg=[]
        peakamp_pos=[]
        peaktt_neg=[]
        peaktt_pos=[]
        az=[]
        baz=[]

        for ir in range(len(receivers)):
            receiver=receivers[ir]
            dist0=dict_in['cc_comp'][comp][receiver]['dist']
            dist.append(dist0)
            lonR.append(dict_in['cc_comp'][comp][receiver]['location'][0])
            latR.append(dict_in['cc_comp'][comp][receiver]['location'][1])
            eleR.append(0.0)
            az.append(dict_in['cc_comp'][comp][receiver]['az'])
            baz.append(dict_in['cc_comp'][comp][receiver]['baz'])
            peakamp_neg.append(np.array(dict_in['cc_comp'][comp][receiver]['peak_amplitude'])[0])
            peakamp_pos.append(np.array(dict_in['cc_comp'][comp][receiver]['peak_amplitude'])[1])
            peaktt_neg.append(np.array(dict_in['cc_comp'][comp][receiver]['peak_amplitude_time'])[0])
            peaktt_pos.append(np.array(dict_in['cc_comp'][comp][receiver]['peak_amplitude_time'])[1])


        outDF=pd.DataFrame({'source':source_out,'lonS':lonS,'latS':latS,'eleS':eleS,
                           'receiver':receivers,'lonR':lonR,'latR':latR,'eleR':eleR,
                           'az':az,'baz':baz,'dist':dist,'peakamp_neg':peakamp_neg,
                            'peakamp_pos':peakamp_pos,'peaktt_neg':peaktt_neg,
                            'peaktt_pos':peaktt_pos,'comp':comp_out})
        fname=filenamebase+'_'+comp+'_peakamp.txt'
        outDF.to_csv(fname,index=False)
        print('data was saved to: '+fname)

def get_stationpairs(ccfiles,getcclist=True,flag=False):
    """
    Extract unique station pairs from all cc files in ASDF format.

    ====PARAMETERS===
    ccfiles: a list of cc files.

    ====RETURNS===
    pairs_all: all netstaion pairs in the format of NET1.STA1_NET2.STA2
    netsta_all: all net.sta (unique list)
    ccomp_all: all unique list of cc components.
    """
    if isinstance(ccfiles,str):ccfiles=[ccfiles]
    pairs_all = []
    ccomp_all=[]
    for f in ccfiles:
        # load the data from daily compilation
        ds=pyasdf.ASDFDataSet(f,mpi=False,mode='r')
        try:
            pairlist   = ds.auxiliary_data.list()
            if getcclist:
                for p in pairlist:
                    chanlist=ds.auxiliary_data[p].list()
                    for c in chanlist:
                        c1,c2=c.split('_')
                        ccomp_all.extend(c1[-1]+c2[-1])
                ccomp_all=sorted(set(ccomp_all))

            pairs_all.extend(pairlist)
            pairs_all=sorted(set(pairs_all))

        except Exception:
            if flag:print('continue! no data in %s'%(f))
            continue

    netsta_all=[]
    for p in pairs_all:
        netsta=p.split('_')
        netsta_all.extend(netsta)

    netsta_all=sorted(set(netsta_all))

    if getcclist:
        return pairs_all,netsta_all,ccomp_all
    else:
        return pairs_all,netsta_all

def extract_corrdata(sfile,pair=None,comp=['all']):
    '''
    extract the 2D matrix of the cross-correlation functions and the metadata for a certain time-chunck.
    PARAMETERS:
    --------------------------
    sfile: cross-correlation functions outputed by SeisGo cross-correlation workflow
    pair: net1.sta1-net2.sta2 pair to extract, default is to extract all pairs.
    comp: cross-correlation component or a list of components to extract, default is all components.

    RETURN:
    --------------------------
    corrdict: a dictionary that contains all extracted correlations, which each key as the station pair name.
                for each station pair, the correlaitons are saved as a list of CorrData objects.
    USAGE:
    --------------------------
    extract_corrdata('temp.h5',comp='ZZ')
    '''
    #check help or not at the very beginning

    # open data for read
    if isinstance(pair,str): pair=[pair]
    if isinstance(comp,str): comp=[comp]
    corrdict=dict()

    try:
        ds = pyasdf.ASDFDataSet(sfile,mpi=False,mode='r')
        # extract common variables
        spairs_all = ds.auxiliary_data.list()
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()
    if pair is None: pair=spairs_all

    for spair in list(set(pair) & set(spairs_all)):
        ttr = spair.split('_')
        snet,ssta = ttr[0].split('.')
        rnet,rsta = ttr[1].split('.')
        path_lists = ds.auxiliary_data[spair].list()
        corrdict[spair]=dict()
        for ipath in path_lists:
            schan,rchan = ipath.split('_')
            cc_comp=schan[-1]+rchan[-1]
            if cc_comp in comp or comp == ['all'] or comp ==['ALL']:
                try:
                    para=ds.auxiliary_data[spair][ipath].parameters
                    flag,ngood,ttime,dt,maxlag,az,baz,cc_method,dist,slat,slon,rlat,rlon = \
                                [para['substack'],para['ngood'],para['time'],\
                                para['dt'],para['maxlag'],para['azi'],para['baz'],\
                                para['cc_method'],para['dist'],para['latS'],para['lonS'],\
                                para['latR'],para['lonR']]
                    if "eleS" in  list(para.keys()):
                        sele = para['eleS']
                    else:
                        sele = 0.0
                    if "eleR" in  list(para.keys()):
                        rele = para['eleR']
                    else:
                        rele = 0.0
                    if "cc_len" in  list(para.keys()):
                        cc_len = para['cc_len']
                    else:
                        cc_len = None
                    if "cc_step" in  list(para.keys()):
                        cc_step = para['cc_step']
                    else:
                        cc_step = None
                    if flag:
                        data = ds.auxiliary_data[spair][ipath].data[:,:]
                    else:
                        data = ds.auxiliary_data[spair][ipath].data[:]
                except Exception:
                    print('continue! something wrong with %s %s'%(spair,ipath))
                    continue
                corrdict[spair][cc_comp]=CorrData(net=[snet,rnet],sta=[ssta,rsta],loc=['',''],\
                                                chan=[schan,rchan],lon=[slon,rlon],lat=[slat,rlat],
                                                ele=[sele,rele],cc_comp=cc_comp,dt=dt,lag=maxlag,
                                                cc_len=cc_len,cc_step=cc_step,dist=dist,az=az,
                                                baz=baz,ngood=ngood,time=ttime,data=data,
                                                substack=flag,misc=para)
                if "type" in  list(para.keys()): corrdict[spair][cc_comp].type=para['type']

    return corrdict

def save_corrfile_to_sac(cfile,rootdir='.',pair=None,comp=['all'],v=True):
    """
    Save correlation files in ASDF to sac files.

    === PARAMETERS ===
    cfile: correlation file from SeisGo workflow. It could be a list of files.
    rootdir: folder to save the converted sac files. this is the root folder, not
            the folder for individual sources/receivers, which will be created
            by this function. Default is the current directory.
    pair: net1.sta1_net2.sta2 pair to extract, default is to extract all pairs.
    comp: cross-correlation component or a list of components to extract, default is 'all'.
    v: verbose or not, default is True.
    """
    if isinstance(cfile,str):cfile=[cfile]
    if isinstance(pair,str): pair=[pair]

    nfile=len(cfile)

    for cf in cfile:
        if v: print('working on file: '+cf.split('/')[-1])

        corrdict=extract_corrdata(cf)
        pairs_all=list(corrdict.keys())
        if pair is None:
            extract_pair=pairs_all
        else:
            extract_pair=pair

        for p in extract_pair:
            if p in pairs_all:
                netsta1,netsta2=p.split('_')
                outdir=os.path.join(rootdir,netsta1,netsta2)

                comp_all=list(corrdict[p].keys())
                for c in comp_all:
                    if c in comp or comp == ['all'] or comp ==['ALL']:
                        corrdict[p][c].to_sac(outdir=outdir)
            else:
                print('Pair %s not found. Skip.'%(p))
                continue
########################################################
################ MONITORING FUNCTIONS ##################
########################################################

'''
a compilation of all available core functions for computing phase delays based on ambient noise interferometry

quick index of dv/v methods:
1) stretching (time stretching; Weaver et al (2011))
2) dtw_dvv (Dynamic Time Warping; Mikesell et al. 2015)
3) mwcs_dvv (Moving Window Cross Spectrum; Clark et al., 2011)
4) mwcc_dvv (Moving Window Cross Correlation; Snieder et al., 2012)
5) wts_dvv (Wavelet Streching; Yuan et al., in prep)
6) wxs_dvv (Wavelet Xross Spectrum; Mao et al., 2019)
7) wdw_dvv (Wavelet Dynamic Warping; Yuan et al., in prep)
'''

def stretching(ref, cur, dv_range, nbtrial, para):

    """
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.

    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    dv_range: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change ('float')
    nbtrial: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
    para: vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
    For error computation, we need parameters:
        fmin: minimum frequency of the data
        fmax: maximum frequency of the data
        tmin: minimum time window where the dv/v is computed
        tmax: maximum time window where the dv/v is computed
    RETURNS:
    ----------------
    dv: Relative velocity change dv/v (in %)
    cc: correlation coefficient between the reference waveform and the best stretched/compressed current waveform
    cdp: correlation coefficient between the reference waveform and the initial current waveform
    error: Errors in the dv/v measurements based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)

    Note: The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values.
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .

    Originally by L. Viens 04/26/2018 (Viens et al., 2018 JGR)
    modified by Chengxin Jiang
    """
    # load common variables from dictionary
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin,tmax,dt)

    # make useful one for measurements
    dvmin = -np.abs(dv_range)
    dvmax = np.abs(dv_range)
    Eps = 1+(np.linspace(dvmin, dvmax, nbtrial))
    cof = np.zeros(Eps.shape,dtype=np.float32)

    # Set of stretched/compressed current waveforms
    for ii in range(len(Eps)):
        nt = tvec*Eps[ii]
        s = np.interp(x=tvec, xp=nt, fp=cur)
        waveform_ref = ref
        waveform_cur = s
        cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cdp = np.corrcoef(cur, ref)[0, 1] # correlation coefficient between the reference and initial current waveforms

    # find the maximum correlation coefficient
    imax = np.nanargmax(cof)
    if imax >= len(Eps)-2:
        imax = imax - 2
    if imax <= 2:
        imax = imax + 2

    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = np.linspace(Eps[imax-2], Eps[imax+2], 100)
    ncof    = np.zeros(dtfiner.shape,dtype=np.float32)
    for ii in range(len(dtfiner)):
        nt = tvec*dtfiner[ii]
        s = np.interp(x=tvec, xp=nt, fp=cur)
        waveform_ref = ref
        waveform_cur = s
        ncof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cc = np.max(ncof) # Find maximum correlation coefficient of the refined  analysis
    dv = 100. * dtfiner[np.argmax(ncof)]-100 # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

    # Error computation based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, error, cc, cdp


def dtw_dvv(ref, cur, para, maxLag, b, direction):
    """
    Dynamic time warping for dv/v estimation.

    PARAMETERS:
    ----------------
    ref : reference signal (np.array, size N)
    cur : current signal (np.array, size N)
    para: dict containing useful parameters about the data window and targeted frequency
    maxLag : max number of points to search forward and backward.
            Suggest setting it larger if window is set larger.
    b : b-value to limit strain, which is to limit the maximum velocity perturbation.
            See equation 11 in (Mikesell et al. 2015)
    direction: direction to accumulate errors (1=forward, -1=backward)
    RETURNS:
    ------------------
    -m0 : estimated dv/v
    em0 : error of dv/v estimation

    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)
    """
    twin = para['twin']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    tvect = np.arange(tmin,tmax,dt)

    # setup other parameters
    npts = len(ref) # number of time samples

    # compute error function over lags, which is independent of strain limit 'b'.
    err = computeErrorFunction( cur, ref, npts, maxLag )

    # direction to accumulate errors (1=forward, -1=backward)
    dist  = accumulateErrorFunction( direction, err, npts, maxLag, b )
    stbar = backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )
    stbarTime = stbar * dt   # convert from samples to time

    # cut the first and last 5% for better regression
    indx = np.where((tvect>=0.05*npts*dt) & (tvect<=0.95*npts*dt))[0]

    # linear regression to get dv/v
    if npts >2:

        # weights
        w = np.ones(npts)
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(tvect.flatten()[indx], stbarTime.flatten()[indx], w.flatten()[indx], intercept_origin=True)

    else:
        print('not enough points to estimate dv/v for dtw')
        m0=0;em0=0

    return m0*100,em0*100,dist


def mwcs_dvv(ref, cur, moving_window_length, slide_step, para, smoothing_half_win=5):
    """
    Moving Window Cross Spectrum method to measure dv/v (relying on phi=2*pi*f*t in freq domain)

    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    moving_window_length: moving window length to calculate cross-spectrum (np.float, in sec)
    slide_step: steps in time to shift the moving window (np.float, in seconds)
    para: a dict containing parameters about input data window and frequency info, including
        delta->The sampling rate of the input timeseries (in Hz)
        window-> The target window for measuring dt/t
        freq-> The frequency bound to compute the dephasing (in Hz)
        tmin: The leftmost time lag (used to compute the "time lags array")
    smoothing_half_win: If different from 0, defines the half length of the smoothing hanning window.

    RETURNS:
    ------------------
    time_axis: the central times of the windows.
    delta_t: dt
    delta_err:error
    delta_mcoh: mean coherence

    Copied from MSNoise (https://github.com/ROBelgium/MSNoise/tree/master/msnoise)
    Modified by Chengxin Jiang
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvect = np.arange(tmin,tmax,dt)

    # parameter initialize
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    # info on the moving window
    window_length_samples = np.int(moving_window_length/dt)
    padd = int(2 ** (utils.nextpow2(window_length_samples) + 2))
    count = 0
    tp = cosine_taper(window_length_samples, 0.15)

    minind = 0
    maxind = window_length_samples

    # loop through all sub-windows
    while maxind <= len(ref):
        cci = cur[minind:maxind]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[minind:maxind]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step/dt)
        maxind += int(slide_step/dt)

        # do fft
        fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
        fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # get cross-spectrum & do filtering
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
            dcur = np.sqrt(smooth2(fcur2, window='hanning',half_win=smoothing_half_win))
            dref = np.sqrt(smooth2(fref2, window='hanning',half_win=smoothing_half_win))
            X = smooth2(X, window='hanning',half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, dt)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= fmin,freq_vec <= fmax))

        # Get Coherence and its mean value
        coh = getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[index_range])

        # Get Weights
        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        # Frequency array:
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin; weights for the WLS must be the variance !
        m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())
        delta_t.append(m)

        # print phi.shape, v.shape, w.shape
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2)

        delta_err.append(e)
        delta_mcoh.append(np.real(mcoh))
        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(cur) + int(slide_step/dt):
        print("The last window was too small, but was computed")

    # ensure all matrix are np array
    delta_t = np.array(delta_t)
    delta_err = np.array(delta_err)
    delta_mcoh = np.array(delta_mcoh)
    time_axis  = np.array(time_axis)

    # ready for linear regression
    delta_mincho = 0.65
    delta_maxerr = 0.1
    delta_maxdt  = 0.1
    indx1 = np.where(delta_mcoh>delta_mincho)
    indx2 = np.where(delta_err<delta_maxerr)
    indx3 = np.where(delta_t<delta_maxdt)

    #-----find good dt measurements-----
    indx = np.intersect1d(indx1,indx2)
    indx = np.intersect1d(indx,indx3)

    if len(indx) >2:

        #----estimate weight for regression----
        w = 1/delta_err[indx]
        w[~np.isfinite(w)] = 1.0

        #---------do linear regression-----------
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=True)

    else:
        print('not enough points to estimate dv/v for mwcs')
        m0=0;em0=0

    return -m0*100,em0*100


def WCC_dvv(ref, cur, moving_window_length, slide_step, para):
    """
    Windowed cross correlation (WCC) for dt or dv/v mesurement (Snieder et al. 2012)

    Parameters:
    -----------
    ref: The "Reference" timeseries
    cur: The "Current" timeseries
    moving_window_length: The moving window length (in seconds)
    slide_step: The step to jump for the moving window (in seconds)
    para: a dict containing freq/time info of the data matrix

    Returns:
    ------------
    time_axis: central times of the moving window
    delta_t: dt
    delta_err: error
    delta_mcoh: mean coherence for each window

    Written by Congcong Yuan (1 July, 2019)
    """
    # common variables
    twin = para['twin']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)

    # parameter initialize
    delta_t = []
    delta_t_coef = []
    time_axis = []

    # info on the moving window
    window_length_samples = np.int(moving_window_length/dt)
    count = 0
    tp = cosine_taper(window_length_samples, 0.15)

    minind = 0
    maxind = window_length_samples

    # loop through all sub-windows
    while maxind <= len(ref):
        cci = cur[minind:maxind]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[minind:maxind]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step/dt)
        maxind += int(slide_step/dt)

        # normalize signals before cross correlation
        cci = (cci - cci.mean()) / cci.std()
        cri = (cri - cri.mean()) / cri.std()

        # get maximum correlation coefficient and its index
        cc2 = np.correlate(cci, cri, mode='same')
        cc2 = cc2 / np.sqrt((cci**2).sum() * (cri**2).sum())

        imaxcc2 = np.where(cc2==np.max(cc2))[0]
        maxcc2 = np.max(cc2)

        # get the time shift
        m = (imaxcc2-((maxind-minind)//2))*dt
        delta_t.append(m)
        delta_t_coef.append(maxcc2)

        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

    del cci, cri, cc2, imaxcc2, maxcc2
    del m

    if maxind > len(cur) + int(slide_step/dt):
        print("The last window was too small, but was computed")

    delta_t = np.array(delta_t)
    delta_t_coef = np.array(delta_t_coef)
    time_axis  = np.array(time_axis)

    # linear regression to get dv/v
    if count >2:
        # simple weight
        w = np.ones(count)
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis.flatten(), delta_t.flatten(), w.flatten(),intercept_origin=True)

    else:
        print('not enough points to estimate dv/v for wcc')
        m0=0;em0=0

    return -m0*100,em0*100


def wxs_dvv(ref,cur,allfreq,para,dj=1/12, s0=-1, J=-1, sig=False, wvn='morlet',unwrapflag=False):
    """
    Compute dt or dv/v in time and frequency domain from wavelet cross spectrum (wxs).
    for all frequecies in an interest range

    Parameters
    --------------
    ref: The "Reference" timeseries (numpy.ndarray)
    cur: The "Current" timeseries (numpy.ndarray)
    allfreq: a boolen variable to make measurements on all frequency range or not
    para: a dict containing freq/time info of the data matrix
    dj, s0, J, sig, wvn: common parameters used in 'wavelet.wct'
    unwrapflag: True - unwrap phase delays. Default is False

    RETURNS:
    ------------------
    dvv*100 : estimated dv/v in %
    err*100 : error of dv/v estimation in %

    Originally written by Tim Clements (1 March, 2019)
    Modified by Congcong Yuan (30 June, 2019) based on (Mao et al. 2019).
    Updated by Chengxin Jiang (10 Oct, 2019) to merge the functionality for mesurements across all frequency and one freq range
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin,tmax,dt)
    npts = len(tvec)

    # perform cross coherent analysis, modified from function 'wavelet.cwt'
    WCT, aWCT, coi, freq, sig = wct_modified(ref, cur, dt, dj=dj, s0=s0, J=J, sig=sig, wavelet=wvn, normalize=True)

    if unwrapflag:
        phase = np.unwrap(aWCT,axis=-1) # axis=0, upwrap along time; axis=-1, unwrap along frequency
    else:
        phase = aWCT

    # zero out data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]

    # follow MWCS to do two steps of linear regression
    if not allfreq:

        delta_t_m, delta_t_unc = np.zeros(npts,dtype=np.float32),np.zeros(npts,dtype=np.float32)
        # assume the tvec is the time window to measure dt
        for it in range(npts):
            w = 1/WCT[freq_indin,it]
            w[~np.isfinite(w)] = 1.
            delta_t_m[it],delta_t_unc[it] = linear_regression(freq[freq_indin]*2*np.pi, phase[freq_indin,it], w)

        # new weights for regression
        w2 = 1/np.mean(WCT[freq_indin,:],axis=0)
        w2[~np.isfinite(w2)] = 1.

        # now use dt and t to get dv/v
        if len(w2)>2:
            if not np.any(delta_t_m):
                dvv, err = np.nan,np.nan
            m, em = linear_regression(tvec, delta_t_m, w2, intercept_origin=True)
            dvv, err = -m, em
        else:
            print('not enough points to estimate dv/v for wts')
            dvv, err=np.nan, np.nan

        return dvv*100,err*100

    # convert phase directly to delta_t for all frequencies
    else:

        # convert phase delay to time delay
        delta_t = phase / (2*np.pi*freq[:,None]) # normalize phase by (2*pi*frequency)
        dvv, err = np.zeros(freq_indin.shape), np.zeros(freq_indin.shape)

        # loop through freq for linear regression
        for ii, ifreq in enumerate(freq_indin):
            if len(tvec)>2:
                if not np.any(delta_t[ifreq]):
                    continue

                # how to better approach the uncertainty of delta_t
                w = 1/WCT[ifreq]
                w[~np.isfinite(w)] = 1.0

                #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
                m, em = linear_regression(tvec, delta_t[ifreq], w, intercept_origin=True)
                dvv[ii], err[ii] = -m, em
            else:
                print('not enough points to estimate dv/v for wts')
                dvv[ii], err[ii]=np.nan, np.nan

        return freq[freq_indin], dvv*100, err*100


def wts_dvv(ref,cur,allfreq,para,dv_range,nbtrial,dj=1/12,s0=-1,J=-1,wvn='morlet',normalize=True):
    """
    Apply stretching method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range

    Parameters
    --------------
    ref: The "Reference" timeseries (numpy.ndarray)
    cur: The "Current" timeseries (numpy.ndarray)
    allfreq: a boolen variable to make measurements on all frequency range or not
    para: a dict containing freq/time info of the data matrix
    dv_range: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change (float)
    nbtrial: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  (float)
    dj, s0, J, sig, wvn: common parameters used in 'wavelet.wct'
    normalize: normalize the wavelet spectrum or not. Default is True

    RETURNS:
    ------------------
    dvv: estimated dv/v
    err: error of dv/v estimation

    Written by Congcong Yuan (30 Jun, 2019)
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin,tmax,dt)

    # apply cwt on two traces
    cwt1, sj, freq, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)

    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)

    # zero out data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]

    # convert wavelet domain back to time domain (~filtering)
    if not allfreq:

        # inverse cwt to time domain
        icwt1 = pycwt.icwt(cwt1[freq_indin], sj[freq_indin], dt, dj, wvn)
        icwt2 = pycwt.icwt(cwt2[freq_indin], sj[freq_indin], dt, dj, wvn)

        # assume all time window is used
        wcwt1, wcwt2 = np.real(icwt1), np.real(icwt2)

        # Normalizes both signals, if appropriate.
        if normalize:
            ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
            ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
        else:
            ncwt1 = wcwt1
            ncwt2 = wcwt2

        # run stretching
        dvv, err, cc, cdp = stretching(ncwt2, ncwt1, dv_range, nbtrial, para)
        return dvv, err

    # directly take advantage of the
    else:
        # initialize variable
        nfreq=len(freq_indin)
        dvv, cc, cdp, err = np.zeros(nfreq,dtype=np.float32), np.zeros(nfreq,dtype=np.float32),\
            np.zeros(nfreq,dtype=np.float32),np.zeros(nfreq,dtype=np.float32)

        # loop through each freq
        for ii, ifreq in enumerate(freq_indin):

            # prepare windowed data
            wcwt1, wcwt2 = rcwt1[ifreq], rcwt2[ifreq]

            # Normalizes both signals, if appropriate.
            if normalize:
                ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
                ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
            else:
                ncwt1 = wcwt1
                ncwt2 = wcwt2

            # run stretching
            dv, error, c1, c2 = stretching(ncwt2, ncwt1, dv_range, nbtrial, para)
            dvv[ii], cc[ii], cdp[ii], err[ii]=dv, c1, c2, error

        return freq[freq_indin], dvv, err


def wtdtw_allfreq(ref,cur,allfreq,para,maxLag,b,direction,dj=1/12,s0=-1,J=-1,wvn='morlet',normalize=True):
    """
    Apply dynamic time warping method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range

    Parameters
    --------------
    ref: The "Reference" timeseries (numpy.ndarray)
    cur: The "Current" timeseries (numpy.ndarray)
    allfreq: a boolen variable to make measurements on all frequency range or not
    maxLag: max number of points to search forward and backward.
    b: b-value to limit strain, which is to limit the maximum velocity perturbation. See equation 11 in (Mikesell et al. 2015)
    direction: direction to accumulate errors (1=forward, -1=backward)
    dj, s0, J, sig, wvn: common parameters used in 'wavelet.wct'
    normalize: normalize the wavelet spectrum or not. Default is True

    RETURNS:
    ------------------
    dvv: estimated dv/v
    err: error of dv/v estimation

    Written by Congcong Yuan (30 Jun, 2019)
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin,tmax)*dt

    # apply cwt on two traces
    cwt1, sj, freq, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)

    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)

    # zero out cone of influence and data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]

        # Use DTW method to extract dvv
        nfreq=len(freq_indin)
        dvv, err = np.zeros(nfreq,dtype=np.float32), np.zeros(nfreq,dtype=np.float32)

        for ii,ifreq in enumerate(freq_indin):

            # prepare windowed data
            wcwt1, wcwt2 = rcwt1[ifreq], rcwt2[ifreq]
            # Normalizes both signals, if appropriate.
            if normalize:
                ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
                ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
            else:
                ncwt1 = wcwt1
                ncwt2 = wcwt2

            # run dtw
            dv, error, dist  = dtw_dvv(ncwt2, ncwt1, para, maxLag, b, direction)
            dvv[ii], err[ii] = dv, error

    del cwt1, cwt2, rcwt1, rcwt2, ncwt1, ncwt2, wcwt1, wcwt2, coi, sj, dist

    if not allfreq:
        return np.mean(dvv),np.mean(err)
    else:
        return freq[freq_indin], dvv, err





#############################################################
################ MONITORING UTILITY FUNCTIONS ###############
#############################################################

'''
below are assembly of the monitoring utility functions called by monitoring functions
'''

def smooth2(x, window='boxcar', half_win=3):
    """
    performs smoothing in interested time window

    Parameters
    --------------
    x: timeseris data
    window: types of window to do smoothing
    half_win: half window length

    RETURNS:
    ------------------
    y: smoothed time window
    """
    # TODO: docsting
    window_len = 2 * half_win + 1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype('complex')
    else:
        w = scipy.signal.hanning(window_len).astype('complex')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[half_win:len(y) - half_win]


def getCoherence(dcs, ds1, ds2):
    """
    get cross coherence between reference and current waveforms following equation of A3 in Clark et al., 2011

    Parameters
    --------------
    dcs: amplitude of the cross spectrum
    ds1: amplitude of the spectrum of current waveform
    ds2: amplitude of the spectrum of reference waveform

    RETURNS:
    ------------------
    coh: cohrerency matrix used for estimate the robustness of the cross spectrum
    """
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2) > 0))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh


def computeErrorFunction(u1, u0, nSample, lag, norm='L2'):
    """
    compute Error Function used in DTW. The error function is equation 1 in Hale, 2013. You could uncomment the
    L1 norm and comment the L2 norm if you want on Line 29

    Parameters
    --------------
    u1:  trace that we want to warp; size = (nsamp,1)
    u0:  reference trace to compare with: size = (nsamp,1)
    nSample: numer of points to compare in the traces
    lag: maximum lag in sample number to search
    norm: 'L2' or 'L1' (default is 'L2')

    RETURNS:
    ------------------
    err: the 2D error function; size = (nsamp,2*lag+1)

    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)

    """

    if lag >= nSample:
        raise ValueError('computeErrorFunction:lagProblem','lag must be smaller than nSample')

    # Allocate error function variable
    err = np.zeros([nSample, 2 * lag + 1])

    # initial error calculation
    # loop over lags
    for ll in np.arange(-lag,lag + 1):
        thisLag = ll + lag

        # loop over samples
        for ii in range(nSample):

            # skip corners for now, we will come back to these
            if (ii + ll >= 0) & (ii + ll < nSample):
                err[ii,thisLag] = u1[ii] - u0[ii + ll]

    if norm == 'L2':
        err = err**2
    elif norm == 'L1':
        err = np.abs(err)

    # Now fix corners with constant extrapolation
    for ll in np.arange(-lag,lag + 1):
        thisLag = ll + lag

        for ii in range(nSample):
            if ii + ll < 0:
                err[ii, thisLag] = err[-ll, thisLag]

            elif ii + ll > nSample - 1:
                err[ii,thisLag] = err[nSample - ll - 1,thisLag]

    return err


def accumulateErrorFunction(dir, err, nSample, lag, b ):
    """
    accumulation of the error, which follows the equation 6 in Hale, 2013.

    Parameters
    --------------
    dir: accumulation direction ( dir > 0 = forward in time, dir <= 0 = backward in time)
    err: the 2D error function; size = (nsamp,2*lag+1)
    nSample: numer of points to compare in the traces
    lag: maximum lag in sample number to search
    b: strain limit (integer value >= 1)

    RETURNS:
    ------------------
    d: the 2D distance function; size = (nsamp,2*lag+1)

    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)

    """

    # number of lags from [ -lag : +lag ]
    nLag = ( 2 * lag ) + 1

    # allocate distance matrix
    d = np.zeros([nSample, nLag])

    # Setup indices based on forward or backward accumulation direction
    if dir > 0: # FORWARD
        iBegin, iEnd, iInc = 0, nSample - 1, 1
    else: # BACKWARD
        iBegin, iEnd, iInc = nSample - 1, 0, -1

    # Loop through all times ii in forward or backward direction
    for ii in range(iBegin,iEnd + iInc,iInc):

        # min/max to account for the edges/boundaries
        ji = max([0, min([nSample - 1, ii - iInc])])
        jb = max([0, min([nSample - 1, ii - iInc * b])])

        # loop through all lag
        for ll in range(nLag):

            # check limits on lag indices
            lMinus1 = ll - 1

            # check lag index is greater than 0
            if lMinus1 < 0:
                lMinus1 = 0 # make lag = first lag

            lPlus1 = ll + 1 # lag at l+1

            # check lag index less than max lag
            if lPlus1 > nLag - 1:
                lPlus1 = nLag - 1

            # get distance at lags (ll-1, ll, ll+1)
            distLminus1 = d[jb, lMinus1] # minus:  d[i-b, j-1]
            distL = d[ji,ll] # actual d[i-1, j]
            distLplus1 = d[jb, lPlus1] # plus d[i-b, j+1]

            if ji != jb: # equation 10 in Hale, 2013
                for kb in range(ji,jb + iInc - 1, -iInc):
                    distLminus1 = distLminus1 + err[kb, lMinus1]
                    distLplus1 = distLplus1 + err[kb, lPlus1]

            # equation 6 (if b=1) or 10 (if b>1) in Hale (2013) after treating boundaries
            d[ii, ll] = err[ii,ll] + min([distLminus1, distL, distLplus1])

    return d


def backtrackDistanceFunction(dir, d, err, lmin, b):
    """
    The function is equation 2 in Hale, 2013.

    Parameters
    --------------
    dir: side to start minimization ( dir > 0 = front, dir <= 0 =  back)
    d : the 2D distance function; size = (nsamp,2*lag+1)
    err: the 2D error function; size = (nsamp,2*lag+1)
    lmin: minimum lag to search over
    b : strain limit (integer value >= 1)

    RETURNS:
    ------------------
    stbar: vector of integer shifts subject to |u(i)-u(i-1)| <= 1/b

    Original by Di Yang
    Last modified by Dylan Mikesell (19 Dec. 2014)

    Translated to python by Tim Clements (17 Aug. 2018)

    """

    nSample, nLag = d.shape
    stbar = np.zeros(nSample)

    # Setup indices based on forward or backward accumulation direction
    if dir > 0: # FORWARD
        iBegin, iEnd, iInc = 0, nSample - 1, 1
    else: # BACKWARD
        iBegin, iEnd, iInc = nSample - 1, 0, -1

    # start from the end (front or back)
    ll = np.argmin(d[iBegin,:]) # find minimum accumulated distance at front or back depending on 'dir'
    stbar[iBegin] = ll + lmin # absolute value of integer shift

    # move through all time samples in forward or backward direction
    ii = iBegin

    while ii != iEnd:

        # min/max for edges/boundaries
        ji = np.max([0, np.min([nSample - 1, ii + iInc])])
        jb = np.max([0, np.min([nSample - 1, ii + iInc * b])])

        # check limits on lag indices
        lMinus1 = ll - 1

        if lMinus1 < 0: # check lag index is greater than 1
            lMinus1 = 0 # make lag = first lag

        lPlus1 = ll + 1

        if lPlus1 > nLag - 1: # check lag index less than max lag
            lPlus1 = nLag - 1

        # get distance at lags (ll-1, ll, ll+1)
        distLminus1 = d[jb, lMinus1] # minus:  d[i-b, j-1]
        distL = d[ji,ll] # actual d[i-1, j]
        distLplus1 = d[jb, lPlus1] # plus d[i-b, j+1]

        # equation 10 in Hale (2013)
        # sum errors over i-1:i-b+1
        if ji != jb:
            for kb in range(ji, jb - iInc - 1, iInc):
                distLminus1 = distLminus1 + err[kb, lMinus1]
                distLplus1  = distLplus1  + err[kb, lPlus1]

        # update minimum distance to previous sample
        dl = np.min([distLminus1, distL, distLplus1 ])

        if dl != distL: # then ll ~= ll and we check forward and backward
            if dl == distLminus1:
                ll = lMinus1
            else:
                ll = lPlus1

        # assume ii = ii - 1
        ii += iInc

        # absolute integer of lag
        stbar[ii] = ll + lmin

        # now move to correct time index, if smoothing difference over many
        # time samples using 'b'
        if (ll == lMinus1) | (ll == lPlus1): # check edges to see about b values
            if ji != jb: # if b>1 then need to move more steps
                for kb in range(ji, jb - iInc - 1, iInc):
                    ii = ii + iInc # move from i-1:i-b-1
                    stbar[ii] = ll + lmin  # constant lag over that time

    return stbar


def wct_modified(y1, y2, dt, dj=1/12, s0=-1, J=-1, sig=True, significance_level=0.95, wavelet='morlet', normalize=True, **kwargs):
    '''
    Wavelet coherence transform (WCT).
​
    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.
​
    Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
    significance_level (float, optional) :
        Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.
​
    Returns
    -------
    TODO: Something TBA and TBC
​
    See also
    --------
    cwt, xwt
​
    '''

    wavelet = pycwt.wavelet._check_parameter_wavelet(wavelet)
    # Checking some input parameters
    if s0 == -1:
        # Number of scales
        s0 = 2 * dt / wavelet.flambda()
    if J == -1:
        # Number of scales
        J = np.int(np.round(np.log2(y1.size * dt / s0) / dj))

    # Makes sure input signals are numpy arrays.
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    # Calculates the standard deviation of both input signals.
    std1 = y1.std()
    std2 = y2.std()
    # Normalizes both signals, if appropriate.
    if normalize:
        y1_normal = (y1 - y1.mean()) / std1
        y2_normal = (y2 - y2.mean()) / std2
    else:
        y1_normal = y1
        y2_normal = y2

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = pycwt.cwt(y1_normal, dt, **_kwargs)
    W2, sj, freq, coi, _, _ = pycwt.cwt(y2_normal, dt, **_kwargs)

    scales1 = np.ones([1, y1.size]) * sj[:, None]
    scales2 = np.ones([1, y2.size]) * sj[:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = wavelet.smooth(np.abs(W1) ** 2 / scales1, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(W2) ** 2 / scales2, dt, dj, sj)

    # Now the wavelet transform coherence
    W12 = W1 * W2.conj()
    scales = np.ones([1, y1.size]) * sj[:, None]
    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
    WCT = np.abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(W12)

    # Calculate cross spectrum & its amplitude
    WXS, WXA = W12, np.abs(S12)

    # Calculates the significance using Monte Carlo simulations with 95%
    # confidence as a function of scale.

    if sig:
        a1, b1, c1 = pycwt.ar1(y1)
        a2, b2, c2 = pycwt.ar1(y2)
        sig = pycwt.wct_significance(a1, a2, dt=dt, dj=dj, s0=s0, J=J,
                               significance_level=significance_level,
                               wavelet=wavelet, **kwargs)
    else:
        sig = np.asarray([0])

    return WCT, aWCT, coi, freq, sig


#####################################################
#########BACK UP FUNCTIONS , NOT CURRENTLY USED######
#####################################################
def correlate_bkp(fft1,fft2,D,Nfft,dataS_t):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)
    PARAMETERS:
    ---------------------
    fft1: FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    dataS_t: matrix of datetime object.
    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function
    '''
    #----load paramters----
    dt      = D['dt']
    maxlag  = D['maxlag']
    method  = D['cc_method']
    cc_len  = D['cc_len']
    substack= D['substack']
    substack_len  = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    fft1_smoothed_abs=np.conj(fft1)
    nwin  = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size,)*fft2.reshape(fft2.size,)

    if method == "coherency":
        temp = utils.moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    if substack:
        if substack_len == cc_len:
            # choose to keep all fft data for a day
            s_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)   # stacked correlation
            ampmax = np.zeros(nwin,dtype=np.float32)
            n_corr = np.zeros(nwin,dtype=np.int16)                  # number of correlations for each substack
            t_corr = dataS_t                                        # timestamp
            crap   = np.zeros(Nfft,dtype=np.complex64)
            for i in range(nwin):
                n_corr[i]= 1
                crap[:Nfft2] = corr[i,:]
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = dataS_t[-1]-dataS_t[0]             # total duration of what we have now
            tstart = dataS_t[0]

            nstack = int(np.round(Ttotal/substack_len))
            ampmax = np.zeros(nstack,dtype=np.float32)
            s_corr = np.zeros(shape=(nstack,Nfft),dtype=np.float32)
            n_corr = np.zeros(nstack,dtype=np.int)
            t_corr = np.zeros(nstack,dtype=np.float)
            crap   = np.zeros(Nfft,dtype=np.complex64)

            for istack in range(nstack):
                # find the indexes of all of the windows that start or end within
                itime = np.where( (dataS_t >= tstart) & (dataS_t < tstart+substack_len) )[0]
                if len(itime)==0:tstart+=substack_len;continue

                crap[:Nfft2] = np.mean(corr[itime,:],axis=0)   # linear average of the correlation
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[istack,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(itime)               # number of windows stacks
                t_corr[istack] = tstart                   # save the time stamps
                tstart += substack_len
                #print('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        ampmax = np.max(corr,axis=1)
        tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
        n_corr = nwin
        s_corr = np.zeros(Nfft,dtype=np.float32)
        t_corr = dataS_t[0]
        crap   = np.zeros(Nfft,dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr[tindx],axis=0)
        crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2],axis=0)
        crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
        s_corr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2+1, Nfft2)*dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim==1:
        s_corr = s_corr[ind]
    elif s_corr.ndim==2:
        s_corr = s_corr[:,ind]
    return s_corr,t_corr,n_corr

def noise_processing(dataS,time_norm='no',freq_norm='no',smooth_N=20):
    '''
    this function performs time domain and frequency domain normalization if needed. in real case, we prefer use include
    the normalization in the cross-correaltion steps by selecting coherency or decon (Prieto et al, 2008, 2009; Denolle et al, 2013)
    PARMAETERS:
    ------------------------
    fft_para: dictionary containing all useful variables used for fft and cc
    dataS: 2D matrix of all segmented noise data
    # OUTPUT VARIABLES:
    source_white: 2D matrix of data spectra
    '''
    N = dataS.shape[0]

    #------to normalize in time or not------
    if time_norm != 'no':

        if time_norm == 'one_bit': 	# sign normalization
            white = np.sign(dataS)
        elif time_norm == 'rma': # running mean: normalization over smoothed absolute average
            white = np.zeros(shape=dataS.shape,dtype=dataS.dtype)
            for kkk in range(N):
                white[kkk,:] = dataS[kkk,:]/utils.moving_ave(np.abs(dataS[kkk,:]),smooth_N)

    else:	# don't normalize
        white = dataS

    #-----to whiten or not------
    if freq_norm != 'no':
        source_white = utils.whiten(white,fft_para)	# whiten and return FFT
    else:
        Nfft = int(next_fast_len(int(dataS.shape[1])))
        source_white = scipy.fftpack.fft(white, Nfft, axis=1) # return FFT

    return source_white
