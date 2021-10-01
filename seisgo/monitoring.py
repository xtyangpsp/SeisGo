import obspy,scipy,time,pycwt
import numpy as np
from obspy.signal.invsim import cosine_taper
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.regression import linear_regression
from obspy.signal.filter import bandpass
from obspy import UTCDateTime
from seisgo.types import DvvData
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pyasdf

'''
This dvv module is written to realize the measurements of velocity perturbation dv/v. In general,
the modules are organized based on their functionality in the following way. it includes:

1) monitoring functions representing different methods to measure dv/v;
2) monitoring utility functions used by the monitoring functions.

Original by: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
             Marine Denolle (mdenolle@fas.harvard.edu)
Updated by:
             Congcong Yuan (cyuan@fas.harvard.edu)
Adapted for SeisGo by:
            Xiaotao Yang (xtyang@purdue.edu)

several utility functions are modified based on https://github.com/tclements/noise
'''

########################################################
################ MONITORING FUNCTIONS ##################
########################################################

'''
a compilation of all available core functions for computing phase delays based on ambient noise interferometry

quick index of dv/v methods:
1) wcc_dvv (Moving Window Cross Correlation; Snieder et al., 2012)
2) ts_dvv (Trace stretching; Weaver et al (2011))
3) dtw_dvv (Dynamic Time Warping; Mikesell et al. 2015)
4) mwcs_dvv (Moving Window Cross Spectrum; Clark et al., 2011)
5) wxs_dvv (Wavelet Xross Spectrum; Mao et al., 2019)
6) wts_dvv (Wavelet Streching; Yuan et al., in prep)
7) wdw_dvv (Wavelet Dynamic Warping; Yuan et al., in prep)
'''

####
def get_dvv(corrdata,freq,win,stack_method='linear',offset=1.0,resolution=None,
            vmin=1.0,normalize=True,method='wts',dvmax=0.05,subfreq=True,
            plot=False,figsize=(8,8),savefig=False,figdir='.',save=False,outdir='.',nproc=None):
    """
    Compute dvv with given corrdata object, with options to save dvvdata to file.

    =====PARAMETERS=======
    corrdata: CorrData object that stores the correlaiton result.
    freq: minimum and maximum frequencies for dvv measurements.
    win: window length for dvv measurements.
    dvmax: maximum dvv searching range. default: 0.05 (5%).
    vmin: minimum velocity for the main phase, only considered in cross-correlations between
        two stations. default 1.0 km/s.
    offset: offset from 0.0 seconds (for autocorr) and from the maximum arrival time of the main
        phase (for xcorr)
    resolution: in seconds, specifying the temporal resolution (resampling/substacking) before measuring
        dvv.
    stack_method: stacking method to get the reference trace.
    normalize: Ture or False for data normalization in measuring dvv.
    method: dvv measuring method.
    subfreq: keep all frequencies in the dvv result. default is True. Otherwise, only get one dvv result.
    plot: Default is False. It determines whether plots the corrdata and the measuring time windows.
        Plotting for dvvdata is currently seperated as a dvvdata.plot() method.
    savefig: Default False. Save plot or not.
    figdir: directory to save the figure.
    save: this flag is for the dvvdata result. if true, the result will be saved to an ASDF file.
        Othersie, it returns the dvvdata object. Default is False.
    outdir: this is the directory to save the dvvdata.
    """
    if method.lower()!="wts":
        raise ValueError(method+" is not available yet. Please change to 'wts' for now!")
    # load stacked and sub-stacked waveforms
    cdata=corrdata.copy()

    cdata.stack(resolution)
    ref=cdata.stack(method=stack_method,overwrite=False)

    nwin=cdata.data.shape[0]

    # make conda window based on vmin
    if cdata.sta[0]!=cdata.sta[1]: #xcorr
        tmin = offset+cdata.dist/vmin
    else:
        tmin=offset

    twin = [tmin,tmin+win]
    if twin[1] > cdata.lag:
        raise ValueError('proposed window exceeds limit! reduce %d'%win)

    # ref and tvec
    tvec_all = np.arange(-cdata.lag,cdata.lag+cdata.dt,cdata.dt)
    zero_indx0 = np.where((tvec_all> -cdata.dt)&(tvec_all<cdata.dt))[0]
    zero_indx = zero_indx0[np.argmin(np.abs(tvec_all[zero_indx0]))]
    tvec_half=tvec_all[zero_indx:]
    # casual and acasual coda window
    pwin_indx = np.where((tvec_all>=np.min(twin))&(tvec_all<np.max(twin)))[0]
    nwin_indx = np.where((tvec_all<=-np.min(twin))&(tvec_all>=-np.max(twin)))[0]
    pcor_cc = np.zeros(shape=(nwin),dtype=np.float32)
    ncor_cc = np.zeros(shape=(nwin),dtype=np.float32)
    pcur=np.zeros(shape=(nwin,zero_indx+1),dtype=np.float32)
    ncur=np.zeros(shape=(nwin,zero_indx+1),dtype=np.float32)
    pref=np.zeros(shape=(nwin,zero_indx+1),dtype=np.float32)
    nref=np.zeros(shape=(nwin,zero_indx+1),dtype=np.float32)
    # allocate matrix for cur and ref waveforms and corr coefficient
    cur  = cdata.data #cdata.data
    # load all current waveforms and get corr-coeff
    if normalize:
        ref /= np.max(np.abs(ref))
        # loop through each cur waveforms
        for ii in range(nwin):
            cur[ii] /= np.max(np.abs(cur[ii]))
    for ii in range(nwin):
        # get cc coeffient
        pcor_cc[ii] = np.corrcoef(ref[pwin_indx],cur[ii,pwin_indx])[0,1]
        ncor_cc[ii] = np.corrcoef(ref[nwin_indx],cur[ii,nwin_indx])[0,1]

        pcur[ii] = cur[ii,zero_indx:]
        ncur[ii] = np.flip(cur[ii,:zero_indx+1])
        pref[ii] = ref[zero_indx:]
        nref[ii] = np.flip(ref[:zero_indx+1])
    #######################
    ##### MONITORING #####
    dvv_pos,dvv_neg,freqall,maxcc_p,maxcc_n,error_p,error_n=[],[],[],[],[],[],[]
    if nproc is None or nproc<2: #regular loop
        # loop through each win again
        for ii in range(nwin):
            # casual and acasual lags for both ref and cur waveforms
            print('working on window: '+str(UTCDateTime(cdata.time[ii]))+" ... "+str(ii+1)+"/"+str(nwin))
            if method.lower()=="wts":
                freq_p,dvv_p,dvv_error_p,cc_p,cdp_p = wts_dvv(pref[ii],pcur[ii],tvec_half,twin,freq,\
                                                                         allfreq=subfreq,dvmax=dvmax)
                freq_n,dvv_n,dvv_error_n,cc_n,cdp_n = wts_dvv(pref[ii],pcur[ii],tvec_half,twin,freq,\
                                                                             allfreq=subfreq,dvmax=dvmax)
            else:
                raise ValueError(method+" is not available yet. Please change to 'wts' for now!")
            if ii==0: freqall=freq_p
            dvv_pos.append(dvv_p)
            dvv_neg.append(dvv_n)
            maxcc_p.append(cc_p)
            maxcc_n.append(cc_n)
            error_p.append(dvv_error_p)
            error_n.append(dvv_error_n)
    else: #use multiple processor for parallel preprocessing
        #parallel
        print('working on %d windows with %d workers.'%(nwin,nproc))
        p=Pool(int(nproc))
        presults=p.starmap(wts_dvv,[(pref[ii],pcur[ii],tvec_half,\
                                    twin,freq,subfreq,dvmax) for ii in range(nwin)])
        nresults=p.starmap(wts_dvv,[(nref[ii],ncur[ii],tvec_half,\
                                    twin,freq,subfreq,dvmax) for ii in range(nwin)])
        p.close()
        for ii in range(nwin):
            if ii==0: freqall=presults[ii][0]
            dvv_pos.append(presults[ii][1])
            dvv_neg.append(nresults[ii][1])
            error_p.append(presults[ii][2])
            error_n.append(nresults[ii][2])
            maxcc_p.append(presults[ii][3])
            maxcc_n.append(nresults[ii][3])
    #
    del pcur,ncur,pref,nref
    maxcc_p=np.array(maxcc_p)
    maxcc_n=np.array(maxcc_n)
    error_p=np.array(error_p)
    error_n=np.array(error_n)
    dvvdata=DvvData(cdata,freq=freqall,cc1=ncor_cc,cc2=pcor_cc,maxcc1=maxcc_n,maxcc2=maxcc_p,
                        method=method,stack_method=stack_method,error1=error_n,error2=error_p,
                        window=twin,data1=np.array(dvv_neg),data2=np.array(dvv_pos))
    if save:
        dvvdata.to_asdf(outdir=outdir)

    ######plotting
    if plot:
        disp_indx = np.where(np.abs(tvec_all)<=np.max(twin)+10)[0]
        tvec_disp=tvec_all[disp_indx]
        # tick inc for plotting
        if nwin>100:
            tick_inc = int(nwin/10)
        elif nwin>10:
            tick_inc = int(nwin/5)
        else:
            tick_inc = 2
        #filter corrdata:
        for i in range(nwin):
            if freq[1]==0.5/cdata.dt:
                cur[i,:]=bandpass(cur[i,:],freq[0],0.995*freq[1],df=1/cdata.dt,corners=4,zerophase=True)
            else:
                cur[i,:]=bandpass(cur[i,:],freq[0],freq[1],df=1/cdata.dt,corners=4,zerophase=True)


        fig=plt.figure(figsize=figsize, facecolor = 'white')
        ax0= fig.add_subplot(8,1,(1,4))
        # 2D waveform matrix
        ax0.matshow(cur[:,disp_indx],cmap='seismic',extent=[tvec_disp[0],tvec_disp[-1],nwin,0],
                    aspect='auto')
        ax0.plot([0,0],[0,nwin],'k--',linewidth=2)
        ax0.set_title('%s, dist:%5.2fkm, %4.2f-%4.2f Hz' % (cdata.id,cdata.dist,freq[0],freq[1]))
        ax0.set_xlabel('time [s]')
        ax0.set_ylabel('waveforms')
        ax0.set_yticks(np.arange(0,nwin,step=tick_inc))
        # shade the coda part
        ax0.fill(np.concatenate((tvec_all[nwin_indx],np.flip(tvec_all[nwin_indx],axis=0)),axis=0), \
            np.concatenate((np.ones(len(nwin_indx))*0,np.ones(len(nwin_indx))*nwin),axis=0),'c', alpha=0.3,linewidth=1)
        ax0.fill(np.concatenate((tvec_all[pwin_indx],np.flip(tvec_all[pwin_indx],axis=0)),axis=0), \
            np.concatenate((np.ones(len(nwin_indx))*0,np.ones(len(nwin_indx))*nwin),axis=0),'y', alpha=0.3)
        ax0.xaxis.set_ticks_position('bottom')
        # reference waveform
        ax1 = fig.add_subplot(8,1,(5,6))
        ax1.plot(tvec_disp,ref[disp_indx],'k-',linewidth=1)
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.grid(True)
        ax1.legend(['reference'],loc='upper right')

        # the cross-correlation coefficient
        xticks=np.int16(np.linspace(0,nwin-1,6))
        xticklabel=[]
        for x in xticks:
            xticklabel.append(str(UTCDateTime(cdata.time[x]))[:10])
        ax2 = fig.add_subplot(8,1,(7,8))
        ax2.plot(cdata.time,pcor_cc,'yo-',markersize=2,linewidth=1)
        ax2.plot(cdata.time,ncor_cc,'co-',markersize=2,linewidth=1)
        ax2.set_xticks(cdata.time[xticks])
        ax2.set_xticklabels(xticklabel,fontsize=12)
        # ax2.set_xticks(timestamp[0:nwin:tick_inc])
        ax2.set_xlim([min(cdata.time),max(cdata.time)])
        ax2.set_ylabel('cc coeff')
        ax2.legend(['positive','negative'],loc='upper right')

        plt.tight_layout()

        ###################
        ##### SAVING ######
        if savefig:
            if not os.path.isdir(figdir):os.mkdir(figdir)
            ccfilebase=ccfile
            outfname = figdir+'/xcc_'+cdata.id
            plt.savefig(outfname+'_'+cc_comp+'.'+format, format=format, dpi=300, facecolor = 'white')
            plt.close()
        else:
            plt.show()

    if not save: return dvvdata
#
def extract_dvvdata(sfile,pair=None,comp=['all']):
    """
    Extracts DvvData from ASDF file.

    PARAMETERS:
    --------------------------
    dfile: ASDF file containing DvvData saved through DvvData.to_asdf().
    pair: net1.sta1-net2.sta2 pair to extract, default is to extract all pairs.
    comp: cross-correlation component or a list of components to extract, default is all components.

    RETURN:
    --------------------------
    dvvdict: a dictionary that contains all extracted DvvData objects, which each key as the station
                pair name. for each station pair, the dvvdata are saved as a list of DvvData objects.
    USAGE:
    --------------------------
    extract_dvvdata('temp.h5',comp='ZZ')
    """
    #check help or not at the very beginning

    # open data for read
    if isinstance(pair,str): pair=[pair]
    if isinstance(comp,str): comp=[comp]
    dvvdict=dict()

    try:
        ds = pyasdf.ASDFDataSet(sfile,mpi=False,mode='r')
        # extract common variables
        spairs_all = ds.auxiliary_data.list()
    except Exception:
        raise IOError("exit! cannot open %s to read"%sfile)
    if pair is None: pair=spairs_all

    for spair in list(set(pair) & set(spairs_all)):
        ttr = spair.split('_')
        snet,ssta = ttr[0].split('.')
        rnet,rsta = ttr[1].split('.')
        path_lists = ds.auxiliary_data[spair].list()
        dvvdict[spair]=dict()
        for ipath in path_lists:
            schan,rchan = ipath.split('_')
            cc_comp=schan[-1]+rchan[-1]
            if cc_comp in comp or comp == ['all'] or comp ==['ALL']:
                try:
                    para=ds.auxiliary_data[spair][ipath].parameters
                    dt,dist,dist_unit,azi,baz,slon,slat,sele,rlon,rlat,rele,\
                    window,stack_method,method,normalize,ttime,comp,freq,\
                    net,sta,chan,side,cc1,cc2,maxcc1,maxcc2,error1,error2=\
                        para['dt'],para['dist'],para['dist_unit'],para['azi'],para['baz'],\
                        para['lonS'],para['latS'],para['eleS'],para['lonR'],para['latR'],para['eleR'],\
                        para['window'],para['stack_method'],para['method'],para['normalize'],\
                        para['time'],para['comp'],para['freq'],\
                        para['net'],para['sta'],para['chan'],para['side'],para['cc1'],para['cc2'],\
                        para['maxcc1'],para['maxcc2'],para['error1'],para['error2']

                    if side.lower() == 'a':
                        data1 = ds.auxiliary_data[spair][ipath].data[0].copy()
                        data2 = ds.auxiliary_data[spair][ipath].data[1].copy()
                    elif side.lower()=='n':
                        data1 = ds.auxiliary_data[spair][ipath].data.copy()
                        data2=None
                    elif side.lower()=='p':
                        data1=None
                        data2 = ds.auxiliary_data[spair][ipath].data.copy()

                except Exception:
                    print('continue! something wrong with %s %s'%(spair,ipath))
                    continue
                dvvdict[spair][cc_comp]=DvvData(net=[snet,rnet],sta=[ssta,rsta],loc=['',''],\
                                                chan=chan,lon=[slon,rlon],lat=[slat,rlat],
                                                ele=[sele,rele],cc_comp=comp,dt=dt,dist=dist,
                                                dist_unit=dist_unit,az=azi,baz=baz,time=ttime,normalize=normalize,
                                                freq=freq,cc1=cc1,cc2=cc2,maxcc1=maxcc1,maxcc2=maxcc2,
                                                error1=error1,error2=error2,window=window,
                                                stack_method=stack_method,method=method,
                                                data1=data1,data2=data2)
    return dvvdict

def wcc_dvv(ref, cur, moving_window_length, slide_step, para):
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

def ts_dvv(ref, cur, dv_max, ndv, para):

    """
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.

    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    dv_max: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change ('float')
    ndv: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
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

    Note: The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "ndv" values.
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .

    Originally by L. Viens 04/26/2018 (Viens et al., 2018 JGR)
    modified by Chengxin Jiang
    """
    # load common variables from dictionary
    t = para['t']
    twin = para['twin']
    freq = para['freq']
    dt   = t[1]-t[0]
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    itvec = np.arange(np.int((tmin-t.min())/dt)+1, np.int((tmax-t.min())/dt)+1)
    tvec = t[itvec]

    # make useful one for measurements
    dvmin = -np.abs(dv_max)
    dvmax = np.abs(dv_max)
    Eps = 1+(np.linspace(dvmin, dvmax, ndv))
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
    t = para['t']
    twin = para['twin']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    itvec = np.arange(np.int((tmin-t.min())/dt)+1, np.int((tmax-t.min())/dt)+1)
    tvec = t[itvec]

    # setup other parameters
    npts = len(ref) # number of time samples

    # compute error function over lags, which is independent of strain limit 'b'.
    err = computeErrorFunction( cur, ref, npts, maxLag )

    # direction to accumulate errors (1=forward, -1=backward)
    dist  = accumulateErrorFunction( direction, err, npts, maxLag, b )
    stbar = backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )
    stbarTime = stbar * dt   # convert from samples to time

    # cut the first and last 5% for better regression
    # indx = np.where((tvec>=0.05*npts*dt) & (tvec<=0.95*npts*dt))[0]
    indx = np.where((tvec>=(0.05*npts*dt+tmin)) & (tvec<=(0.95*npts*dt+tmin)))[0]

    # linear regression to get dv/v
    if npts >2:

        # weights
        w = np.ones(npts)
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(tvec.flatten()[indx], stbarTime.flatten()[indx], w.flatten()[indx], intercept_origin=True)

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
    t = para['t']
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
    padd = int(2 ** (nextpow2(window_length_samples) + 2))
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
            dcur = np.sqrt(smooth(fcur2, window='hanning',half_win=smoothing_half_win))
            dref = np.sqrt(smooth(fref2, window='hanning',half_win=smoothing_half_win))
            X = smooth(X, window='hanning',half_win=smoothing_half_win)
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
    t = para['t']
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    itvec = np.arange(np.int((tmin-t.min())/dt)+1, np.int((tmax-t.min())/dt)+1)
    tvec = t[itvec]
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
            w = 1/WCT[freq_indin,itvec[it]]
            w[~np.isfinite(w)] = 1.
            delta_t_m[it],delta_t_unc[it] = linear_regression(freq[freq_indin]*2*np.pi, phase[freq_indin,itvec[it]], w)

        # new weights for regression
        wWCT = WCT[:,itvec]
        w2 = 1/np.mean(wWCT[freq_indin,],axis=0)
        w2[~np.isfinite(w2)] = 1.

        # now use dt and t to get dv/v
        if len(w2)>2:
            if not np.any(delta_t_m):
                dvv, err = np.nan,np.nan
            m, em = linear_regression(tvec, delta_t_m, w2, intercept_origin=True)
            dvv, err = -m, em
        else:
            print('not enough points to estimate dv/v for wxs')
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
                w = 1/WCT[ifreq, itvec]
                w[~np.isfinite(w)] = 1.0

                #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
                m, em = linear_regression(tvec, delta_t[ifreq, itvec], w, intercept_origin=True)
                dvv[ii], err[ii] = -m, em
            else:
                print('not enough points to estimate dv/v for wxs')
                dvv[ii], err[ii]=np.nan, np.nan

        return freq[freq_indin], dvv*100, err*100

def wts_dvv(ref,cur,t,twin,freq,allfreq=True,dvmax=0.05,normalize=True,ndv=100,dj=1/12,s0=-1,J=-1,wvn='morlet'):
    """
    Apply stretching method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range

    Parameters
    --------------
    ref: The complete "Reference" time series (numpy.ndarray)
    cur: The complete "Current" time series (numpy.ndarray)
    allfreq: a boolen variable to make measurements on all frequency range or not
    para: a dict containing freq/time info of the data matrix
    dvmax: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change (float)
    ndv: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  (float)
    dj, s0, J, sig, wvn: common parameters used in 'wavelet.wct'. Defaults are dj=1/12,s0=-1,J=-1,wvn='morlet'
    normalize: normalize the wavelet spectrum or not. Default is True

    RETURNS:
    ------------------
    dvv: estimated dv/v
    err: error of dv/v estimation

    Written by Congcong Yuan (30 Jun, 2019)
    """
    # common variables
    para={"t":t,"twin":twin,"freq":freq}
    dt   = t[1]-t[0]
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    itvec = np.arange(np.int((tmin-t.min())/dt)+1, np.int((tmax-t.min())/dt)+1)

    # apply cwt on two traces
    cwt1, sj, f, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, f, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)

    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)

    # zero out data outside frequency band
    if (fmax> np.max(f)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        f_ind = np.where((f >= fmin) & (f <= fmax))[0]

    # convert wavelet domain back to time domain (~filtering)
    if not allfreq:

        # inverse cwt to time domain
        icwt1 = pycwt.icwt(cwt1[f_ind], sj[f_ind], dt, dj, wvn)
        icwt2 = pycwt.icwt(cwt2[f_ind], sj[f_ind], dt, dj, wvn)

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
        dvv, err, cc, cdp = ts_dvv(ncwt2[itvec], ncwt1[itvec], dvmax, ndv, para)
        return dvv, err, cc, cdp

    # directly take advantage of the real-valued parts of wavelet transforms
    else:
        # initialize variable
        nfreq=len(f_ind)
        dvv, cc, cdp, err = np.zeros(nfreq,dtype=np.float32), np.zeros(nfreq,dtype=np.float32),\
            np.zeros(nfreq,dtype=np.float32),np.zeros(nfreq,dtype=np.float32)

        # loop through each freq
        for ii, ifreq in enumerate(f_ind):

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
            dv, error, c1, c2 = ts_dvv(ncwt2[itvec], ncwt1[itvec], dvmax, ndv, para)
            dvv[ii], cc[ii], cdp[ii], err[ii]=dv, c1, c2, error

        return f[f_ind], dvv, err, cc, cdp

def wtdtw_dvv(ref,cur,allfreq,para,maxLag,b,direction,dj=1/12,s0=-1,J=-1,wvn='morlet',normalize=True):
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
    t = para['t']
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    itvec = np.arange(np.int((tmin-t.min())/dt)+1, np.int((tmax-t.min())/dt)+1)
    tvec = t[itvec]

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

        # run dtw
        dv, error, dist  = dtw_dvv(ncwt2[itvec], ncwt1[itvec], para, maxLag, b, direction)
        dvv, err = dv, error

        return dvv, err

    # directly take advantage of the real-valued parts of wavelet transforms
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

            # run dtw
            dv, error, dist  = dtw_dvv(ncwt2[itvec], ncwt1[itvec], para, maxLag, b, direction)
            dvv[ii], err[ii] = dv, error

        return freq[freq_indin], dvv, err

#############################################################
################ MONITORING UTILITY FUNCTIONS ###############
#############################################################

'''
below are assembly of the monitoring utility functions called by monitoring functions
'''

def smooth(x, window='boxcar', half_win=3):
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


def nextpow2(x):
    """
    Returns the next power of 2 of x.
    """
    return int(np.ceil(np.log2(np.abs(x))))


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
