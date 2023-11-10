import obspy,scipy,time,pycwt,pickle,os
import numpy as np
from obspy.signal.invsim import cosine_taper
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.filter import bandpass
from obspy import UTCDateTime
from seisgo.types import DvvData
from seisgo import utils,helpers
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pyasdf
from scipy.signal import correlate,correlation_lags
from scipy.stats import linregress

'''
The dvv measuring functions are from Congcong Yuan at Harvard, originally written by Chengxin Jiang and Marine Denolle.
Xiaotao Yang adapted them for SeisGo and added wrapper functions.

Note by Congcong: several utility functions are modified based on https://github.com/tclements/noise
'''

########################################################
################ WRAPPER FUNCTIONS ##################
########################################################
def get_dvv(corrdata,freq,win,ref=None,stack_method='linear',offset=1.0,resolution=None,
            vmin=1.0,normalize=True,whiten='no',whiten_smooth=20, whiten_pad=100,
            method='wts',dvmax=0.05,subfreq=True,plot=False,figsize=(8,8),
            savefig=False,figdir='.',figname=None,figformat='png',save=False,outdir='.',outfile=None,
            format=None,nproc=None,v=False):
    """
    Compute dvv with given corrdata object, with options to save dvvdata to file.

    =====PARAMETERS=======
    corrdata: CorrData object that stores the correlaiton result.
    freq: minimum and maximum frequencies for dvv measurements.
    win: window length for dvv measurements.
    ref: reference trace/stack. default is None (will get by stacking all in the data)
    dvmax: maximum dvv searching range. default: 0.05 (5%).
    vmin: minimum velocity for the main phase, only considered in cross-correlations between
        two stations. default 1.0 km/s.
    offset: offset from 0.0 seconds (for autocorr) and from the maximum arrival time of the main
        phase (for xcorr)
    resolution: in seconds, specifying the temporal resolution (resampling/substacking) before measuring
        dvv.
    stack_method: stacking method to get the reference trace (if not given) and the short-window substacks. default is 'linear'.
    normalize: Ture or False for data normalization in measuring dvv.
    whiten='no',whiten_smooth=20, whiten_pad=100: parameters for whitening the trace before measuring dv/v.
                whiten: default is 'no', could be 'phase_only' or 'rma'.
    method: dvv measuring method.
    subfreq: keep all frequencies in the dvv result. default is True. Otherwise, only get one dvv result.
    plot: Default is False. It determines whether plots the corrdata and the measuring time windows.
        Plotting for dvvdata is currently seperated as a dvvdata.plot() method.
    savefig: Default False. Save plot or not.
    figdir: directory to save the figure.
    figname: figure name, excluding directory.
    save: this flag is for the dvvdata result. if true, the result will be saved to an ASDF file.
        Othersie, it returns the dvvdata object. Default is False.
    outdir: this is the directory to save the dvvdata.
    outfile: specify the file name to save the dvvdata. the file format can be indicated with the
            extension or "format"
    format: data file format: "asdf" or "pickle". Default is None, which will be determined by file extension.
    v: verbose. Default False.
    """
    method_all=helpers.dvv_methods() #list of all available methods.
    if method.lower() == "ts":
        subfreq=False
    if method.lower() not in method_all:
        raise ValueError(method+" is not available yet. Please change to one of: "+str(method_all))
    if corrdata.side.lower() != "a":
        raise ValueError("only works for now on corrdata with both sides. here: corrdata.side="+corrdata.side)
    # load stacked and sub-stacked waveforms
    cdata=corrdata.copy()
    #demean and detrend
    datatemp=cdata.data.copy()
    cdata.data=utils.demean(utils.detrend(datatemp))
    if resolution is not None: cdata.stack(win_len=resolution,method=stack_method,overwrite=True)
    if ref is None: ref=cdata.stack(method=stack_method,overwrite=False)

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
    tvec_all = np.arange(-cdata.lag,cdata.lag+0.5*cdata.dt,cdata.dt)
    zero_indx0 = np.where((tvec_all> -0.5*cdata.dt)&(tvec_all<0.5*cdata.dt))[0]
    zero_indx = zero_indx0[np.argmin(np.abs(tvec_all[zero_indx0]))]
    tvec_half=tvec_all[zero_indx:]
    # casual and acasual coda window
    pwin_indx = np.where((tvec_all>=np.min(twin))&(tvec_all<=np.max(twin)))[0]
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
    #do whitening if specified.
    if whiten != 'no':
        ref = utils.whiten(ref,cdata.dt,freq[0],freq[1],method=whiten,smooth=whiten_smooth,pad=whiten_pad)
        cur = utils.whiten(cur,cdata.dt,freq[0],freq[1],method=whiten,smooth=whiten_smooth,pad=whiten_pad)
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
            if v: print('working on window: '+str(UTCDateTime(cdata.time[ii]))+" ... "+str(ii+1)+"/"+str(nwin))
            if method.lower()=="wts":
                freq_p,dvv_p,dvv_error_p,cc_p,_ = wts_dvv(pref[ii],pcur[ii],tvec_half,twin,freq,\
                                                                         subfreq=subfreq,dvmax=dvmax)
                _,dvv_n,dvv_error_n,cc_n,_ = wts_dvv(nref[ii],ncur[ii],tvec_half,twin,freq,\
                                                                             subfreq=subfreq,dvmax=dvmax)
            elif method.lower()=="ts":
                dvv_p,dvv_error_p,cc_p,_ = ts_dvv(pref[ii],pcur[ii],tvec_half,twin,freq,\
                                                                         dvmax=dvmax)
                dvv_n,dvv_error_n,cc_n,_ = ts_dvv(nref[ii],ncur[ii],tvec_half,twin,freq,\
                                                                             dvmax=dvmax)
                #
                freq_p = freq
            else:
                raise ValueError(method+" is not available yet. Please change to one of: "+str(method_all))
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
        if method.lower()=="wts":
            presults=p.starmap(wts_dvv,[(pref[ii],pcur[ii],tvec_half,\
                                        twin,freq,subfreq,dvmax) for ii in range(nwin)])
            nresults=p.starmap(wts_dvv,[(nref[ii],ncur[ii],tvec_half,\
                                        twin,freq,subfreq,dvmax) for ii in range(nwin)])
            for ii in range(nwin):
                if ii==0: freqall=presults[ii][0]
                dvv_pos.append(presults[ii][1])
                dvv_neg.append(nresults[ii][1])
                error_p.append(presults[ii][2])
                error_n.append(nresults[ii][2])
                maxcc_p.append(presults[ii][3])
                maxcc_n.append(nresults[ii][3])
        elif method.lower()=="ts":
            #set filter to True when using as a standalone
            presults=p.starmap(ts_dvv,[(pref[ii],pcur[ii],tvec_half,\
                                        twin,freq,dvmax) for ii in range(nwin)])
            nresults=p.starmap(ts_dvv,[(nref[ii],ncur[ii],tvec_half,\
                                        twin,freq,dvmax) for ii in range(nwin)])
            for ii in range(nwin):
                if ii==0: freqall=freq
                dvv_pos.append(presults[ii][0])
                dvv_neg.append(nresults[ii][0])
                error_p.append(presults[ii][1])
                error_n.append(nresults[ii][1])
                maxcc_p.append(presults[ii][2])
                maxcc_n.append(nresults[ii][2])
        else:
            raise ValueError(method+" is not available yet. Please change to one of: "+str(method_all))
        p.close()

    #
    del pcur,ncur,pref,nref
    maxcc_p=np.array(maxcc_p)
    maxcc_n=np.array(maxcc_n)
    error_p=np.array(error_p)
    error_n=np.array(error_n)
    #for now, if errors are negative, assign np.nan to dvv data.
    dvv_neg=np.array(dvv_neg)
    dvv_pos=np.array(dvv_pos)
    idx1=np.where((error_n<0))
    idx2=np.where((error_p<0))
    error_n[idx1]=np.nan
    error_p[idx2]=np.nan
    maxcc_n[idx1]=np.nan
    maxcc_p[idx2]=np.nan
    dvv_neg[idx1]=np.nan
    dvv_pos[idx2]=np.nan
    dvvdata=DvvData(cdata,subfreq=subfreq,freq=freqall,cc1=ncor_cc,cc2=pcor_cc,maxcc1=maxcc_n,maxcc2=maxcc_p,
                        method=method,stack_method=stack_method,error1=error_n,error2=error_p,
                        window=twin,normalize=normalize,data1=np.array(dvv_neg),data2=np.array(dvv_pos))
    if save:
        dvvdata.save(outdir=outdir,file=outfile,format=format)

    ######plotting
    if plot:
        disp_indx = np.where(np.abs(tvec_all)<=np.max(twin)+0.2*win)[0]
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
        ref = bandpass(ref,freq[0],freq[1],df=1/cdata.dt,corners=4,zerophase=True)

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
            np.concatenate((np.ones(len(pwin_indx))*0,np.ones(len(pwin_indx))*nwin),axis=0),'y', alpha=0.3)
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
        ax2.plot(cdata.time,ncor_cc,'yo-',markersize=2,linewidth=1)
        ax2.plot(cdata.time,pcor_cc,'co-',markersize=2,linewidth=1)
        ax2.set_xticks(cdata.time[xticks])
        ax2.set_xticklabels(xticklabel,fontsize=12)
        ax2.set_title('substacked with '+stack_method,fontsize=12)
        # ax2.set_xticks(timestamp[0:nwin:tick_inc])
        ax2.set_xlim([min(cdata.time),max(cdata.time)])
        ax2.set_ylabel('cc coeff')
        ax2.legend(['negative','positive'],loc='upper right')

        plt.tight_layout()

        ###################
        ##### SAVING ######
        if savefig:
            if not os.path.isdir(figdir):os.mkdir(figdir)
            if figname is None:
                figname = 'xcc_'+cdata.id+'_'+cc_comp+'.'+figformat
            plt.savefig(figdir+'/'+figname, format=figformat, dpi=300, facecolor = 'white')
            plt.close()
        else:
            plt.show()

    if not save: return dvvdata
#
def extract_dvvdata(sfile,pair=None,comp=['all'],format=None):
    """
    Extracts DvvData from a file.

    PARAMETERS:
    --------------------------
    dfile: data file containing DvvData saved through DvvData.save().
    pair: net1.sta1-net2.sta2 pair to extract, default is to extract all pairs.
    comp: cross-correlation component or a list of components to extract, default is all components.
    format: if the file format is known, specify here. otherwise, the program determines it from the file extension.
            You can specify as "asdf" or "pickle"

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

    #
    if format is None: #automatically determine the format.
        fext=sfile[-2:]
        if fext.lower() == "h5":
            format="asdf"
        elif fext.lower() == "pk":
            format="pickle"
        else:
            raise ValueError("file extension ["+fext+"] is not supported. Specify the format if you know it.")

    dvvdict=dict()
    if format.lower() == "asdf":
        try:
            ds = pyasdf.ASDFDataSet(sfile,mpi=False,mode='r')
            # extract common variables
            spairs_all = ds.auxiliary_data.list()
        except Exception:
            raise IOError("exit! cannot open %s to read"%sfile)
    elif format.lower() == "pickle":
        datain=pickle.load(open(sfile,"rb"))
        spairs_all=list(datain.keys())

    if pair is None: pair=spairs_all

    for spair in list(set(pair) & set(spairs_all)):
        ttr = spair.split('_')
        snet,ssta = ttr[0].split('.')
        rnet,rsta = ttr[1].split('.')
        if format.lower() == "asdf":
            path_lists = ds.auxiliary_data[spair].list()
        elif format.lower() == "pickle":
            path_lists = list(datain[spair].keys())
        #
        dvvdict[spair]=dict()
        for ipath in path_lists:
            schan,rchan = ipath.split('_')
            cc_comp=schan[-1]+rchan[-1]
            if cc_comp in comp or comp == ['all'] or comp ==['ALL']:
                try:
                    if format.lower() == "asdf":
                        para=ds.auxiliary_data[spair][ipath].parameters
                    elif format.lower() == "pickle":
                        para=datain[spair][ipath]["parameters"]
                    #
                    dt,dist,dist_unit,azi,baz,slon,slat,sele,rlon,rlat,rele,\
                    window,stack_method,method,normalize,ttime,comp,freq,\
                    net,sta,chan,side,cc1,cc2,maxcc1,maxcc2,error1,error2=\
                        para['dt'],para['dist'],para['dist_unit'],para['azi'],para['baz'],\
                        para['lonS'],para['latS'],para['eleS'],para['lonR'],para['latR'],para['eleR'],\
                        para['window'],para['stack_method'],para['method'],para['normalize'],\
                        para['time'],para['comp'],para['freq'],\
                        para['net'],para['sta'],para['chan'],para['side'],para['cc1'],para['cc2'],\
                        para['maxcc1'],para['maxcc2'],para['error1'],para['error2']

                    ##special handling of time, in case time_mean is saved to reduce
                    #the attribute memory_size
                    if "time_mean" in list(para.keys()):
                        tmean=para["time_mean"]
                        ttime = np.float64(ttime) + tmean
                    if "subfreq" in list(para.keys()):
                        subfreq=para['subfreq']
                    else:
                        subfreq=True #to be compatible with old usage, only wts with subfreq True.
                    if format.lower() == "asdf":
                        datamatrix=ds.auxiliary_data[spair][ipath].data
                    elif format.lower() == "pickle":
                        datamatrix=datain[spair][ipath]["data"]

                    if side.lower() == 'a':
                        data1 = datamatrix[0].copy()
                        data2 = datamatrix[1].copy()
                    elif side.lower()=='n':
                        data1 = datamatrix.copy()
                        data2=None
                    elif side.lower()=='p':
                        data1=None
                        data2 = datamatrix.copy()

                except Exception:
                    print('continue! something wrong with %s %s'%(spair,ipath))
                    continue
                dvvdict[spair][cc_comp]=DvvData(net=[snet,rnet],sta=[ssta,rsta],loc=['',''],\
                                                chan=chan,lon=[slon,rlon],lat=[slat,rlat],
                                                ele=[sele,rele],cc_comp=comp,dt=dt,dist=dist,
                                                dist_unit=dist_unit,az=azi,baz=baz,time=ttime,normalize=normalize,
                                                freq=freq,cc1=cc1,cc2=cc2,maxcc1=maxcc1,maxcc2=maxcc2,
                                                error1=error1,error2=error2,window=window,subfreq=subfreq,
                                                stack_method=stack_method,method=method,
                                                data1=data1,data2=data2)
        return dvvdict
    else:
        raise ValueError(format+" is not supported yet.")

##########################################################################
################ MONITORING FUNCTIONS ADAPTED FOR SEISGO ##################
##########################################################################
def ts_dvv(ref, cur, t,twin, freq, dvmax=0.05, ndv=100, filter=True):
    """
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.
    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    freq: [min,max] frequency frequency of the data
    t: time vector for the data.
    twin: time window for the measurements of dv/v
    dvmax: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change ('float'). Default=0.05.
    ndv: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float'). Default is 100.
    filter: apply filter with the specified frequency range. Default is True.
    RETURNS:
    ----------------
    dv: Relative velocity change dv/v (in %)
    cc: correlation coefficient between the reference waveform and the best stretched/compressed current waveform
    cdp: correlation coefficient between the reference waveform and the initial current waveform
    error: Errors in the dv/v measurements based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)
    Note: The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "ndv" values.
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .
    Originally by L. Viens 04/26/2018 (Viens et al., 2018 JGR)
    modified by Chengxin Jiang, Xiaotao Yang
    """

    # load common variables from dictionary
    dt   = t[1]-t[0]
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    itvec = np.arange(int((tmin-t.min())/dt)+1, int((tmax-t.min())/dt)+1)
    tvec = t[itvec]

    # make useful one for measurements
    dvmin = -np.abs(dvmax)
    dvmax = np.abs(dvmax)
    Eps = 1+(np.linspace(dvmin, dvmax, ndv))
    cof = np.zeros(Eps.shape,dtype=np.float32)

    #apply filter if requested
    if filter: #this is important when the data was not filtered before calling ts_dvv().
        ref=bandpass(ref,freq[0],0.998*freq[1],df=1/dt,corners=4,zerophase=True)
        cur=bandpass(cur,freq[0],0.998*freq[1],df=1/dt,corners=4,zerophase=True)
    refwin_temp=ref[itvec]
    curwin_temp=cur[itvec]
    refwin=refwin_temp/np.max(np.abs(refwin_temp))
    curwin=curwin_temp/np.max(np.abs(curwin_temp))
    # Set of stretched/compressed current waveforms
    # plt.figure()
    # plt.plot(tvec,refwin,'r',label='ref')
    # plt.plot(tvec,curwin,'b',label='cur')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # plt.figure()
    for ii in range(len(Eps)):
        nt = tvec*Eps[ii]
        s = np.interp(x=tvec, xp=nt, fp=curwin)
        tcheckmin=np.max([nt[0],tvec[0]]) #trim the zero paddings due to extrapolation.
        tcheckmax=np.min([nt[-1],tvec[-1]])
        tcheckidx=np.where((tvec>= tcheckmin) & (tvec<=tcheckmax))[0]
        waveform_ref = refwin[tcheckidx]
        waveform_cur = s[tcheckidx]
        # plt.plot(tvec[tcheckidx],ii+waveform_ref,'r',label='ref')
        # plt.plot(tvec[tcheckidx],ii+waveform_cur,'b')
        cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]
#         cof[ii] = np.sum(np.power(waveform_ref-waveform_cur,2))
    #
    # plt.grid()
    # plt.show()
    cdp = np.corrcoef(curwin, refwin)[0, 1] # correlation coefficient between the reference and initial current waveforms

    # find the maximum correlation coefficient
    imax = np.nanargmax(cof)
    if imax >= len(Eps)-2:
        imax = imax - 2
    if imax <= 2:
        imax = imax + 2
    #
    # plt.plot(Eps,cof,'o')
    # plt.show()
    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = np.linspace(Eps[imax-2], Eps[imax+2], ndv)
    ncof    = np.zeros(dtfiner.shape,dtype=np.float32)

    for ii in range(len(dtfiner)):
        nt = tvec*dtfiner[ii]
        s = np.interp(x=tvec, xp=nt, fp=curwin)
        tcheckmin=np.max([nt[0],tvec[0]])
        tcheckmax=np.min([nt[-1],tvec[-1]])
        tcheckidx=np.where((tvec>= tcheckmin) & (tvec<=tcheckmax))[0]
        waveform_ref = refwin[tcheckidx]
        waveform_cur = s[tcheckidx]
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

def wts_dvv(ref,cur,t,twin,freq,subfreq=True,dvmax=0.05,normalize=True,ndv=100,dj=1/12,s0=-1,J=-1,wvn='morlet'):
    """
    Apply stretching method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range

    Parameters
    --------------
    ref: The complete "Reference" time series (numpy.ndarray)
    cur: The complete "Current" time series (numpy.ndarray)
    t: time vector (one side)
    twin: time window to measure dv/v
    freq: frequence range for measuring.
    subfreq: a boolen variable to make measurements on all frequency range or not. Default True.
    dvmax: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change (float).
            Default: 0.05 (5%).
    ndv: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  (Default)
    dj, s0, J, sig, wvn: common parameters used in 'wavelet.wct'. Defaults are dj=1/12,s0=-1,J=-1,wvn='morlet'
    normalize: normalize the wavelet spectrum or not. Default is True

    RETURNS:
    ------------------
    dvv: estimated dv/v
    err: error of dv/v estimation

    Written by Congcong Yuan (30 Jun, 2019)
    Adapted by Xiaotao Yang for SeisGo (July, 2021)
    """
    # common variables
    dt   = t[1]-t[0]
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    itvec = np.arange(np.int((tmin-t.min())/dt)+1, np.int((tmax-t.min())/dt)+1)

    # apply cwt on two traces
    cwt1, sj, f, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, f, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)

    # zero out data outside frequency band
    if (fmax> np.max(f)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        f_ind = np.where((f >= fmin) & (f <= fmax))[0]

    # convert wavelet domain back to time domain (~filtering)
    if not subfreq:

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
        dvv, err, cc, cdp = ts_dvv(ncwt2, ncwt1, t,twin,freq,dvmax=dvmax, ndv=ndv,filter=False)
        return freq, dvv, err, cc, cdp

    # directly take advantage of the real-valued parts of wavelet transforms
    else:
        # extract real values of cwt
        rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)
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
            #update the frequency to be passed to ts_dvv(). the frequency range is used to compute the errors.
            if ii >0:
                if ii < len(f_ind) - 1:
                    newfreq=[np.mean([f[ifreq],f[f_ind[ii-1]]]),np.mean([f[ifreq],f[f_ind[ii+1]]])]
                else:
                    df=np.abs(f[f_ind[ii]] - f[f_ind[ii-1]])
                    newfreq=[np.mean([f[ifreq],f[f_ind[ii-1]]]),f[ifreq]+0.5*df]
            else:
                df=np.abs(f[f_ind[ii+1]] - f[f_ind[ii]])
                newfreq=[f[ifreq]-0.5*df,np.mean([f[ifreq],f[f_ind[ii+1]]])]
            # run stretching
            dv, error, c1, c2 = ts_dvv(ncwt2, ncwt1, t,twin,newfreq, dvmax=dvmax, ndv=ndv,filter=False)
            dvv[ii], err[ii], cc[ii], cdp[ii]=dv, error,c1, c2

        return f[f_ind], dvv, err, cc, cdp


#
def xc_dvv(ref, cur, t,twin, freq, filter=True,plot=False):
    """
    This function compares the waveform differences through moving cross-correlations. It might suffer from
    cycle skipping.
    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    freq: [min,max] frequency frequency of the data
    t: time vector for the data.
    twin: time window for the measurements of dv/v
    filter: apply filter with the specified frequency range. Default is True.
    RETURNS:
    ----------------
    dv: Relative velocity change dv/v (in %)
    cc: is the R-squared from the linear fit.
    cdp: correlation coefficient between the reference waveform and the initial current waveform
    error: Errors in the dv/v measurements based on linear regression.
    """

    # load common variables from dictionary
    dt   = t[1]-t[0]
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    itvec = np.arange(int((tmin-t.min())/dt)+1, int((tmax-t.min())/dt)+1)
    tvec = t[itvec]

    #apply filter if requested
    if filter: #this is important when the data was not filtered before calling ts_dvv().
        ref=bandpass(ref,freq[0],0.998*freq[1],df=1/dt,corners=4,zerophase=True)
        cur=bandpass(cur,freq[0],0.998*freq[1],df=1/dt,corners=4,zerophase=True)
    refwin_temp=ref[itvec]
    curwin_temp=cur[itvec]
    refwin=refwin_temp/np.max(np.abs(refwin_temp))
    curwin=curwin_temp/np.max(np.abs(curwin_temp))
    cdp = np.corrcoef(curwin, refwin)[0, 1] # correlation coefficient between the reference and initial current waveforms

    #
    winlen=int(2/freq[0]/dt)
    step=int(winlen/2)
    sliceref,slicen,sliceidx=utils.sliding_window(refwin,winlen,ss=step,getindex=True)
    slicecur,slicen,sliceidx=utils.sliding_window(curwin,winlen,ss=step,getindex=True)
    lags = dt*correlation_lags(winlen, winlen)
    dtarray=[]
    tarray=[]
    cc_all=[]
    for i in range(slicen-1):
        xc = correlate(slicecur[i],sliceref[i])
        cc_all.append(np.max(xc))

        xcmax=np.nanargmax(xc)
        xcmaxlag=lags[xcmax]
        dtarray.append(xcmaxlag)
        tarray.append(tvec[sliceidx[i]])

    #
    tarray=np.array(tarray)
    dtarray=np.array(dtarray)
    res=linregress(tarray,dtarray)
    if plot:
        plt.figure()
        plt.plot(tarray,dtarray,'o')
        plt.plot(tarray, res.intercept + res.slope*tarray, 'r', label='fitted line')
        plt.show()

    dv=-100*res.slope
    error=100*res.stderr
    cc=np.power(res.rvalue,2)

    return dv, error, cc, cdp
