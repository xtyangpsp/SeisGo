import os,sys,obspy,scipy,pyasdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass
from seisgo import noise, stacking,utils
import pygmt as gmt
from obspy import UTCDateTime
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm

def get_color_cycle(cmap, N=None, use_index="auto"):
    """
    This function was original posted by ImportanceOfBeingErnest
    as get_cycle() as an answer to a question on stackoverflow.com. URL:
    https://stackoverflow.com/questions/30079590/use-matplotlib-color-map-for-color-cycle.
    ==PARAMETERS==
    cmap: str of cmap name
    N: number of colors. default None.
    use_index: whether use index when returning the color cycle. default "auto"

    ==RETURN==
    plt.cycler object.

    ==USAGE CASES==
    Named cmap: plt.rcParams["axes.prop_cycle"] = get_color_cycle("viridis",7)
    Indexed cmap: plt.rcParams["axes.prop_cycle"] = get_color_cycle("tab20c")
    """
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return plt.cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return plt.cycler("color",colors)

def plot_eventsequence(cat,figsize=(12,4),ytype='magnitude',figname=None,
                       yrange=None,save=False,stem=True):
    if isinstance(cat,obspy.core.event.catalog.Catalog):
        cat=pd.DataFrame(utils.qml2list(cat))
    elif isinstance(cat,list):
        cat=pd.DataFrame(cat)
    #All magnitudes greater than or equal to the limit will be plotted

    plt.figure(figsize=figsize)
    plt.title(ytype+" vs. time")
    plt.xlabel("Date (UTC)")

    if yrange is not None:
        ymin,ymax=yrange
        if ytype.lower()=="magnitude":
            cat2=cat[(cat.magnitude>=yrange[0]) & (cat.magnitude<=yrange[1]) ]
        elif ytype.lower()=="depth":
            cat2=cat[(cat.depth>=yrange[0]) & (cat.depth<=yrange[1]) ]
    else:
        cat2=cat
        if ytype.lower()=="magnitude":
            ymin=np.min(cat2.magnitude)*0.9
            ymax=np.max(cat2.magnitude)*1.1
        elif ytype.lower()=="depth":
            ymin=np.min(cat2.depth)*0.9
            ymax=np.max(cat2.depth)*1.1
    t=[]
    for i in range(len(cat2)):
        tTime=obspy.UTCDateTime(cat2.iloc[i]["datetime"])
        t.append(tTime.datetime)

    if stem:
        if ytype.lower()=="magnitude":
            markerline, stemlines, baseline=plt.stem(t,cat2.magnitude,linefmt='k-',markerfmt="o",
                                                     bottom=ymin)
            plt.ylabel("magnitue")

        elif ytype.lower()=="depth":
            markerline, stemlines, baseline=plt.stem(t,cat2.depth,linefmt='k-',markerfmt="o",
                                                     bottom=ymin)
            plt.ylabel("depth (km)")
        markerline.set_markerfacecolor('r')
        markerline.set_markeredgecolor('r')
    else:
        if ytype.lower()=="magnitude":
            plt.scatter(t,cat2.magnitude,5,'k')
            plt.ylabel("magnitue")
        elif ytype.lower()=="depth":
            plt.scatter(t,cat2.depth,cat2.magnitude,'k')
            plt.ylabel("depth (km)")

        #
    plt.grid(axis="both")
    plt.ylim([ymin,ymax])

    if save:
        if figname is not None:
            plt.savefig(figname)
        else:
            plt.savefig(ytype+"_vs_time.png")
    else:
        plt.show()

def plot_stations(lon,lat,region=None,markersize="c0.2c",title="station map",style="fancy",figname=None,
                  format='png',distance=None,projection="M5i", xshift="6i",frame="af",save=False):
    """
    lon, lat: could be list of vectors contaning multiple sets of stations. The number of sets must be the same
            as the length of the marker list.
    marker: a list specifying the symbols for each station set.
    region: [minlon,maxlon,minlat,maxlat] for map view
    """
    nsta=len(lon)
    if isinstance(markersize,str):
        markersize=[markersize]*nsta
    #
    if region is None:
        region="%6.2f/%6.2f/%5.2f/%5.2f"%(np.min(lon),np.max(lon),np.min(lat),np.max(lat))
    fig = gmt.Figure()
    gmt.config(MAP_FRAME_TYPE=style)
    for i in range(nsta):
        if i==0:
            fig.coast(region=region, resolution="f",projection=projection, rivers='rivers',
                      water="cyan",frame=frame,land="white",
                      borders=["1/0.5p,gray,2/1p,gray"])
            fig.basemap(frame='+t"'+title+'"')
        fig.plot(
            x=lon[i],
            y=lat[i],
            style=markersize[i],
            color="red",
        )

    if save:
        if figname is None:
            figname='stationmap.'+format
        fig.savefig(figname)
        print('plot was saved to: '+figname)
    else:
        fig.show()


##plot power spectral density
def plot_psd(data,dt,labels=None,xrange=None,cmap='jet',normalize=True,figsize=(13,5),\
            save=False,figname=None,tick_inc=None,db=False):
    """
    Plot the power specctral density of the data array.

    =PARAMETERS=
    data: 2-D array containing the data. the data to be plotted should be on axis 1 (second dimention)
    dt: sampling inverval in time.
    labels: row labels of the data, default is None.
    cmap: colormap, default is 'jet'
    time_format: format to show time marks, default is: '%Y-%m-%dT%H'
    normalize: whether normalize the PSD in plotting, default is True
    figsize: figure size, default: (13,5)
    """
    data=np.array(data)
    if data.ndim > 2:
        raise ValueError('only plot 1-d arrya or 2d matrix for now. the input data has a dimention of %d'%(data.ndim))

    f,psd=utils.psd(data,1/dt)
    if db:
        psd=10*np.log10(np.abs(psd))
    f=f[1:]

    plt.figure(figsize=figsize)
    ax=plt.subplot(111)
    if data.ndim==2:
        nwin=data.shape[0]
        if tick_inc is None:
            if nwin>10:
                tick_inc = int(nwin/5)
            else:
                tick_inc = 2

        psdN=np.ndarray((psd.shape[0],psd.shape[1]-1))
        for i in range(psd.shape[0]):
            if normalize: psdN[i,:]=psd[i,1:]/np.max(np.abs(psd[i,1:]))
            else: psdN[i,:]=psd[i,1:]

        plt.imshow(psdN,aspect='auto',extent=[f.min(),f.max(),psdN.shape[0],0],cmap=cmap)
        ax.set_yticks(np.arange(0,nwin,step=tick_inc))
        if labels is not None: ax.set_yticklabels(labels[0:nwin:tick_inc])
        if normalize: plt.colorbar(label='normalized PSD')
        else: plt.colorbar(label='PSD')
    else:
        if normalize: psdN=psd[1:]/np.max(np.abs(psd[1:]))
        else: psdN[i,:]=psd[1:]

        plt.plot(f,psdN)
    if xrange is None:plt.xlim([f[1],f[-1]])
    else:
        plt.xlim(xrange)
    plt.xscale('log')
    plt.xlabel('frequency (Hz)')
    plt.title('PSD')

    if save:
        if figname is not None:
            plt.savefig(figname)
        else:
            plt.savefig("PSD.png")
    else:
        plt.show()
#############################################################################
############### PLOTTING RAW SEISMIC WAVEFORMS ##########################
#############################################################################
'''
Inherited and modified from the plotting functions in the plotting_module of
NoisePy (https://github.com/mdenolle/NoisePy) by Chengxin Jiang and Marine Denolle.
'''
def plot_waveform(sfile,net,sta,freqmin,freqmax,save=False,figdir=None,format='pdf'):
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
    if ncomp==0:
        print('no data found for the specified net.sta.')
        return None
    tr   = ds.waveforms[tsta][tcomp[0]]
    dt   = tr[0].stats.delta
    npts = tr[0].stats.npts
    tt   = np.arange(0,npts)*dt
    if ncomp == 1:
        data = tr[0].data
        data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
        fig=plt.figure(figsize=(9,3))
        plt.plot(tt,data,'k-',linewidth=1)
        plt.title('T\u2080:%s   %s.%s.%s   @%5.3f-%5.2f Hz' % (tr[0].stats.starttime,net,sta,tcomp[0].split('_')[0].upper(),freqmin,freqmax))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
    else:
        data = np.zeros(shape=(ncomp,npts),dtype=np.float32)
        for ii in range(ncomp):
            data[ii] = ds.waveforms[tsta][tcomp[ii]][0].data
            data[ii] = bandpass(data[ii],freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
        fig=plt.figure(figsize=(9,6))

        for c in range(ncomp):
            if c==0:
                plt.subplot(ncomp,1,1)
                plt.plot(tt,data[0],'k-',linewidth=1)
                plt.title('T\u2080:%s   %s.%s   @%5.3f-%5.2f Hz' % (tr[0].stats.starttime,net,sta,freqmin,freqmax))
                plt.legend([tcomp[0].split('_')[0].upper()],loc='upper left')
                plt.xlabel('Time [s]')
            else:
                plt.subplot(ncomp,1,c+1)
                plt.plot(tt,data[c],'k-',linewidth=1)
                plt.legend([tcomp[c].split('_')[0].upper()],loc='upper left')
                plt.xlabel('Time [s]')

        fig.tight_layout()

    if save:
        if not os.path.isdir(figdir):os.mkdir(figdir)
        sfilebase=sfile.split('/')[-1]
        outfname = figdir+'/{0:s}_{1:s}.{2:s}'.format(sfilebase.split('.')[0],net,sta)
        fig.savefig(outfname+'.'+format, format=format, dpi=300)
        plt.close()
    else:
        fig.show()

#############################################################################
###############PLOTTING XCORR RESULTS AS THE OUTPUT OF SEISGO ##########################
#############################################################################
def plot_xcorr_substack(sfile,freqmin,freqmax,lag=None,comp='ZZ',
                        save=True,figdir=None):
    '''
    display the 2D matrix of the cross-correlation functions for a certain time-chunck.
    PARAMETERS:
    --------------------------
    sfile: cross-correlation functions outputed by SeisGo workflow
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    lag: time ranges for display
    USAGE:
    --------------------------
    plot_xcorr_substack('temp.h5',0.1,1,100,True,'./')
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
    lag0=np.min([1.0*lag,maxlag])
    if lag>maxlag:raise ValueError('lag excceds maxlag!')

    # t is the time labels for plotting
    if lag>=5:
        tstep=int(int(lag)/5)
        t1=np.arange(-int(lag),0,step=tstep)
        t2=np.arange(0,int(lag+0.5*tstep),step=tstep)
        t=np.concatenate((t1,t2))
    else:
        tstep=lag/5
        t1=np.arange(-lag,0,step=tstep)
        t2=np.arange(0,lag+0.5*tstep,step=tstep)
        t=np.concatenate((t1,t2))

    indx1 = int((maxlag-lag0)/dt)
    indx2 = indx1+2*int(lag0/dt)+1
    for spair in spairs:
        ttr = spair.split('_')
        net1,sta1 = ttr[0].split('.')
        net2,sta2 = ttr[1].split('.')
        path_lists = ds.auxiliary_data[spair].list()
        for ipath in path_lists:
            chan1,chan2 = ipath.split('_')
            cc_comp=chan1[-1]+chan2[-1]
            if cc_comp == comp or comp=='all' or comp=='ALL':
                try:
                    dist = ds.auxiliary_data[spair][ipath].parameters['dist']
                    ngood= ds.auxiliary_data[spair][ipath].parameters['ngood']
                    ttime= ds.auxiliary_data[spair][ipath].parameters['time']
                except Exception:
                    print('continue! something wrong with %s %s'%(spair,ipath))
                    continue

                # cc matrix
                timestamp = np.empty(ttime.size,dtype='datetime64[s]')
                data = ds.auxiliary_data[spair][ipath].data[:,indx1:indx2]
                # print(data.shape)
                nwin = data.shape[0]
                amax = np.zeros(nwin,dtype=np.float32)
                if nwin==0 or len(ngood)==1: print('continue! no enough substacks!');continue

                tmarks = []
                data_normalizd=data

                # load cc for each station-pair
                for ii in range(nwin):
                    data[ii] = bandpass(data[ii],freqmin,freqmax,1/dt,corners=4, zerophase=True)
                    data[ii] = data[ii]-np.mean(data[ii])
                    amax[ii] = np.max(np.abs(data[ii]))
                    data_normalizd[ii] = data[ii]/amax[ii]
                    timestamp[ii] = obspy.UTCDateTime(ttime[ii])
                    tmarks.append(obspy.UTCDateTime(ttime[ii]).strftime('%Y-%m-%dT%H:%M:%S'))

                dstack_mean=np.mean(data,axis=0)
                dstack_robust=stacking.robust_stack(data)[0]

                # plotting
                if nwin>10:
                    tick_inc = int(nwin/5)
                else:
                    tick_inc = 2

                fig = plt.figure(figsize=(10,6))
                ax = fig.add_subplot(5,1,(1,3))
                ax.matshow(data_normalizd,cmap='seismic',extent=[-lag0,lag0,nwin,0],aspect='auto')
                ax.plot((0,0),(nwin,0),'k-')
                ax.set_title('%s.%s.%s  %s.%s.%s  dist:%5.2fkm' % (net1,sta1,chan1,net2,sta2,chan2,dist))
                ax.set_xlabel('time [s]')
                ax.set_xticks(t)
                ax.set_yticks(np.arange(0,nwin,step=tick_inc))
#                 ax.set_yticklabels(np.arange(0,nwin,step=tick_inc))
                ax.set_yticklabels(tmarks[0:nwin:tick_inc])
                ax.set_xlim([-lag,lag])
                ax.xaxis.set_ticks_position('bottom')

                ax1 = fig.add_subplot(5,1,(4,5))
                ax1.set_title('stack at %4.2f-%4.2f Hz'%(freqmin,freqmax))
                tstack=np.arange(-lag0,lag0+0.5*dt,dt)
                if len(tstack)>len(dstack_mean):tstack=tstack[:-1]
                ax1.plot(tstack,dstack_mean,'b-',linewidth=1,label='mean')
                ax1.plot(tstack,dstack_robust,'r-',linewidth=1,label='robust')
                ax1.set_xlabel('time [s]')
                ax1.set_xticks(t)
                ax1.set_xlim([-lag,lag])
                ylim=ax1.get_ylim()
                ax1.plot((0,0),ylim,'k-')

                ax1.set_ylim(ylim)
                ax1.legend(loc='upper right')
                ax1.grid()
#                 ax2 = fig.add_subplot(414)
#                 ax2.plot(amax/min(amax),'r-')
#                 ax2.plot(ngood,'b-')
#                 ax2.set_xlabel('waveform number')
#                 ax2.set_xticks(np.arange(0,nwin,step=tick_inc))
#                 ax2.set_xticklabels(tmarks[0:nwin:tick_inc])
#                 #for tick in ax[2].get_xticklabels():
#                 #    tick.set_rotation(30)
#                 ax2.legend(['relative amp','ngood'],loc='upper right')
                fig.tight_layout()

                # save figure or just show
                if save:
                    if figdir==None:figdir = sfile.split('.')[0]
                    if not os.path.isdir(figdir):os.mkdir(figdir)
                    outfname = figdir+\
                    '/{0:s}.{1:s}.{2:s}_{3:s}.{4:s}.{5:s}_{6:s}-{7:s}Hz.png'.format(net1,sta1,\
                                                                      chan1,net2,\
                                                                      sta2,chan2,
                                                                     str(freqmin),str(freqmax))
                    fig.savefig(outfname, format='png', dpi=400)
                    print('saved to: '+outfname)
                    plt.close()
                else:
                    fig.show()

def plot_corrfile(sfile,freqmin,freqmax,lag=None,comp='ZZ',
                        save=True,figname=None,format='png',figdir=None):
    '''
    display the 2D matrix of the cross-correlation functions for a certain time-chunck.
    PARAMETERS:
    --------------------------
    sfile: cross-correlation functions outputed by SeisGo workflow
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    lag: time ranges for display
    USAGE:
    --------------------------
    plot_corrfile('temp.h5',0.1,1,100,True,'./')
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    corrdict=noise.extract_corrdata(sfile,comp=comp)
    clist=list(corrdict.keys())
    for c in clist:
        corr=corrdict[c]
        if comp in list(corr.keys()):
            corr[comp].plot(freqmin=freqmin,freqmax=freqmax,lag=lag,save=save,figdir=figdir,
                            figname=figname,format=format)


def plot_corrdata(corr,freqmin=None,freqmax=None,lag=None,save=False,figdir=None,figsize=(10,8)):
    '''
    display the 2D matrix of the cross-correlation functions for a certain time-chunck.
    PARAMETERS:
    --------------------------
    corr: : class:`~seisgo.types.CorrData`
            CorrData object containing the correlation functions and the metadata.
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    lag: time ranges for display

    USAGE:
    --------------------------
    plot_corrdata(corr,0.1,1,100,save=True,figdir='./')
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    netstachan1 = corr.net[0]+'.'+corr.sta[0]+'.'+corr.loc[0]+'.'+corr.chan[0]
    netstachan2 = corr.net[1]+'.'+corr.sta[1]+'.'+corr.loc[1]+'.'+corr.chan[1]

    dt,maxlag,dist,ngood,ttime,substack = [corr.dt,corr.lag,corr.dist,corr.ngood,corr.time,corr.substack]

   # lags for display
    if not lag:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')
    lag0=np.min([1.0*lag,maxlag])

    # t is the time labels for plotting
    if lag>=5:
        tstep=int(int(lag)/5)
        t1=np.arange(-int(lag),0,step=tstep);t2=np.arange(0,int(lag+0.5*tstep),step=tstep)
        t=np.concatenate((t1,t2))
    else:
        tstep=lag/5
        t1=np.arange(-lag,0,step=tstep);t2=np.arange(0,lag+0.5*tstep,step=tstep)
        t=np.concatenate((t1,t2))

    indx1 = int((maxlag-lag0)/dt);indx2 = indx1+2*int(lag0/dt)+1

    # cc matrix
    if substack:
        data = corr.data[:,indx1:indx2]
        timestamp = np.empty(ttime.size,dtype='datetime64[s]')
        # print(data.shape)
        nwin = data.shape[0]
        amax = np.zeros(nwin,dtype=np.float32)
        if nwin==0 or len(ngood)==1:
            print('continue! no enough trace to plot!')
            return

        tmarks = []
        data_normalizd=data

        # load cc for each station-pair
        for ii in range(nwin):
            if freqmin is not None and freqmax is not None:
                data[ii] = bandpass(data[ii],freqmin,freqmax,1/dt,corners=4, zerophase=True)
            data[ii] = data[ii]-np.mean(data[ii])
            amax[ii] = np.max(np.abs(data[ii]))
            data_normalizd[ii] = data[ii]/amax[ii]
            timestamp[ii] = obspy.UTCDateTime(ttime[ii])
            tmarks.append(obspy.UTCDateTime(ttime[ii]).strftime('%Y-%m-%dT%H:%M:%S'))

        dstack_mean=np.mean(data,axis=0)
#         dstack_robust=stack.robust_stack(data)[0]

        # plotting
        if nwin>10:
            tick_inc = int(nwin/5)
        else:
            tick_inc = 2

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(6,1,(1,4))
        ax.matshow(data_normalizd,cmap='seismic',extent=[-lag0,lag0,nwin,0],aspect='auto')
        ax.plot((0,0),(nwin,0),'k-')
        if freqmin is not None and freqmax is not None:
            ax.set_title('%s-%s : dist : %5.2f km : %4.2f-%4.2f Hz' % (netstachan1,netstachan2,
                                                                       dist,freqmin,freqmax))
        else:
            ax.set_title('%s-%s : dist : %5.2f km : unfiltered' % (netstachan1,netstachan2,dist))
        ax.set_xlabel('time [s]')
        ax.set_xticks(t)
        ax.set_yticks(np.arange(0,nwin,step=tick_inc))
        ax.set_yticklabels(tmarks[0:nwin:tick_inc])
        ax.set_xlim([-lag,lag])
        ax.xaxis.set_ticks_position('bottom')

        ax1 = fig.add_subplot(6,1,(5,6))
        if freqmin is not None and freqmax is not None:
            ax1.set_title('stack at %4.2f-%4.2f Hz'%(freqmin,freqmax))
        else:
            ax1.set_title('stack: unfiltered')
        tstack=np.arange(-lag0,lag0+0.5*dt,dt)
        if len(tstack)>len(dstack_mean):tstack=tstack[:-1]
        ax1.plot(tstack,dstack_mean,'b-',linewidth=1,label='mean')
#         ax1.plot(tstack,dstack_robust,'r-',linewidth=1,label='robust')
        ax1.set_xlabel('time [s]')
        ax1.set_xticks(t)
        ax1.set_xlim([-lag,lag])
        ylim=ax1.get_ylim()
        ax1.plot((0,0),ylim,'k-')

        ax1.set_ylim(ylim)
        ax1.legend(loc='upper right')
        ax1.grid()

        fig.tight_layout()
    else: #only one trace available
        data = corr.data[indx1:indx2]

        # load cc for each station-pair
        if freqmin is not None and freqmax is not None:
            data = bandpass(data,freqmin,freqmax,1/dt,corners=4, zerophase=True)
        data = data-np.mean(data)
        amax = np.max(np.abs(data))
        data /= amax
        timestamp = obspy.UTCDateTime(ttime)
        tmarks=obspy.UTCDateTime(ttime).strftime('%Y-%m-%dT%H:%M:%S')

        tx=np.arange(-lag0,lag0+0.5*dt,dt)
        if len(tx)>len(data):tx=tx[:-1]
        plt.figure(figsize=figsize)
        ax=plt.gca()
        plt.plot(tx,data,'k-',linewidth=1)
        if freqmin is not None and freqmax is not None:
            plt.title('%s-%s : dist : %5.2f km : %4.2f-%4.2f Hz' % (netstachan1,netstachan2,
                                                                       dist,freqmin,freqmax))
        else:
            plt.title('%s-%s : dist : %5.2f km : unfiltered' % (netstachan1,netstachan2,dist))
        plt.xlabel('time [s]')
        plt.xticks(t)
        ylim=ax.get_ylim()
        plt.plot((0,0),ylim,'k-')

        plt.ylim(ylim)
        plt.xlim([-lag,lag])
        ax.grid()

    # save figure or just show
    if save:
        if figdir==None:figdir = sfile.split('.')[0]
        if not os.path.isdir(figdir):os.mkdir(figdir)
        outfname = figdir+\
        '/{0:s}_{1:s}_{2:s}-{3:s}Hz.png'.format(netstachan1,netstachan2,
                                                         str(freqmin),str(freqmax))
        plt.savefig(outfname, format='png', dpi=300)
        print('saved to: '+outfname)
        plt.close()
    else:
        plt.show()

#############################################################################
###############PLOTTING THE POST-STACKING XCORR FUNCTIONS##########################
#############################################################################
'''
Modified from the plotting functions in the plotting_module of NoisePy (https://github.com/mdenolle/NoisePy).
Credits should be given to the development team for NoisePy (Chengxin Jiang and Marine Denolle).
'''
def plot_xcorr_moveout_heatmap(sfiles,sta,dtype,freq,comp,dist_inc,lag=None,save=False,\
                                figsize=None,format='png',figdir=None):
    '''
    display the moveout (2D matrix) of the cross-correlation functions stacked for all time chuncks.
    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: station name as the virtual source.
    dtype: datatype either 'Allstack_pws' or 'Allstack_linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    comp:   cross component
    dist_inc: distance bins to stack over
    lag: lag times for displaying
    save: set True to save the figures (in pdf format)
    figdir: diresied directory to save the figure (if not provided, save to default dir)
    USAGE:
    ----------------------
    plot_xcorr_moveout_heatmap('temp.h5','sta','Allstack_pws',0.1,0.2,1,'ZZ',200,True,'./temp')
    '''
    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')
    if not isinstance(freq[0],list):freq=[freq]
    freq=np.array(freq)
    figlabels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    if freq.shape[0]>9:
        raise ValueError('freq includes more than 9 (maximum allowed for now) elements!')
    elif freq.shape[0]==9:
        subplot=[3,3]
        figsize0=[14,7.5]
    elif freq.shape[0] >=7 and freq.shape[0] <=8:
        subplot=[2,4]
        figsize0=[18,10]
    elif freq.shape[0] >=5 and freq.shape[0] <=6:
        subplot=[2,3]
        figsize0=[14,7.5]
    elif freq.shape[0] ==4:
        subplot=[2,2]
        figsize0=[10,6]
    else:
        subplot=[1,freq.shape[0]]
        if freq.shape[0]==3:
            figsize0=[13,3]
        elif freq.shape[0]==2:
            figsize0=[8,3]
        else:
            figsize0=[4,3]
    if figsize is None:figsize=figsize0

    path  = comp

    receiver = sta+'.h5'
    stack_method = dtype.split('_')[-1]
    # extract common variables
    try:
        ds    = pyasdf.ASDFDataSet(sfiles[0],mpi=False,mode='r')
        dt    = ds.auxiliary_data[dtype][path].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][path].parameters['maxlag']
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

    data0 = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
    dist = np.zeros(nwin,dtype=np.float32)
    ngood= np.zeros(nwin,dtype=np.int16)

    # load cc and parameter matrix
    for ii in range(len(sfiles)):
        sfile = sfiles[ii]
        treceiver = sfile.split('_')[-1]

        ds = pyasdf.ASDFDataSet(sfile,mpi=False,mode='r')
        try:
            # load data to variables
            dist[ii] = ds.auxiliary_data[dtype][path].parameters['dist']
            ngood[ii]= ds.auxiliary_data[dtype][path].parameters['ngood']
            tdata    = ds.auxiliary_data[dtype][path].data[indx1:indx2]
            if treceiver == receiver: tdata=np.flip(tdata,axis=0)
        except Exception:
            print("continue! cannot read %s "%sfile);continue

        data0[ii] = tdata

    ntrace = int(np.round(np.max(dist)+0.51)/dist_inc)

    fig=plt.figure(figsize=figsize)

    for f in range(len(freq)):
        freqmin=freq[f][0]
        freqmax=freq[f][1]
        data = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
        for i2 in range(data0.shape[0]):
            data[i2]=bandpass(data0[i2],freqmin,freqmax,1/dt,corners=4, zerophase=True)

        # average cc
        ndata  = np.zeros(shape=(ntrace,indx2-indx1),dtype=np.float32)
        ndist  = np.zeros(ntrace,dtype=np.float32)
        for td in range(ndata.shape[0]):
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
        ax=fig.add_subplot(subplot[0],subplot[1],f+1)
        ax.matshow(ndata,cmap='seismic',extent=[-lag,lag,ndist[-1],ndist[0]],aspect='auto')
        ax.set_title('%s %s stack %s %5.3f-%5.2f Hz'%(figlabels[f],sta,stack_method,freqmin,freqmax))
        ax.set_xlabel('time [s]')
        ax.set_ylabel('distance [km]')
        ax.set_xticks(t)
        ax.xaxis.set_ticks_position('bottom')
        #ax.text(np.ones(len(ndist))*(lag-5),dist[ndist],ngood[ndist],fontsize=8)

    plt.tight_layout()
    # save figure or show
    if save:
        outfname = figdir+'/moveout_'+sta+'_heatmap_'+str(stack_method)+'_'+str(dist_inc)+'kmbin_'+comp+'.'+format
        plt.savefig(outfname, format=format, dpi=300)
        plt.close()
    else:
        plt.show()


#test functions
def plot_xcorr_moveout_wiggle(sfiles,sta,dtype,freq,ccomp=None,scale=1.0,lag=None,\
                            ylim=None,save=False,figsize=None,figdir=None,format='png',minsnr=None):
    '''
    display the moveout waveforms of the cross-correlation functions stacked for all time chuncks.
    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: source station name
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    ccomp: x-correlation component names, could be a string or a list of strings.
    scale: plot the waveforms with scaled amplitudes
    lag: lag times for displaying
    save: set True to save the figures (in pdf format)
    figdir: diresied directory to save the figure (if not provided, save to default dir)
    minsnr: mimumum SNR as a QC criterion, the SNR is computed as max(abs(trace))/mean(abs(trace)),
            without signal and noise windows.
    USAGE:
    ----------------------
    plot_xcorr_moveout_wiggle('temp.h5','Allstack0pws',0.1,0.2,'ZZ',200,True,'./temp')
    '''
    if not isinstance(freq[0],list):freq=[freq]
    freq=np.array(freq)
    figlabels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    if freq.shape[0]>9:
        raise ValueError('freq includes more than 9 (maximum allowed for now) elements!')
    elif freq.shape[0]==9:
        subplot=[3,3]
        figsize0=[14,7.5]
    elif freq.shape[0] >=7 and freq.shape[0] <=8:
        subplot=[2,4]
        figsize0=[18,10]
    elif freq.shape[0] >=5 and freq.shape[0] <=6:
        subplot=[2,3]
        figsize0=[14,7.5]
    elif freq.shape[0] ==4:
        subplot=[2,2]
        figsize0=[10,6]
    else:
        subplot=[1,freq.shape[0]]
        if freq.shape[0]==3:
            figsize0=[13,3]
        elif freq.shape[0]==2:
            figsize0=[8,3]
        else:
            figsize0=[4,3]
    if figsize is None:figsize=figsize0
    #
    qc=False
    if minsnr is not None:
        qc=True

    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    receiver = sta+'.h5'
    stack_method = dtype.split('_')[-1]
    if isinstance(ccomp,str):ccomp=[ccomp]

    # extract common variables
    try:
        ds    = pyasdf.ASDFDataSet(sfiles[0],mpi=False,mode='r')
        complist=ds.auxiliary_data[dtype].list()
        dt    = ds.auxiliary_data[dtype][complist[0]].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][complist[0]].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read"%sfiles[0]);sys.exit()
    if ccomp is None:ccomp=complist
    # lags for display
    if lag is None:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')
    tt = np.arange(-lag,lag+dt,dt)
    indx0= int(maxlag/dt) #zero time index
    indx1 = int((maxlag-lag)/dt)
    indx2 = indx1+2*int(lag/dt)+1

    # load cc and parameter matrix
    for ic in range(len(ccomp)):
        comp = ccomp[ic]

        data0 = np.zeros(shape=(len(sfiles),indx2-indx1),dtype=np.float32)
        dist = np.zeros(len(sfiles),dtype=np.float32)
        snrneg = np.zeros(len(sfiles),dtype=np.float32)
        snrpos = np.zeros(len(sfiles),dtype=np.float32)
        iflip = np.zeros(len(sfiles),dtype=np.int16)
        for ii in range(len(sfiles)):
            sfile = sfiles[ii]
            iflip[ii] = 0
            treceiver = sfile.split('_')[-1]
            if treceiver == receiver:
                iflip[ii] = 1

            ds = pyasdf.ASDFDataSet(sfile,mpi=False,mode='r')
            try:
                # load data to variables
                dist[ii] = ds.auxiliary_data[dtype][comp].parameters['dist']
                ngood= ds.auxiliary_data[dtype][comp].parameters['ngood']
                data0[ii]  = ds.auxiliary_data[dtype][comp].data[indx1:indx2]

                if qc:
                    #get the pseudo-SNR: maximum absolute amplitude/mean absolute amplitude.
                    dneg=ds.auxiliary_data[dtype][comp].data[indx1:indx0-1]
                    dpos=ds.auxiliary_data[dtype][comp].data[indx0+1:indx2]
                    snrneg[ii]=np.max(np.abs(dneg))/np.mean(np.abs(dneg))
                    snrpos[ii]=np.max(np.abs(dpos))/np.mean(np.abs(dpos))
#                     print([snrneg,snrpos])
            except Exception as e:
                print("continue! error working on %s "%sfile);
                print(e)
                continue

        mdist=np.max(dist)
        mindist=np.min(dist)
        plt.figure(figsize=figsize)
        for f in range(freq.shape[0]):
            freqmin=freq[f][0]
            freqmax=freq[f][1]
            plt.subplot(subplot[0],subplot[1],f+1)

            for i2 in range(data0.shape[0]):
                tdata = bandpass(data0[i2],freqmin,freqmax,1/dt,corners=4, zerophase=True)
                tdata /= np.max(tdata,axis=0)

                if ylim is not None:
                    if dist[i2]>ylim[1] or dist[i2]<ylim[0]:
                        continue
                if qc:
                    if np.max([snrneg[i2],snrpos[i2]]) < minsnr:
                        continue

                if iflip[i2]:
                    plt.plot(tt,scale*np.flip(tdata,axis=0)+dist[i2],'k',linewidth=0.8)
                else:
                    plt.plot(tt,scale*tdata+dist[i2],'k',linewidth=0.8)
            plt.title('%s %s filtered %5.3f-%5.3f Hz' % (figlabels[f],sta,freqmin,freqmax))
            plt.xlabel('time (s)')
            plt.ylabel('offset (km)')

            plt.xlim([-1.0*lag,lag])
            if ylim is None:
                ylim=[0.8*mindist,1.1*mdist]
            plt.plot([0,0],ylim,'b--',linewidth=1)

            plt.ylim(ylim)
            font = {'family': 'serif', 'color':  'red', 'weight': 'bold','size': 14}
            plt.text(lag*0.75,ylim[0]+0.07*(ylim[1]-ylim[0]),comp,fontdict=font,
                     bbox=dict(facecolor='white',edgecolor='none',alpha=0.85))
        plt.tight_layout()

        # save figure or show
        if save:
            if len(ccomp)>1:
                outfname = figdir+'/moveout_'+sta+'_wiggle_'+str(stack_method)+'_'+str(len(ccomp))+\
                            'ccomp_minsnr'+str(minsnr)+'.'+format
            else:
                outfname = figdir+'/moveout_'+sta+'_wiggle_'+str(stack_method)+'_'+ccomp[0]+\
                            '_minsnr'+str(minsnr)+'.'+format
            plt.savefig(outfname, format=format, dpi=300)
            plt.close()
        else:
            plt.show()

#get peak amplitudes
def get_xcorr_peakamplitudes(sfiles,sta,dtype,freq,ccomp=['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ'],
                              scale=1.0,lag=None,ylim=None,save=False,figdir=None,minsnr=None,
                        velocity=[1.0,5.0]):
    '''
    display the moveout waveforms of the cross-correlation functions stacked for all time chuncks.
    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: source station name
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freq: [freqmin,freqmax] as a filter.
    ccomp: xcorr components to extract.
    scale: scale of the waveforms in plotting the traces.
    lag: lag times for displaying
    save: set True to save the figures (in pdf format)
    figdir: diresied directory to save the figure (if not provided, save to default dir)
    minsnr: SNR cutoff. the SNR is computed with the given velocity range.
    velocity: velocity range for the main phase used to estimate the signal windows.

    RETURNS:
    -----------------------
    A dictionary that contains the following keys: source, receivers. Source is a dictionary containing
    the 'name', 'location' of the virtual source. Receivers is a dictionary
    containing the 'name' keys of an eight element array for the 'longitude', 'latitude', 'elevation' of
    each receiver and the 'distance', 'az','baz',peak_amplitude', 'peak_amplitude_time', 'snr' of the each receiver.

    USAGE:
    ----------------------
    get_xcorr_peakamplitudes('temp.h5','Allstack0pws',0.1,0.2,'ZZ',200,True,'./temp')
    '''

    #initialize out dictionary
    outdic=dict()
    outdic['source']=dict()
    outdic['source']['name']=sta
    outdic['source']['location']=np.empty((1,3,)) #three-element array of longitude, latitude, and elevation/depth
    outdic['cc_comp']=dict()
    qc=False
    if minsnr is not None:
        qc=True

    # open data for read
    if save:
        if figdir==None:print('no path selected! save figures in the default path')

    freqmin=freq[0]
    freqmax=freq[1]
    source = sta
    stack_method = dtype.split('_')[-1]
    typeofcomp=str(type(ccomp)).split("'")[1]
    ccomptemp=[]
    if typeofcomp=='str':
        ccomptemp.append(ccomp)
        ccomp=ccomptemp
    # print(ccomp)

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

    # extract common variables
    try:
        ds    = pyasdf.ASDFDataSet(sfiles[0],mpi=False,mode='r')
        dt    = ds.auxiliary_data[dtype][ccomp[0]].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][ccomp[0]].parameters['maxlag']
        iflip = 0
        treceiver_tmp = sfiles[0].split('_')[-1]
        treceiver=treceiver_tmp.split('.')[0]+'.'+treceiver_tmp.split('.')[1]
        if treceiver == source:
            iflip = 1
        if iflip:
            outdic['source']['location']=[ds.auxiliary_data[dtype][ccomp[0]].parameters['lonR'],
                                         ds.auxiliary_data[dtype][ccomp[0]].parameters['latR'],0.0]
        else:
            outdic['source']['location']=[ds.auxiliary_data[dtype][ccomp[0]].parameters['lonS'],
                                         ds.auxiliary_data[dtype][ccomp[0]].parameters['latS'],0.0]
    except Exception:
        print("exit! cannot open %s to read"%sfiles[0]);sys.exit()

    # lags for display
    if lag is None:lag=maxlag
    if lag>maxlag:raise ValueError('lag excceds maxlag!')
    tt = np.arange(-int(lag),int(lag)+dt,dt)
    indx0= int(maxlag/dt) #zero time index
    indx1 = int((maxlag-lag)/dt)
    indx2 = indx1+2*int(lag/dt)+1

    # load cc and parameter matrix
    plt.figure(figsize=figsize)
    for ic in range(len(ccomp)):
        comp = ccomp[ic]
        outdic['cc_comp'][comp]=dict() #keys of the 'receivers' dictionary are the station names, saving an eight-element array
        #for 'longitude', 'latitude', 'elevation','distance','az','baz', 'peak_amplitude', 'peak_amplitude_time', 'snr'.
        #

        plt.subplot(subplot[0],subplot[1],ic+1)
        mdist=0
        peakamp=np.empty((len(sfiles),2,))
        peakamp.fill(np.nan)
        peaktt=np.empty((len(sfiles),2,))
        peaktt.fill(np.nan)
        distall=np.empty((len(sfiles),))
        distall.fill(np.nan)
        outdict_tmp=dict()
        for ii in range(len(sfiles)):
            sfile = sfiles[ii]
            iflip = 0
            treceiver_tmp = sfile.split('_')[-1]
            treceiver=treceiver_tmp.split('.')[0]+'.'+treceiver_tmp.split('.')[1]
            tsource=sfile.split('_')[0]
            if treceiver == source:
                iflip = 1
                treceiver=tsource

            ds = pyasdf.ASDFDataSet(sfile,mpi=False,mode='r')
            try:
                # load data to variables
                dist = ds.auxiliary_data[dtype][comp].parameters['dist']
                distall[ii]=dist
                ngood= ds.auxiliary_data[dtype][comp].parameters['ngood']
                tdata  = ds.auxiliary_data[dtype][comp].data[indx1:indx2]

                #get key metadata parameters
                if iflip:
                    az=ds.auxiliary_data[dtype][comp].parameters['baz']
                    baz=ds.auxiliary_data[dtype][comp].parameters['azi']
                    lonR=ds.auxiliary_data[dtype][comp].parameters['lonS']
                    latR=ds.auxiliary_data[dtype][comp].parameters['latS']
                else:
                    az=ds.auxiliary_data[dtype][comp].parameters['azi']
                    baz=ds.auxiliary_data[dtype][comp].parameters['baz']
                    lonR=ds.auxiliary_data[dtype][comp].parameters['lonR']
                    latR=ds.auxiliary_data[dtype][comp].parameters['latR']
            except Exception as e:
                print("continue! error working on %s "%sfile);
                print(e)
                continue

            if ylim is not None:
                if dist>ylim[1] or dist<ylim[0]:
                    continue
            elif dist>mdist:
                mdist=dist

            #get signal window: start and end indices
            signal_neg=[indx0-int(dist/velocity[0]/dt)-indx1,indx0-int(dist/velocity[1]/dt)-indx1]
            signal_pos=[int(dist/velocity[1]/dt)+indx0-indx1,int(dist/velocity[0]/dt)+indx0-indx1]

            tdata = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)

            if dist/velocity[0] > lag:
                print('Signal window %6.1f is larger than the max lag %6.1f specified by the user' %
                     (dist/velocity[0],lag))
                continue


            if iflip:
                dtemp=np.flip(tdata,axis=0)
                dn=dtemp[signal_neg[0]:signal_neg[1]] #negative data section
                dp=dtemp[signal_pos[0]:signal_pos[1]] #positive dta section

                if qc:
                    #get the pseudo-SNR: maximum absolute amplitude/mean absolute amplitude.
                    snrneg=np.max(np.abs(dn))/np.mean(np.abs(tdata[0:indx0-1-indx1]))
                    snrpos=np.max(np.abs(dp))/np.mean(np.abs(tdata[indx0+1-indx1:-1]))
                    if np.nanmax([snrneg,snrpos]) < minsnr:
                        continue
                #get maximum index
                maxidx=[np.argmax(np.abs(dn)),np.argmax(np.abs(dp))]
                if maxidx[0] >0 and maxidx[0]<len(dn)-1:
                    peakamp[ii,0]=np.max(np.abs(dn))
                    peaktt[ii,0]=tt[maxidx[0]+signal_neg[0]]
                if maxidx[1] >0 and maxidx[1]<len(dn)-1:
                    peakamp[ii,1]=np.max(np.abs(dp))
                    peaktt[ii,1]=tt[maxidx[1]+signal_pos[0]]
                #normalize for plotting
                plt.plot(tt,dist + scale*dtemp/np.max(dtemp,axis=0),'k',linewidth=0.5)
            else:
                dn=tdata[signal_neg[0]:signal_neg[1]] #negative data section
                dp=tdata[signal_pos[0]:signal_pos[1]] #positive dta section

                if qc:
                    #get the pseudo-SNR: maximum absolute amplitude/mean absolute amplitude.
                    snrneg=np.max(np.abs(dn))/np.mean(np.abs(tdata[0:indx0-1-indx1]))
                    snrpos=np.max(np.abs(dp))/np.mean(np.abs(tdata[indx0+1-indx1:-1]))
                    if np.nanmax([snrneg,snrpos]) < minsnr:
                        continue
                #get maximum index
                maxidx=[np.argmax(np.abs(dn)),np.argmax(np.abs(dp))]
                if maxidx[0] >0 and maxidx[0]<len(dn)-1:
                    peakamp[ii,0]=np.max(np.abs(dn))
                    peaktt[ii,0]=tt[maxidx[0]+signal_neg[0]]
                if maxidx[1] >0 and maxidx[1]<len(dn)-1:
                    peakamp[ii,1]=np.max(np.abs(dp))
                    peaktt[ii,1]=tt[maxidx[1]+signal_pos[0]]

                plt.plot(tt,dist + scale*tdata/np.max(tdata,axis=0),'k',linewidth=0.5)


            #save to out dictionary
            #initialize the receiver element.
            outdic['cc_comp'][comp][treceiver]=dict()
            outdic['cc_comp'][comp][treceiver]['location']=[lonR,latR,0.0]
            outdic['cc_comp'][comp][treceiver]['az']=az
            outdic['cc_comp'][comp][treceiver]['baz']=baz
            outdic['cc_comp'][comp][treceiver]['dist']=dist
            outdic['cc_comp'][comp][treceiver]['peak_amplitude']=peakamp[ii,:]
            outdic['cc_comp'][comp][treceiver]['peak_amplitude_time']=peaktt[ii,:]

        #
        for jj in range(len(sfiles)):
            plt.plot(peaktt[jj,:],[distall[jj],distall[jj]],'.r',markersize=2)
        plt.xlim([-1.0*lag,lag])
        if ylim is None:
            ylim=[0.0,mdist]
        plt.plot([0,0],ylim,'b--',linewidth=1)

        #plot the bounding lines for signal windows.
        plt.plot([0, ylim[1]/velocity[1]],[0, ylim[1]],'c-',linewidth=0.5) #postive lag starting bound
        plt.plot([0, ylim[1]/velocity[0]],[0, ylim[1]],'c-',linewidth=0.5) #postive lag ending bound
        plt.plot([0, -ylim[1]/velocity[1]],[0, ylim[1]],'c-',linewidth=0.5) #negative lag starting bound
        plt.plot([0, -ylim[1]/velocity[0]],[0, ylim[1]],'c-',linewidth=0.5) #negative lag ending bound

        plt.ylim(ylim)
        font = {'family': 'serif', 'color':  'red', 'weight': 'bold','size': 10}
        plt.text(lag*0.75,ylim[0]+0.07*(ylim[1]-ylim[0]),comp,fontdict=font,
                 bbox=dict(facecolor='white',edgecolor='none',alpha=0.85))
        plt.title('%s filtered @%5.3f-%5.3f Hz' % (sta,freqmin,freqmax))
        plt.xlabel('time (s)')
        plt.ylabel('offset (km)')
    plt.tight_layout()

    # save figure or show
    if save:
        outfname = figdir+'/moveout_'+sta+'_wiggle_'+str(stack_method)+'_'+str(freqmin)+'_'+str(freqmax)+'Hz_'+str(len(ccomp))+'ccomp_minsnr'+str(minsnr)+'_peakamp.png'
        plt.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        plt.show()

    return outdic


#####
def plot_xcorr_amplitudes(dict_in,region,fignamebase=None,format='png',distance=None,
                   projection="M5i", xshift="6i",frame="af"):
    """
    This function plots the peak amplitude maps for both negative and positive lags,
    for each xcorr component pair. The map views plot amplitudes corrected for geometric
    spreading for surface waves. This function calls pygmt package for plotting. It also plots
    peak amplitudes v.s. distance, without correcting the amplitudes for geometric spreading.

    PARAMETERS:
    ----------------------------
    dict_in: dictionary containing peak amplitude information from one virtual source to all other receivers.
            This can be the output of get_xcorr_peakamplitudes().
    region: [minlon,maxlon,minlat,maxlat] for map view

    DEPENDENCIES:
    ----------------------------
    PyGMT: for plotting map view with geographical projections, which can be specified as arguments.

    """
    source=dict_in['source']['name']
    lonS,latS,eleS=dict_in['source']['location']
    mindatapoints=2 #at least two receivers having data. otherwise, skip.
    #
    if fignamebase is None:
        fignamebase = source

    cc_comp=list(dict_in['cc_comp'].keys())

    for ic in range(len(cc_comp)):
        comp = cc_comp[ic]
        receivers=list(dict_in['cc_comp'][comp].keys())
        lonR=[]
        latR=[]
        dist=[]
        peakamp_neg=[]
        peakamp_pos=[]
        peaktt_neg=[]
        peaktt_pos=[]

        for ir in range(len(receivers)):
            receiver=receivers[ir]
            dist0=dict_in['cc_comp'][comp][receiver]['dist']
            if distance is not None:
                if dist0<distance[0] or dist0>distance[1]:
                    continue
            dist.append(dist0)
            lonR.append(dict_in['cc_comp'][comp][receiver]['location'][0])
            latR.append(dict_in['cc_comp'][comp][receiver]['location'][1])
            peakamp_neg.append(np.array(dict_in['cc_comp'][comp][receiver]['peak_amplitude'])[0]*dist0)
            peakamp_pos.append(np.array(dict_in['cc_comp'][comp][receiver]['peak_amplitude'])[1]*dist0)
            peaktt_neg.append(np.array(dict_in['cc_comp'][comp][receiver]['peak_amplitude_time'])[0])
            peaktt_pos.append(np.array(dict_in['cc_comp'][comp][receiver]['peak_amplitude_time'])[1])

        if len(peakamp_neg) >= mindatapoints:
            #amplitudes map views
            panelstring=['(a) negative lag','(b) positive lag']
            fig = gmt.Figure()
            for d,dat in enumerate([peakamp_neg,peakamp_pos]):
                if d>0:
                    fig.shift_origin(xshift=xshift)
                fig.coast(region=region, projection=projection, frame=frame,land="gray",
                          shorelines=True,borders=["1/1p,black","2/0.5p,white"])
                fig.basemap(frame='+t"'+fignamebase.split('/')[-1]+'_'+comp+':'+panelstring[d]+'"')
                fig.plot(
                    x=lonS,
                    y=latS,
                    style="a0.5c",
                    color="black",
                )
                gmt.makecpt(cmap="viridis", series=[np.min(dat), np.max(dat)])
                fig.plot(
                    x=lonR,
                    y=latR,
                    color=dat,
                    cmap=True,
                    style="c0.3c",
                    pen="black",
                )
                fig.colorbar(frame='af+l"Amplitude"')

            figname=fignamebase+'_'+comp+'_peakamp_map.'+format
            fig.savefig(figname)
            print('plot was saved to: '+figname)

            #peak amplitude arrival times
            fig = gmt.Figure()
            for d,dat in enumerate([peaktt_neg,peaktt_pos]):
                if d>0:
                    fig.shift_origin(xshift=xshift)
                if d==0:
                    dat=np.multiply(dat,-1.0)
                fig.coast(region=region, projection=projection, frame=frame,land="gray",
                          shorelines=True,borders=["1/1p,black","2/0.5p,white"])
                fig.basemap(frame='+t"'+fignamebase.split('/')[-1]+'_'+comp+':'+panelstring[d]+'"')
                fig.plot(
                    x=lonS,
                    y=latS,
                    style="a0.5c",
                    color="black",
                )
                gmt.makecpt(cmap="viridis", series=[np.min(dat), np.max(dat)])
                fig.plot(
                    x=lonR,
                    y=latR,
                    color=dat,
                    cmap=True,
                    style="c0.3c",
                    pen="black",
                )
                fig.colorbar(frame='af+l"Arrival time (s)"')

            figname=fignamebase+'_'+comp+'_peaktt_map.'+format
            fig.savefig(figname)
            print('plot was saved to: '+figname)

            #plot amplitudes v.s. distance
            plt.figure(figsize=[8,6])
            plt.plot(dist,np.divide(peakamp_neg,dist),'ob',fillstyle='none',markersize=5,label='negative')
            plt.plot(dist,np.divide(peakamp_pos,dist),'or',markersize=5,label='positive')
            plt.title(fignamebase.split('/')[-1]+'_'+comp)
            plt.xlabel('distance (km)')
            plt.ylabel('Peak amplitudes')
            plt.legend(loc='upper right')
            figname=fignamebase+'_'+comp+'_peakamp_dist.'+format
            plt.savefig(figname)
            print('plot was saved to: '+figname)
        else:
            print('less than '+str(mindatapoints)+' receivers with data. Skip!')

##plot velocity clustering results.
def plot_vmodel_cluster(data,series=True,centers=True,map_plotly=True,map_gmt=True,
                        gmt_region=None,gmt_projection="M4i",gmt_frame_style="plain",
                        gmt_frame=None,cmap="turbo",figbase=None,save=False):
    """
    Plots velocity model clustering results. It may work better seperating each plot into different functions.
    It is hard to provide all controling parameters for all plots in one function.
    =======PARAMETERS=====
    data: the dictionary containing all clustering results.
    series=True,centers=True,map_plotly=True,map_gmt=True: determine which plot to produce. default is all True.
    gmt_region=None,gmt_projection="M4i",gmt_frame_style="plain",gmt_frame=None,cmap="turbo": parameters for gmt plots.
    figbase: fig name base when saving. Default: "vmodel_cluster+k"+str(ncluster)+"_"+method
    save: Default False.
    """
    depth,pred,df,tag,method=[data['depth'],data['pred'],
                            data['cluster_map'],data['tag'],data['method']]
    paralist=list(gmt_para.keys())

    if gmt_region is None:
        minlon=np.min(df['lon'])
        maxlon=np.max(df['lon'])
        minlat=np.min(df['lat'])
        maxlat=np.max(df['lat'])
        gmt_region=str(minlon)+"/"+str(maxlon)+"/"+str(minlat)+"/"+str(maxlat)

    ncluster=len(pred)
    if figbase is None:
        figbase="vmodel_cluster_k"+str(ncluster)+"_"+method
    ###########
    #### plotting clustered data/time series.
    ###########
    if series:
        if ncluster<4:
            plt.figure(figsize=(13, 4),facecolor='w')
        else:
            plt.figure(figsize=(13, 9),facecolor='w')
        for yi in range(ncluster):
            if ncluster<4:
                plt.subplot(1, ncluster, yi + 1)
            elif ncluster<9:
                plt.subplot(int(np.ceil(ncluster/2)), 2, yi + 1)
            elif ncluster < 16:
                plt.subplot(int(np.ceil(ncluster/3)), 3, yi + 1)
            elif ncluster < 21:
                plt.subplot(int(np.ceil(ncluster/4)), 4, yi + 1)
            elif ncluster < 26:
                plt.subplot(int(np.ceil(ncluster/5)), 5, yi + 1)
            else:
                plt.subplot(int(np.ceil(ncluster/6)), 6, yi + 1)
            for xx in pred[yi]:
                plt.plot(depth,xx, "k-", alpha=.2)
            plt.plot(depth,np.average(np.vstack(pred[yi]),axis=0),c="red")
            plt.title(f"Cluster {yi+1}: "+method)
            plt.xlabel('depth (km)')
            plt.ylabel('Vs (km/s)')
        plt.tight_layout()

        if savefig:
            plt.savefig(figbase+"_clusters.png",format="png")
            plt.close()
        else:
            plt.show()
    ###########
    #### plotting cluster centers.
    ###########
    if centers:
        plt.figure(figsize=(7,5),facecolor='w')
        plt.rcParams["axes.prop_cycle"] = get_color_cycle(cmap,ncluster)
        for yi in range(ncluster):
            plt.plot(depth,np.average(np.vstack(pred[yi]),axis=0),label="cluster "+str(yi+1))
        plt.legend()

        plt.xlabel("depth (km)")
        plt.ylabel("Vs (km/s)")
        if savefig:
            plt.savefig(figbase+"_clustercenters.png",format="png")
            plt.close()
        else:
            plt.show()
    #####################
    ######## plot map view of clusters.
    ####################
    # Create map using plotly
    if map_plotly:
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon='lon',
            color='cluster',
            size_max=13,
            zoom=3,
            width=900,
            height=800,
        )

        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "source": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"]
                }
            ]
        )
        if savefig:
            #
            fig.write_image(figbase+"_clustermap.png",format="png")
        else:
            fig.show()

    ##gmt map
    if map_gmt:
        title=tag+":"+method
        gridsize=np.diff(sorted(df['lat'].unique())[0:2])[0]
        ss=0.1*gridsize/0.25
        ss_str="r%3.2f/%3.2f"%(ss,1.3*ss)
        fig = gmt.Figure()
        gmt.config(MAP_FRAME_TYPE=gmt_frame_style)
        gmt.config(FORMAT_GEO_MAP="ddd")
        gmt.config(FONT_TITLE='14')
        gmt.makecpt(cmap=cmap, series=[df['cluster'].min()-0.5, df['cluster'].max()+0.5,1])
    #         grd=gmt.xyz2grd(region=region,x=df['lon'],y=df['lat'],z=df['cluster'],projection=projection,
    #                 spacing=str(gridsize)+"/"+str(gridsize))
        fig.coast(region=gmt_region, resolution="l",projection=gmt_projection,
                  water="white",frame=frame,land="grey")
        fig.basemap(frame='+t"'+title+'"')
        fig.plot(
            x=df['lon'],
            y=df['lat'],
            style=ss_str,
            color=df['cluster'],
            cmap=True,
        )
        fig.coast(region=gmt_region, resolution="l",projection=gmt_projection,
                  borders=["1/0.5p,gray,2/1p,gray"],shorelines=True)

        fig.colorbar(frame='af+l"Cluster"')

        if savefig:
            figname=figbase+"_clustermap_gmt.png"
            fig.savefig(figname)
        else:
            fig.show()
