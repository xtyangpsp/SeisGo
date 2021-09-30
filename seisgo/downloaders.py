import obspy
import os, glob
import time
import pyasdf
import sys
import pandas as pd
from obspy import read_events, UTCDateTime
from obspy.core import Trace, Stream
from obspy.clients.fdsn import Client
from seisgo import utils
from seisgo.utils import get_tracetag, save2asdf
import numpy as np

def get_sta_list(net_list, sta_list, chan_list, starttime, endtime, fname=None,\
                maxseischan=3,source='IRIS',lamin= None, lamax= None, \
                lomin= None, lomax= None, pressure_chan=None):
    """
    Function to get station list with given parameters. It is a wrapper of the obspy function "get_stations()".
    it is a practical applicaiton of get_stations() for mass downloading.

    ===PARAMETERS====
    net_list,sta_list,chan_list: targeted station information, wildcards are accepted.
    starttime,endtime: could be string (e.g., "2021_09_01_0_0_0") or obspy.UTCDateTime() objects.
    fname: save station list to file. Not save if None (default). Returns a Pandas dataframe of the list if None.
    maxseischan: default 3. this is used to avoid duplicates when, for example, both EH* and BH* channels are listed.
                The funciton will search by order in chan_list, and stop when maximum seismic channels are found.
                pressure channels are dealt with seperatedly.
    source: obspy Client source. Check obspy for the list of viable sources. Default is "IRIS"
    lamin, lamax,lomin,lomax: box region for the search. optional.
    pressure_chan: list of pressure channels. This is used when counting the number of seismic channels. pressure channels
                are not counted. we could avoid duplicates but still keep the channels we want.
    """
    sta = [];
    net = [];
    chan = [];
    location = [];
    lon = [];
    lat = [];
    elev = []
    if source == 'IRISPH5':
        client=Client(service_mappings={'station':'http://service.iris.edu/ph5ws/station/1'})
    else:
        client=Client(source)
    # time tags
    if not isinstance(starttime,obspy.core.utcdatetime.UTCDateTime) and starttime is not None:
        starttime_UTC = obspy.UTCDateTime(starttime)
    else:
        starttime_UTC = starttime
    if not isinstance(endtime,obspy.core.utcdatetime.UTCDateTime)  and endtime is not None:
        endtime_UTC   = obspy.UTCDateTime(endtime)
    else:
        endtime_UTC = endtime
    # loop through specified network, station and channel lists
    chanhistory = {}

    if isinstance(net_list,str):net_list=[net_list]
    if isinstance(sta_list,str):sta_list=[sta_list]
    if isinstance(chan_list,str):chan_list=[chan_list]
    if isinstance(pressure_chan,str):pressure_chan=[pressure_chan]

    for inet in net_list:
        for ista in sta_list:
            dataflag = 0
            for ichan in chan_list:
                # gather station info
                try:
                    inv = client.get_stations(network=inet, station=ista, channel=ichan, location='*', \
                                              starttime=starttime_UTC, endtime=endtime_UTC, minlatitude=lamin,
                                              maxlatitude=lamax, \
                                              minlongitude=lomin, maxlongitude=lomax, level='response')
                    dataflag = 1
                except Exception as e:
                    if ichan == chan_list[-1] and dataflag == 0:
                        print(inet + '.' + ista + '.' + ichan + ': Abort due to ' + str(e))
                        break
                    else:
                        continue

                for K in inv:
                    for tsta in K:
                        # if ichan in pressure_chan or len(chanhistory)<maxseischan:
                        ckeys = chanhistory.keys()
                        netsta = K.code + '.' + tsta.code
                        for c in tsta.get_contents()['channels']:
                            chan_this=c.split('.')[-1]
                            if netsta not in ckeys:
                                chanhistory[netsta] = []
                                sta.append(tsta.code)
                                net.append(K.code)
                                chan.append(chan_this)
                                lon.append(tsta.longitude)
                                lat.append(tsta.latitude)
                                elev.append(tsta.elevation)
                                # sometimes one station has many locations and here we only get the first location
                                if tsta[0].location_code:
                                    location.append(tsta[0].location_code)
                                else:
                                    location.append('*')
                                if pressure_chan != None:
                                    if ichan not in pressure_chan:
                                        chanhistory[netsta].append(chan_this)
                                else:
                                    chanhistory[netsta].append(chan_this)
                            elif len(chanhistory[netsta]) < maxseischan or (pressure_chan !=None and chan_this in pressure_chan):
                                sta.append(tsta.code)
                                net.append(K.code)
                                chan.append(chan_this)
                                lon.append(tsta.longitude)
                                lat.append(tsta.latitude)
                                elev.append(tsta.elevation)
                                # sometimes one station has many locations and here we only get the first location
                                if tsta[0].location_code:
                                    location.append(tsta[0].location_code)
                                else:
                                    location.append('*')
                                if pressure_chan != None:
                                    if ichan not in pressure_chan:
                                        chanhistory[netsta].append(chan_this)
                                else:
                                    chanhistory[netsta].append(chan_this)

    # output station list
    dict = {'network': net, 'station': sta, 'channel': chan, 'latitude': lat, 'longitude': lon, 'elevation': elev}
    locs0 = pd.DataFrame(dict)
    locs=locs0.drop_duplicates(ignore_index=True) #remove duplicate inputs/stations
    if fname is not None:
        fsplit=fname.split('/')[:-1]
        fdir=os.path.join(*fsplit)
        if not os.path.isdir(fdir):os.makedirs(fdir)
        locs.to_csv(fname, index=False)

    return locs

#
def getdata(net,sta,starttime,endtime,chan,source='IRIS',samp_freq=None,
            rmresp=True,rmresp_output='VEL',pre_filt=None,debug=False,
            sacheader=False,getstainv=False,verbose=False):
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
    debug : bool
            Plot raw waveforms before and after preprocessing.
    sacheader : bool
            Key sacheader information in a dictionary using the SAC header naming convention.
    """

    if source == 'IRISPH5':
        client=Client(service_mappings={'dataselect':'http://service.iris.edu/ph5ws/dataselect/1',
                        'station':'http://service.iris.edu/ph5ws/station/1'})
    else:
        client=Client(source)
    tr = None
    sac=dict() #place holder to save some sac headers.
    #check arguments
    if rmresp:
        if pre_filt is None:
            pre_filt = set_filter(samp_freq, 0.001)
            print("getdata(): pre_filt is not specified. Use 0.001-0.5*samp_freq as "+\
                    "the filter range when removing response.")
    """
    a. Downloading
    """
    if sacheader or getstainv or rmresp:
        inv = client.get_stations(network=net,station=sta,
                        channel=chan,location="*",starttime=starttime,endtime=endtime,
                        level='response')
        if sacheader:
            tempsta,tempnet,stlo, stla,stel,temploc=utils.sta_info_from_inv(inv)
            sac['knetwk']=tempnet
            sac['kstnm']=tempsta
            sac['stlo']=stlo
            sac['stla']=stla
            sac['stel']=stel
            sac['kcmpnm']=chan
            sac['khole']=temploc

    # pressure channel
    tr=client.get_waveforms(network=net,station=sta,
                    channel=chan,location="*",starttime=starttime,endtime=endtime)
#     trP[0].detrend()
    if verbose:print('number of segments downloaded: '+str(len(tr)))

    tr[0].stats['sac']=sac

    if verbose:print("station "+net+"."+sta+" --> seismic channel: "+chan)

    if debug:
        year = tr[0].stats.starttime.year
        julday = tr[0].stats.starttime.julday
        hour = tr[0].stats.starttime.hour
        mnt = tr[0].stats.starttime.minute
        sec = tr[0].stats.starttime.second
        tstamp = str(year) + '.' + str(julday)+'T'+str(hour)+'-'+str(mnt)+'-'+str(sec)
        trlabels=[net+"."+sta+"."+tr[0].stats.channel]
    """
    b. Resampling
    """
    if samp_freq is not None:
        sps=int(tr[0].stats.sampling_rate)
        delta = tr[0].stats.delta
        #assume pressure and vertical channels have the same sampling rat
        # make downsampling if needed
        if sps > samp_freq:
            if verbose:print("  downsamping from "+str(sps)+" to "+str(samp_freq))
            for r in tr:
                if np.sum(np.isnan(r.data))>0:
                    raise(Exception('NaN found in trace'))
                else:
                    r.interpolate(samp_freq,method='weighted_average_slopes')
                    # when starttimes are between sampling points
                    fric = r.stats.starttime.microsecond%(delta*1E6)
                    if fric>1E-4:
                        r.data = utils.segment_interpolate(np.float32(r.data),float(fric/(delta*1E6)))
                        #--reset the time to remove the discrepancy---
                        r.stats.starttime-=(fric*1E-6)
                # print('new sampling rate:'+str(tr.stats.sampling_rate))

    """
    c. Plot raw data before removing responses.
    """
    if debug:
        utils.plot_trace([tr],size=(12,3),title=trlabels,freq=[0.005,0.1],ylabels=["raw"],
                        outfile=net+"."+sta+"_"+tstamp+"_raw.png")

    """
    d. Remove responses
    """
    if rmresp:
        for r in tr:
            if np.sum(np.isnan(r.data))>0:
                raise(Exception('NaN found in trace'))
            else:
                try:
                    if verbose:print('  removing response using inv for '+net+"."+sta+"."+r.stats.channel)
                    r.attach_response(inv)
                    r.remove_response(output=rmresp_output,pre_filt=pre_filt,
                                              water_level=60,zero_mean=True,plot=False)
                except Exception as e:
                    print(e)
                    r = []
    for r in tr:
        r.detrend('demean')
        r.detrend('linear')
        r.taper(0.005)

    if len(tr.get_gaps())>0:
        if verbose:print('merging segments with gaps')
        tr.merge(fill_value=0)
    tr=tr[0]
    """
    e. Plot raw data after removing responses.
    """
    if debug:
        plot_trace([tr],size=(12,3),title=trlabels,freq=[0.005,0.1],ylabels=[rmresp_output],
                   outfile=net+"."+sta+"_"+tstamp+"_raw_rmresp.png")

    #
    if getstainv:return tr,inv
    else: return tr

def cleantargetdir(rawdatadir):
    if not os.path.isdir(rawdatadir):
        print("target directory not found")
    dfiles1 = glob.glob(os.path.join(rawdatadir, '*.h5'))
    if len(dfiles1) > 0:
        print('Cleaning up raw data directory before downloading ...')
        for df1 in dfiles1:
            os.remove(df1)

def in_directory(fname, sta, net, tag):
    # Returns False if not already in directory
    if not os.path.isfile(fname):
        return False
    else:
        # Checks if file w/ same channel in directory
        with pyasdf.ASDFDataSet(fname, mpi=False, mode='r') as rds:
            read_tnames = rds.waveforms.list()
            read_tags= [sta.get_waveform_tags() for sta in rds.waveforms]
            read_records = {read_tnames[i]: read_tags[i][0] for i in range(len(read_tnames))}
            tname = net + '.' + sta
            # print(read_records)
            # if tname in read_records and tag == tname[tag]:
            # if {tname:tag}.items() <= read_records.items():
            if ((tname, tag) in read_records.items()):
                return True
            else:
                return False

def set_filter(samp_freq, pfreqmin,pfreqmax=None):
    if pfreqmax is None:
        pfreqmax = samp_freq / 2
    f1 = 0.95 * pfreqmin;
    f2 = pfreqmin
    if 1.05 * pfreqmax > 0.48 * samp_freq:
        f3 = 0.45 * samp_freq
        f4 = 0.48 * samp_freq
    else:
        f3 = pfreqmax
        f4 = 1.05 * pfreqmax
    pre_filt = [f1, f2, f3, f4]
    return pre_filt

def download(starttime, endtime, stationinfo=None, network=None, station=None,channel=None,
                source='IRIS',rawdatadir=None,sacheader=False, getstainv=True, max_tries=10,
                savetofile=False,pressure_chan=None,samp_freq=None,freqmin=0.001,freqmax=None,
                rmresp=True, rmresp_out='DISP',respdir=None,qc=True,event=None,verbose=False):
    """
    starttime, endtime: timing duration for the download.
    stationinfo:
            This could be a CSV file name, a Pandas DataFrame, or a dictionary. All data should
            include network, station, (optional) channel, (optional) location. Note, if stationinfo is NOT None,
            the station information in stationinfo will overwrite the network,station, and channel,
            if specified individually.
    network,station,channel: those will be ignored if stationinfo is NOT None.
    source: obspy source name. default is "IRIS"
    rawdatadir: directory to save the downloaded data. only needed when savetofile is True.
    sacheader: default False. get sacheader or not. this is useful when you want to save data to sac later or check some metadata.
    getstainv: default True. get station inventory. If true, funciton returns two variables: trlist,invlist
    max_tries: defautl 10. will try multiple times incase there is server connection issue.
    savetofile: default is False. Save to ASDF file if True.
    pressure_chan: this is needed when downloading OBS data with pressure channel. this influences the output of
            response removal. Use "VEL" for pressure channels to get comparable amplitudes as "DISP" for seismic channels.
    samp_freq: downsample data to target sampling rate.
    freqmin,freqmax: frequency range in removing response. default freqmin is 0.001.
    rmresp: default True. remove response.
    rmresp_out: default "DISP". use "VEL" for pressure chan.
    respdir: directory containing response information. currently NOT used in this function.
    qc: When True, does QC to clean up the trace. default True.
    event: ObsPy Event object for earthquake data. the event qml data will be saved into ASDF.
    =============RETURNS============
    trlist: Obspy Stream containing all traces. Note that when savetofile is True, the return will be an empty Stream.
    sta_inv_list: inventory list of the stations. Empty when getstainv is False.
    """
    ######################read in station information first.
    if stationinfo is not None:
        if isinstance(stationinfo,str): #assume stationinfo is a CSV file containing network,station, channel columns
            if not os.path.isfile(stationinfo):raise IOError('file %s not exist! double check!' % stationinfo)
            # read station info from list
            locs = pd.read_csv(stationinfo)
        elif isinstance(stationinfo,pd.DataFrame) or isinstance(stationinfo,dict):
            locs = stationinfo

        network  = list(locs['network'])
        station  = list(locs['station'])
        try:
            channel = list(locs['channel'])
        except Exception as e:
            channel = ['*']*len(station)
        # location info: useful for some occasion
        try:
            location = list(locs['location'])
        except Exception as e:
            location = ['*']*len(station)
    elif None in [network,station,channel]:
        raise IOError('Must specify network, station, and channel if stationinfo is None!')
    ##########################################
    if rawdatadir is not None:
        savetofile=True
        if not os.path.isdir(rawdatadir):os.makedirs(rawdatadir)

    if event is None:
        type = "continuous"
    else:
        if event.event_type == "earthquake":
            type = "earthquake"
        else:
            type = "other event"

    # if user passes a string instead of a list, make a list of one string
    # if station is None: station = ['*']*len(network)
    if isinstance(station, str): station = [station]
    # if channel is None: channel = ['*']*len(station)
    if isinstance(channel, str): channel = [channel]
    if isinstance(network, str): network = [network]
    if isinstance(pressure_chan,str):pressure_chan=[pressure_chan]

    pre_filt = set_filter(samp_freq, freqmin,freqmax)

    # dtlist = utils.split_datetimestr(starttime, endtime, inc_hours)
    # print(dtlist)
    # for idt in range(len(dtlist) - 1):
    if isinstance(starttime,str):
        sdatetime = obspy.UTCDateTime(starttime)
    else:
        sdatetime = starttime
    if isinstance(endtime,str):
        edatetime = obspy.UTCDateTime(endtime)
    else:
        edatetime = endtime

    if savetofile:
        if type == "continuous":
            fname = os.path.join(rawdatadir,
                                 str(sdatetime).replace(':', '-') + 'T' + str(edatetime).replace(':', '-') + '.h5')
        elif type == 'earthquake':
            fname = os.path.join(rawdatadir, str(event.origins[0].time) + "_M" + str(event.magnitudes[0].mag) + '.h5')
        else:
            fname = os.path.join(rawdatadir, str(event.origins[0].time) + '.h5')

    """
    Start downloading.
    """
    trlist=[]
    sta_inv_list=[]
    #loop through all stations.
    for i in range(len(station)):
        inet=network[i]
        ista=station[i]
        ichan=channel[i]

        for nt in range(max_tries):
            if verbose:print(inet+'.'+ista + '.' + ichan + '  downloading ... try ' + str(nt + 1))

            t0 = time.time()

            rmresp_out_tmp=rmresp_out
            if pressure_chan is not None and ichan in pressure_chan:
                rmresp_out_tmp='VEL'
            try:
                output = getdata(inet, ista, sdatetime, edatetime, chan=ichan, source=source,
                                        samp_freq=samp_freq, rmresp=rmresp, rmresp_output=rmresp_out_tmp,
                                       pre_filt=pre_filt, sacheader=sacheader, getstainv=getstainv)
            except Exception as e:
                print(e, 'for', ista)
                time.sleep(1)  # sleep for 1 second before next try.
                continue
            if getstainv == True or sacheader == True:
                sta_inv = output[1]
                tr = output[0]
            else:
                tr = output
                sta_inv = None

            ta = time.time() - t0
            if verbose:print('  downloaded ' + inet+"." + ista + "." + ichan + " in " + str(ta) + " seconds.")
            tag = get_tracetag(tr)
            chan = tr.stats.channel

            """
            Add cleanup
            """
            if qc:
                if chan[-1].lower() == 'h': tag_type= "trP"; hasPressure=True
                elif chan[-1].lower() == '1' or chan[-1].lower() == 'e':tag_type="tr1"
                elif chan[-1].lower() == '2' or chan[-1].lower() == 'n':tag_type="tr2"
                elif chan[-1].lower() == 'z':tag_type="trZ"
                else: print('  No seismic channels found. Drop the station: '+ista); break

                #sanity check.
                badtrace=False
                if not isinstance(tr, Trace):
                    print("  "+str(tr)+" is not a Trace object. "+ista)
                    badtrace=True
                    break
                elif np.sum(np.isnan(tr.data))>0:
                    print('  NaN found in trace: '+str(tr)+". "+ista)
                    badtrace=True
                    break
                elif np.count_nonzero(tr.data) < 1:
                    print('  All zeros in trace: '+str(tr)+". "+ista)
                    badtrace=True
                    break

                if badtrace:
                    if savetofile:
                        in_dir = in_directory(fname, ista, inet, tag)
                        """
                        Save to ASDF file.
                        """
                        if in_dir == True:
                            print(inet + '.' + ista + '.' + chan + '  exists. Continue to next!')
                            break
                        else:
                            print("  Encountered bad trace for " + ista + ". Save as is without processing!")
                            save2asdf(fname,tr,tag,sta_inv=sta_inv, event=event)
                            break
                    else:
                        trlist.append(tr)
                        sta_inv_list.append(sta_inv)
                        break
                else:
                    if savetofile:
                        in_dir = in_directory(fname, ista, inet, tag)
                        """
                        Save to ASDF file.
                        """
                        if in_dir == True:
                            print(inet + '.' + ista + "." + chan + '  exists. Continue to next!')
                            break
                        else:
                            print(" Saving data for", inet + '.' + ista  + '.' + chan )
                            save2asdf(fname,[tr],[tag],sta_inv=sta_inv, event=event)
                            break
                    else:
                        trlist.append(tr)
                        sta_inv_list.append(sta_inv)
                        break
            else:  #not QC
                if savetofile:
                    print(" Saving data for", inet + '.' + ista  + '.' + chan )
                    save2asdf(fname,[tr],[tag],sta_inv=sta_inv, event=event)
                    break
                else:
                    trlist.append(tr)
                    sta_inv_list.append(sta_inv)
                    break

    return Stream(trlist),sta_inv_list

def read_data(files,rm_resp='no',respdir='.',freqmin=None,freqmax=None,rm_resp_out='VEL',
                stainv=True,samp_freq=None):
    """
    Wrapper to read local data and (optionally) remove instrument response, and gather station inventory.

    ==== PARAMETERS ====
    files: local data file or a list of files.
    rm_resp: 'no'[default], 'spectrum', 'RESP', or 'polozeros'.
    respdir: directory for response files, default is '.'
    freqmin: minimum frequency in removing responses. default is 0.001
    freqmax: maximum frequency in removing responses. default is 0.499*sample_rate
    rm_resp_out: the ouptut unit for removing response, default is 'VEL', could be "DIS"
    stainv: get station inventory or not, default is True.

    ==== RETURNS ====
    tr_all: all traces as a list of obspy Trace objects.
    inv_all: all inventory objects, if stainv flag is True. Otherwise, only tr_all will be returned.
    """
    if isinstance(files,str):files=[files]
    tr_all=[]
    inv_all=[]
    for f in files:
        tr=obspy.read(f, debug_headers=True)

        fs=tr[0].stats.sampling_rate
        if freqmin is None: freqmin=0.001
        if freqmax is None: freqmax=0.499*fs
        pre_filt = set_filter(fs, freqmin,freqmax)
        net=tr[0].stats.network
        sta=tr[0].stats.station
        chan=tr[0].stats.channel
        netstachan=net+"."+sta+"."+chan
        date_info = {'starttime':tr[0].stats.starttime,'endtime':tr[0].stats.endtime}
        if rm_resp != 'no':
            if rm_resp == 'spectrum':
                print('remove response using spectrum')
                specfile = glob.glob(os.path.join(respdir,'*'+netstachan+'*'))
                if len(specfile)==0:
                    raise ValueError('no response sepctrum found for %s' % netstachan)
                tr = utils.resp_spectrum(tr,specfile[0],fs,pre_filt)

            elif rm_resp == 'RESP':
                print('remove response using RESP files')
                resp = glob.glob(os.path.join(respdir,'RESP.'+netstachan+'*'))
                print(resp)
                if len(resp)==0:
                    raise ValueError('no RESP files found for %s' % netstachan)
                seedresp = {'filename':resp[0],'date':date_info['starttime'],'units':rm_resp_out}
                tr.simulate(paz_remove=None,pre_filt=pre_filt,seedresp=seedresp)

            elif rm_resp == 'polozeros':
                print('remove response using polos and zeros')
                paz_sts = glob.glob(os.path.join(respdir,'*'+netstachan+'*'))
                if len(paz_sts)==0:
                    raise ValueError('no polozeros found for %s' % netstachan)
                tr.simulate(paz_remove=paz_sts[0],pre_filt=pre_filt)
            else:
                raise ValueError('no such option for rm_resp! please double check!')

        if samp_freq is not None:
            sps=int(tr[0].stats.sampling_rate)
            delta = tr[0].stats.delta
            #assume pressure and vertical channels have the same sampling rat
            # make downsampling if needed
            if sps > samp_freq:
                print("  downsamping from "+str(sps)+" to "+str(samp_freq))
                if np.sum(np.isnan(tr[0].data))>0:
                    raise(Exception('NaN found in trace'))
                else:
                    tr[0].interpolate(samp_freq,method='weighted_average_slopes')
                    # when starttimes are between sampling points
                    fric = tr[0].stats.starttime.microsecond%(delta*1E6)
                    if fric>1E-4:
                        tr[0].data = utils.segment_interpolate(np.float32(tr[0].data),float(fric/(delta*1E6)))
                        #--reset the time to remove the discrepancy---
                        tr[0].stats.starttime-=(fric*1E-6)
        tr[0].detrend('demean')
        tr[0].detrend('linear')
        tr[0].taper(0.005)
        tr_all.append(tr[0])

        if stainv:
            inv_all.append(utils.stats2inv(tr[0].stats))
    #
    #return
    if stainv:
        return Stream(tr_all),inv_all
    else:
        return Stream(tr_all)


def get_events(start,end,minlon=-180,maxlon=180,minlat=-90,maxlat=90,minmag=0,maxmag=10,
                    magstep=1.0,source="USGS",v=False):
    """
    Download event catalog within a box from USGS or ISC catalogs.
    """
    #elist is a list of panda dataframes
    t0=time.time()

    #Download the catalog by magnitudes
    maglist=np.arange(minmag,maxmag+0.5*magstep,magstep)
    events=[]
    for i in range(len(maglist)-1):
        if i>0:
            minM=str(maglist[i]+0.0001) #to avoid duplicates with intersecting magnitude thresholds
        else:
            minM=str(maglist[i])
        maxM=str(maglist[i+1])
        if v: print(minM,maxM)
        if(source=="ISC"):
            quake_url="http://isc-mirror.iris.washington.edu/fdsnws/event/1/query?starttime="+\
            start+"&endtime="+end+"&minlatitude="+str(minlat)+"&maxlatitude="+str(maxlat)+"&minlongitude="+\
            str(minlon)+"&maxlongitude="+str(maxlon)+"&minmagnitude="+minM+"&maxmagnitude="+maxM+""
        elif(source=="USGS"):
            quake_url="https://earthquake.usgs.gov/fdsnws/event/1/query?format=xml&starttime="+\
            start+"&endtime="+end+"&minmagnitude="+minM+"&maxmagnitude="+maxM+"&minlatitude="+\
            str(minlat)+"&maxlatitude="+str(maxlat)+"&minlongitude="+str(minlon)+"&maxlongitude="+str(maxlon)+""

        try:
            event = read_events(quake_url)
            if v: print("Catalog: " + str(event.count()))
            events.append(event)
        except Exception as e:
            print(e)

    catalog = events[0]
    for i in np.arange(1,len(events)):
            catalog.extend(events[i])
    #Print the data frame, display total, and total time to complete task
    if v: print("It took "+str(time.time()-t0)+" seconds to download the catalog")

    return catalog

def get_event_waveforms(event,stainfo,window=150,offset=-50,arrival_type='first',maxtry=3,source='IRIS',rawdatadir=None,
                        sacheader=False, getstainv=False, savetofile=False,pressure_chan=None,
                        samp_freq=None,freqmin=0.001,freqmax=None,rmresp=True, rmresp_out='DISP',
                        respdir=None,qc=True):
    """
    Download seismic event waveforms. This function downloads one earthquake for one station.

    ====PARAMETERS====
    event: Obspy Event object.
    stainfo: Pandas dateframe contaning station information.
    window: length of the waveform in seconds
    offset: offset in seconds relative to the specified arrival.
    arrival_type: default is first arrival time, could be any specified phase string.
    other options are the same as download().

    savetofile: return Obspy Stream if False.
    """
    event_t = event.origins[0].time
    event_long = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    depth_km = event.origins[0].depth / 1000
    network=list(stainfo.network)
    station=list(stainfo.station)
    try:
        channel = list(stainfo.channel)
    except Exception as e:
        channel = ['*']*len(station)
    trall=[]
    invall=[]
    # Download for each station
    for i in range(len(network)):
        sta_lat = stainfo.latitude[i]
        sta_long = stainfo.longitude[i]

        travel_time = utils.get_tt(event_lat, event_long, sta_lat, sta_long, depth_km,type='first')[0]
        sta_start = event_t + travel_time + offset
        sta_end = sta_start + window
        if getstainv:
            tr,inv=download(sta_start,sta_end,network=network[i],station=station[i],channel=channel[i],
                                source=source,rawdatadir=rawdatadir,sacheader=sacheader,
                                getstainv=getstainv, max_tries=maxtry,savetofile=savetofile,
                                pressure_chan=pressure_chan,samp_freq=samp_freq,freqmin=freqmin,
                                freqmax=freqmax,rmresp=rmresp, rmresp_out=rmresp_out,
                                respdir=respdir,qc=qc,event=event)

            if not savetofile:
                if len(tr)>0:
                    trall.append(tr[0])
                    invall.append(inv)
        else:
            tr=download(sta_start,sta_end,network=network[i],station=station[i],channel=channel[i],
                                source=source,rawdatadir=rawdatadir,sacheader=sacheader,
                                getstainv=getstainv, max_tries=maxtry,savetofile=savetofile,
                                pressure_chan=pressure_chan,samp_freq=samp_freq,freqmin=freqmin,
                                freqmax=freqmax,rmresp=rmresp, rmresp_out=rmresp_out,
                                respdir=respdir,qc=qc,event=event)

            if not savetofile:
                if len(tr)>0:
                    trall.append(tr[0])

    ###
    ###
    if not savetofile:
        if getstainv:
            return Stream(trall),invall
        else:
            return Stream(trall)
