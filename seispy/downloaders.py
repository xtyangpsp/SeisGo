import obspy
import os, glob
import time
import pyasdf
import sys
import pandas as pd
from obspy.core import Trace, Stream
from obspy.clients.fdsn import Client
from seispy import utils
from seispy.utils import get_tracetag, save2asdf
import numpy as np

def get_sta_list(net_list, sta_list, chan_list, starttime, endtime, fname=None,maxseischan=3,source='IRIS',
                lamin= None, lamax= None, lomin= None, lomax= None, pressure_chan=None):
    """
    Function to get station list with given parameters. It is a wrapper of the obspy function "get_stations()".
    it is a practical applicaiton of get_stations() for mass downloading.
    """
    sta = [];
    net = [];
    chan = [];
    location = [];
    lon = [];
    lat = [];
    elev = []

    client=Client(source)
    # time tags
    starttime_UTC = obspy.UTCDateTime(starttime)
    endtime_UTC   = obspy.UTCDateTime(endtime)
    # loop through specified network, station and channel lists
    chanhistory = {}

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
                            elif len(chanhistory[netsta]) < maxseischan or chan_this in pressure_chan:
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

    # output station list
    dict = {'network': net, 'station': sta, 'channel': chan, 'latitude': lat, 'longitude': lon, 'elevation': elev}
    locs = pd.DataFrame(dict)
    if fname is not None:
        locs.to_csv(fname, index=False)

    return locs

#
def getdata(net,sta,starttime,endtime,chan,source='IRIS',samp_freq=None,
            rmresp=True,rmresp_output='VEL',pre_filt=None,debug=False,
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
    debug : bool
            Plot raw waveforms before and after preprocessing.
    sacheader : bool
            Key sacheader information in a dictionary using the SAC header naming convention.
    """
    client = Client(source)
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
    if sacheader or getstainv:
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
                    channel=chan,location="*",starttime=starttime,endtime=endtime,attach_response=True)
#     trP[0].detrend()
    print('number of segments downloaded: '+str(len(tr)))

    tr[0].stats['sac']=sac

    print("station "+net+"."+sta+" --> seismic channel: "+chan)

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
            print("  downsamping from "+str(sps)+" to "+str(samp_freq))
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
        plot_trace([tr],size=(12,3),title=trlabels,freq=[0.005,0.1],ylabels=["raw"],
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
                    print('  removing response using inv for '+net+"."+sta+"."+r.stats.channel)
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
        print('merging segments with gaps')
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
                rmresp=True, rmresp_out='DISP',respdir=None,qc=True,event=None):
    """
    starttime, endtime: timing duration for the download.
    stationinfo:
            This could be a CSV file name, a Pandas DataFrame, or a dictionary. All data should
            include network, station, (optional) channel, (optional) location. Note, if stationinfo is NOT None,
            the station information in stationinfo will overwrite the network,station, and channel,
            if specified individually.
    network,station,channel: those will be ignored if stationinfo is NOT None.
    qc: When True, does QC to clean up the trace.
    event: ObsPy Event object for earthquake data
    =============RETURNS============
    trlist: Obspy Stream containing all traces. Note that when savetofile is True, the return will be an empty Stream.
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
        type="continuous"
    else:
        type="earthquake"
    # if user passes a string instead of a list, make a list of one string
    # if station is None: station = ['*']*len(network)
    if isinstance(station, str): station = [station]
    # if channel is None: channel = ['*']*len(station)
    if isinstance(channel, str): channel = [channel]
    if isinstance(network, str): network = [network]

    pre_filt = set_filter(samp_freq, freqmin,freqmax)

    # dtlist = utils.split_datetimestr(starttime, endtime, inc_hours)
    # print(dtlist)
    # for idt in range(len(dtlist) - 1):
    sdatetime = obspy.UTCDateTime(starttime)
    edatetime = obspy.UTCDateTime(endtime)
    if savetofile:
        if type == "continuous":
            fname = os.path.join(rawdatadir,
                                 str(sdatetime).replace(':', '-') + 'T' + str(edatetime).replace(':', '-') + '.h5')
        elif type == 'earthquake':
            fname = os.path.join(rawdatadir, str(eq_source.origins[0].time) + "_M" + str(eq_source.magnitudes[0].mag) + '.h5')

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
            print(inet+'.'+ista + '.' + ichan + '  downloading ... try ' + str(nt + 1))

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
            print('  downloaded ' + inet+"." + ista + "." + ichan + " in " + str(ta) + " seconds.")
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
                            save2asdf(fname,tr,tag,sta_inv=sta_inv)
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
                            save2asdf(fname,[tr],[tag],sta_inv=sta_inv)
                            break
                    else:
                        trlist.append(tr)
                        sta_inv_list.append(sta_inv)
                        break
            else:  #not QC
                if savetofile:
                    print(" Saving data for", inet + '.' + ista  + '.' + chan )
                    save2asdf(fname,[tr],[tag],sta_inv=sta_inv)
                    break
                else:
                    trlist.append(tr)
                    sta_inv_list.append(sta_inv)
                    break

    return trlist,sta_inv_list
