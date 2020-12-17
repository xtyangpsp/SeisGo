from seispy import utils
import obspy
import os, glob
import time
import pyasdf
from seispy.utils import get_tracetag, save2asdf
import sys
import pandas as pd
from obspy.core import Trace
import numpy as np

def get_sta_list(fname, client, net_list, sta_list, chan_list, starttime, endtime, maxseischan, prepro_para, lamin= None, lamax= None, lomin= None, lomax= None, pressure_chan=None):

    sta = [];
    net = [];
    chan = [];
    location = [];
    lon = [];
    lat = [];
    elev = []

    # loop through specified network, station and channel lists
    chanhistory = {}

    for inet in net_list:
        for ista in sta_list:
            dataflag = 0
            for ichan in chan_list:
                # gather station info
                try:
                    inv = client.get_stations(network=inet, station=ista, channel=ichan, location='*', \
                                              starttime=starttime, endtime=endtime, minlatitude=lamin,
                                              maxlatitude=lamax, \
                                              minlongitude=lomin, maxlongitude=lomax, level='response')
                    dataflag = 1
                except Exception as e:
                    if ichan == chan_list[-1] and dataflag == 0:
                        print(inet + '.' + ista + '.' + ichan + ': Abort at L137 in S0A due to ' + str(e))
                        sys.exit()
                    else:
                        continue

                for K in inv:
                    for tsta in K:
                        # if ichan in pressure_chan or len(chanhistory)<maxseischan:
                        ckeys = chanhistory.keys()
                        netsta = K.code + '.' + tsta.code
                        print(K.code + '.' + tsta.code + '.' + ichan)
                        if netsta not in ckeys:
                            chanhistory[netsta] = []
                            sta.append(tsta.code)
                            net.append(K.code)
                            chan.append(ichan)
                            lon.append(tsta.longitude)
                            lat.append(tsta.latitude)
                            elev.append(tsta.elevation)
                            # sometimes one station has many locations and here we only get the first location
                            if tsta[0].location_code:
                                location.append(tsta[0].location_code)
                            else:
                                location.append('*')
                            if ichan not in pressure_chan:
                                chanhistory[netsta].append(ichan)
                        elif len(chanhistory[netsta]) < maxseischan or ichan in pressure_chan:
                            sta.append(tsta.code)
                            net.append(K.code)
                            chan.append(ichan)
                            lon.append(tsta.longitude)
                            lat.append(tsta.latitude)
                            elev.append(tsta.elevation)
                            # sometimes one station has many locations and here we only get the first location
                            if tsta[0].location_code:
                                location.append(tsta[0].location_code)
                            else:
                                location.append('*')

                            if ichan not in pressure_chan:
                                chanhistory[netsta].append(ichan)

    nsta = len(sta)
    prepro_para['nsta'] = nsta

    # output station list
    dict = {'network': net, 'station': sta, 'channel': chan, 'latitude': lat, 'longitude': lon, 'elevation': elev}
    locs = pd.DataFrame(dict)
    locs.to_csv(fname, index=False)

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

def butterworth(samp_freq, pfreqmin):
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


def download(rawdatadir, starttime, endtime, inc_hours, net, stalist,
                                            chanlist=['*'],source='IRIS',
                                            samp_freq=None, plot=False, rmresp=True, rmresp_output='VEL',
                                            pre_filt=None, sacheader=False, getstainv=True, max_tries=10):

    # if user passes a string instead of a list, make a list of one string
    if isinstance(stalist, str): stalist = [stalist]
    if isinstance(chanlist, str): chanlist = [chanlist]
    if isinstance(net, str): net = [net]

    dtlist = utils.split_datetimestr(starttime, endtime, inc_hours)
    print(dtlist)
    for idt in range(len(dtlist) - 1):
        sdatetime = obspy.UTCDateTime(dtlist[idt])
        edatetime = obspy.UTCDateTime(dtlist[idt + 1])
    fname = os.path.join(rawdatadir, str(sdatetime) + 'T' + str(edatetime) + '.h5')

    """
    Start downloading.
    """
    for inet in net:
        for ista in stalist:

            #print('Downloading ' + inet + "." + ista + " ...")
            """
            3a. Request data.
            """
            for chan in chanlist:
                for nt in range(max_tries):
                    print(ista + '.' + chan + '  downloading ... try ' + str(nt + 1))
                    try:
                        t0 = time.time()

                        output = utils.getdata(inet, ista, sdatetime, edatetime, chan=chan, source=source,
                                                                                samp_freq=samp_freq, plot=plot, rmresp=rmresp, rmresp_output=rmresp_output,
                                                                               pre_filt=pre_filt, sacheader=sacheader, getstainv=getstainv)

                        if getstainv == True or sacheader == True:
                            sta_inv = output[1]
                            tr = output[0]
                        else:
                            tr = output
                            sta_inv = None

                        ta = time.time() - t0
                        print('  downloaded ' + "." + ista + "." + chan + " in " + str(ta) + " seconds.")
                        tag = get_tracetag(tr)
                        chan = tr.stats.channel

                        """
                        Add cleanup
                        """

                        if chan[-1].lower() == 'h': tag_type= "trP"; hasPressure=True
                        elif chan[-1].lower() == '1' or chan[-1].lower() == 'e':tag_type="tr1"
                        elif chan[-1].lower() == '2' or chan[-1].lower() == 'n':tag_type="tr2"
                        elif chan[-1].lower() == 'z':tag_type="trZ"
                        else: print('  No seismic channels found. Drop the station: '+ista); break

                        #sanity check.
                        badtrace=False
                        hasPressure=False
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

                        in_dir = in_directory(fname, ista, inet, tag)
                        if badtrace:
                            if not drop_if_has_badtrace:
                                """
                                3b. Save to ASDF file.
                                """
                                if in_dir == True:
                                    print(inet + '.' + ista + '.' + chan + '  exists. Continue to next!')
                                    break
                                else:
                                    print("  Not enough good traces for TC removal! Save as is without processing!")

                                    save2asdf(fname,tr,tag,sta_inv=sta_inv)

                                    break
                            else:
                                print("  Encountered bad trace for " + ista + ". Skipped!")
                                break
                        else:
                            """
                            3b. Save to ASDF file.
                            """
                            if in_dir == True:
                                print(inet + '.' + ista + "." + chan + '  exists. Continue to next!')
                                break
                            else:
                                print(" Saving data for", inet + '.' + ista  + '.' + chan )
                                save2asdf(fname,[tr],[tag],sta_inv=sta_inv)
                                break

                    except Exception as e:
                        print(e, 'for', ista)
                        time.sleep(0.05)  # sleep for 50ms before next try.
                        continue