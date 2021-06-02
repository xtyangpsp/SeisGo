
import pandas as pd
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel
from seisgo.downloaders import download
from seisgo.downloaders import get_sta_list
import numpy as np
from obspy import read_events
import time



def sta_travel_time(event_lat, event_long, sta_lat, sta_long, depth_km):
    sta_t = locations2degrees(event_lat, event_long, sta_lat, sta_long)
    taup = TauPyModel(model="iasp91")
    arrivals = taup.get_travel_times(source_depth_in_km=depth_km,distance_in_degree=sta_t)
    travel_time = arrivals[0].time
    return travel_time

# =========================================================================
#                           Download Catalog
# =========================================================================

# Catalog params
start = "2011-03-11"
end = "2011-03-15"
minlat = 32
maxlat = 41
minlon = 35
maxlon = 146
minmag = 8
maxmag = 9

events = []
magstep = 1.0
maglist = np.arange(minmag, maxmag + 0.5 * magstep, magstep)
for i in range(len(maglist) - 1):
    if i > 0:
        minM = str(maglist[i] + 0.0001)  # to avoid duplicates with intersecting magnitude thresholds
    else:
        minM = str(maglist[i])
    maxM = str(maglist[i + 1])

    print(minM, maxM)

    URL = "http://isc-mirror.iris.washington.edu/fdsnws/event/1/query?starttime=" + \
          start + "&endtime=" + end + "&minmagnitude=" + minM + "&maxmagnitude=" + maxM + "&minlatitude=" + \
          str(minlat) + "&maxlatitude=" + str(maxlat) + "&minlongitude=" + str(minlon) + "&maxlongitude=" + str(maxlon) + ""

    event = read_events(URL)
    print("Catalog: " + str(event.count()))
    events.append(event)
    print("Download loop #" + str(i) + " complete")

catalog = events[0]
for i in range(len(events) - 1):
    if (i == 0):
        skip = True
    else:
        catalog.extend(events[i])

# =========================================================================
git
#Station list parameters
fname = None
chan_list = ["*"]
net_list  = ["*"]
sta_list  = ["*"]
lamin,lamax,lomin,lomax= 27.372,46.073,126.563,150.82
max_tries = 10
source = 'IRIS'
maxseischan = 3
window_len = 120

# Downloading parameters
freqmin   = 0.01
freqmax   = 100
samp_freq= 200
sta_fname = None
download_dir= "../../EQData"

for event in catalog:

    # Get station list using seisgo.downloaders.get_sta_list()
    event_t = event.origins[0].time
    event_long = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    depth_km = event.origins[0].depth / 1000
    edatetime = event_t + window_len
    sdatetime = event_t.datetime
    edatetime = edatetime.datetime
    sta_list = get_sta_list(fname, net_list, sta_list, chan_list,
                            sdatetime, edatetime, maxseischan, source='IRIS',
                            lamin=lamin, lamax=lamax, lomin=lomin, lomax=lomax)

    try:
        pd.write_csv(sta_fname+str(event.origins[0].time)+str(event.magnitudes[0].mag))
    except:
        pass
    sta_list = sta_list.assign(starttime=np.nan, endtime=np.nan)
    # Travel time correction using function defined in script
    for i, sta in enumerate(sta_list.iterrows()):
        sta_lat = sta_list.iloc[i].latitude
        sta_long = sta_list.iloc[i].longitude
        travel_time = sta_travel_time(event_lat, event_long, sta_lat, sta_long, depth_km)
        sta_list.at[i, 'starttime'] = event_t + travel_time
        sta_list.at[i, 'endtime'] = sta_list.starttime[i] + window_len
        # Download for each station
        # makes 1 asdf file per earthquake event, and appends new data
        # saves event info and station info with each trace

        download(rawdatadir=download_dir, starttime=sta_list.starttime[i], endtime=sta_list.endtime[i],
                 network=sta_list.network[i], station=sta_list.station[i],
                 channel=sta_list.channel[i], source=source, max_tries=max_tries,
                 samp_freq=samp_freq, freqmax=freqmax, freqmin=freqmin,
                 event=event)
