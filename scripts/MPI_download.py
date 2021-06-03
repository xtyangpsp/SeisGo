
import downloaders
import obspy
import os
import time
from obspy.clients.fdsn import Client
import numpy as np
import pandas as pd
from mpi4py import MPI
from seisgo.utils import split_datetimestr

# Code to run this file from within a virtual environment
#########################################################
    # import sys
    # syspath = os.path.dirname(os.path.realpath(__file__))
    # print('Running code')
    # com = 'mpirun -n 4 ' + str(sys.executable) + " " + syspath + "/MPI_download.py"
    # print(com)
    # os.system(com)
    # print('Done')
#########################################################


#########################################################
################ PARAMETER SECTION ######################
#########################################################
tt0=time.time()

# paths and filenames
rootpath = os.path.dirname(os.path.abspath("MPI_Seismic_Download.py")) # roothpath for the project
direc  = os.path.join(rootpath,'Raw')                      # where to store the downloaded data
if not os.path.isdir(direc): os.mkdir(direc)
down_list  = os.path.join(direc,'station.txt')

# CSV file for station location info

# download parameters
# download parameters
client    = Client('IRIS')                                     # client/data center. see https://docs.obspy.org/packages/obspy.clients.fdsn.html for a list
max_tries = 10                                                  #maximum number of tries when downloading, in case the server returns errors.
use_down_list = False                                                # download stations from a pre-compiled list or not
flag      = True                                               # print progress when running the script; recommend to use it at the begining
samp_freq = .5                                                  # targeted sampling rate at X samples per seconds
rm_resp   = 'inv'                                               # select 'no' to not remove response and use 'inv','spectrum','RESP', or 'polozeros' to remove response
rm_resp_out = 'DISP'
pressure_chan = [None]				#Added by Xiaotao Yang. This is needed when downloading some special channels, e.g., pressure data. VEL output for these channels.
respdir   = os.path.join(rootpath,'resp')                       # directory where resp files are located (required if rm_resp is neither 'no' nor 'inv')
freqmin   = 0.01                                                # pre filtering frequency bandwidth
freqmax   = .25
pre_filt = downloaders.butterworth(samp_freq, freqmin)
# note this cannot exceed Nquist freq

# targeted region/station information: only needed when use_down_list is False
lamin,lamax,lomin,lomax= 36.9623,44.1776,-89.9757,-80.4395                # regional box: min lat, min lon, max lat, max lon (-114.0)
chan_list = ["LH*"]
net_list  = ["N4"] #["7D","X9","TA","XT","UW"]                                              # network list
sta_list  = ["*"]                                               # station (using a station list is way either compared to specifying stations one by one)
start_date = "2015_01_01_0_0_0"                               # start date of download
end_date   = "2015_04_01_0_0_0"                               # end date of download
inc_hours  = 8                                                 # length of data for each request (in hour)
maxseischan = 3                                                  # the maximum number of seismic channels, excluding pressure channels for OBS stations.
ncomp      = maxseischan #len(chan_list)

# get rough estimate of memory needs to ensure it now below up in S1
cc_len    = 7200                                                # basic unit of data length for fft (s)
step      = 7200                                                 # overlapping between each cc_len (s)
MAX_MEM   = 5.0                                                 # maximum memory allowed per core in GB

nseg_chunk = int(np.floor((inc_hours/24*86400-cc_len)/step))+1
npts_chunk = int(nseg_chunk*cc_len*samp_freq)
##################################################
# we expect no parameters need to be changed below

# time tags
starttime_UTC = obspy.UTCDateTime(start_date)
endtime_UTC   = obspy.UTCDateTime(end_date)
if flag:
    print('station.list selected [%s] for data from %s to %s with %sh interval'%(use_down_list,starttime_UTC,endtime_UTC,inc_hours))

# assemble parameters used for pre-processing
prepro_para = {'rm_resp':rm_resp,'rm_resp_out':rm_resp_out,'respdir':respdir,'freqmin':freqmin,'freqmax':freqmax,'samp_freq':samp_freq,'start_date':\
    start_date,'end_date':end_date,'inc_hours':inc_hours,'cc_len':cc_len,'step':step,'MAX_MEM':MAX_MEM,'lamin':lamin,\
    'lamax':lamax,'lomin':lomin,'lomax':lomax,'ncomp':ncomp}
metadata = os.path.join(direc,'download_info.txt')

downlist_kwargs = {"client":client, 'net_list':net_list, "sta_list":sta_list, "chan_list":chan_list, \
                    "starttime":starttime_UTC, "endtime":endtime_UTC, "maxseischan":maxseischan, "lamin":lamin, "lamax":lamax, \
                    "lomin":lomin, "lomax":lomax, "pressure_chan":pressure_chan, "prepro_para":prepro_para, "fname":down_list}


downloaders.get_sta_list(**downlist_kwargs) # saves station list to "down_list" file
                                          # here, file name is "station.txt"

########################################################
#################DOWNLOAD SECTION#######################
########################################################

#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    if not os.path.isdir(rootpath):os.mkdir(rootpath)
    if not os.path.isdir(direc):os.mkdir(direc)

    # save parameters for future reference
    fout = open(metadata,'w')
    fout.write(str(prepro_para));fout.close()

    all_chunk = split_datetimestr(start_date,end_date,inc_hours)
    if len(all_chunk)<1:
        raise ValueError('Abort! no data chunk between %s and %s' % (start_date,end_date))
    splits = len(all_chunk)-1
else:
    splits,all_chunk = [None for _ in range(2)]

# broadcast the variables
splits = comm.bcast(splits,root=0)
all_chunk  = comm.bcast(all_chunk,root=0)
extra = splits % size


tp = 0
# MPI: loop through each time chunk
for ick in range(rank,splits,size):

    s1= all_chunk[ick]
    s2=all_chunk[ick+1]
    date_info = {'starttime':obspy.UTCDateTime(s1),'endtime':obspy.UTCDateTime(s2)}

    #read in station list.
    if not os.path.isfile(down_list):
        raise IOError('file %s not exist! double check!' % down_list)

    # read station info from list
    locs = pd.read_csv(down_list)
    chan = list(locs.iloc[:]['channel'])
    net  = list(locs.iloc[:]['network'])
    sta  = list(locs.iloc[:]['station'])
    lat  = list(locs.iloc[:]['latitude'])
    lon  = list(locs.iloc[:]['longitude'])
    nsta = len(sta)

    # location info: useful for some occasion
    try:
        location = list(locs.iloc[:]['location'])
    except Exception as e:
        location = ['*']*nsta

    # rough estimation on memory needs (assume float32 dtype)
    memory_size = nsta*npts_chunk*4/1024**3
    if memory_size > MAX_MEM:
        raise ValueError('Require %5.3fG memory but only %5.3fG provided)! Reduce inc_hours to avoid this issue!' % (memory_size,MAX_MEM))

    for ista in range(nsta):

        download_kwargs = {"rawdatadir": direc, "starttime": s1, \
                  "endtime": s2, "inc_hours": inc_hours, \
                  "net": net[ista], "stalist": sta[ista], \
                  "chanlist": chan[ista], "pre_filt": pre_filt, \
                  "samp_freq": samp_freq, "rmresp_output": "VEL", \
                  "getstainv": True, "max_tries":max_tries}

        # Download for ick
        downloaders.download(**download_kwargs)

tt1=time.time()
print('downloading step takes %6.2f s with %6.2f for preprocess' %(tt1-tt0, tp))

comm.barrier()
if rank == 0:
    sys.exit()
