import sys
import obspy
import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
from seisgo.utils import split_datetimestr
from seisgo import downloaders
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
rootpath = "data_decatur" # roothpath for the project
direc  = os.path.join(rootpath,'Raw')                   # where to store the downloaded data
#if not os.path.isdir(direc): os.mkdir(direc)
down_list  = os.path.join(direc,'station.txt')
# CSV file for station location info

# download parameters
source='IRIS'                                 # client/data center. see https://docs.obspy.org/packages/obspy.clients.fdsn.html for a list
max_tries = 10                                                  #maximum number of tries when downloading, in case the server returns errors.
use_down_list = False                                                # download stations from a pre-compiled list or not
flag      = True                                               # print progress when running the script; recommend to use it at the begining
samp_freq = 50                                                  # targeted sampling rate at X samples per seconds
rmresp   = True
rmresp_out = 'DISP'
pressure_chan = [None]				#Added by Xiaotao Yang. This is needed when downloading some special channels, e.g., pressure data. VEL output for these channels.
respdir   = os.path.join(rootpath,'resp')                       # directory where resp files are located (required if rm_resp is neither 'no' nor 'inv')
freqmin   = 0.01                                                # pre filtering frequency bandwidth
freqmax   = 25
# note this cannot exceed Nquist freq

# targeted region/station information: only needed when use_down_list is False
lamin,lamax,lomin,lomax= 39,40,-89,-88                # regional box: min lat, min lon, max lat, max lon (-114.0)
chan_list = ["HHZ"]
net_list  = ["GS"] #["7D","X9","TA","XT","UW"]                                              # network list
sta_list  = ["DEC05","DEC06","DEC07","DEC10"]                                               # station (using a station list is way either compared to specifying stations one by one)
start_date = "2014_01_01_0_0_0"                               # start date of download
end_date   = "2014_01_11_0_0_0"                               # end date of download
inc_hours  = 24                                                 # length of data for each request (in hour)
maxseischan = 1                                                  # the maximum number of seismic channels, excluding pressure channels for OBS stations.
ncomp      = maxseischan #len(chan_list)

# get rough estimate of memory needs to ensure it now below up in noise cross-correlations
cc_len    = 3600                                                # basic unit of data length for fft (s)
step      = 1800                                                 # overlapping between each cc_len (s)
MAX_MEM   = 8.0                                                 # maximum memory allowed per core in GB

##################################################
# we expect no parameters need to be changed below
# assemble parameters used for pre-processing waveforms in downloading
prepro_para = {'rmresp':rmresp,'rmresp_out':rmresp_out,'respdir':respdir,'freqmin':freqmin,'freqmax':freqmax,\
                'samp_freq':samp_freq}

downlist_kwargs = {"source":source, 'net_list':net_list, "sta_list":sta_list, "chan_list":chan_list, \
                    "starttime":start_date, "endtime":end_date, "maxseischan":maxseischan, "lamin":lamin, "lamax":lamax, \
                    "lomin":lomin, "lomax":lomax, "pressure_chan":pressure_chan, "fname":down_list}

########################################################
#################DOWNLOAD SECTION#######################
########################################################
#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    if flag:
        print('station.list selected [%s] for data from %s to %s with %sh interval'%(use_down_list,start_date,end_date,inc_hours))

    if not os.path.isdir(direc):os.makedirs(direc)
    if use_down_list:
        stalist=pd.read_csv(down_list)
    else:
        stalist=downloaders.get_sta_list(**downlist_kwargs) # saves station list to "down_list" file
                                              # here, file name is "station.txt"
    # rough estimation on memory needs (assume float32 dtype)
    memory_size=noise.cc_memory(inc_hours,samp_freq,len(stalist.station),ncomp,cc_len,step)
    if memory_size > MAX_MEM:
        raise ValueError('Require %5.3fG memory but only %5.3fG provided)! Reduce inc_hours to avoid this issue!' % (memory_size,MAX_MEM))

    # save parameters for future reference
    metadata = os.path.join(direc,'download_info.txt')
    fout = open(metadata,'w')
    fout.write(str({**prepro_para,**downlist_kwargs,'inc_hours':inc_hours,'ncomp':ncomp}));fout.close()

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

# MPI: loop through each time chunk
for ick in range(rank,splits,size):
    s1= all_chunk[ick]
    s2=all_chunk[ick+1]

    download_kwargs = {"source":source,"rawdatadir": direc, "starttime": s1, "endtime": s2, \
              "stationinfo": stalist,**prepro_para}

    # Download for ick
    downloaders.download(**download_kwargs)

tt1=time.time()
print('downloading step takes %6.2f s' %(tt1-tt0))

comm.barrier()
if rank == 0:
    sys.exit()
