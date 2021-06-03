#!/usr/bin/env python
# coding: utf-8
"""
Download OBS date including all four components (seismic channels and the pressure channel).
This light-weight script mainly aims at downloading test data for some seisgo modules, instead of a comprehensive
and robust downloading wrapper. It is recommended to use NoisePy's downloading script for
more comprehensive downloading (https://github.com/mdenolle/NoisePy).
"""
#import needed packages.
from seisgo import utils
from seisgo import obsmaster as obs
import sys
import time
import scipy
import obspy
import pyasdf
import datetime
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from obspy import UTCDateTime
from obspy.core import Stream,Trace
from obspy.clients.fdsn import Client

# In[ ]:
"""
1. Set parameters for downloading data.
"""
rawdatadir = '../data/raw'
if not os.path.isdir(rawdatadir): os.mkdir(rawdatadir)

source='IRIS'
client=Client(source)
# get data from IRIS web service
net="7D"
stalist=["FN07A","G30A"]#["G03A","J35A","J44A","J65A"]

starttime = "2012_02_02_0_0_0"
endtime   = "2012_02_05_0_0_0"
inc_hours = 8

"""
2. Preprocessing parameters
"""
rmresp=True #remove instrument response
# parameters for butterworth filter
samp_freq=10
pfreqmin=0.002
pfreqmax=samp_freq/2

# prefilter information used when removing instrument responses
f1 = 0.95*pfreqmin;f2=pfreqmin
if 1.05*pfreqmax > 0.48*samp_freq:
    f3 = 0.45*samp_freq
    f4 = 0.48*samp_freq
else:
    f3 = pfreqmax
    f4= 1.05*pfreqmax
pre_filt  = [f1,f2,f3,f4]


"""
3. Download by looping through datetime list.
***** The users usually don't need to chance the following lines ****
"""
dtlist = utils.split_datetimestr(starttime,endtime,inc_hours)
print(dtlist)
for idt in range(len(dtlist)-1):
    sdatetime = obspy.UTCDateTime(dtlist[idt])
    edatetime   = obspy.UTCDateTime(dtlist[idt+1])

    fname = os.path.join(rawdatadir,dtlist[idt]+'T'+dtlist[idt+1]+'.h5')

    """
    Start downloading.
    """
    for ista in stalist:
        print('Downloading '+net+"."+ista+" ...")
        t0=time.time()
        """
        3a. Request data.
        """
        tr1,tr2,trZ,trP = obs.getdata(net,ista,sdatetime,edatetime,source=source,samp_freq=samp_freq,
                                      plot=False,rmresp=rmresp,pre_filt=pre_filt)
        sta_inv=client.get_stations(network=net,station=ista,
                                    starttime=sdatetime,endtime=edatetime,
                                    location='*',level='response')
        ta=time.time() - t0
        print('  downloaded '+net+"."+ista+" in "+str(ta)+" seconds.")
        """
        3b. Save to ASDF file.
        """
        tags=[]
        for itr,tr in enumerate([tr1,tr2,trZ,trP],1):
            if len(tr.stats.location) == 0:
                tlocation='00'
            else:
                tlocation=tr.stats.location

            tags.append(tr.stats.channel.lower()+'_'+tlocation.lower())

        print('  saving to '+fname)
        utils.save2asdf(fname,Stream(traces=[tr1,tr2,trZ,trP]),tags,sta_inv=sta_inv)
