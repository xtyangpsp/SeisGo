#!/usr/bin/env python
# coding: utf-8

# # Tilt and Compliance Corrections for OBS Data: Continuous
# ### Xiaotao Yang @ Harvard University
# This notebook contains examples of compliance corrections using local data on the disk.
# The functions for tilt and compliance corrections are in module seisgo.obsmaster.

# ## Step 0. Load needed packages.
# Some functions are imported from the utils.py and the obsmaster.py.

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
# import matplotlib.pyplot  as plt
# from obspy import UTCDateTime
from obspy.core import Stream, Trace

"""
1. Set global data path parameters.
"""
rawdatadir = '../data/raw'
if not os.path.isdir(rawdatadir): os.mkdir(rawdatadir)
#directory to save the data after TC removal
tcdatadir = '../data/tcremoval'
if not os.path.isdir(tcdatadir): os.mkdir(tcdatadir)
####################################
#### Optional clean-up block ####
####################################
cleantartgetdir=True #change to False or remove/comment this block if needed.
dfiles0 = glob.glob(os.path.join(tcdatadir,'*.h5'))
if cleantartgetdir and len(dfiles0)>0:
    print('Cleaning up TC removal directory before running ...')
    for df0 in dfiles0:os.remove(df0)
####################################
##### End of clean-up block #####
####################################

####################################
#### Optional downloading block ####
####################################
downloadexample=False #change to False or remove/comment this block if needed.
if downloadexample:
    print('Cleaning up raw data directory before downloading ...')
    dfiles1 = glob.glob(os.path.join(rawdatadir,'*.h5'))
    for df1 in dfiles1:os.remove(df1)
    os.system('python seisgo_download_obsdata.py')
####################################
##### End of downloading block #####
####################################

"""
2. Tilt and compliance removal parameters
"""
window=3600
overlap=0.2
taper=0.08
qc_freq=[0.004, 1]
plot_correction=True
normalizecorrectionplot=True
tc_subset=['ZP-H']
#assemble all parameters into a dictionary.
tcpara={'window':window,'overlap':overlap,'taper':taper,'qc_freq':qc_freq,
        'tc_subset':tc_subset}
print(tcpara)
"""
3. Read local data and do correction
We use the wrapper function for tilt and compliance corrections.

Steps:
a. read in file list
b. loop through all files
c. loop through all stations
    c-1. read waveform tags and list
    c-2. read station info if available. skip land stations or stations with only vertical.
    c-3. assemble all four components for OBS stations
    c-4. do correction work flow and plot the result if applicable
    c-5. save auxiliary data, e.g., tilt direction and angle, and TC removal parameters.
    c-6. save to original file name in different directory
"""
dfiles = glob.glob(os.path.join(rawdatadir,'*.h5'))
nfiles = len(dfiles)
splits  = nfiles
if nfiles==0:
    raise IOError('Abort! no available seismic files in '+rawdatadir)

t0=time.time()
for ifile in range(nfiles):
    df=dfiles[ifile]
    print('Working on: '+df+' ... ['+str(ifile+1)+'/'+str(nfiles)+']')
    dfbase=os.path.split(df)[-1]
    df_tc=os.path.join(tcdatadir,dfbase)

    ds=pyasdf.ASDFDataSet(df,mpi=False,mode='r')
    netstalist = ds.waveforms.list()
    nsta = len(netstalist)

    tilt=[]
    sta_processed=[]
    for ista in netstalist:
        print('  station: '+ista)
        """
        Get the four-component data
        """
        try:
            inv = ds.waveforms[ista]['StationXML']
        except Exception as e:
            print('  No stationxml for %s in file %s'%(ista,df))
            inv = None

        all_tags = ds.waveforms[ista].get_waveform_tags()
        print(all_tags)
        if len(all_tags)!=4:
            print("  Wrong number of components. Has to be four (4) channels! Skip!")
            continue

        tr1=None
        tr2=None
        trZ=None
        trP=None
        #assign components by waveform tags.
        #This step may fail if the tags don't reflect the real channel information
        newtags=['-','-','-','-']
        for tg in all_tags:
            tr_temp = ds.waveforms[ista][tg][0]
            chan = tr_temp.stats.channel
            if chan[-1].lower() == 'h':trP=tr_temp;newtags[3]=tg
            elif chan[-1].lower() == '1' or chan[-1].lower() == 'e':tr1=tr_temp;newtags[0]=tg
            elif chan[-1].lower() == '2' or chan[-1].lower() == 'n':tr2=tr_temp;newtags[1]=tg
            elif chan[-1].lower() == 'z':trZ=tr_temp;newtags[2]=tg

        #sanity check.
        for tr in [tr1, tr2, trZ, trP]:
            if not isinstance(tr, Trace):
                print(str(tr)+" is not a Trace object")

        """
        Call correction wrapper
        """
        spectra,transfunc,correct=obs.TCremoval_wrapper(
            tr1,tr2,trZ,trP,window=window,overlap=overlap,merge_taper=taper,
            qc_freq=qc_freq,qc_spectra=True,fig_spectra=False,
            save_spectrafig=False,fig_transfunc=False,correctlist=tc_subset)
        tilt.append(spectra['rotation'].tilt)
        sta_processed.append(ista)
        if plot_correction:
            obs.plotcorrection(trZ,correct,normalize=normalizecorrectionplot,freq=[0.005,0.1],
                               size=(12,3),save=True,form='png')

        """
        Save to ASDF file.
        """
        trZtc,tgtemp=obs.correctdict2stream(trZ,correct,tc_subset)
        print('  saving to: '+df_tc)
        utils.save2asdf(df_tc,Stream(traces=[tr1,tr2,trZtc[0],trP]),newtags,sta_inv=inv)

    #save auxiliary data to file.
    print('  saving auxiliary data to: '+df_tc)
    tcpara_temp=tcpara
    tcpara_temp['tilt_stations']=sta_processed
    utils.save2asdf(df_tc,np.array(tilt),None,group='auxiliary',para={'data_type':'tcremoval',
                                                'data_path':'tiltdir',
                                                'parameters':tcpara_temp})

tend=time.time() - t0
print('Finished all files in '+str(tend)+' seconds, or '+str(tend/3600)+' hours')

# In[ ]:
