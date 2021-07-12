#!/usr/bin/env python
# coding: utf-8
# This script cleans up the raw seismic waveform data with specified number of channels.

# In[ ]:


#import needed packages.
from seisgo import utils
from seisgo import obsmaster as obs
import sys
import time
import scipy
import obspy
import pyasdf
import datetime
from mpi4py import MPI
import os, glob
import numpy as np
# import pandas as pd
# import matplotlib.pyplot  as plt
# from obspy import UTCDateTime
from obspy.core import Stream, Trace

t0=time.time()
"""
1. Set global parameters.
"""
rootpath='../data'
rawdatadir = os.path.join(rootpath,'raw')
downloadexample=False #change to False or remove/comment this block if needed.

#directory to save the data after TC removal
outdatadir = os.path.join(rootpath,'raw_clean')
cleanoutdatadir=True #If True, the program will remove all *.h5 files under `tcdatafir` before running.

#parameters for orientation correcttions for horizontal components
correct_obs_orient=True
obs_orient_file='OBS_orientation_Cascadia.csv'

#how to deal with stations with bad traces
drop_if_has_badtrace=True

######################################################################
#### Normally, no changes are needed for the following processing ####
######################################################################
"""
2. Read local data and do correction
We use the wrapper function for tilt and compliance corrections. The data after noise removal
will be saved to the original file name BUT in different directory defined by `outdatadir`.
"""
#-------- Set MPI parameters --------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank==0:
    ####################################
    #### Optional clean-up block ####
    ####################################
    if not os.path.isdir(rawdatadir):
        comm.barrier()
        IOError('Abort! Directory for raw data NOT found: '+rawdatadir)
        sys.exit()

    if not os.path.isdir(outdatadir): os.mkdir(outdatadir)
    dfilesTC0 = glob.glob(os.path.join(outdatadir,'*.h5'))
    if cleanoutdatadir and len(dfilesTC0)>0:
        print('Cleaning up output directory before running ...')
        for df0 in dfilesTC0:os.remove(df0)
    ####################################
    ##### End of clean-up block #####
    ####################################

    dfiles0 = glob.glob(os.path.join(rawdatadir,'*.h5'))
    nfiles = len(dfiles0)
    splits0  = nfiles
    if nfiles < 1:
        raise IOError('Abort! no available seismic files in '+rawdatadir)
        sys.exit()
else:
    splits0,dfiles0 = [None for _ in range(2)]

# broadcast the variables
splits = comm.bcast(splits0,root=0)
dfiles  = comm.bcast(dfiles0,root=0)
#--------End of setting MPI parameters -----------------------
for ifile in range(rank,splits,size):
    #read obs orientation data.
    if correct_obs_orient:
        try:
            obs_orient_data=obs.get_orientations(obs_orient_file)
        except Exception as e:
            print(e)
            sys.exit()
    df=dfiles[ifile]
    print('Working on: '+df+' ... ['+str(ifile+1)+'/'+str(len(dfiles))+']')
    dfbase=os.path.split(df)[-1]
    df_out=os.path.join(outdatadir,dfbase)

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
        if len(all_tags) < 1 or len(all_tags) > 4: continue #empty waveform group.
        else: print(all_tags)

        tr1, tr2, trZ, trP=[None for _ in range(4)]
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
        badtrace=False
        hasPressure=False
        if isinstance(trP,Trace):
            hasPressure=True
        if not isinstance(tr1, Trace) and not isinstance(tr2, Trace) and not isinstance(trZ, Trace):
                print('  No seismic channels found. Drop the station: '+ista)
                continue
        for tr in [tr1, tr2, trZ]:
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
            if not drop_if_has_badtrace:
                print("  Not enough good traces for TC removal! Save as is without processing!")
                outtrace=[]
                for tg in all_tags:
                    outtrace.append(ds.waveforms[ista][tg][0])
                utils.save2asdf(df_out,Stream(traces=outtrace),all_tags,sta_inv=inv)
            else:
                print("  Encountered bad trace for "+ista+". Skipped!")
            continue
        else:
            newtags_tmp=[]
            if isinstance(tr1, Trace) and isinstance(tr2, Trace) and correct_obs_orient and ista in obs_orient_data.keys():
                #correct horizontal orientations if in the obs_orient_data list.
                print("  Correctting horizontal orientations for: "+ista)
                trE,trN = obs.correct_orientations(tr1,tr2,obs_orient_data)
                newtags_tmp.append(utils.get_tracetag(trE))
                newtags_tmp.append(utils.get_tracetag(trN))
                if isinstance(trP,Trace):
                    outstream=Stream(traces=[trE,trN,trZ,trP])
                else:
                    outstream=Stream(traces=[trE,trN,trZ])
            else: #save the station as is if it is not in the orientation database, assuming it is a land station.
                newtags_tmp.append(utils.get_tracetag(tr1))
                newtags_tmp.append(utils.get_tracetag(tr2))
                if isinstance(trP,Trace):
                    outstream=Stream(traces=[tr1,tr2,trZ,trP])
                else:
                    outstream=Stream(traces=[tr1,tr2,trZ])

            newtags_tmp.append(utils.get_tracetag(trZ))
            if isinstance(trP,Trace): newtags_tmp.append(utils.get_tracetag(trP))
            print(newtags_tmp)
            print('  Saving '+ista+' to: '+df_out)
            utils.save2asdf(df_out,outstream,newtags_tmp,sta_inv=inv)

###############################################
comm.barrier()
if rank == 0:
    tend=time.time() - t0
    print('*************************************')
    print('<<< Finished all files in %7.1f seconds, or %6.2f hours for %d files >>>' %(tend,tend/3600,len(dfiles)))
    sys.exit()
