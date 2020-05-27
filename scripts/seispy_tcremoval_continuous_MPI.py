#!/usr/bin/env python
# coding: utf-8

# # Tilt and Compliance Corrections for OBS Data: Continuous
# ### Xiaotao Yang @ Harvard University
# This notebook contains examples of compliance corrections using local data on the disk. The functions for tilt and compliance corrections are in module seispy.obsmaster.

# ## Step 0. Load needed packages.
# Some functions are imported from the utils.py and the obsmaster.py.

# In[ ]:


#import needed packages.
from seispy import utils
from seispy import obsmaster as obs
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
1. Set global data path parameters.
"""
rootpath='../data'
rawdatadir = os.path.join(rootpath,'raw')
downloadexample=False #change to False or remove/comment this block if needed.

#directory to save the data after TC removal
tcdatadir = os.path.join(rootpath,'tcremoval')
cleantcdatadir=True #If True, the program will remove all *.h5 files under `tcdatafir` before running.

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
savetcpara=True                     #If True, the parameters are saved to a text file
                                    #in the `tcdatadir` directory.
tcparaoutfile=os.path.join(tcdatadir,'tcparameters.txt')
#assemble all parameters into a dictionary.
tcpara={'window':window,'overlap':overlap,'taper':taper,'qc_freq':qc_freq,
        'tc_subset':tc_subset}

######################################################################
#### Normally, no changes are needed for the following processing ####
######################################################################
"""
3. Read local data and do correction
We use the wrapper function for tilt and compliance corrections. The data after noise removal
will be saved to the original file name BUT in different directory defined by `tcdatadir`.
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
        raise IOError('Abort! Directory for raw data NOT found: '+rawdatadir)
        sys.exit()

    if not os.path.isdir(tcdatadir): os.mkdir(tcdatadir)
    dfilesTC0 = glob.glob(os.path.join(tcdatadir,'*.h5'))
    if cleantcdatadir and len(dfilesTC0)>0:
        print('Cleaning up TC removal directory before running ...')
        for df0 in dfilesTC0:os.remove(df0)
    ####################################
    ##### End of clean-up block #####
    ####################################
    print(tcpara)
    if savetcpara:
        fout = open(tcparaoutfile,'w')
        fout.write(str(tcpara));
        fout.close()
    dfiles0 = glob.glob(os.path.join(rawdatadir,'*.h5'))
    nfiles = len(dfiles0)
    splits0  = nfiles
    if nfiles < 1:
        raise IOError('Abort! no available seismic files in '+rawdatadir)
else:
    splits0,dfiles0 = [None for _ in range(2)]

# broadcast the variables
splits = comm.bcast(splits0,root=0)
dfiles  = comm.bcast(dfiles0,root=0)
#--------End of setting MPI parameters -----------------------
for ifile in range(rank,splits,size):
    df=dfiles[ifile]
    print('Working on: '+df+' ... ['+str(ifile+1)+'/'+str(len(dfiles))+']')
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
        for tr in [tr1, tr2, trZ, trP]:
            if not isinstance(tr, Trace):
                print(str(tr)+" is not a Trace object. Save as is without processing: "+ista)
                badtrace=True
        if badtrace or len(all_tags) < 4:
            print("  Not enough good traces for TC removal! Save as is without processing!")
            outtrace=[]
            for tg in all_tags:
                outtrace.append(ds.waveforms[ista][tg][0])
            utils.save2asdf(df_tc,Stream(traces=outtrace),all_tags,sta_inv=inv)
            continue

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

        outstream=Stream(traces=[tr1,tr2,trZtc[0],trP])
        """
        Save to ASDF file.
        """
        trZtc,tgtemp=obs.correctdict2stream(trZ,correct,tc_subset)
        print('  saving to: '+df_tc)
        utils.save2asdf(df_tc,outstream,newtags,sta_inv=inv)

    #save auxiliary data to file.
    if len(tilt) > 0:
        print('  saving auxiliary data to: '+df_tc)
        tcpara_temp=tcpara
        tcpara_temp['tilt_stations']=sta_processed
        utils.save2asdf(df_tc,np.array(tilt),None,group='auxiliary',para={'data_type':'tcremoval',
                                                    'data_path':'tiltdir',
                                                    'parameters':tcpara_temp})
###############################################
comm.barrier()
if rank == 0:
    tend=time.time() - t0
    print('*************************************')
    print('<<< Finished all files in %7.1f seconds, or %6.2f hours for %d files >>>' %(tend,tend/3600,len(dfiles)))
    sys.exit()
