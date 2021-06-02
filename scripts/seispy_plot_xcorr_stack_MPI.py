#!/usr/bin/env python
# coding: utf-8

# # This notebook reads and plots the stacked cross-correlation results
#
# ## Xiaotao Yang @ Purdue University 2020

# ## 0. import needed packages.

# In[10]:


from seisgo import utils
from seisgo import obsmaster as obs
from seisgo import noise as ns
import sys
import time
import scipy
import obspy
import pyasdf
import datetime
from mpi4py import MPI
import os, glob
import numpy as np
import pandas as pd
from obspy.core import Stream, Trace
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from obspy.signal.filter import bandpass
import pygmt as gmt

# ## 1. Set global parameters and get the list of virtual sources
t0=time.time()
#
rootpath='./data'
databasename='Stack_TC_coh'
datatimeflag='2012'
maparea=[-132,-118,39,52]
complist=['ZZ']
#set frequency bands
freqs=np.array([[0.03, 0.1],[0.05, 0.5],[0.1, 1]])
velocities=np.array([[1.7, 5],[1.5, 4.5],[1.2, 4]])
minsnr=[5, 6, 6]
datapath=os.path.join(rootpath,databasename,datatimeflag)
outfigdir=os.path.join(rootpath,databasename+'_figs',datatimeflag)

#-------- Set MPI parameters --------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank==0:
    if not os.path.isdir(outfigdir):
        os.makedirs(outfigdir)
    srclist0 = glob.glob(os.path.join(datapath,'*'))
    print(srclist0)

    nsrc=len(srclist0)
    splits0 = nsrc
    if nsrc < 1:
        raise IOError('Abort! No xcorr data in the direction:'+datapath)
        sys.exit()
else:
    splits0,srclist0 = [None for _ in range(2)]

# broadcast the variables
splits = comm.bcast(splits0,root=0)
srclist  = comm.bcast(srclist0,root=0)
#--------End of setting MPI parameters -----------------------

# ## 2. Loop through all virtual sources in the data path, with inner loop for all receivers
for idir in range(rank,splits,size):
    d=srclist[idir]
    src=d.split('/')[-1]
    print('Plotting ... '+src)

    filelist = glob.glob(os.path.join(d,'*.h5'))
        # print(filelist)
    if len(filelist)>0:
        for p in range(freqs.shape[0]):
            ns.plot_xcorr_moveout_heatmap(filelist,src,'Allstack_robust',freqs[p,:],complist[0],
                                    20,lag=400,save=True,figdir=outfigdir)
            ns.plot_xcorr_moveout_wiggle(filelist,src,'Allstack_robust',freqs[p,:],ccomp=complist,
                                scale=9,lag=400,ylim=[50,650],save=True,figdir=outfigdir,
                               minsnr=20)
            peakampdict=ns.get_xcorr_peakamplitudes(filelist,src,'Allstack_robust',freqs[p,:],ccomp=complist,
                                scale=7,lag=500,ylim=[50,550],save=True,figdir=outfigdir,
                               minsnr=minsnr[p],velocity=velocities[p,:])
            try:
                ns.plot_xcorr_amplitudes(peakampdict,region=maparea,fignamebase=outfigdir+'/'+src+'_'+str(freqs[p,0])+
                                '_'+str(freqs[p,1])+'Hz',projection='M3.5i',xshift='4.5i',frame='a6f3',
                               distance=[3.5/freqs[p,0],550])
            except Exception as e:
                print(e)
                print('failed plotting the amplitudes. Skip!')
            ns.save_xcorr_amplitudes(peakampdict,filenamebase=outfigdir+'/'+src+'_'+str(freqs[p,0])+
                            '_'+str(freqs[p,1])+'Hz')
    else:
        print('no files found. Continue!')
        continue

###############################################
comm.barrier()
if rank == 0:
    tend=time.time() - t0
    print('*************************************')
    print('<<< Finished all files in %7.1f seconds, or %6.2f hours for %d files >>>' %(tend,tend/3600,len(srclist)))
    sys.exit()
