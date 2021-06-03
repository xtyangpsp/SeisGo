#!/usr/bin/env python
# coding: utf-8
import os,sys,glob
from seisgo.noise import compute_fft,correlate
from seisgo import utils,downloaders

rootdir='.'
respdir='.'
sacfiles = sorted(glob.glob(os.path.join(rootdir,'*.SAC')))
rm_resp='RESP'
#for removing responses.
freqmin=0.01
freqmax=100

tr,inv=downloaders.read_data(sacfiles,rm_resp=rm_resp,freqmin=freqmin,freqmax=freqmax,stainv=True)
tr1,tr2=tr;inv1,inv2=inv

#trimming is needed for this data set, which there is one sample difference in the starting time.
cstart=max([tr1.stats.starttime,tr2.stats.starttime])
cend=min([tr1.stats.endtime,tr2.stats.endtime])
tr1.trim(starttime=cstart,endtime=cend,nearest_sample=True)
tr2.trim(starttime=cstart,endtime=cend,nearest_sample=True)

print('cross-correlation ...')
cc_len    = 3600                                                            # basic unit of data length for fft (sec)
cc_step      = 900                                                             # overlapping between each cc_len (sec)
maxlag         = 100                                                        # lags of cross-correlation to save (sec)
freq_norm='rma'
time_norm='no'

#for whitening
freqmin=0.02
freqmax=2
#get FFT, #do correlation
fftdata1=compute_fft(tr1,cc_len,cc_step,stainv=inv1,
                        freq_norm=freq_norm,freqmin=freqmin,freqmax=freqmax,
                     time_norm=time_norm,smooth=500)
fftdata2=compute_fft(tr2,cc_len,cc_step,stainv=inv2,
                       freq_norm=freq_norm,freqmin=freqmin,freqmax=freqmax,
                     time_norm=time_norm,smooth=500)
corrdata=correlate(fftdata1,fftdata2,maxlag,substack=True)

#plot xcorr result
freqs=[[0.05,0.1],[0.07,0.1],[0.1,0.5],[0.1,1],[0.5,1],[1,2]]
for i in range(len(freqs)):
    corrdata.plot(freqmin=freqs[i][0],freqmax=freqs[i][1],lag=50,stack_method='robust',save=True)

corrdata.to_asdf('2020.087_xcorr.h5')

corrdata.to_sac()
