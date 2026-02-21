import os,glob,copy,obspy,scipy,time,pycwt,pyasdf,datetime
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression
from obspy.core.util.base import _get_function_from_entry_point
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.filter import bandpass,lowpass
import matplotlib.pyplot as plt
from pysurf96 import surf96
"""
This is a planned module, to be developed.
"""
################################################################
################ DISPERSION EXTRACTION FUNCTIONS ###############
################################################################
def get_dispersion_waveforms_cwt(d, dt,fmin,fmax,dj=1/12, s0=-1, J=-1, wvn='morlet'):
    """
    Produce dispersion wavefroms with continuous wavelet tranform.

    ===parameters===
    d: 1-d array data.
    df: time interval.
    fmin, fmax: frequency range.
    dj=1/12, s0=-1, J=-1, wvn='morlet': pycwt.cwt parameters.

    ==returns===
    dout, fout: narrowband-filtered waveforms and the frequency vector.
    """
    ds_cwt, sj, f, coi, _, _ = pycwt.cwt(d, dt, dj, s0, J, wvn)
    f_ind = np.where((f >= fmin) & (f <= fmax))[0]
    dout=[]
    fout=[]
    for ii in range(len(f_ind)):
        if ii>0 and ii<len(f_ind)-1: f_ind_temp=f_ind[ii-1:ii+1]
        elif ii==len(f_ind)-1: f_ind_temp=f_ind[ii-1:ii]
        elif ii==0:f_ind_temp=f_ind[ii:ii+1]
        fout.append(np.mean(f[f_ind_temp]))
        rds_cwt=np.real(pycwt.icwt(ds_cwt[f_ind_temp], sj[f_ind_temp], dt, dj, wvn))
        ds_win=np.power(rds_cwt,2)
        dout.append(ds_win/np.max(ds_win))
    return np.flip(np.array(dout),axis=0), np.flip(fout)

def narrowband_waveforms(d, dt,pmin,pmax,dp=1,pscale='ln',extend=10):
    """
    Produce dispersion wavefroms with narrowband filters.

    ===parameters===
    d: 1-d array data.
    dt: sampling interval.
    pmin, pmax: period range.
    dp: period increment in seconds. default 1 s.
    pscale: period scales. "ln" for linear [default]. "nln" for non-linear scale.
    extend: extend individual period value to form a band range. default: 5 scale steps.

    ==returns===
    dout, pout: narrowband-filtered waveforms and the period vector.
    """
    period=np.array([pmin - extend*dp,pmax + extend*dp])
    if period[0] < 2*dt: period[0]=2.01*dt

    if pscale=="ln":
        # f_all=np.arange(fmin-extend*df,fmax+extend*df,df)
        ptest=np.arange(period.min(),period.max(),dp)
    elif pscale=="nln":
        ptest=2 ** np.arange(np.log2(0.1*period.min()),
                    np.log2(2*period.max()),dp)
    f_all=np.flip(1/ptest)
    fout_temp=[]
    dout_temp=[]
    din=d.copy()

    for ii in range(len(f_all)-extend):
        if f_all[ii]>=1/(2*dt) or f_all[ii+extend]>=1/(2*dt): continue
        ds_win=bandpass(din,f_all[ii],f_all[ii+extend],1/dt,corners=4, zerophase=True)
        dout_temp.append(ds_win/np.max(np.abs(ds_win)))
        fout_temp.append(np.mean([f_all[ii],f_all[ii+extend]])) #center frequency
    fout_temp=np.array(fout_temp)
    f_ind=np.where((fout_temp>=1/pmax) & (fout_temp<=1/pmin))[0]
    fout=fout_temp[f_ind]
    dout_temp=np.array(dout_temp)
    dout = dout_temp[f_ind]
    pout = 1/fout
    return dout, pout
##
def get_dispersion_image(g,t,d,pmin,pmax,vmin,vmax,dp=1,dv=0.1,window=1,pscale='ln',pband_extend=5,
                        verbose=False,min_trace=5,min_wavelength=1.5,energy_type='power_sum',get_best_v=False,
                        plot=False,figsize=None,cmap='jet',clim=[0,1]):
    """
    Uses phase-shift method. Park et al. (1998): http://www.masw.com/files/DispersionImaingScheme-1.pdf

    =====PARAMETERS====
    g: waveform gather for all distances (traces). It should be a numpy array.
    t: time vector.
    d: distance vector corresponding to the waveforms in `g`
    pmin: minimum period.
    pmax: maximum period.
    vmin: minimum phase velocity to search.
    vmax: maximum phase velocity to search.
    dp: period increment. default is 1.
    dv: velocity increment in searching. default is 0.1
    window: number of wavelength when slicing the time segments in computing summed energy. default is 1.
            Window can be a two-element array [min,max], when the window size will be interpolated between
            the minimum and the maximum.
    pscale: period vector scale in applying narrowband filters. default is 'ln' for linear scale.
    pband_extend: number of period increments to extend in filtering. defult is 5.
    verbose: verbose mode. default False.
    min_trace: minimum trace to consider. default 5.
    min_wavelength: minimum wavelength to satisfy far-field. default 1.5.
    energy_type: method to compute maximum energy, 'envelope' or 'power_sum'. Default is 'power_sum'
    get_best_v: pick best velocity for each period. Default False.
    plot: plot dispersion image or not. Default is False.
    figsize: specify figsize. Decides automatically if not specified.
    cmap: colormap. Default is 'jet'
    clim: color value limit. Default is [0,1]

    =====RETURNS====
    dout: dispersion information showing the normalized energy for each velocity value for each frequency.
    vout: velocity vector used in searching.
    pout: period vector.
    best_v: best velocity (group velocity). Only return if get_best_v is True.
    """
    #validate options.
    energy_type_list=['power_sum','envelope']
    if energy_type.lower() not in energy_type_list:
        raise ValueError(energy_type+" is not a recoganized energy type. Use one of "+str(energy_type_list))
    if len(np.array(window).shape) < 1:
        window=[window,window]

    dt=np.abs(t[1]-t[0])
    if t[0]<-1.0*dt and t[-1]> dt: #two sides.
        side='a'
        zero_idx=int((len(t)-1)/2)
        if figsize is None:
            figsize=(8,3)
    elif t[0]<-1.0*dt:
        side='n'
        zero_idx=len(t)-1
        if figsize is None:
            figsize=(5,4)
    elif t[-1]>dt:
        side='p'
        zero_idx=0
        if figsize is None:
            figsize=(5,4)

    if verbose: print('working on side: '+side)
    dfiltered_all=[]
    dist_final=[]
    for k in range(g.shape[0]):
        dtemp,pout=narrowband_waveforms(g[k]/np.max(np.abs(g[k])),dt,pmin,
                                        pmax,dp=dp,pscale=pscale,extend=pband_extend)
        dfiltered_all.append(dtemp)
    dfiltered_all=np.array(dfiltered_all)
    vout=np.arange(vmin,vmax+0.5*dv,dv)
    dout_n_all=[]
    dout_p_all=[]
    window_vector=np.linspace(window[1],window[0],len(pout))
    best_v_n = []
    best_v_p = []
    for k in range(len(pout)):
        win_length=window_vector[k]*pout[k]
        win_len_samples=int(win_length/dt)+1
        dout_n=[]
        dout_p=[]

        d_in=dfiltered_all[:,k,:]
        for i,v in enumerate(vout):
            #subset by distance
            mindist=min_wavelength*v*pout[k] #at least 1.5 wavelength.
            dist_idx=np.where((d >= mindist))[0]
            if len(dist_idx) >min_trace:
                if side=='a' or side=='n':
                    dvec=[]
                    for j in dist_idx: #distance, loop through traces
                        tmin=d[j]/v
                        tmin_idx=zero_idx - int(tmin/dt)
                        dsec=d_in[j][tmin_idx - win_len_samples : tmin_idx]
                        if not any(np.isnan(dsec)) and len(dsec)==win_len_samples:
                            dvec.append(dsec)
                    if energy_type.lower() == 'power_sum':
                        peak_energy=np.sum(np.power(np.mean(dvec,axis=1),2))
                    elif energy_type.lower() == 'envelope':
                        peak_energy=np.max(np.abs(hilbert(np.mean(dvec,axis=1))))
                    dout_n.append(peak_energy)

                if side=='a' or side=='p':
                    dvec=[]
                    for j in dist_idx: #distance, loop through traces
                        tmin=d[j]/v
                        tmin_idx=zero_idx + int(tmin/dt)
                        dsec=d_in[j][tmin_idx : tmin_idx + win_len_samples]
                        if not any(np.isnan(dsec)) and len(dsec)==win_len_samples:
                            dvec.append(dsec)
                    #
                    if energy_type.lower() == 'power_sum':
                        peak_energy=np.sum(np.power(np.mean(dvec,axis=1),2))
                    elif energy_type.lower() == 'envelope':
                        peak_energy=np.max(np.abs(hilbert(np.mean(dvec,axis=1))))
                    dout_p.append(peak_energy)
            else:
                if side=='a' or side=='n':
                    dout_n.append(np.nan)

                if side=='a' or side=='p':
                    dout_p.append(np.nan)
        
        if side=='a' or side=='n':
            dout_n /= np.nanmax(dout_n)
            dout_n_all.append(dout_n)
            best_v_n.append(vout[np.nanargmax(dout_n)])
        if side=='a' or side=='p':
            dout_p /= np.nanmax(dout_p)
            dout_p_all.append(dout_p)
            best_v_p.append(vout[np.nanargmax(dout_p)])
        # find the best velocity with maximum energy
        
    # plot or not
    if plot:
        plt.figure(figsize=figsize)
        if side == 'a':
            plt.subplot(1,2,1)
            plt.imshow(np.flip(np.array(dout_n_all).T),cmap=cmap,extent=[pout[-1],pout[0],vout[0],vout[-1]],aspect='auto')
            if get_best_v:
                plt.plot(pout,best_v_n,'k*')
            plt.ylabel('velocity (km/s)',fontsize=12)
            plt.xlabel('period (s)',fontsize=12)
            plt.xticks(np.linspace(pmin,pmax,5),fontsize=12)
            plt.yticks(np.linspace(vmin,vmax,5),fontsize=12)
            plt.clim(clim)
            ax=plt.colorbar()
            ax.set_label('normalized energy (%s)'%(energy_type.replace('_',' ')))
            plt.title('negative lag: '+energy_type.replace('_',' '),fontsize=13)

            plt.subplot(1,2,2)
            plt.imshow(np.flip(np.array(dout_p_all).T),cmap=cmap,extent=[pout[-1],pout[0],vout[0],vout[-1]],aspect='auto')
            if get_best_v:
                plt.plot(pout,best_v_p,'k*')
            plt.ylabel('velocity (km/s)',fontsize=12)
            plt.xlabel('period (s)',fontsize=12)
            plt.xticks(np.linspace(pmin,pmax,5),fontsize=12)
            plt.yticks(np.linspace(vmin,vmax,5),fontsize=12)
            plt.clim(clim)
            ax=plt.colorbar()
            ax.set_label('normalized energy (%s)'%(energy_type.replace('_',' ')))
            plt.title('positive lag: '+energy_type.replace('_',' '),fontsize=13)

            plt.tight_layout()
        elif side == 'n':
            plt.imshow(np.flip(np.array(dout_n_all).T),cmap=cmap,extent=[pout[-1],pout[0],vout[0],vout[-1]],aspect='auto')
            if get_best_v:
                plt.plot(pout,best_v_n,'k*')
            plt.ylabel('velocity (km/s)',fontsize=12)
            plt.xlabel('period (s)',fontsize=12)
            plt.xticks(np.linspace(pmin,pmax,5),fontsize=12)
            plt.yticks(np.linspace(vmin,vmax,5),fontsize=12)
            plt.clim(clim)
            ax=plt.colorbar()
            ax.set_label('normalized energy (%s)'%(energy_type.replace('_',' ')))
            plt.title('negative lag: '+energy_type.replace('_',' '),fontsize=13)
        elif side == 'p':
            plt.imshow(np.flip(np.array(dout_p_all).T),cmap=cmap,extent=[pout[-1],pout[0],vout[0],vout[-1]],aspect='auto')
            if get_best_v:
                plt.plot(pout,best_v_p,'k*')
            plt.ylabel('velocity (km/s)',fontsize=12)
            plt.xlabel('period (s)',fontsize=12)
            plt.xticks(np.linspace(pmin,pmax,5),fontsize=12)
            plt.yticks(np.linspace(vmin,vmax,5),fontsize=12)
            plt.clim(clim)
            ax=plt.colorbar()
            ax.set_label('normalized energy (%s)'%(energy_type.replace('_',' ')))
            plt.title('positive lag: '+energy_type.replace('_',' '),fontsize=13)
        #
        plt.show()

    if side=='a':
        dout=np.squeeze(np.array([dout_n_all,dout_p_all],dtype=np.float64))
        best_v = np.squeeze(np.array([best_v_n,best_v_p],dtype=np.float64))
    elif side == 'p':
        dout=np.squeeze(np.array(dout_p_all,dtype=np.float64))
        best_v = np.squeeze(np.array(best_v_p,dtype=np.float64))
    elif side == 'n':
        dout=np.squeeze(np.array(dout_n_all,dtype=np.float64))
        best_v = np.squeeze(np.array(best_v_n,dtype=np.float64))
    if get_best_v:
        return dout,vout,pout,best_v
    else:
        return dout,vout,pout

def forward_solver(vs, periods, thickness, wave_type='rayleigh', mode=1, velocity_type='group'):
    """
    Wrapper for surf96 to compute synthetic group velocity dispersion curve. 
    Maps Vs to Vp and Density using standard geophysical relations.

    ==PARAMETERS==
    vs: Vs for each layer in km/s.
    periods: periods in s.
    thickness: layer thickness in 1-d array in km.
    wave_type: rayleigh or love.
    mode: wave mode. default 1 (fundamental mode).
    velocity_type: group or phase. default group.

    ==RETURN==
    output of surf96 program. see manual of surf96 for details.
    """
    # Assumptions for poorly constrained parameters:
    vp = vs * 1.75             # Vp/Vs ratio
    rho = 0.77 + 0.32 * vp          # Nafe-Drake density relation
        
    return surf96(thickness, vp, vs, rho, periods, 
                  wave=wave_type, mode=mode, velocity=velocity_type)

def inversion(periods, velocity, thickness, initial_vs, 
                  iterations=8, damp=0.1, smooth=0.5,
                  wave_type='rayleigh', mode=1, velocity_type='group',
                  maxdv=0.02):
    """
    Performs 1-D damped least-squares inversion with smoothness.

    ==PARAMETER==
    periods: wave periods in 1-d array
    velocity: observed velocity from disperson analysis in km/s.
    thickness: layer thickness in 1-d array in km.
    initial_vs: stating Vs for each layer.
    iterations: maximum number of iterations. default 8.
    damp: Damping (stability). default 0.1.
    smooth: Smoothness (geological plausibility). default 0.5.
    maxdv: maximum velocity perturbation in km/s.

    ==RETURN==
    vs_curr: inverted velocity for each layer.
    """
    vs_curr = np.copy(initial_vs)
    n = len(vs_curr)
    m = len(periods)
    
    # Second-difference matrix for smoothness (L)
    L = np.zeros((n-2, n))
    for i in range(n-2):
        L[i, i] = 1; L[i, i+1] = -2; L[i, i+2] = 1
        
    for i in range(iterations):
        # Current prediction
        pred_u = forward_solver(vs_curr, periods, thickness,
                                wave_type=wave_type, mode=mode, velocity_type=velocity_type)
        residual = velocity - pred_u
        
        # Build Numerical Jacobian (Sensitivity Matrix)
        J = np.zeros((m, n))
        for j in range(n):
            v_tmp = np.copy(vs_curr)
            v_tmp[j] += maxdv
            up_u = forward_solver(v_tmp, periods, thickness)
            J[:, j] = (up_u - pred_u) / maxdv
            
        # Solve: (J.T@J + damping + smoothness) * dm = J.T @ residual
        lhs = J.T @ J + damp * np.eye(n) + smooth * (L.T @ L)
        rhs = J.T @ residual
        delta_m = np.linalg.solve(lhs, rhs)
        
        vs_curr += delta_m
        print(f"Iteration {i+1}: RMSE = {np.sqrt(np.mean(residual**2)):.5f}")
        
    return vs_curr
