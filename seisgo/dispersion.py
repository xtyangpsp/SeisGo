"""
dispersion.py -- module for surface wave dispersion analysis, with automated frequency-
time analysis (AFTAN). This includes theAFTAN measurement pipeline (DispData, aftan(), aftan_pmf(),
assemble_dispersion(), plot_dispersion_matrix()), the synthetic-test-signal
generators used to validate it (make_dispersive_synthetic(),
make_synthetic_from_phase_velocity()), and dispersion_to_1d_model(), a thin connector
from a measured DispData curve straight into the inversion().

===Credits===
The AFTAN half of this file follows:
    Levshin, A. L., & Ritzwoller, M. H. (2001). Automated detection, extraction, and
        measurement of regional surface waves. PAGEOPH, 158(8), 1531-1545.
    Bensen, G. D., et al. (2007). Processing seismic ambient noise data to obtain
        reliable broad-band surface wave dispersion measurements. GJI, 169(3), 1239-1260.
    Feng, L., & Ritzwoller, M. H. (2017). The effect of sedimentary basins on surface
        waves that pass through them. GJI, 211(1), 572-592.
    Feng, L., & Ritzwoller, M. H. (2019). A 3-D shear velocity model of the crust and
        uppermost mantle beneath Alaska including apparent radial anisotropy. JGR: Solid
        Earth, 124(10), 10468-10497.
The overall narrow-band-filter/group-velocity-picking/phase-matched-filter/
phase-velocity workflow mirrors the one implemented (with a compiled Fortran77 core)
in pyaftan by Lili Feng (https://github.com/lfengmle/pyaftan); in particular, the
phase-to-phase-velocity ambiguity-resolution recursion in _phase_velocity() below
follows the same construction as pyaftan's _phtovel()/__get_phase_vel(). Everything
here, however, is re-derived and re-implemented from scratch in pure NumPy/SciPy (no
Fortran, no numba requirement).

Drafted with the assistance of Claude (Anthropic, https://claude.ai) working from the
SeisGo CorrData class (seisgo/types.py) and the pyaftan source above.
"""
import pycwt, json, h5py
import numpy as np
from scipy.signal import detrend, hilbert
from obspy.signal.invsim import cosine_taper
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.filter import bandpass
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from pysurf96 import surf96
# alias used throughout the AFTAN block below
_obspy_bandpass = bandpass
# DispData lives in seisgo.types 
from seisgo.types import DispData

################################################################
################ DISPERSION EXTRACTION FUNCTIONS ###############
################################################################
# CWT-based narrowband/dispersion-image extraction, and the surf96-based
# 1-D forward solver/inversion that dispersion_to_1d_model() below calls into.
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

################################################################
###################### AFTAN FUNCTIONS ##########################
################################################################
def read_dispdata(filename):
    """
    Load a DispData object from an input .h file.

    ===PARAMETERS===
    filename: input .h5 path, as written by DispData.save().

    ===RETURNS===
    a DispData object, equivalent to the one that was saved (envelope/phase_matrix
    are present only if the saved object had them, i.e. it was created with
    store_image=True).
    """
    with h5py.File(filename, 'r') as f:
        dist = float(f.attrs['dist'])
        dt = float(f.attrs['dt'])
        method = f.attrs['method'] or None
        side = json.loads(f.attrs['side_json'])
        params = json.loads(f.attrs['params_json'])
        # station-pair metadata: absent in files written before this field was added
        # (see DispData.save()) -- default every sub-field to None in that case rather
        # than failing to read an older file.
        station = json.loads(f.attrs['station_json']) if 'station_json' in f.attrs else {}

        def _get(name):
            return np.array(f[name]) if name in f else None

        period = _get('period')
        group_velocity = _get('group_velocity')
        phase_velocity = _get('phase_velocity')
        amplitude = _get('amplitude')
        snr = _get('snr')
        inst_period = _get('inst_period')
        arrival_time = _get('arrival_time')
        phase_pick = _get('phase_pick')
        envelope = _get('envelope')
        phase_matrix = _get('phase_matrix')
    return DispData(period, group_velocity, amplitude, snr, inst_period, dist, dt, side,
                     params, phase_velocity=phase_velocity, method=method,
                     arrival_time=arrival_time, phase_pick=phase_pick,
                     envelope=envelope, phase_matrix=phase_matrix,
                     src_net=station.get('src_net'), src_sta=station.get('src_sta'),
                     src_lon=station.get('src_lon'), src_lat=station.get('src_lat'),
                     rcv_net=station.get('rcv_net'), rcv_sta=station.get('rcv_sta'),
                     rcv_lon=station.get('rcv_lon'), rcv_lat=station.get('rcv_lat'))

def _simple_snr(d, dt, dist, max_vel=6.0, noise_window=None):
    """
    A fast, broadband SNR estimate for one side (causal or acausal, one-sided/
    zero-lag-first) of a cross-correlation waveform, used to weight the two sides
    when combining them into a symmetric waveform (see weighted_symmetric_average()).

    The waveform is first trimmed to start at a predicted minimum arrival time
    tmin = dist/max_vel, discarding everything before it -- this avoids the
    near-zero-lag numerical/instrument artifacts (and any residual source-time
    spike) common in cross-correlations, which would otherwise dominate a naive
    peak-amplitude estimate. "Signal" is then the peak absolute amplitude of the
    trimmed waveform; "noise" is the standard deviation of the trimmed waveform with
    a short window around that peak excluded (or of a fixed trailing `noise_window`
    right after the peak, if given).

    ===PARAMETERS===
    d: 1-D one-sided waveform (zero lag first).
    dt: sampling interval (s).
    dist: source-receiver distance (km).
    max_vel: reference group velocity (km/s) used to predict the earliest
            physically-meaningful arrival time tmin = dist/max_vel and trim the
            waveform before it. default 6.0 km/s (a fast/near-upper-bound crustal
            group velocity, safely ahead of any real surface- or body-wave arrival,
            so this just clears the zero-lag region without cutting into real
            signal). Note: it is the *maximum* plausible velocity that predicts the
            *minimum* (earliest) arrival time, since tmin = dist/velocity decreases
            as velocity increases.
    noise_window: length (s) of a trailing window (starting right after the peak)
            used for the noise estimate. default None: use the whole trimmed
            waveform, excluding a small window around the peak itself.

    ===RETURNS===
    snr: linear (not dB) peak-amplitude-to-noise-std ratio. 0.0 if the waveform is
            too short to trim/estimate.
    """
    npts = len(d)
    t = np.arange(npts) * dt
    tmin = dist / max_vel if max_vel > 0 else 0.0
    i0 = int(np.searchsorted(t, tmin))
    i0 = min(max(i0, 0), npts - 1)
    trimmed = d[i0:]
    if len(trimmed) < 4:
        return 0.0
    ipeak = int(np.argmax(np.abs(trimmed)))
    peak = float(np.abs(trimmed[ipeak]))
    if noise_window is not None:
        n_noise = max(int(round(noise_window / dt)), 1)
        noise_seg = trimmed[ipeak + 1: ipeak + 1 + n_noise]
        if len(noise_seg) < 4:
            noise_seg = np.delete(trimmed, ipeak)
    else:
        pad = max(int(round(1.0 / dt)), 1)  # exclude ~1s around the peak
        noise_seg = np.delete(trimmed, np.arange(max(0, ipeak - pad), min(len(trimmed), ipeak + pad + 1)))
    noise = float(np.std(noise_seg)) if len(noise_seg) > 0 else 0.0
    if noise <= 0:
        noise = 1e-12
    return peak / noise


def weighted_symmetric_average(data_n, data_p, dt, dist, max_vel=6.0, noise_window=None):
    """
    Combine the causal (positive-lag) and acausal (negative-lag) sides of a
    cross-correlation into a single symmetric waveform, weighting each side by its
    own estimated SNR (see _simple_snr()) instead of a plain 0.5/0.5 average -- useful
    when the two sides have visibly different data quality (e.g. an asymmetric noise
    source distribution), so the noisier side doesn't degrade the combination as much
    as an unweighted average would.

    ===PARAMETERS===
    data_n,data_p: 1-D negative-lag and positive-lag waveforms, both already in
            one-sided/causal format (zero lag first, time increasing away from zero),
            equal length -- e.g. as stored directly in two separate single-sided
            CorrData objects, or as produced by CorrData.split().
    dt: sampling interval (s).
    dist: source-receiver distance (km).
    max_vel: reference group velocity (km/s) used to predict the earliest
            physically-meaningful arrival time (dist/max_vel) and trim near-zero-lag
            samples before estimating each side's SNR. default 6.0. see
            _simple_snr().
    noise_window: passed to _simple_snr(). default None.

    ===RETURNS===
    sym: the SNR-weighted symmetric waveform, same length as data_n/data_p.
    snr_n,snr_p: each side's estimated (linear) SNR, for reference/diagnostics.
    """
    data_n = np.asarray(data_n, dtype=np.float64)
    data_p = np.asarray(data_p, dtype=np.float64)
    if len(data_n) != len(data_p):
        raise ValueError("data_n and data_p must be the same length to average (got %d and %d)."
                          % (len(data_n), len(data_p)))
    snr_n = _simple_snr(data_n, dt, dist, max_vel=max_vel, noise_window=noise_window)
    snr_p = _simple_snr(data_p, dt, dist, max_vel=max_vel, noise_window=noise_window)
    wn, wp = snr_n, snr_p
    if wn + wp <= 0:
        wn = wp = 1.0  # both sides unusable by this metric: fall back to a plain average
    sym = (wn * data_n + wp * data_p) / (wn + wp)
    return sym, snr_n, snr_p


def _far_field_mask(periods, dist, vmax, min_wavelengths=1.0, far_field_vel=None):
    """
    Boolean mask over `periods`, True where a period FAILS a simple far-field/
    near-field validity check and should be discarded from the dispersion curve: a
    period is only trusted if the source-receiver distance spans at least
    `min_wavelengths` wavelengths at that period, i.e. if
        period <= dist / (min_wavelengths * far_field_vel)
    Long periods on short paths violate the (approximately) plane-wave, far-field
    assumptions the narrow-band-filter/ridge-picking approach relies on, and can give
    a spuriously smooth-looking but physically meaningless "measurement" -- this is a
    coarse guard against that, not a substitute for a proper resolution/wavelength
    analysis.

    ===PARAMETERS===
    periods: period vector (s).
    dist: source-receiver distance (km).
    vmax: the aftan()/aftan_pmf() group-velocity search upper bound -- used as the
            default far_field_vel when none is given.
    min_wavelengths: minimum number of wavelengths the distance must span. default 1.0.
    far_field_vel: velocity (km/s) used to convert period to wavelength for this
            check. default None: uses vmax.

    ===RETURNS===
    fail: boolean array, same length as periods; True = fails the check (too long a
            period for this distance) and should be masked to NaN.
    """
    vel = far_field_vel if far_field_vel is not None else vmax
    periods = np.asarray(periods, dtype=np.float64)
    if vel is None or vel <= 0 or min_wavelengths <= 0 or dist is None or dist <= 0:
        return np.zeros(len(periods), dtype=bool)
    max_period = dist / (min_wavelengths * vel)
    return periods > max_period


def _resolve_piover4(piover4, corrdata, verbose=False):
    """
    Resolve aftan()/aftan_pmf()'s `piover4` far-field phase-shift term (see
    _pick_group_velocity()) when the caller leaves it at its default (None),
    using seisgo.types.CorrData's `type` tag to tell whether `corrdata` already
    holds an empirical Green's function (EGF) or the raw, undifferentiated noise
    correlation function.

    Background: an EGF built as EGF(t) = -dC/dt(t) from a correlation function
    C(t) has a phase spectrum offset from C's own by a CONSTANT pi/2 at every
    frequency (differentiation multiplies the spectrum by i*omega, and i has a
    frequency-independent phase of pi/2). A constant phase offset contributes
    zero extra group delay (d(const)/domega = 0), so group-velocity measurements
    are unaffected either way -- but phase-velocity measurements need that pi/2
    compensated, since piover4's -pi/4-type far-field term is only correct for
    one specific convention. Verified numerically (not just derived): analyzing
    a signal built directly from a known phase-velocity curve (i.e. an EGF-like
    signal) recovers the truth with piover4=1.0; integrating that same signal
    once (undoing the -d/dt, i.e. a raw-correlation-like signal) and analyzing
    IT needs piover4=-1.0 for the same accuracy -- a difference of exactly 2
    units of pi/4 = pi/2, matching the theory.

    ===PARAMETERS===
    piover4: the value passed by the caller. If not None, returned unchanged --
            an explicit value always wins over auto-detection.
    corrdata: the seisgo.types.CorrData object passed to aftan()/aftan_pmf(), or
            None if the caller used raw data/dt/dist instead.
    verbose: print which convention was auto-detected. default False.

    ===RETURNS===
    piover4: resolved float. If `corrdata` is None (raw-array usage) or has no
            recognizable `type` tag, defaults to 1.0 (the EGF convention) --
            preserving this module's original default/backward-compatible
            behavior for callers not using CorrData's type tag at all.
    """
    if piover4 is not None:
        return piover4
    ctype = getattr(corrdata, "type", None) if corrdata is not None else None
    if ctype == "Empirical Green's Functions":
        resolved = 1.0
        reason = "corrdata.type == \"Empirical Green's Functions\""
    elif ctype is not None:
        resolved = -1.0
        reason = "corrdata.type == %r (not yet converted to an EGF)" % ctype
    else:
        resolved = 1.0
        reason = "no corrdata/type tag available -- assuming EGF convention (this module's original default)"
    if verbose:
        print("piover4 auto-detected as %.1f (%s)" % (resolved, reason))
    return resolved


def _get_aftan_waveform(corrdata, side=None, stack_index=None, sym_weighted=False,
                         sym_max_vel=6.0):
    """
    Extract a single 1-D, one-sided (causal, starting at zero lag) waveform, its
    sampling interval, and the station-pair distance from a seisgo.types.CorrData
    object, ready to feed to aftan()/aftan_pmf().

    ===PARAMETERS===
    corrdata: a seisgo.types.CorrData object.
    side: which lag to use when corrdata.side=='A' (both sides stored): 'p' positive
            [default], 'n' negative, or 'sym'/'s' for the symmetric (average) waveform.
            Ignored when corrdata already stores a single side.
    stack_index: when corrdata.data is a substack (2-D, shape [nwin,npts]), select this
            window index. default None: average (linear stack) over all windows.
    sym_weighted: when side is 'sym', combine the two sides using
            weighted_symmetric_average() (SNR-weighted) instead of a plain 0.5/0.5
            average. default False (plain average, the original/backward-compatible
            behavior). Ignored unless side is 'sym'.
    sym_max_vel: the max_vel passed to weighted_symmetric_average() when
            sym_weighted=True. default 6.0 km/s.

    ===RETURNS===
    dout,dt,dist,side: the 1-D waveform (float64), sampling interval, distance (km),
            and the side label actually used.
    """
    if corrdata.data is None:
        raise ValueError("corrdata.data is empty.")
    dt = corrdata.dt
    dist = corrdata.dist
    d = np.asarray(corrdata.data)
    if d.ndim == 2:  # substack: default to a linear stack over all windows.
        d = d[stack_index] if stack_index is not None else np.mean(d, axis=0)

    cside = str(getattr(corrdata, "side", "o")).lower()
    if cside == "a":
        side = (side or "p").lower()
        n = len(d)
        nhalf = n // 2
        if side == "p":
            dout = d[nhalf:]
        elif side == "n":
            dout = np.flip(d[:nhalf + 1])
        elif side in ("sym", "s", "symmetric"):
            d_p = d[nhalf:]
            d_n = np.flip(d[:nhalf + 1])
            if sym_weighted:
                dout, _, _ = weighted_symmetric_average(d_n, d_p, dt, dist, max_vel=sym_max_vel)
            else:
                dout = 0.5 * (d_p + d_n)
        else:
            raise ValueError("side must be one of 'p','n','sym' when corrdata.side=='A'. Got: %s" % side)
    else:
        dout = d
        side = cside
    return np.array(dout, dtype=np.float64), dt, dist, side


def _gaussian_filter_spectrum(spec, omega, omega0, alpha):
    """
    Narrow-band Gaussian filter applied to the FFT spectrum of a real signal, forming
    the band-limited analytic-signal spectrum used in AFTAN
    (Levshin & Ritzwoller, 2001; Bensen et al., 2007).

    ===PARAMETERS===
    spec: complex FFT spectrum (full length nfft) of a REAL input signal.
    omega: angular-frequency vector (rad/s), same length as spec (2*pi*fftfreq).
    omega0: central angular frequency of the filter (rad/s).
    alpha: Gaussian filter width parameter (larger alpha -> narrower band, better
            frequency resolution but worse time resolution). Typical values 50-200
            for ambient-noise surface waves; can be scaled with distance for
            teleseismic applications.

    ===RETURNS===
    the filtered, one-sided (positive-frequency, doubled) analytic-signal spectrum.
    """
    filt = np.exp(-alpha * ((omega - omega0) / omega0) ** 2)
    filt[omega < 0] = 0.0  # keep only positive frequencies -> analytic signal
    out = spec * filt
    out[omega == 0] *= 0.5
    return 2.0 * out

def _aftan_narrowband(data, dt, periods, alpha=100.0, nfft=None):
    """
    Apply a bank of narrow-band Gaussian filters to `data` and return the envelope and
    unwrapped phase of the resulting analytic signal for every period in `periods`.

    ===PARAMETERS===
    data: 1-D real waveform (already detrended/tapered).
    dt: sampling interval (s).
    periods: 1-D array of central periods (s) to analyze.
    alpha: Gaussian filter width parameter. see _gaussian_filter_spectrum().
    nfft: FFT length. default None: zero-padded to next_fast_len(2*npts) for better
            frequency resolution.

    ===RETURNS===
    envelope,phase: 2-D arrays, shape (len(periods), npts).
    """
    npts = len(data)
    if nfft is None:
        nfft = next_fast_len(2 * npts)
    spec = fft(data, n=nfft)
    omega = 2 * np.pi * np.fft.fftfreq(nfft, d=dt)

    envelope = np.zeros((len(periods), npts))
    phase = np.zeros((len(periods), npts))
    for i, p in enumerate(periods):
        omega0 = 2 * np.pi / p
        fspec = _gaussian_filter_spectrum(spec, omega, omega0, alpha)
        asig = ifft(fspec, n=nfft)[:npts]
        envelope[i] = np.abs(asig)
        phase[i] = np.unwrap(np.angle(asig))
    return envelope, phase

def _aftan_narrowband_bandpass(data, dt, periods, pscale='log', pband_extend=5, corners=4):
    """
    Narrow-band filter bank using actual zero-phase Butterworth bandpass
    filters (obspy.signal.filter.bandpass) plus a Hilbert-transform analytic signal,
    instead of the frequency-domain Gaussian filter used by _aftan_narrowband(). This
    mirrors seisgo.dispersion.narrowband_waveforms()/get_dispersion_image()'s
    pscale='ln'/'nln' option (https://github.com/xtyangpsp/SeisGo), ported here to
    single-trace AFTAN-style analysis: for each period, the two band-edge periods are
    taken `pband_extend` grid steps away *within this same, already-computed
    `periods` array* (extrapolating geometrically/linearly beyond its ends as
    needed) -- so the passband width automatically follows whatever pscale `periods`
    itself was built with.

    This matters because a passband built from a FIXED, linear step in period
    (pscale='linear' here, matching seisgo's 'ln') gets progressively narrower in
    *relative* bandwidth as period grows (e.g. a 5 s-wide band is +/-100% around a
    5 s center period but only +/-12.5% around a 40 s one) -- a narrower relative
    band means a longer time-domain impulse response, which can smear/leak
    near-zero-lag energy into the analysis window at long periods (this is what
    "too narrow [a relative band] at long periods" refers to, and what was
    contributing to the near-vmax edge artifacts seen on short-baseline real data).
    Using pscale='log' (matching seisgo's 'nln') instead keeps a fixed grid-step
    *ratio* between band edges, i.e. constant relative bandwidth/logarithmic width,
    across the whole period range -- avoiding that long-period narrowing. Note the
    default Gaussian filter (_aftan_narrowband(), alpha fixed) already has this
    constant-relative-bandwidth property built in by construction (its width scales
    with 1/omega0 for any period), same as pyaftan's own default; this bandpass
    alternative is offered for direct comparison/consistency with seisgo's other
    (multi-trace, phase-shift) dispersion-image method and as another lever against
    long-period instability on noisy real data, not because the Gaussian default is
    architecturally biased toward narrow long-period bands.

    ===PARAMETERS===
    data: 1-D real waveform (already detrended/tapered).
    dt: sampling interval (s).
    periods: 1-D array of central periods (s) to analyze (as produced by aftan()'s
            own pscale-driven period vector).
    pscale: 'log' [default] (a.k.a. seisgo's 'nln'): band edges are `pband_extend`
            steps away in log(period) -- constant relative bandwidth/logarithmic
            width, recommended especially when analyzing a wide period range.
            'linear' (a.k.a. seisgo's 'ln'): band edges are `pband_extend` steps
            away in linear period -- reproduces the narrowing-at-long-period
            behavior described above; offered for direct comparison.
    pband_extend: number of steps (in the `periods` grid, in log- or linear-period
            space per `pscale`) to each side of the center period to form the
            passband. default 5 (matches seisgo's pband_extend default).
    corners: Butterworth filter order (per side; zero-phase applies it twice).
            default 4 (matches seisgo's narrowband_waveforms()).

    ===RETURNS===
    envelope,phase: 2-D arrays, shape (len(periods), npts). A period whose passband
            could not be formed (too close to/above Nyquist) is left as NaN in both.
    """
    npts = len(data)
    fs = 1.0 / dt
    nyq = 0.5 * fs
    n = len(periods)
    order = np.argsort(periods)
    psorted = np.asarray(periods, dtype=np.float64)[order]

    if pscale.lower().startswith('log'):
        grid = np.log(psorted)
    else:
        grid = psorted
    step = float(np.median(np.diff(grid))) if n > 1 else 0.1

    envelope = np.full((n, npts), np.nan)
    phase = np.full((n, npts), np.nan)
    for k in range(n):
        i = order[k]
        lo_idx, hi_idx = k - pband_extend, k + pband_extend
        g_lo = grid[lo_idx] if lo_idx >= 0 else grid[0] + lo_idx * step
        g_hi = grid[hi_idx] if hi_idx < n else grid[-1] + (hi_idx - (n - 1)) * step
        if pscale.lower().startswith('log'):
            p_lo, p_hi = np.exp(g_lo), np.exp(g_hi)
        else:
            p_lo, p_hi = g_lo, g_hi
        p_lo = max(p_lo, 2.01 * dt)  # stay below Nyquist period
        f_lo, f_hi = 1.0 / p_hi, 1.0 / p_lo
        f_hi = min(f_hi, 0.999 * nyq)
        if f_lo <= 0 or f_hi <= f_lo:
            continue  # leave this period as NaN: no valid passband could be formed
        filtered = _obspy_bandpass(data, f_lo, f_hi, fs, corners=corners, zerophase=True)
        asig = hilbert(filtered)
        envelope[i] = np.abs(asig)
        phase[i] = np.unwrap(np.angle(asig))
    return envelope, phase


def _parabolic_peak(y, i):
    """
    Parabolic (3-point) interpolation for a sub-sample peak location/value of `y`
    around index `i`.

    The correction is clamped to +/-0.5 samples: a proper 3-point parabolic fit to a
    true local maximum at/near index i always yields |delta|<=0.5, so a larger value
    only occurs when the three points are nearly colinear (a tiny, noise-dominated
    curvature `denom`) and the fit is numerically unreliable -- in that regime the
    unclamped formula can return an arbitrarily large delta (even for `denom` many
    orders of magnitude above exact zero, so a simple `denom == 0` guard is not
    enough). Clamping keeps the
    correction to its intended sub-sample-refinement role.

    ===RETURNS===
    delta: sub-sample offset (in samples) from index i to the interpolated peak,
            clamped to [-0.5, 0.5].
    ypeak: interpolated peak value.
    """
    if i <= 0 or i >= len(y) - 1:
        return 0.0, y[i]
    yl, yc, yr = y[i - 1], y[i], y[i + 1]
    denom = (yl - 2 * yc + yr)
    if not (denom < 0):  # not a proper local-max curvature (denom==0, or noise-flipped sign)
        return 0.0, yc
    delta = 0.5 * (yl - yr) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    ypeak = yc - 0.25 * (yl - yr) * delta
    return delta, ypeak


def _pick_group_velocity(envelope, phase, periods, dt, dist, vmin, vmax,
                          snr_min=5.0, vg_step_max=0.5, noise_window=100.0, piover4=1.0):
    """
    Pick and track the group-velocity dispersion curve from the narrow-band envelopes,
    scanning periods from long to short and favoring continuity (bounded group-velocity
    jumps between adjacent periods) over always taking the single largest peak -- a
    simple, fast alternative to the classical jump-detection/backtracking algorithm
    used by the Fortran aftan core.

    Also records, for each accepted pick, the arrival time and the (far-field-corrected)
    phase there, which is what _phase_velocity() needs to resolve phase velocity.

    ===PARAMETERS===
    envelope,phase: outputs of _aftan_narrowband().
    periods: period vector (s) matching envelope/phase's first axis.
    dt: sampling interval (s).
    dist: source-receiver distance (km).
    vmin,vmax: group-velocity search range (km/s).
    snr_min: preferred minimum SNR (dB) for a pick.
    vg_step_max: maximum allowed group-velocity change (km/s) between adjacent
            (in period-tracking order) accepted picks.
    noise_window: length (s) of the window right after the group-velocity search
            window used to estimate the noise level for SNR.
    piover4: far-field phase-shift term (as a multiple of pi/4) added to the raw
            measured phase before it is used for phase-velocity determination, to
            compensate for the asymptotic far-field phase shift of the cylindrically
            spreading surface-wave Green's function. default +1.0 (i.e. +pi/4) --
            note this is the opposite sign from pyaftan's own default (-pi/4);
            pyaftan's phase convention comes from a Fortran core built around an
            e^{-i*omega*t} time convention, while the narrow-band filtering here is
            built around numpy's e^{+i*omega*t} ifft convention, which flips the sign
            of this term. +1.0 was verified against a synthetic dispersive wave train
            built from a known phase-velocity curve (see the module's tests); adjust
            only if you have reason to believe your convention differs.

    Tracking starts from an ANCHOR period -- the period whose best candidate peak
    has the highest SNR anywhere in the analyzed range (falling back to the single
    highest-amplitude candidate anywhere, only if no candidate anywhere passes
    snr_min) -- and expands outward from there in both directions (toward longer
    periods, and separately toward shorter periods), rather than always starting
    at the longest period and walking toward the shortest. Starting at the most
    reliable pick available anywhere in the range, rather than wherever the period
    grid happens to end, avoids a failure mode seen on real, noisy single-sided
    (causal- or acausal-only) correlations: if the *longest* usable period happens
    to have poor SNR, a fixed long-to-short walk can anchor on a weak/wrong local
    peak there and then, because vg_step_max bounds every subsequent step to the
    immediately preceding pick, never be able to jump back to the real, much
    stronger dispersion ridge that only becomes distinguishable at shorter
    periods -- the whole curve stays trapped on the wrong branch. Anchoring at
    the best pick available anywhere sidesteps this: a period with strong,
    unambiguous SNR is far less likely to be sitting on a spurious peak in the
    first place, and both expansion directions inherit that good starting point
    instead of only one direction ever seeing it.

    ===RETURNS===
    grvel,amp,snr,instper,arrival_time,phase_pick: 1-D arrays (len(periods)), in the
            same order as `periods` (NOT re-sorted). NaN where no pick could be
            made/tracked.
    """
    nper, npts = envelope.shape
    t = np.arange(npts) * dt

    grvel = np.full(nper, np.nan)
    amp = np.full(nper, np.nan)
    snr = np.full(nper, np.nan)
    instper = np.full(nper, np.nan)
    arrival_time = np.full(nper, np.nan)
    phase_pick = np.full(nper, np.nan)

    tmin = dist / vmax if vmax > 0 else 0.0
    tmax = min(dist / vmin, t[-1]) if vmin > 0 else t[-1]

    def _candidates(idx):
        """
        Build this period's candidate peak(s) -- (cand_t,cand_a,cand_snr,cand_delta,
        cand_v,local_idx), all filtered to the [vmin,vmax] velocity range -- or None
        if this period has no usable envelope/search window/in-range candidate at
        all. Pure lookup/computation, no continuity constraint applied here (that
        happens once in _pick_from(), so the same candidate set can be reused for
        both the anchor-selection scan and the actual tracking pass).
        """
        env = envelope[idx]
        if np.all(np.isnan(env)):
            return None  # no narrowband filter could be formed for this period
                          # (see _aftan_narrowband_bandpass()).
        if tmin >= tmax:
            return None
        win = np.where((t >= tmin) & (t <= tmax))[0]
        if len(win) < 3:
            return None

        inner = win[1:-1]
        is_peak = (env[inner] > env[inner - 1]) & (env[inner] > env[inner + 1])
        local_idx = inner[is_peak]
        if len(local_idx) == 0:
            # No genuine interior local maximum inside the search window (the
            # envelope is monotonic, or effectively flat, across it). Only fall
            # back to the window's own maximum sample when that maximum is itself
            # an interior sample (e.g. a broad, numerically-flat-topped bump where
            # the strict ">" test above missed a tie) -- if the window max sits
            # right at one of the window's own edges (win[0]/win[-1]), the
            # envelope is still rising/falling *into* the boundary and the true
            # peak lies outside [vmin,vmax] entirely; reporting that boundary
            # sample as "the pick" is not a real detection; it previously showed
            # up as suspiciously confident (high-SNR) picks sitting exactly at
            # vmin or vmax, most visibly downstream of aftan_pmf()'s re-analysis
            # of a re-dispersed pulse. Skip the period instead (leave it NaN).
            argmax_win = win[np.argmax(env[win])]
            if argmax_win in (win[0], win[-1]):
                return None
            local_idx = np.array([argmax_win])

        # noise level: trailing window right after the search window.
        noise_mask = np.where((t >= tmax) & (t <= min(t[-1], tmax + noise_window)))[0]
        if len(noise_mask) < 5:
            noise_mask = np.where(t > tmax)[0]
        noise_amp = np.std(env[noise_mask]) if len(noise_mask) > 0 else np.std(env)
        if noise_amp <= 0:
            noise_amp = 1e-12

        cand_t, cand_a, cand_snr, cand_delta = [], [], [], []
        for li in local_idx:
            delta, ypeak = _parabolic_peak(env, li)
            cand_t.append(t[li] + delta * dt)
            cand_a.append(ypeak)
            cand_delta.append(delta)
            cand_snr.append(20 * np.log10(ypeak / noise_amp) if ypeak > 0 else -np.inf)
        cand_t = np.array(cand_t)
        cand_a = np.array(cand_a)
        cand_snr = np.array(cand_snr)
        cand_delta = np.array(cand_delta)
        cand_v = dist / cand_t

        # Safety net: a candidate's velocity must stay within the requested
        # [vmin,vmax] search range. The sub-sample parabolic correction is clamped
        # (see _parabolic_peak()) so this should already hold except at the very
        # edge of the search window, where the correction can still nudge a
        # boundary sample's implied arrival time just outside [tmin,tmax] -- catch
        # that here rather than ever returning/propagating an out-of-range "group
        # velocity" (this was the root cause of values like -40 km/s previously
        # leaking through, most visibly in aftan_pmf()'s re-analysis of a short,
        # heavily edge-affected re-dispersed pulse).
        in_range = (cand_v >= vmin) & (cand_v <= vmax)
        if not np.any(in_range):
            return None
        return (cand_t[in_range], cand_a[in_range], cand_snr[in_range],
                cand_delta[in_range], cand_v[in_range], local_idx[in_range])

    # Compute every period's candidates once, up front -- reused for both the
    # anchor scan below and the tracking pass, so no window/peak/SNR work is
    # ever repeated for a given period.
    cand_cache = [_candidates(idx) for idx in range(nper)]

    # 1. Pick the tracking ANCHOR: the period with the single highest-SNR
    # candidate anywhere in the range (preferred), or, only if literally no
    # candidate anywhere passes snr_min, the single highest-amplitude candidate
    # anywhere -- see this function's docstring for why this replaces always
    # anchoring at the longest period.
    anchor_idx, anchor_val, anchor_has_snr_pass = None, -np.inf, False
    for idx, cand in enumerate(cand_cache):
        if cand is None:
            continue
        _, cand_a, cand_snr, _, _, _ = cand
        passing = cand_snr >= snr_min
        if np.any(passing):
            best_here = np.max(cand_snr[passing])
            if not anchor_has_snr_pass or best_here > anchor_val:
                anchor_idx, anchor_val, anchor_has_snr_pass = idx, best_here, True
        elif not anchor_has_snr_pass:
            best_here = np.max(cand_a)
            if best_here > anchor_val:
                anchor_idx, anchor_val = idx, best_here

    if anchor_idx is None:
        return grvel, amp, snr, instper, arrival_time, phase_pick  # nothing usable anywhere

    def _pick_from(idx, prev_v):
        """
        Apply the continuity-constrained pick at period `idx` given the previous
        (already-accepted, in the current walking direction) group velocity
        `prev_v` (None only for the anchor's own first pick), writing the result
        into grvel/amp/snr/instper/arrival_time/phase_pick. Returns the velocity
        to use as `prev_v` for the next period in this same walking direction --
        unchanged from the input on a gap (no acceptable candidate), exactly as
        the original single-pass tracker did, so a gap doesn't reset continuity.
        """
        cand = cand_cache[idx]
        if cand is None:
            return prev_v
        cand_t, cand_a, cand_snr, cand_delta, cand_v, local_idx = cand

        if prev_v is None:
            good = np.where(cand_snr >= snr_min)[0]
            pool = good if len(good) > 0 else np.arange(len(cand_v))
            sel = pool[np.argmax(cand_a[pool])]
        else:
            dv = np.abs(cand_v - prev_v)
            good = np.where((cand_snr >= snr_min) & (dv <= vg_step_max))[0]
            if len(good) == 0:
                good = np.where(dv <= vg_step_max)[0]
            if len(good) == 0:
                return prev_v  # gap: no acceptable continuation; leave NaN, keep prev_v.
            sel = good[np.argmin(dv[good])]

        grvel[idx] = cand_v[sel]
        amp[idx] = cand_a[sel]
        snr[idx] = cand_snr[sel]
        arrival_time[idx] = cand_t[sel]

        p = periods[idx]
        pha = phase[idx]
        li_pick = local_idx[sel]
        if 0 < li_pick < npts - 1:
            dphidt = (pha[li_pick + 1] - pha[li_pick - 1]) / (2 * dt)
            phase_at_sample = pha[li_pick] + dphidt * cand_delta[sel] * dt
        else:
            dphidt = 2 * np.pi / p
            phase_at_sample = pha[li_pick]
        instper[idx] = 2 * np.pi / abs(dphidt) if dphidt != 0 else p
        phase_pick[idx] = phase_at_sample + piover4 * np.pi / 4.0
        return cand_v[sel]

    # 2. Pick the anchor itself, then expand outward toward shorter and (again,
    # independently, both seeded from the anchor's own picked velocity) toward
    # longer periods.
    anchor_v = _pick_from(anchor_idx, None)
    order_asc = np.argsort(periods)
    anchor_pos = int(np.where(order_asc == anchor_idx)[0][0])

    prev_v = anchor_v
    for pos in range(anchor_pos - 1, -1, -1):  # toward shorter periods
        prev_v = _pick_from(order_asc[pos], prev_v)

    prev_v = anchor_v
    for pos in range(anchor_pos + 1, nper):  # toward longer periods
        prev_v = _pick_from(order_asc[pos], prev_v)

    return grvel, amp, snr, instper, arrival_time, phase_pick


def _phase_velocity(period, group_velocity, arrival_time, phase_pick, dist,
                     ref_period=None, ref_velocity=None):
    """
    Resolve the 2*pi cycle-skipping ambiguity in the measured phase to obtain phase
    velocity, using a group-velocity-guided recursive continuation from the longest to
    the shortest period. This follows the same construction as the classical AFTAN
    phase-velocity step (Levshin & Ritzwoller, 2001; Bensen et al., 2007), matching the
    approach implemented (with a Fortran core) in pyaftan's _phtovel()/
    __get_phase_vel() (https://github.com/lfengmle/pyaftan).

    At the longest period, the integer ambiguity is resolved against a predicted phase
    velocity Vpred: either a supplied reference/predicted dispersion curve
    (ref_period,ref_velocity -- e.g. from a regional/global reference model), or, if
    none is supplied, the observed group velocity at that period as a rough proxy. A
    real reference model should be supplied whenever available, especially for long
    paths (dist spanning many wavelengths), where the group-velocity proxy is more
    likely to pick the wrong 2*pi cycle. For each shorter period, Vpred is instead
    obtained from the previous (already resolved) phase velocity and the local group
    slowness, which keeps the ambiguity resolution self-consistent along the curve.

    ===PARAMETERS===
    period,group_velocity,arrival_time,phase_pick: arrays of valid (non-NaN)
            group-velocity picks and the associated arrival time (s) and measured,
            far-field-corrected phase (rad) at each pick (see _pick_group_velocity()).
            Any input order is fine; sorted internally by ascending period.
    dist: source-receiver distance (km).
    ref_period,ref_velocity: an optional reference/predicted phase-velocity dispersion
            curve (same units) used to anchor the ambiguity at the longest period.

    ===RETURNS===
    phase_velocity: array, in the same order as the inputs, km/s.
    """
    period = np.asarray(period, dtype=np.float64)
    order = np.argsort(period)
    per = period[order]
    U = np.asarray(group_velocity, dtype=np.float64)[order]
    T = np.asarray(arrival_time, dtype=np.float64)[order]
    pha = np.asarray(phase_pick, dtype=np.float64)[order]
    n = len(per)
    if n < 2:
        raise ValueError("_phase_velocity needs at least 2 valid periods.")

    omega = 2 * np.pi / per
    sU = 1.0 / U
    phV = np.zeros(n)

    if ref_period is not None and ref_velocity is not None and len(ref_period) >= 2:
        rp = np.asarray(ref_period, dtype=np.float64)
        rv = np.asarray(ref_velocity, dtype=np.float64)
        srt = np.argsort(rp)
        rp, rv = rp[srt], rv[srt]
        spl = CubicSpline(rp, rv, extrapolate=True)
        vpred0 = float(spl(np.clip(per[-1], rp[0], rp[-1])))
    else:
        vpred0 = U[-1]

    phpred = omega[-1] * (T[-1] - dist / vpred0)
    k = round((phpred - pha[-1]) / (2 * np.pi))
    phV[-1] = dist / (T[-1] - (pha[-1] + 2 * np.pi * k) / omega[-1])

    for m in range(n - 2, -1, -1):
        vpred = 1.0 / (((sU[m] + sU[m + 1]) * (omega[m] - omega[m + 1]) / 2.0
                         + omega[m + 1] / phV[m + 1]) / omega[m])
        phpred = omega[m] * (T[m] - dist / vpred)
        k = round((phpred - pha[m]) / (2 * np.pi))
        phV[m] = dist / (T[m] - (pha[m] + 2 * np.pi * k) / omega[m])

    out = np.empty(n)
    out[order] = phV
    return out


def _band_taper(period, pmin, pmax, roll=0.15):
    """
    Smooth (raised-cosine, in log-period) amplitude taper that is ~1 within
    [pmin,pmax] and rolls off to 0 over a fractional (`roll`) extension beyond each
    edge. Used to band-limit the phase-matched filter to the periods actually being
    analyzed.
    """
    lp = np.log(np.maximum(period, 1e-6))
    lpmin, lpmax = np.log(pmin), np.log(pmax)
    width = (lpmax - lpmin) * roll
    lo, hi = lpmin - width, lpmax + width
    x = np.clip((lp - lo) / (hi - lo), 0.0, 1.0)
    w = np.sin(np.pi * x) ** 2
    w[(lp >= lpmin) & (lp <= lpmax)] = 1.0
    w[(lp < lo) | (lp > hi)] = 0.0
    return w


def _phase_match_filter(data, dt, dist, ref_period, ref_velocity, pmin, pmax, nfft=None):
    """
    Build and apply a phase-matched filter (PMF) that removes the dispersion predicted
    by a reference group-velocity curve, compressing the dispersed surface-wave train
    into a compact pulse (Levshin & Ritzwoller, 2001; the "aftanipg"/PMF stage of the
    classic AFTAN scheme, see also pyaftan's `_tgauss`/`aftanipg` for the equivalent
    Fortran-wrapped step).

    ===PARAMETERS===
    data: 1-D real waveform (already detrended/tapered).
    dt: sampling interval (s).
    dist: source-receiver distance (km).
    ref_period,ref_velocity: reference/preliminary group-velocity dispersion curve
            (e.g. from a first-pass aftan()) used to build the de-dispersion phase.
    pmin,pmax: period band actually being analyzed (used to band-limit the filter).
    nfft: FFT length. default None: zero-padded to next_fast_len(2*npts).

    ===RETURNS===
    compressed: the phase-matched-filtered (de-dispersed/compressed) waveform, length
            npts (same length as `data`).
    dphi,posmask,nfft: the phase correction (rad, for the positive-frequency bins
            marked by posmask) and the FFT length used, needed later by
            _phase_match_restore() to invert the filter.
    t_ref: the reference time (s) the filter was designed to compress the dispersed
            wave train onto (the mean predicted group delay across the analyzed
            band). When the reference curve reasonably matches the data, the compact
            pulse should appear near this time -- used to keep _isolate_pulse()'s
            peak search local to where compression is actually expected, instead of
            an unconstrained global maximum that can lock onto an unrelated artifact
            elsewhere in the record (e.g. a taper edge, or residual energy where the
            reference curve fits poorly).
    """
    npts = len(data)
    if nfft is None:
        nfft = next_fast_len(2 * npts)
    spec = fft(data, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=dt)
    omega = 2 * np.pi * freqs
    posmask = freqs > 1e-8
    om_pos = omega[posmask]
    per_pos = 2 * np.pi / om_pos

    rp = np.asarray(ref_period, dtype=np.float64)
    rv = np.asarray(ref_velocity, dtype=np.float64)
    srt = np.argsort(rp)
    rp, rv = rp[srt], rv[srt]
    spl = CubicSpline(rp, rv, extrapolate=True)
    vg = spl(np.clip(per_pos, rp[0], rp[-1]))
    tau = dist / vg  # predicted group delay (s) at each positive frequency

    # accumulate phase(omega) with d(phase)/d(omega) = tau(omega), by trapezoidal
    # integration in ascending-omega order (same construction used to build a
    # synthetic dispersive wave train from a prescribed group-velocity curve).
    o_order = np.argsort(om_pos)
    om_s = om_pos[o_order]
    tau_s = tau[o_order]
    intphase_s = np.concatenate(([0.0], np.cumsum(0.5 * (tau_s[1:] + tau_s[:-1]) * np.diff(om_s))))
    intphase = np.empty_like(om_pos)
    intphase[o_order] = intphase_s

    t_ref = float(np.mean(tau))  # compress the pulse to sit near this reference time
    dphi = intphase - t_ref * om_pos
    band = _band_taper(per_pos, pmin, pmax)

    spec_c = np.zeros(nfft, dtype=complex)
    spec_c[posmask] = spec[posmask] * np.exp(1j * dphi) * band
    compressed = 2.0 * np.real(ifft(spec_c, n=nfft))[:npts]
    return compressed, dphi, posmask, nfft, t_ref


def _phase_match_restore(pulse, nfft, dphi, posmask, npts):
    """
    Invert _phase_match_filter(): re-apply the original (frequency-dependent)
    dispersion to a cleaned/tapered compact pulse, returning it to a normal dispersed
    waveform so it can be re-analyzed with the standard narrow-band AFTAN.
    """
    spec2 = fft(pulse, n=nfft)
    spec_r = np.zeros(nfft, dtype=complex)
    spec_r[posmask] = spec2[posmask] * np.exp(-1j * dphi)
    restored = 2.0 * np.real(ifft(spec_r, n=nfft))[:npts]
    return restored


def _isolate_pulse(x, dt, min_ratio=0.05, taper_frac=0.05, search_center=None, search_radius=None):
    """
    Isolate the single compact pulse of a phase-matched-filtered waveform: keep only
    the samples around the envelope maximum down to where the envelope first drops
    below `min_ratio` of the peak on each side, with a short cosine taper at the cut,
    zeroing everything else. This is a simplified, fast stand-in for the classical
    AFTAN PMF tapering step (cf. pyaftan's `_tgauss`), which additionally hunts for
    the best side-lobe minima and adapts the taper based on SNR; here a single
    amplitude-ratio threshold is used for simplicity.

    ===PARAMETERS===
    x: the phase-matched-filtered (compressed) waveform.
    dt: sampling interval (s).
    min_ratio,taper_frac: see above.
    search_center,search_radius: if both given, the envelope peak is searched for
            only within [search_center-search_radius, search_center+search_radius]
            (falling back to the whole record if that window is empty), instead of
            an unconstrained global maximum. This matters in practice: on real,
            noisy data the single largest envelope sample in the whole record can
            sit somewhere the filter was never intended to compress energy to (a
            taper edge, residual energy where the reference dispersion curve fits
            poorly, coda) -- searching only near the filter's own target time
            (_phase_match_filter()'s `t_ref`) keeps the isolated pulse anchored to
            where compression was actually designed to happen. default None: search
            the whole record (only safe when you already know the pulse dominates).

    ===RETURNS===
    out: the isolated/tapered pulse, same length as x.
    (il,ir): the sample-index bounds of the kept segment.
    """
    env = np.abs(hilbert(x))
    n = len(x)
    if search_center is not None and search_radius is not None:
        t = np.arange(n) * dt
        window = np.where(np.abs(t - search_center) <= search_radius)[0]
        ipeak = int(window[np.argmax(env[window])]) if len(window) > 0 else int(np.argmax(env))
    else:
        ipeak = int(np.argmax(env))
    thresh = min_ratio * env[ipeak]
    il = ipeak
    while il > 0 and env[il] > thresh:
        il -= 1
    ir = ipeak
    while ir < n - 1 and env[ir] > thresh:
        ir += 1
    out = np.zeros_like(x)
    seg = x[il:ir + 1].copy()
    if len(seg) > 4:
        seg = seg * cosine_taper(len(seg), p=min(0.5, taper_frac))
    out[il:ir + 1] = seg
    return out, (il, ir)


def aftan(corrdata=None, data=None, dt=None, dist=None, side=None, stack_index=None,
          pmin=5.0, pmax=50.0, nper=64, pscale='log',
          vmin=1.0, vmax=5.0, alpha=100.0, snr_min=5.0, vg_step_max=0.5,
          taper_frac=0.05, noise_window=100.0, nfft=None, piover4=None,
          phase_velocity=True, ref_period=None, ref_velocity=None,
          sym_weighted=False, sym_max_vel=6.0,
          min_wavelengths=1.0, far_field_vel=None,
          filter_method='bandpass', pband_extend=5, filter_corners=4,
          store_image=True, verbose=False,
          src_net=None, src_sta=None, src_lon=None, src_lat=None,
          rcv_net=None, rcv_sta=None, rcv_lon=None, rcv_lat=None):
    """
    Automatic Frequency-Time Analysis (AFTAN) for surface-wave group- and phase-
    velocity dispersion measurement, following the narrow-band Gaussian filtering
    approach of Levshin & Ritzwoller (2001) and Bensen et al. (2007). Pure Python/
    NumPy/SciPy implementation (no Fortran core), meant as the aftan() entrance for
    seisgo.dispersion, operating directly on SeisGo <CorrData> objects
    (see seisgo.types.CorrData; https://github.com/xtyangpsp/SeisGo). For comparison,
    the existing Fortran-wrapping python interface is pyaftan
    (https://github.com/lfengmle/pyaftan); see this module's top-of-file docstring for
    full credits/references. For a second (phase-matched-filter) refinement stage, see
    aftan_pmf().

    ===PARAMETERS===
    corrdata: a seisgo.types.CorrData object. When given, <data>,<dt>,<dist> are
            extracted from it automatically (the explicit data/dt/dist arguments below
            are then ignored).
    data,dt,dist: use these instead of <corrdata> to run aftan() on a raw 1-D array.
            data: 1-D real waveform, one-sided/causal, starting at zero lag.
            dt: sampling interval (s).
            dist: source-receiver distance (km).
    side: which lag to analyze when corrdata.side=='A' (both sides stored): 'p'
            positive [default], 'n' negative, or 'sym'/'s' for the symmetric
            (averaged) component. Ignored when corrdata already has a single side, or
            when using raw <data>.
    stack_index: when corrdata.data is a substack (2-D), select this window index.
            default None: average (linear stack) over all windows.
    sym_weighted: when side='sym', combine the causal/acausal sides using an
            SNR-weighted average (weighted_symmetric_average()) instead of a plain
            0.5/0.5 average. default False. Ignored unless side='sym'.
    sym_max_vel: fast/near-upper-bound reference group velocity (km/s) used to
            predict the earliest physically-meaningful arrival time (dist/sym_max_vel)
            and trim near-zero-lag samples before estimating each side's SNR when
            sym_weighted=True. default 6.0. see
            weighted_symmetric_average()/_simple_snr().
    pmin,pmax: period range to analyze (s). default 5-50 s.
    nper: number of periods sampled between pmin and pmax. default 64.
    pscale: 'log' [default] or 'linear' period sampling.
    vmin,vmax: group-velocity search range (km/s). default 1.0-5.0 km/s.
    alpha: Gaussian narrow-band filter width parameter (larger -> narrower band,
            better frequency but worse time resolution). default 100.
    snr_min: preferred minimum SNR (dB) when picking/tracking the group-velocity
            local maximum. default 5.
    vg_step_max: maximum allowed group-velocity jump (km/s) between adjacent periods
            while tracking the dispersion curve. default 0.5.
    taper_frac: fraction of the record cosine-tapered at each end before the FFT.
            default 0.05.
    noise_window: length (s) of the window right after the group-velocity search
            window, used to estimate the noise level for SNR. default 100.
            NOTE on empirical Green's functions (EGF): if `corrdata` already holds
            an EGF (built as -dC/dt from the raw noise correlation C), its
            spectrum is tilted by a factor of omega relative to C's own --
            differentiation suppresses long-period/low-frequency amplitude and
            boosts short-period/high-frequency amplitude. This does NOT bias
            group- or phase-velocity picking itself (confirmed numerically: SNR,
            being a signal-vs-local-noise ratio computed on the same,
            consistently-filtered trace, stays comparable either way), but it
            does mean genuinely low-amplitude long periods have proportionally
            even less energy in an EGF than in the raw correlation -- so `pmax`
            should stay safely inside the correlation's own well-resolved period
            range rather than pushed to its very edge, and this DispData's
            `amplitude` values are on a different absolute scale (by roughly a
            factor of 2*pi/period) than one computed from the undifferentiated
            correlation function -- don't compare the two directly.
    nfft: FFT length. default None: chosen automatically (zero-padded) from the
            record length.
    piover4: far-field phase-shift term (as a multiple of pi/4) added to the
            picked phase before phase-velocity's 2*pi ambiguity resolution --
            see _pick_group_velocity()/_resolve_piover4(). default None:
            auto-detected from corrdata.type (set by seisgo.types.CorrData) --
            1.0 if corrdata.type == "Empirical Green's Functions" (this
            module's original default/assumed convention), -1.0 for any other
            type tag (i.e. corrdata still holds the raw, undifferentiated noise
            correlation function), or 1.0 if there's no corrdata/type tag to
            check at all (raw-array usage). Only affects phase velocity -- group
            velocity is identical either way, since a differentiation's implied
            phase shift is a CONSTANT across frequency and so contributes zero
            group delay. Pass an explicit value to override auto-detection.
    phase_velocity: also compute phase velocity. default True.
    ref_period,ref_velocity: an optional reference/predicted phase-velocity dispersion
            curve used to resolve the 2*pi cycle-skipping ambiguity at the longest
            period (see _phase_velocity()). default None: uses the observed group
            velocity at the longest period as a rough proxy -- fine for modest
            distances, but supply a real reference model for long paths whenever
            possible.
    min_wavelengths,far_field_vel: far-field/near-field validity check. A period is
            only trusted if the source-receiver distance spans at least
            `min_wavelengths` wavelengths at that period, i.e. if
            period <= dist / (min_wavelengths * far_field_vel); group_velocity,
            phase_velocity, inst_period, amplitude and snr are all set to NaN for any
            period that fails this check (too close to the near field/single
            wavelength for the plane/far-field-wave assumptions AFTAN relies on to be
            trustworthy). far_field_vel is the velocity (km/s) used to convert period
            to wavelength for this check; default None uses `vmax`. min_wavelengths
            default 1.0 (require at least one full wavelength); increase (e.g. to 2-3)
            for a stricter far-field requirement, particularly for short paths.
    filter_method: narrow-band filter bank used to build the envelope/phase image.
            'bandpass' [default]: actual zero-phase Butterworth bandpass filters plus
            a Hilbert-transform analytic signal (_aftan_narrowband_bandpass()),
            mirroring seisgo.dispersion.get_dispersion_image()'s/
            narrowband_waveforms()'s pscale='ln'/'nln' option
            (https://github.com/xtyangpsp/SeisGo): with `pscale='log'` (the default
            stepping method, matching seisgo's 'nln') the band edges keep a fixed
            ratio (in period) to the center period at every period analyzed -- i.e.
            constant relative bandwidth/logarithmic width, avoiding the
            progressively-narrower-in-relative-terms bands a purely linear period
            grid would give at long periods (`pscale` here reuses the same
            'log'/'linear' choice already used to build the period vector above).
            Made the default after real-data testing showed it consistently gives
            far more stable, fully-tracked dispersion curves than the Gaussian
            option below, especially at longer periods and on noisy/short-baseline
            correlations -- see the module's real-data test scripts for side-by-side
            comparisons. 'gaussian': the classical frequency-domain Gaussian filter
            (_gaussian_filter_spectrum(); same form as pyaftan's own default,
            constant relative bandwidth by construction -- see `alpha`); kept
            available for comparison/backward compatibility, but on real data it
            tends to saturate with an edge artifact at longer periods that
            'bandpass' avoids.
    pband_extend: only used when filter_method='bandpass'. Number of steps (in the
            period grid, spaced per `pscale`) to each side of the center period used
            to form its passband. default 5 (matches seisgo's own default).
    filter_corners: only used when filter_method='bandpass'. Butterworth filter
            order (per side; zero-phase applies it twice). default 4 (matches
            seisgo's narrowband_waveforms()).
    store_image: keep the full 2-D narrow-band envelope/phase matrices (shape
            [nper,npts]) on the returned DispData (its .envelope/.phase_matrix
            attributes), needed to plot a dispersion image or do other image-based
            analysis (see DispData.get_image()/.plot_image()). default True; set
            False to save memory in large batch runs where only the picked curves
            are needed.
    verbose: print a one-line summary when done. default False.

    ===RETURNS===
    result: a DispData object with the group- (and, if requested, phase-) velocity
            dispersion curve and diagnostics (period, group_velocity, phase_velocity,
            inst_period, amplitude, snr), and, if store_image=True, the full
            envelope/phase_matrix dispersion-image arrays. Periods failing the
            far-field check (see min_wavelengths/far_field_vel) are NaN throughout.

    ===EXAMPLE===
    >>> from seisgo import dispersion
    >>> result = dispersion.aftan(corrdata, side='sym', pmin=5, pmax=40)
    >>> result.plot(snr_min=8)
    """
    # 1. get the waveform, sampling interval, and distance to use.
    if corrdata is not None:
        data, dt, dist, side = _get_aftan_waveform(corrdata, side=side, stack_index=stack_index,
                                                     sym_weighted=sym_weighted, sym_max_vel=sym_max_vel)
        # auto-fill station-pair metadata from corrdata unless the caller already
        # gave explicit values (which always win). See DispData's docstring for why
        # this is worth carrying: it's what an eikonal-tomography phase-velocity map
        # (seisgo.imaging.eikonal) needs to place each measurement on the map.
        cnet = getattr(corrdata, 'net', [None, None])
        csta = getattr(corrdata, 'sta', [None, None])
        clon = getattr(corrdata, 'lon', [None, None])
        clat = getattr(corrdata, 'lat', [None, None])
        if src_net is None: src_net = cnet[0]
        if src_sta is None: src_sta = csta[0]
        if src_lon is None: src_lon = clon[0]
        if src_lat is None: src_lat = clat[0]
        if rcv_net is None: rcv_net = cnet[1]
        if rcv_sta is None: rcv_sta = csta[1]
        if rcv_lon is None: rcv_lon = clon[1]
        if rcv_lat is None: rcv_lat = clat[1]
    else:
        if data is None or dt is None or dist is None:
            raise ValueError("Provide either <corrdata> or all of <data>,<dt>,<dist>.")
        data = np.asarray(data, dtype=np.float64)

    if dist is None or dist <= 0:
        raise ValueError("distance (dist) must be positive for group-velocity computation.")
    npts = len(data)
    if npts < 10:
        raise ValueError("waveform is too short for AFTAN (npts=%d)." % npts)
    if pmin < 2 * dt:
        raise ValueError("pmin=%g s is at/below the Nyquist period (2*dt=%g s)." % (pmin, 2 * dt))

    # 1b. resolve piover4 (see _resolve_piover4()) from corrdata.type when the
    # caller left it at its default -- only affects phase velocity, never group
    # velocity (see that function's docstring).
    piover4 = _resolve_piover4(piover4, corrdata, verbose=verbose)

    # 2. detrend and taper.
    d = detrend(data, type='linear')
    if taper_frac > 0:
        d = d * cosine_taper(npts, p=taper_frac)

    # 3. period vector.
    if pscale.lower().startswith('log'):
        periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), nper))
    else:
        periods = np.linspace(pmin, pmax, nper)

    # 4. narrow-band filtering -> envelope & phase for every period.
    if filter_method.lower().startswith('band'):
        envelope, phase = _aftan_narrowband_bandpass(d, dt, periods, pscale=pscale,
                                                       pband_extend=pband_extend,
                                                       corners=filter_corners)
    else:
        envelope, phase = _aftan_narrowband(d, dt, periods, alpha=alpha, nfft=nfft)

    # 5. pick + track the group-velocity dispersion curve.
    grvel, amp, snr, instper, arrtime, phpick = _pick_group_velocity(
        envelope, phase, periods, dt, dist, vmin, vmax,
        snr_min=snr_min, vg_step_max=vg_step_max, noise_window=noise_window, piover4=piover4)

    # 5b. near-field guard: discard (NaN) periods whose wavelength (at far_field_vel,
    # default vmax) isn't covered at least min_wavelengths times by this distance --
    # done before phase velocity so a discarded period is never used as its anchor.
    ff_fail = _far_field_mask(periods, dist, vmax, min_wavelengths=min_wavelengths,
                               far_field_vel=far_field_vel)
    if np.any(ff_fail):
        grvel[ff_fail] = np.nan
        amp[ff_fail] = np.nan
        snr[ff_fail] = np.nan
        instper[ff_fail] = np.nan
        if verbose:
            print("aftan(): masked %d/%d period(s) failing the far-field check "
                  "(min_wavelengths=%g, far_field_vel=%g km/s)." %
                  (int(ff_fail.sum()), nper, min_wavelengths,
                   far_field_vel if far_field_vel is not None else vmax))

    # 6. optionally resolve phase velocity from the picked phases.
    phvel = np.full(nper, np.nan)
    if phase_velocity:
        good = ~np.isnan(grvel)
        if good.sum() >= 2:
            phvel[good] = _phase_velocity(periods[good], grvel[good], arrtime[good], phpick[good],
                                           dist, ref_period=ref_period, ref_velocity=ref_velocity)
        elif verbose:
            print("aftan(): fewer than 2 valid group-velocity picks; skipping phase velocity.")

    params = dict(pmin=pmin, pmax=pmax, nper=nper, pscale=pscale, vmin=vmin, vmax=vmax,
                  alpha=alpha, snr_min=snr_min, vg_step_max=vg_step_max, taper_frac=taper_frac,
                  noise_window=noise_window, piover4=piover4, sym_weighted=sym_weighted,
                  sym_max_vel=sym_max_vel, min_wavelengths=min_wavelengths, far_field_vel=far_field_vel,
                  filter_method=filter_method, pband_extend=pband_extend, filter_corners=filter_corners)
    result = DispData(periods, grvel, amp, snr, instper, dist, dt, side, params,
                       phase_velocity=phvel, method='aftan', arrival_time=arrtime, phase_pick=phpick,
                       envelope=envelope if store_image else None,
                       phase_matrix=phase if store_image else None,
                       src_net=src_net, src_sta=src_sta, src_lon=src_lon, src_lat=src_lat,
                       rcv_net=rcv_net, rcv_sta=rcv_sta, rcv_lon=rcv_lon, rcv_lat=rcv_lat)
    if verbose:
        print(result)
    return result


def _smooth_curve(period, velocity, weight=None, degree=3):
    """
    Robust-ish smooth curve through (period,velocity) points, fit as a low-degree
    polynomial in log(period) weighted by `weight` (e.g. SNR). Used to turn a
    preliminary, possibly locally erratic (e.g. hijacked by a stray high-amplitude
    arrival on a few periods) group-velocity pick set into a smooth curve suitable for
    building a phase-matched filter -- isolated bad picks are down-weighted relative
    to the overall trend rather than taken at face value.

    ===RETURNS===
    a callable: period (s) -> smoothed velocity (km/s).
    """
    lp = np.log(np.asarray(period, dtype=np.float64))
    v = np.asarray(velocity, dtype=np.float64)
    deg = int(min(degree, len(lp) - 1))
    deg = max(deg, 1)
    coeffs = np.polyfit(lp, v, deg=deg, w=weight)
    poly = np.poly1d(coeffs)
    return lambda p: poly(np.log(np.asarray(p, dtype=np.float64)))


def aftan_pmf(corrdata=None, data=None, dt=None, dist=None, side=None, stack_index=None,
              ref_result=None, ref_period=None, ref_velocity=None, smooth_ref=True,
              smooth_degree=3, pmin=5.0, pmax=50.0, nper=64, pscale='log', vmin=1.0, vmax=5.0,
              alpha=100.0, snr_min=5.0, vg_step_max=0.5, taper_frac=0.05,
              noise_window=100.0, nfft=None, piover4=None, phase_velocity=True,
              sym_weighted=False, sym_max_vel=6.0,
              min_wavelengths=1.0, far_field_vel=None,
              filter_method='bandpass', pband_extend=5, filter_corners=4,
              pulse_min_ratio=0.05, pulse_taper_frac=0.05, store_image=True, verbose=False,
              src_net=None, src_sta=None, src_lon=None, src_lat=None,
              rcv_net=None, rcv_sta=None, rcv_lon=None, rcv_lat=None):
    """
    Two-stage AFTAN using a Phase-Matched Filter (PMF): a preliminary group-velocity
    curve is used to build a filter that compresses the dispersed surface-wave train
    into a single compact pulse, everything else (overtones, multipathing, unrelated
    noise/coda) is tapered away, the compact pulse is re-dispersed, and the standard
    narrow-band AFTAN (aftan()) is re-run on this cleaned waveform. This mirrors the
    "aftanipg" second stage of the classic Levshin & Ritzwoller AFTAN scheme -- see
    this module's top-of-file docstring for full credits/references -- and typically
    improves the dispersion measurement (fewer gaps, higher SNR, cleaner phase
    velocity) for lower-SNR or contaminated correlations, at roughly 2x the cost of a
    single aftan() call.

    ===PARAMETERS===
    corrdata,data,dt,dist,side,stack_index: same as aftan() -- either pass a
            seisgo.types.CorrData object, or raw data/dt/dist.
    ref_result,ref_period,ref_velocity: the preliminary group-velocity dispersion
            curve used to build the phase-matched filter. Priority order: explicit
            ref_period+ref_velocity, then ref_result (a DispData from a prior aftan()
            call, e.g. on a smoothed reference model or an earlier stack), then
            (default) an internal aftan() call on this same waveform with the
            parameters below.
    smooth_ref: when the preliminary curve comes from ref_result or the internal
            aftan() call (i.e. not from an explicit ref_period/ref_velocity), fit a
            smooth, SNR-weighted low-degree polynomial (see _smooth_curve()) through
            its valid picks before using it to build the phase-matched filter. default
            True -- this makes the automatic default noticeably more robust to a
            handful of locally bad preliminary picks (e.g. a stray high-amplitude
            arrival that derails a couple of periods), at the cost of not tracking
            genuinely sharp real dispersion features in the preliminary curve (which
            the final, PMF-refined measurement still resolves normally).
    smooth_degree: polynomial degree used by smooth_ref. default 3.
    pmin,pmax,nper,pscale,vmin,vmax,alpha,snr_min,vg_step_max,taper_frac,noise_window,
    nfft,piover4,phase_velocity: same meaning as in aftan(); used both for the
            (if needed) internal preliminary aftan() call and for the final,
            PMF-refined measurement.
    sym_weighted: when side='sym', combine the causal/acausal sides using an
            SNR-weighted average (weighted_symmetric_average()) instead of a plain
            0.5/0.5 average. default False. Ignored unless side='sym'. Same meaning as
            in aftan(); used for the waveform this function starts from.
    sym_max_vel: fast/near-upper-bound reference group velocity (km/s) used to
            predict the earliest physically-meaningful arrival time (dist/sym_max_vel)
            and trim near-zero-lag samples before estimating each side's SNR when
            sym_weighted=True. default 6.0. see
            weighted_symmetric_average()/_simple_snr().
    min_wavelengths,far_field_vel: far-field/near-field validity check, same meaning
            as in aftan(). Applied both to the internal preliminary aftan() call (so
            the reference curve used to build the phase-matched filter is itself
            far-field-valid before smoothing) and to the final, PMF-refined
            measurement. default min_wavelengths=1.0, far_field_vel=None (uses vmax).
    filter_method,pband_extend,filter_corners: same meaning as in aftan() -- choice
            of narrow-band filter bank ('bandpass' default, mirroring
            seisgo.dispersion.get_dispersion_image()'s pscale='ln'/'nln' option, or
            'gaussian' for the classical constant-alpha filter). Applied both to the
            internal preliminary aftan() call and the final, PMF-refined
            measurement.
    pulse_min_ratio: envelope amplitude ratio (relative to the compressed pulse's
            peak) below which the phase-matched pulse is tapered to zero on each side.
            default 0.05.
    pulse_taper_frac: cosine-taper fraction applied at the edges of the isolated
            pulse. default 0.05.
    store_image: keep the full 2-D narrow-band envelope/phase matrices (of the final,
            PMF-cleaned re-analysis) on the returned DispData. default True; see
            aftan()'s store_image for details. The internal preliminary aftan() call
            (when used) never stores its own image, regardless of this setting.
    verbose: print progress/one-line summaries. default False.

    ===RETURNS===
    result: a DispData object (method='aftan_pmf') with the refined group- (and,
            if requested, phase-) velocity dispersion curve.

    ===EXAMPLE===
    >>> from seisgo import dispersion
    >>> result = dispersion.aftan_pmf(corrdata, side='sym', pmin=5, pmax=40)
    >>> result.plot(snr_min=8)
    """
    # 1. get the waveform, sampling interval, and distance to use (same as aftan()).
    if corrdata is not None:
        data, dt, dist, side = _get_aftan_waveform(corrdata, side=side, stack_index=stack_index,
                                                     sym_weighted=sym_weighted, sym_max_vel=sym_max_vel)
        cnet = getattr(corrdata, 'net', [None, None])
        csta = getattr(corrdata, 'sta', [None, None])
        clon = getattr(corrdata, 'lon', [None, None])
        clat = getattr(corrdata, 'lat', [None, None])
        if src_net is None: src_net = cnet[0]
        if src_sta is None: src_sta = csta[0]
        if src_lon is None: src_lon = clon[0]
        if src_lat is None: src_lat = clat[0]
        if rcv_net is None: rcv_net = cnet[1]
        if rcv_sta is None: rcv_sta = csta[1]
        if rcv_lon is None: rcv_lon = clon[1]
        if rcv_lat is None: rcv_lat = clat[1]
    else:
        if data is None or dt is None or dist is None:
            raise ValueError("Provide either <corrdata> or all of <data>,<dt>,<dist>.")
        data = np.asarray(data, dtype=np.float64)
    if dist is None or dist <= 0:
        raise ValueError("distance (dist) must be positive for group-velocity computation.")
    npts = len(data)
    if npts < 10:
        raise ValueError("waveform is too short for AFTAN (npts=%d)." % npts)
    if pmin < 2 * dt:
        raise ValueError("pmin=%g s is at/below the Nyquist period (2*dt=%g s)." % (pmin, 2 * dt))

    # 1b. resolve piover4 (see _resolve_piover4()) once, up front, so both the
    # preliminary aftan() call below and the final PMF-refined measurement use
    # the same (auto-detected-from-corrdata.type, or explicit) value.
    piover4 = _resolve_piover4(piover4, corrdata, verbose=verbose)

    # 2. get a preliminary group-velocity curve to build the phase-matched filter.
    if ref_period is not None and ref_velocity is not None:
        rp, rv = np.asarray(ref_period, dtype=np.float64), np.asarray(ref_velocity, dtype=np.float64)
    else:
        if ref_result is not None:
            prelim = ref_result
        else:
            prelim = aftan(data=data, dt=dt, dist=dist, pmin=pmin, pmax=pmax, nper=nper, pscale=pscale,
                           vmin=vmin, vmax=vmax, alpha=alpha, snr_min=snr_min, vg_step_max=vg_step_max,
                           taper_frac=taper_frac, noise_window=noise_window, nfft=nfft,
                           min_wavelengths=min_wavelengths, far_field_vel=far_field_vel,
                           filter_method=filter_method, pband_extend=pband_extend,
                           filter_corners=filter_corners,
                           phase_velocity=False, store_image=False)
        good = ~np.isnan(prelim.group_velocity)
        if good.sum() < 3:
            raise RuntimeError("Could not obtain a preliminary group-velocity curve with at least 3 "
                                "valid picks to build the phase-matched filter; supply ref_period/"
                                "ref_velocity (or a better ref_result) explicitly.")
        rp = prelim.period[good]
        rv = prelim.group_velocity[good]
        if smooth_ref:
            w = np.clip(prelim.snr[good], 0.1, None)
            smooth_fn = _smooth_curve(rp, rv, weight=w, degree=smooth_degree)
            rv = smooth_fn(rp)

    # 3. detrend/taper the original waveform exactly like aftan().
    d = detrend(data, type='linear')
    if taper_frac > 0:
        d = d * cosine_taper(npts, p=taper_frac)

    # 4. build + apply the phase-matched filter, isolate the compact pulse, and
    #    re-disperse it so the standard narrow-band AFTAN can be re-run on it.
    compressed, dphi, posmask, nfftused, t_ref = _phase_match_filter(d, dt, dist, rp, rv, pmin, pmax, nfft=nfft)
    # keep the pulse search anchored near where the filter was designed to compress
    # energy to (t_ref), not an unconstrained global search -- on real/noisy data the
    # single largest envelope sample in the whole record need not be the true compact
    # pulse (see _isolate_pulse()). The radius allows for how much the reference
    # curve's own predicted group delay varies across the analyzed band, plus a
    # couple of dominant periods of slop for imperfect compression.
    search_radius = 0.5 * (dist / np.min(rv) - dist / np.max(rv)) + 3.0 * pmax
    pulse, _ = _isolate_pulse(compressed, dt, min_ratio=pulse_min_ratio, taper_frac=pulse_taper_frac,
                               search_center=t_ref, search_radius=max(search_radius, 3.0 * pmax))
    cleaned = _phase_match_restore(pulse, nfftused, dphi, posmask, npts)

    # 5. re-run the standard narrow-band AFTAN on the cleaned waveform.
    if pscale.lower().startswith('log'):
        periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), nper))
    else:
        periods = np.linspace(pmin, pmax, nper)
    if filter_method.lower().startswith('band'):
        envelope, phase = _aftan_narrowband_bandpass(cleaned, dt, periods, pscale=pscale,
                                                       pband_extend=pband_extend,
                                                       corners=filter_corners)
    else:
        envelope, phase = _aftan_narrowband(cleaned, dt, periods, alpha=alpha, nfft=nfft)
    grvel, amp, snr, instper, arrtime, phpick = _pick_group_velocity(
        envelope, phase, periods, dt, dist, vmin, vmax,
        snr_min=snr_min, vg_step_max=vg_step_max, noise_window=noise_window, piover4=piover4)

    # 5b. near-field guard (same as aftan(); see there for details) -- done before
    # phase velocity so a discarded period is never used as its anchor.
    ff_fail = _far_field_mask(periods, dist, vmax, min_wavelengths=min_wavelengths,
                               far_field_vel=far_field_vel)
    if np.any(ff_fail):
        grvel[ff_fail] = np.nan
        amp[ff_fail] = np.nan
        snr[ff_fail] = np.nan
        instper[ff_fail] = np.nan
        if verbose:
            print("aftan_pmf(): masked %d/%d period(s) failing the far-field check "
                  "(min_wavelengths=%g, far_field_vel=%g km/s)." %
                  (int(ff_fail.sum()), nper, min_wavelengths,
                   far_field_vel if far_field_vel is not None else vmax))

    phvel = np.full(nper, np.nan)
    if phase_velocity:
        good = ~np.isnan(grvel)
        if good.sum() >= 2:
            phvel[good] = _phase_velocity(periods[good], grvel[good], arrtime[good], phpick[good],
                                           dist, ref_period=rp, ref_velocity=rv)
        elif verbose:
            print("aftan_pmf(): fewer than 2 valid group-velocity picks; skipping phase velocity.")

    params = dict(pmin=pmin, pmax=pmax, nper=nper, pscale=pscale, vmin=vmin, vmax=vmax,
                  alpha=alpha, snr_min=snr_min, vg_step_max=vg_step_max, taper_frac=taper_frac,
                  noise_window=noise_window, piover4=piover4, pulse_min_ratio=pulse_min_ratio,
                  pulse_taper_frac=pulse_taper_frac, sym_weighted=sym_weighted,
                  sym_max_vel=sym_max_vel, min_wavelengths=min_wavelengths,
                  far_field_vel=far_field_vel, filter_method=filter_method,
                  pband_extend=pband_extend, filter_corners=filter_corners)
    result = DispData(periods, grvel, amp, snr, instper, dist, dt, side, params,
                       phase_velocity=phvel, method='aftan_pmf', arrival_time=arrtime, phase_pick=phpick,
                       envelope=envelope if store_image else None,
                       phase_matrix=phase if store_image else None,
                       src_net=src_net, src_sta=src_sta, src_lon=src_lon, src_lat=src_lat,
                       rcv_net=rcv_net, rcv_sta=rcv_sta, rcv_lon=rcv_lon, rcv_lat=rcv_lat)
    if verbose:
        print(result)
    return result


################################################################
################## ENSEMBLE ASSEMBLY / PLOTTING ##################
################################################################
def _interp_curve_to_grid(period_grid, curve_period, curve_vel):
    """
    Interpolate one dispersion curve (curve_period, curve_vel; curve_vel may
    contain NaN gaps) onto period_grid, linearly, using only the curve's own
    finite (period, velocity) points as anchors. Grid points outside the range
    spanned by those finite anchors are returned as NaN (no extrapolation); grid
    points that fall inside a NaN gap *between* two finite anchors are linearly
    interpolated across that gap (i.e. gaps are bridged, not preserved) -- this
    is deliberate, so a handful of dropped/low-SNR periods in an otherwise good
    curve don't needlessly punch holes in the assembled ensemble.
    """
    curve_period = np.asarray(curve_period, dtype=np.float64)
    curve_vel = np.asarray(curve_vel, dtype=np.float64)
    good = np.isfinite(curve_period) & np.isfinite(curve_vel)
    if good.sum() < 2:
        return np.full(len(period_grid), np.nan)
    f = interp1d(curve_period[good], curve_vel[good], kind='linear', bounds_error=False,
                 fill_value=np.nan)
    return f(period_grid)


def assemble_dispersion(sources, vtype='group', period=None, pmin=None, pmax=None, nper=60,
                         pscale='log', snr_min=None, labels=None):
    """
    Assemble a set of dispersion measurements (DispData objects, and/or saved
    DispData .h5 files, and/or a whole directory of them) onto one common period
    grid, producing a (n_curves, n_periods) velocity matrix -- the basic ensemble
    data structure that plot_dispersion_matrix() and dispersion_to_1d_model() (an
    ensemble-mean curve) both build on.

    ===PARAMETERS===
    sources: any of --
            - a single directory path (str): every "*.h5" file found under it
              (recursively) is loaded with read_dispdata().
            - a list whose entries are DispData objects, and/or .h5 filenames
              (loaded with read_dispdata()) -- the two kinds of entries can be
              mixed freely in one list.
    vtype: 'group' or 'phase' -- which velocity to assemble. default 'group'.
    period: explicit output period vector (s). default None: auto-built from
            pmin/pmax/nper/pscale below.
    pmin,pmax: output period range (s) for the auto-built grid. default None:
            the widest range spanned by the input curves, i.e. min(period.min())
            and max(period.max()) across all successfully-loaded curves.
    nper: number of periods in the auto-built grid. default 60.
    pscale: 'log' or 'linear' spacing for the auto-built grid. default 'log'.
    snr_min: if given, mask (set to NaN) a curve's velocity at any period where
            its own `snr` is below this threshold, before interpolating it onto
            the common grid.
    labels: optional list of labels (e.g. station-pair ids), one per entry of
            `sources`, used to populate the returned `label` list. default None:
            falls back to the input .h5 filename (without directory/extension)
            for file entries, or "curve<i>" for DispData-object entries.

    ===RETURNS===
    a dict with:
        period: the common period vector, shape (nper,).
        velocity: 2-D array, shape (n_curves, nper); NaN where a curve had no
                usable pick at that period (originally NaN, masked by snr_min, or
                outside that curve's own finite-pick range).
        dist: 1-D array, shape (n_curves,) -- each curve's source-receiver
                distance (NaN if a curve's `dist` was None).
        label: list of length n_curves.
        n: number of curves successfully assembled (== velocity.shape[0]); curves
                that failed to load (e.g. a corrupt .h5 file) are skipped with a
                printed warning rather than raising.
    """
    import os
    import glob as _glob

    if isinstance(sources, str):
        entries = sorted(_glob.glob(os.path.join(sources, '**', '*.h5'), recursive=True))
    else:
        entries = list(sources)

    curves, dists, labs = [], [], []
    for i, entry in enumerate(entries):
        try:
            if isinstance(entry, DispData):
                d = entry
                default_lab = "curve%d" % i
            else:
                d = read_dispdata(entry)
                default_lab = os.path.splitext(os.path.basename(entry))[0]
        except Exception as e:
            print("assemble_dispersion(): skipping entry %r (%s)" % (entry, e))
            continue
        curves.append(d)
        dists.append(d.dist if d.dist is not None else np.nan)
        labs.append(labels[i] if labels is not None else default_lab)

    if not curves:
        raise ValueError("assemble_dispersion(): no dispersion curves could be assembled "
                          "from the given `sources`.")

    if period is not None:
        pgrid = np.asarray(period, dtype=np.float64)
    else:
        all_pmin = pmin if pmin is not None else min(np.nanmin(d.period) for d in curves)
        all_pmax = pmax if pmax is not None else max(np.nanmax(d.period) for d in curves)
        if pscale == 'log':
            pgrid = np.exp(np.linspace(np.log(all_pmin), np.log(all_pmax), nper))
            # clip the float64 log/exp round-trip back onto [all_pmin,all_pmax] exactly --
            # otherwise the last (or first) grid point can land a hair outside the range
            # spanned by every curve's own periods, making interp1d(bounds_error=False)
            # correctly-but-uselessly reject it as extrapolation and leave that whole
            # column all-NaN.
            pgrid[0], pgrid[-1] = all_pmin, all_pmax
        else:
            pgrid = np.linspace(all_pmin, all_pmax, nper)

    vel = np.full((len(curves), len(pgrid)), np.nan)
    for i, d in enumerate(curves):
        v = d.phase_velocity if vtype == 'phase' else d.group_velocity
        v = np.asarray(v, dtype=np.float64).copy()
        if snr_min is not None and d.snr is not None:
            v = np.where(np.asarray(d.snr, dtype=np.float64) >= snr_min, v, np.nan)
        vel[i] = _interp_curve_to_grid(pgrid, d.period, v)

    return dict(period=pgrid, velocity=vel, dist=np.asarray(dists, dtype=np.float64),
                label=labs, n=len(curves))


def plot_dispersion_matrix(assembled, ax=None, mode='lines', color_by='dist', cmap='viridis',
                            show_mean=True, mean_kwargs=None, vtype='group', figsize=(7, 5),
                            vmin=None, vmax=None, nvbins=60, **kwargs):
    """
    Visualize the output of assemble_dispersion().

    ===PARAMETERS===
    assembled: dict as returned by assemble_dispersion().
    ax: existing matplotlib Axes. default None: creates a new figure.
    mode: 'lines' [default]: one line per assembled curve (a "spaghetti" plot),
            optionally colored by `color_by`.
          'heatmap': a 2-D histogram of every (period, velocity) sample across
            all curves, binned onto a velocity axis (see vmin/vmax/nvbins) --
            a quick view of the ensemble's dominant trend and spread when there
            are too many curves for individual lines to stay legible.
    color_by: for mode='lines' only -- an array of length assembled['n'] to color
            each line by, or the string 'dist' [default, uses assembled['dist']];
            None disables per-curve coloring (all lines drawn in one color).
    cmap: colormap, for either mode.
    show_mean: overlay the per-period ensemble mean +/- 1 std-dev curve (both
            computed with NaNs ignored). default True.
    mean_kwargs: dict of kwargs for the mean curve's ax.plot() call (color/label/
            etc); the +/-1 std-dev band uses the same color, at alpha=0.2.
    vtype: only used to label the velocity axis ('group'/'phase' velocity).
    figsize: figure size when a new figure is created.
    vmin,vmax,nvbins: for mode='heatmap' -- the velocity-axis histogram range
            (default: the assembled data's own [nanmin,nanmax]) and bin count.
    kwargs: for mode='lines', passed to each per-curve ax.plot() call.

    ===RETURNS===
    ax: the matplotlib Axes used.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    period = assembled['period']
    vel = assembled['velocity']
    show_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_fig = True

    if mode == 'lines':
        if isinstance(color_by, str) and color_by == 'dist':
            cvals = assembled['dist']
        else:
            cvals = color_by
        if cvals is not None:
            finite = np.isfinite(cvals)
            lo = np.nanmin(cvals[finite]) if np.any(finite) else 0.0
            hi = np.nanmax(cvals[finite]) if np.any(finite) else 1.0
            norm = plt.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
            cm = get_cmap(cmap)
        for i in range(vel.shape[0]):
            color = cm(norm(cvals[i])) if cvals is not None and np.isfinite(cvals[i]) else None
            ax.plot(period, vel[i], '-', lw=0.8, alpha=0.6, color=color, **kwargs)
        if cvals is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='distance (km)' if color_by == 'dist' else 'color_by')
    elif mode == 'heatmap':
        pp = np.tile(period, vel.shape[0])
        vv = vel.ravel()
        good = np.isfinite(pp) & np.isfinite(vv)
        vlo = vmin if vmin is not None else (np.nanmin(vv[good]) if good.any() else 0.0)
        vhi = vmax if vmax is not None else (np.nanmax(vv[good]) if good.any() else 1.0)
        H, xedges, yedges = np.histogram2d(pp[good], vv[good], bins=[len(period), nvbins],
                                            range=[[period.min(), period.max()], [vlo, vhi]])
        pc = ax.pcolormesh(xedges, yedges, H.T, shading='auto', cmap=cmap)
        plt.colorbar(pc, ax=ax, label='curve count')
    else:
        raise ValueError("plot_dispersion_matrix(): mode must be 'lines' or 'heatmap' (got %r)"
                          % mode)

    if show_mean:
        mean_kwargs = dict(mean_kwargs or {})
        mean_kwargs.setdefault('color', 'k')
        mean_kwargs.setdefault('lw', 2.0)
        mean_kwargs.setdefault('label', 'ensemble mean')
        mean_v = np.nanmean(vel, axis=0)
        std_v = np.nanstd(vel, axis=0)
        ax.plot(period, mean_v, **mean_kwargs)
        ax.fill_between(period, mean_v - std_v, mean_v + std_v, color=mean_kwargs['color'],
                         alpha=0.2, lw=0)
        ax.legend(loc='best')

    ax.set_xlabel('Period (s)')
    ax.set_ylabel('%s velocity (km/s)' % vtype.capitalize())
    ax.set_title('Assembled %s dispersion (%d curves)' % (vtype, vel.shape[0]))
    if show_fig:
        plt.show()
    return ax


################################################################
############## CONNECTOR TO seisgo.dispersion INVERSION ##############
################################################################
def dispersion_to_1d_model(disp, thickness, initial_vs, vtype='group', wave_type='rayleigh',
                            mode=1, iterations=8, damp=0.1, smooth=0.5, maxdv=0.02,
                            snr_min=None, period=None):
    """
    Invert one measured dispersion curve for a 1-D layered Vs model, by calling this
    same module's OWN forward_solver()/inversion() functions (defined above, in the
    "DISPERSION EXTRACTION FUNCTIONS" section) directly -- this is a thin connector,
    not a reimplementation. It just extracts a plain (periods, velocity) array from
    a DispData object (or an assemble_dispersion() ensemble) and hands it to
    inversion(); no import is needed since forward_solver()/inversion() are plain
    in-file siblings once this file replaces seisgo/dispersion.py. (The pre-merge
    standalone aftan_dispersion.py version of this function instead imports these
    two from an installed seisgo, guarded by a try/except, for use before the merge.)

    ===PARAMETERS===
    disp: a DispData object, OR a dict as returned by assemble_dispersion() -- in
            the latter case the ensemble-mean velocity curve (NaNs ignored) across
            all assembled curves is used as the inversion target.
    thickness: 1-D array of layer thicknesses (km); the last layer's thickness is
            conventionally 0 for a halfspace. Passed straight through to
            inversion() (see that function's own docstring above).
    initial_vs: 1-D array of starting Vs (km/s), one per layer, same length as
            `thickness`. Passed straight through to inversion().
    vtype: 'group' or 'phase' -- which velocity to invert (must match what `disp`
            actually has valid picks for). default 'group'.
    wave_type,mode,iterations,damp,smooth,maxdv: passed straight through to
            inversion() -- see that function's own docstring above for what each
            controls.
    snr_min: only used when `disp` is a DispData -- drop periods with snr below
            this threshold before inverting. (For dict/assemble_dispersion()
            input, filter with assemble_dispersion()'s own snr_min instead, since
            individual curves' snr arrays aren't carried into the assembled dict.)
    period: optional explicit subset of periods to invert on (must match values
            present in `disp.period` / assembled['period']); default None uses
            every period with a finite velocity value.

    ===RETURNS===
    a dict with:
        vs: the inverted Vs profile (km/s), one value per layer -- inversion()'s
                own return value, unmodified.
        thickness: the input `thickness` array, echoed back for convenience.
        periods,velocity: the (period, velocity) arrays actually used as
                inversion()'s target data, after NaN/snr_min/period filtering and
                sorting by period -- useful to check the fit, e.g. by comparing
                against forward_solver(vs, periods, thickness, wave_type=...,
                mode=..., velocity_type=vtype).
    """
    if isinstance(disp, dict) and 'velocity' in disp and 'period' in disp:
        periods_all = np.asarray(disp['period'], dtype=np.float64)
        vel_all = np.nanmean(np.asarray(disp['velocity'], dtype=np.float64), axis=0)
    else:
        periods_all = np.asarray(disp.period, dtype=np.float64)
        vel_all = np.array(disp.phase_velocity if vtype == 'phase' else disp.group_velocity,
                            dtype=np.float64)
        if snr_min is not None and disp.snr is not None:
            vel_all = np.where(np.asarray(disp.snr, dtype=np.float64) >= snr_min, vel_all, np.nan)

    good = np.isfinite(periods_all) & np.isfinite(vel_all)
    if period is not None:
        good &= np.isin(periods_all, np.asarray(period, dtype=np.float64))
    periods = periods_all[good]
    velocity = vel_all[good]
    if len(periods) < 2:
        raise ValueError("dispersion_to_1d_model(): fewer than 2 valid (period, velocity) "
                          "points available to invert (after NaN/snr_min/period filtering).")
    order = np.argsort(periods)
    periods, velocity = periods[order], velocity[order]

    vs = inversion(periods, velocity, thickness, initial_vs, iterations=iterations, damp=damp,
                    smooth=smooth, wave_type=wave_type, mode=mode, velocity_type=vtype,
                    maxdv=maxdv)
    return dict(vs=vs, thickness=np.asarray(thickness, dtype=np.float64),
                periods=periods, velocity=velocity)


################################################################
################ SYNTHETIC TEST-SIGNAL GENERATORS ################
################################################################
# The two functions below build physically self-consistent synthetic dispersive
# wave trains from a PRESCRIBED dispersion curve (group- or phase-velocity), via the
# stationary-phase construction -- the standard way to generate test data with a
# known-truth dispersion curve to validate an AFTAN-style measurement (aftan()/
# aftan_pmf() above) against. Originally written as test-script helpers; moved here
# so any user of this module can generate synthetic validation data without
# duplicating them.
def make_dispersive_synthetic(dist, dt, npts, periods_true, vg_true, noise_amp=0.01):
    """
    Build a synthetic dispersive wave train with a prescribed GROUP-velocity
    dispersion curve, using the stationary-phase construction: build the spectrum
    with phase(w) such that d(phase)/d(w) = group delay tau(w) = dist/vg(w). Then
    x(t) = Re[ifft(spectrum)] has its energy arriving near t=tau(w) for each
    frequency w, by stationary phase -- the standard way to synthesize a test
    signal with a known group-velocity dispersion curve.

    ===PARAMETERS===
    dist: source-receiver distance (km).
    dt: sampling interval (s) of the returned waveform.
    npts: number of samples to return.
    periods_true,vg_true: the prescribed group-velocity dispersion curve (s,
            km/s), used both to build the signal and as ground truth to compare a
            subsequent aftan()/aftan_pmf() measurement against.
    noise_amp: amplitude (relative to the signal's own peak, which is normalized
            to 1) of additive white noise. default 0.01.

    ===RETURNS===
    data: the synthetic waveform, shape (npts,).
    """
    periods_true = np.asarray(periods_true, dtype=np.float64)
    vg_true = np.asarray(vg_true, dtype=np.float64)
    nfft = next_fast_len(2 * npts)
    freqs = np.fft.fftfreq(nfft, d=dt)
    omega = 2 * np.pi * freqs
    posmask = freqs > 1e-6
    om_pos = omega[posmask]
    f_pos = freqs[posmask]
    per_pos = 1.0 / f_pos

    # group delay (s) at each positive frequency, via interpolation/extrapolation
    # of the prescribed dispersion curve (vg vs period).
    vg_interp = np.interp(per_pos, periods_true, vg_true, left=vg_true[0], right=vg_true[-1])
    tau = dist / vg_interp

    order = np.argsort(om_pos)
    om_sorted = om_pos[order]
    tau_sorted = tau[order]
    phase_sorted = np.concatenate(([0.0], np.cumsum(0.5 * (tau_sorted[1:] + tau_sorted[:-1]) *
                                                      np.diff(om_sorted))))
    phase = np.empty_like(om_pos)
    phase[order] = phase_sorted

    # smooth passband amplitude taper (log-period cosine taper) over [pmin,pmax] band
    pmin_b, pmax_b = periods_true.min() * 0.8, periods_true.max() * 1.2
    x = np.clip((np.log(per_pos) - np.log(pmin_b)) / (np.log(pmax_b) - np.log(pmin_b)), 0, 1)
    amp = np.sin(np.pi * x) ** 0.5
    amp[(per_pos < pmin_b) | (per_pos > pmax_b)] = 0.0

    spec = np.zeros(nfft, dtype=complex)
    spec[posmask] = amp * np.exp(-1j * phase)
    x_full = np.fft.ifft(spec, n=nfft)
    data = 2.0 * np.real(x_full)[:npts]
    data /= np.max(np.abs(data))
    data += noise_amp * np.random.randn(npts)
    return data


def make_synthetic_from_phase_velocity(dist, dt, npts, c_period, c_true, pmin, pmax, noise_amp=0.01):
    """
    Build a synthetic dispersive wave train from a prescribed PHASE-velocity curve
    c(period) (rather than group velocity), using theta(w) = -w*dist/c(w) - pi/4
    (the standard far-field phase of a cylindrically-spreading fundamental-mode
    surface wave). The resulting group delay -- and hence the group velocity an
    AFTAN-style analysis will recover -- follows automatically from d(theta)/d(w),
    so this gives a fully self-consistent (c(w), U(w)) pair, e.g. to validate
    _phase_velocity()'s ambiguity resolution against a known truth.

    ===PARAMETERS===
    dist: source-receiver distance (km).
    dt: sampling interval (s) of the returned waveform.
    npts: number of samples to return.
    c_period,c_true: the prescribed phase-velocity dispersion curve (s, km/s).
    pmin,pmax: nominal period band (s) of the synthetic signal's amplitude taper
            (independent of, but normally close to, c_period's own range).
    noise_amp: amplitude (relative to the signal's own peak, which is normalized
            to 1) of additive white noise. default 0.01.

    ===RETURNS===
    data: the synthetic waveform, shape (npts,).
    per_pos: the (unsorted, FFT-bin-ordered) positive-frequency period vector (s)
            that `U_true_at` below corresponds to.
    U_true_at: the true group velocity (km/s) at each period in `per_pos`, derived
            analytically from c_true via 1/U = 1/c - (w/c^2)*dc/dw -- ground truth
            to compare a subsequent aftan()/aftan_pmf() group-velocity measurement
            against, once interpolated onto that measurement's own period grid.
    """
    nfft = next_fast_len(2 * npts)
    freqs = np.fft.fftfreq(nfft, d=dt)
    omega = 2 * np.pi * freqs
    posmask = freqs > 1e-6
    om_pos = omega[posmask]
    per_pos = 2 * np.pi / om_pos

    srt = np.argsort(c_period)
    cp, cv = np.asarray(c_period)[srt], np.asarray(c_true)[srt]
    spl = CubicSpline(cp, cv, extrapolate=True)
    dspl = spl.derivative()
    per_clipped = np.clip(per_pos, cp[0], cp[-1])
    c_om = spl(per_clipped)
    # dc/domega = dc/dper * dper/domega ; per = 2*pi/omega -> dper/domega = -2*pi/omega^2 = -per/omega
    dc_dper = dspl(per_clipped)
    dc_dom = dc_dper * (-per_pos / om_pos)

    theta = -om_pos * dist / c_om - np.pi / 4.0

    pmin_b, pmax_b = pmin * 0.5, pmax * 1.6
    x = np.clip((np.log(per_pos) - np.log(pmin_b)) / (np.log(pmax_b) - np.log(pmin_b)), 0, 1)
    amp = np.sin(np.pi * x) ** 0.5
    amp[(per_pos < pmin_b) | (per_pos > pmax_b)] = 0.0

    spec = np.zeros(nfft, dtype=complex)
    spec[posmask] = amp * np.exp(1j * theta)
    x_full = np.fft.ifft(spec, n=nfft)
    data = 2.0 * np.real(x_full)[:npts]
    data /= np.max(np.abs(data))
    data += noise_amp * np.random.randn(npts)

    # true group velocity, for reference/comparison: 1/U = 1/c - (w/c^2)*dc/dw
    U_true_at = 1.0 / (1.0 / c_om - (om_pos / c_om ** 2) * dc_dom)
    return data, per_pos, U_true_at
