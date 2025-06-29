#!/usr/bin/env python
# coding: utf-8
############################################
##Utility functions used in processing seismic data.
############################################
#import needed packages.
import sys,time,scipy,obspy,pyasdf
import datetime,os, glob, utm
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot  as plt
from collections import OrderedDict
from scipy.signal import hilbert
from scipy.signal.windows import tukey,hann
from obspy.clients.fdsn import Client
from obspy.core import Stream, Trace, read
# from obspy.core.util.base import _get_function_from_entry_point
from obspy.signal.util import _npts2nfft
from obspy.signal.filter import bandpass
from scipy.fftpack import fft,ifft,fftfreq,next_fast_len
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel
from shapely.geometry import MultiPoint, MultiLineString,Polygon, Point
from shapely.ops import unary_union, polygonize
import netCDF4 as nc
from scipy.spatial import Delaunay
from math import sqrt
import math
from seisgo import helpers
#########################################
def rms(d):
    return np.sqrt(np.mean(d**2))
def get_snr(d,t,dist,vmin,vmax,extend=0,offset=20,axis=1,getwindow=False,db=False,
                side="a",shorten_noise=False):
    """
    Get SNRs of the data with given distance, vmin, and vmax. The signal window will be
    computed using vmin and vmax. The noise window will be the same length as the signal
    window shifted toward the end with the given offset.

    ==========
    d,t,dist,vmin,vmax: REQUIRED. data, time vector, distance, minimum velocity, maximum velocity.
    extend: extend the window length from the computed window based on vmin and vmax. default is 20.
    offset: offset between noise and signal windows, in seconds. default is 20.
    axis: axis for the calculation. default 1.
    db: Decibel or not. Default is False.
    getwindow: return the indices of the signal and noise windows. only the start and end indices.
                Default False.
    side: negative (n) and/or positive (p) or both sides (a) for the given data (time vector). Default: "a"
    shorten_noise: force noise window to fit the data length after the signal window. Default False.
                    If True, the noise window will be smaller than the signal window.
    =======RETURNS======
    snr: [negative, positive]
    [sig_idx_p,noise_idx_p],[sig_idx_n,noise_idx_n]: only return these windows when getwindow is True and side=="a".
    When side != "a" only returns the corresponding window indices.
    """
    d=np.array(d)
    #get window index:
    tmin=dist/vmax
    tmax=extend + dist/vmin
    dt=np.abs(t[1]-t[0])
    shift=int(offset/dt)
    if side.lower() == "a":
        halfn=int(len(t)/2) + 1
    else:
        halfn=0
    sig_idx_p=[int(tmin/dt)+halfn,int(tmax/dt)+halfn]
    winlen=sig_idx_p[1]-sig_idx_p[0]+1
    noise_idx_p= [sig_idx_p[0]+shift+winlen,sig_idx_p[1]+shift+winlen]

    if noise_idx_p[1] > len(t) - 1:
        if shorten_noise:
            print("Noise window end [%d]is larger than the data length [%d]. Force it to stop at the end."%(noise_idx_p[1],len(t)-1))
            noise_idx_p[1] = len(t) - 1
        else:
            raise ValueError("Noise window end [%d]is larger than the data length [%d]. Please adjust it."%(noise_idx_p[1],len(t)-1))

    sig_idx_n=[len(t) - sig_idx_p[1], len(t) - sig_idx_p[0]]
    noise_idx_n=[len(t) - noise_idx_p[1], len(t) - noise_idx_p[0]]

    if d.ndim==1:
        #axis is not used in this case
        if side.lower() == "a":
            snr_n=rms(np.abs(d[sig_idx_n[0]:sig_idx_n[1]+1]))/rms(np.abs(d[noise_idx_n[0]:noise_idx_n[1]+1]))
            snr_p=rms(np.abs(d[sig_idx_p[0]:sig_idx_p[1]+1]))/rms(np.abs(d[noise_idx_p[0]:noise_idx_p[1]+1]))
            snr=[snr_n**2,snr_p**2]
        elif side.lower() == "n":
            snr_n=rms(np.abs(d[sig_idx_n[0]:sig_idx_n[1]+1]))/rms(np.abs(d[noise_idx_n[0]:noise_idx_n[1]+1]))
            snr=snr_n**2
        elif side.lower() == "p":
            snr_p=rms(np.abs(d[sig_idx_p[0]:sig_idx_p[1]+1]))/rms(np.abs(d[noise_idx_p[0]:noise_idx_p[1]+1]))
            snr=snr_p**2
        else:
            raise ValueError(side+" is not supported. use one of: "+str(helpers.xcorr_sides()))

    elif d.ndim==2:
        #
        if axis==1:dim=0
        else:dim=1
        if side.lower() == "a":
            snr=np.ndarray((d.shape[dim],2))
            for i in range(d.shape[dim]):
                snr_n=rms(np.abs(d[i,sig_idx_n[0]:sig_idx_n[1]+1]))/rms(np.abs(d[i,noise_idx_n[0]:noise_idx_n[1]+1]))
                snr_p=rms(np.abs(d[i,sig_idx_p[0]:sig_idx_p[1]+1]))/rms(np.abs(d[i,noise_idx_p[0]:noise_idx_p[1]+1]))
                snr[i,:]=[snr_n**2,snr_p**2]
        elif side.lower() == "n":
            snr=np.ndarray((d.shape[dim],1))
            for i in range(d.shape[dim]):
                snr_n=rms(np.abs(d[i,sig_idx_n[0]:sig_idx_n[1]+1]))/rms(np.abs(d[i,noise_idx_n[0]:noise_idx_n[1]+1]))
                snr[i]=snr_n**2
        elif side.lower() == "p":
            snr=np.ndarray((d.shape[dim],1))
            for i in range(d.shape[dim]):
                snr_p=rms(np.abs(d[i,sig_idx_p[0]:sig_idx_p[1]+1]))/rms(np.abs(d[i,noise_idx_p[0]:noise_idx_p[1]+1]))
                snr[i]=snr_p**2
        else:
            raise ValueError(side+" is not supported. use one of: "+str(helpers.xcorr_sides()))
        #
    else:
        raise ValueError("Only handles ndim <=2.")
        snr=None
    if db:
        snr=10*np.log10(snr)
    if getwindow:
        if side.lower() == "a":
            return snr,[sig_idx_p,noise_idx_p],[sig_idx_n,noise_idx_n]
        elif side.lower() == "n":
            return snr,[sig_idx_n,noise_idx_n]
        elif side.lower() == "p":
            return snr,[sig_idx_p,noise_idx_p]
    else:
        return snr
##
def gaussian(dt,width,shift):
    """
    Produce gaussian shaping wavelet.

    Here the equation is consistent with the source time function in FWANT.

    $g = (\exp{-(t - t0)^2}/a^2)/\sqrt{\pi}a$

    where $a$ is the width paramter for gaussian function, i.e., sigma. $t0$ is the
    time shift parameter.

    ===parameters===
    dt: sampling interval.
    width: gaussian sigma.
    shift:time shift of the center.

    ===RETURNS===
    t: time vector of the wavelet.
    g: gaussian function.
    """
    t0=shift
    a=width

    t=np.arange(0,2*t0+0.5*dt,dt)
    nt=len(t)
    g=np.ndarray((nt))
    tmp=np.exp(-np.power(t-t0,2)/(a*a))
    g=tmp/(np.sqrt(np.pi)*a)
    return t,g
##
def ricker(dt,fc,shift):
    """
    Produce Ricker shaping wavelet.

    Here the equation is consistent with the source time function in FWANT.

    ===parameters===
    dt: sampling interval.
    fc: gaussian sigma.
    shift:time shift of the center.

    ===RETURNS===
    t: time vector of the wavelet.
    r: ricker function.
    """
    t0=shift
    f0=np.sqrt(np.pi)/2.0
    t=np.arange(0,2*t0+0.5*dt,dt)

    nt=len(t)

    u=(t - t0)*2.0*np.pi*fc
    r=(np.power(u,2)/4 - 0.5)*np.exp(-np.power(u,2)/4)*f0
    #
    r = -1.0*r #this is to make the main lobe positive.
    return t,r
#
def subsetindex(full,subset):
    """
    Get the indices of the subset of a list.
    """
    if isinstance(subset,str):subset=[subset]
    idx=[]
    for s in subset:
        idx += [i for i, x in enumerate(full) if x == s]

    return idx
#
def get_filelist(dir=None,extension=None,pattern=None,sort=True):
    """
    Get list of files with absolute path, by specifying the format extension.

    ===========PARAMETERS=============
    dir: directory containing the files.
    extension: file extension (the ending format tag), for example "h5" for asdf file.
    pattern: pattern to use in searching. Wildcards are NOT considered here.
    sort: (optional) to sort the list, default is True.

    ============RETURN=============
    flist: the list of file names with paths.
    """
    if dir is None:
        dir="."
    if extension is None:
        flist=[os.path.join(dir,f) for f in os.listdir(dir)]
    else:
        flist=[os.path.join(dir,f) for f in os.listdir(dir) if f[-len(extension):].lower()==extension.lower()]
    if pattern is not None:
        flist2=[]
        for f in flist:
            if f.find(pattern)>=0: flist2.append(f)
        flist=flist2
    if sort:
        return  sorted(flist)
    else:
        return  flist

def slice_list(flist,step,preserve_end=True):
    """
    Slice a lit of values, with given step. Different from utils.sliding_window(), this function
    provides unique segments with NO overlaps. It works with generic Python list object.

    ========PARAMETERS============
    flist: list to be sliced.
    step: step or length of each segment.
    preserve_end: if True the end element will be included, the last segment may have different length.
                    Default is True.
    """

    step=int(step)
    outlist=[]
    if len(flist)<step:
        outlist.append(flist)
    else:
        idxall=np.arange(0,len(flist),step)

        if idxall[-1]<len(flist)-1 and preserve_end:
            idxall=np.append(idxall,len(flist)-1) #make sure all files are considered.

        if len(idxall)==1:
            outlist=[flist[:idxall[0]]]
        else:
            for i in range(len(idxall)-1):
                sublist=[flist[j] for j in np.arange(idxall[i],idxall[i+1])]
                outlist.append(sublist)
    #
    return outlist
#
def correct_orientations(tr1,tr2,orient):
    """
    Correct horizontal orientations with given orientation data. The output traces
    are corrected and renamed to *E and *N convention.

    Parameters
    ----------
    tr1,tr2: :class:`~obspy.core.Trace`
        Seismic traces for horizontals. tr1 corresponds to N, tr2 corresponds to E
    orient:: Dictionary
        Dictionary containing the orientation information for the horizonal
        components for each station in the format of [orient_h1,orient_h2,orient_error].
        This information can be assembed by calling get_orientations().
    Output
    ---------
    trE, trN :class:`~obspy.core.Trace`
        Seismic traces for horizontals.
        The order of output is important, DO NOT change.
        For OBS data, it does NOT need to be E or N. The variable name is set for convenient
    """
    # Check that all traces are valid Trace objects
    for tr in [tr1, tr2]:
        if not isinstance(tr, Trace):
            raise(Exception("Error correct_orientations() - "
                            + str(tr)+" is not a Trace object"))

    #traces after orientation corrections.
    trE=[]
    trN=[]

    #get net and station name for the trace data
    netsta=tr1.stats.network+'.'+tr1.stats.station
    if netsta not in orient.keys():
        print("Error correct_orientations() - "
                    + netsta+" is not in the orientation list.")
        return trE,trN

    oh1,oh2,oerror=orient[netsta]

    chan1=tr1.stats.channel
    chan2=tr2.stats.channel
    data1=tr1.data
    data2=tr2.data

    trE=tr2.copy()
    trE.stats.channel=chan2[0:2]+'E'
    trN=tr1.copy()
    trN.stats.channel=chan2[0:2]+'N'

    angle=np.deg2rad(360 - oh1) #rotation angle to rotate tr1 to trN
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    v12 = np.array([data2, data1])
    vEN = np.tensordot(rot_mat, v12, axes=1)
    trE.data = vEN[0, :]
    trN.data = vEN[1, :]

    return trE,trN

def image_binary_gradient(data,radius=1):
    """
    Gridsearch image pixels to calculate the gradient 0 or 1. 0 means
    the nearby values are the same. 1 means at least one of the nearby
    values is different than the centered reference grid.

    ===PARAMETERS===
    data: 2-d matrix.
    radius: number of grids to extend from each reference grid. Default=1.

    ==RETURNS===
    outg: output gradient matrix, with the same shape as the input data.
    """
    threshold=0.95
    #sanity check
    if radius <1:
        raise ValueError('search radius must be >=1. Current: '+str(radius))
    #
    R,C=data.shape
    outg=np.ndarray((R,C))
    outg.fill(np.nan)

    for i in range(R):
        if i >= radius and i<= R-radius: #skip the edge grids.
            for j in range(C):
                if j >= radius and j <= C-radius: #skip the edge grids.
                    temp=np.concatenate((data[i,int(j-radius):int(j+radius)+1],
                                          data[int(i-radius):int(i+radius)+1,j]))
                    if np.isnan(temp).all():
                        outg[i,j]=0
                    else:
                        vtemp=np.abs(np.nanmax(temp) - np.nanmin(temp))
                        if vtemp > threshold: outg[i,j]=1
                        else: outg[i,j]=0
    #
    return outg
def xyz2matrix(x,y,z):
    """
    Create matrix from the xyz points, without interpolation.
    The data points should have only unique values at each data point.
    """
    xu=np.sort(np.unique(x))
    yu=np.sort(np.unique(y))
    dxmean=np.nanmean(np.abs(np.diff(xu)))
    dymean=np.nanmean(np.abs(np.diff(yu)))

    zout=np.ndarray((len(xu),len(yu)))
    zout.fill(np.nan)
    xout=np.ndarray((len(xu),len(yu)))
    yout=np.ndarray((len(xu),len(yu)))
    for i in range(len(xu)):
        for j in range(len(yu)):
            idx0=np.where((x > xu[i]-0.1*dxmean) & (x < xu[i]+0.1*dxmean) &
                         (y > yu[j]-0.1*dymean) & (y < yu[j]+0.1*dymean))[0]
            if len(idx0) >0: zout[i,j]=np.average(z[idx0])
            xout[i,j]=xu[i]
            yout[i,j]=yu[j]
            #print(z[idx0])
    #
    return zout,xu,yu
def interp3d(x,y,z,v,xq,yq,zq,verbose=False):
    """
    Interpolate 3d matrix by calling the Scipy interpn function for each 3d point.

    PARAMETERS:
    x,y,z: 1-D vectors of the three dimensions.
    v: values of the 3d matrix
    xq,yq,zq: 1-D vectors of the points to resample.

    RETURN:
    vout: 3d matrix with the size of [len(xq),len(yq),len(zq)]
    """
    vout = np.ndarray((len(xq),len(yq),len(zq)))
    vout.fill(np.nan)
    idx=np.where((xq >= np.nanmin(x)) & (xq <= np.nanmax(x)))[0]
    idy=np.where((yq >= np.nanmin(y)) & (yq <= np.nanmax(y)))[0]
    idz=np.where((zq >= np.nanmin(z)) & (zq <= np.nanmax(z)))[0]

    for i in range(len(idx)):
        if verbose:print(str(i)+" of "+str(len(idx)))
        for j in range(len(idy)):
            for k in range(len(idz)):
                vout[idx[i],idy[j],idz[k]] = scipy.interpolate.interpn((x,y,z),v,(xq[idx[i]],yq[idy[j]],zq[idz[k]]))
    return vout
def interp2d(x,y,v,xq,yq,verbose=False):
    """
    Interpolate 2d matrix by calling the Scipy interpn function for each 2d point.

    PARAMETERS:
    x,y: 1-D vectors of the two dimensions.
    v: values of the 2d matrix
    xq,yq: 1-D vectors of the points to resample.

    RETURN:
    vout: 2d matrix with the size of [len(xq),len(yq)]
    """
    vout = np.ndarray((len(xq),len(yq)))
    vout.fill(np.nan)
    idx=np.where((xq >= np.nanmin(x)) & (xq <= np.nanmax(x)))[0]
    idy=np.where((yq >= np.nanmin(y)) & (yq <= np.nanmax(y)))[0]

    for i in range(len(idx)):
        if verbose:print(str(i)+" of "+str(len(idx)))
        for j in range(len(idy)):
            vout[idx[i],idy[j]] = scipy.interpolate.interpn((x,y),v,(xq[idx[i]],yq[idy[j]]))
    return vout
def interp2d_nonregular(x,y,v,xq,yq,verbose=False):
    """
    Interpolate 2d matrix by calling the Scipy interpn function for each 2d point.

    PARAMETERS:
    x,y: 1-D vectors of the two dimensions.
    v: values of the 2d matrix
    xq,yq: 1-D vectors of the points to resample.

    RETURN:
    vout: 2d matrix with the size of [len(xq),len(yq)]
    """
    vout = np.ndarray((len(xq),len(yq)))
    vout.fill(np.nan)
    idx=np.where((xq >= np.nanmin(x)) & (xq <= np.nanmax(x)))[0]
    idy=np.where((yq >= np.nanmin(y)) & (yq <= np.nanmax(y)))[0]

    for i in range(len(idx)):
        if verbose:print(str(i)+" of "+str(len(idx)))
        for j in range(len(idy)):
            vout[idx[i],idy[j]] = scipy.interpolate.interpn((x,y),v,(xq[idx[i]],yq[idy[j]]))
    return vout
    
def generate_points_in_polygon(outline,spacing):
    """
    Generate points in polygon, defined as a shapely.polygon object.

    outline: list of (x,y) points that define the polygon.
    spacing: spacing of the points to be generated.
    """
    poly=Polygon(outline)
    minx, miny, maxx, maxy = poly.bounds
    x0=np.arange(minx-spacing,maxx+spacing,spacing)
    y0=np.arange(miny-spacing,maxy+spacing,spacing)
    pointsx=[]
    pointsy=[]
    for i in range(len(x0)):
        for j in range(len(y0)):
            p = Point(x0[i], y0[j])
            if poly.contains(p):
                pointsx.append(x0[i])
                pointsy.append(y0[j])
    return pointsx,pointsy
#
#
def points_in_polygon(outline,qx,qy):
    """
    Get points that are within the given polygon. Returns the index list.
    poly: list of (x,y) points that define the polygon
    qx,qy: list of the x and y coordinats of the points
            that are to be checked.

    ===RETURNS===
    ix,iy: indices of x and y for points within the polygon.
    """
    poly=Polygon(outline)
    ix=[]
    iy=[]
    for i in range(len(qx)):
        for j in range(len(qy)):
            p = Point(qx[i], qy[j])
            if poly.contains(p):
                ix.append(i)
                iy.append(j)
    return ix,iy
#
#
def read_gmtlines(file,comment="#",segment=">"):
    """
    Read GMT stype lines from text file. By default, the comment lines
    start with "#" and segments are separated by lines starting with ">".
    They can be specified if different than the defaults.

    file - File name.

    =====RETURNS====
    dall, tags - Data (list of all 2-d arrays for all line segments) and tags.
    """
    tags=[]
    data=[]
    dall=[]
    with open(file,'r') as fo:
        for line in fo:
            idn=line.find("\n")
            if idn>0: line=line[:idn] #trim ending LINE RETURN SYMBOL
            if line[0] == comment: #skip comment line
                pass
            elif line[0]== segment:
                tag=str(line[1:])
                if tag.find("-L"):tag=tag[tag.find("-L")+2:]
                tags.append(tag)
                if len(data)>0:dall.append(np.array(data))
                data=[]
            else:
                if line.find("\t") >0:
                    cols = line.split("\t")
                else:
                    cols = line.split(" ")
                data.append([float(i) for i in cols if len(i)>0 ])
        dall.append(np.array(data))

    return dall,tags
#
#
def listvar_ncmodel(dfile,metadata=False):
    """
    Read 3D seismic model from netCDF file that follows IRIS EMC format.

    dfile - Data file name.
    var - variable name.

    ===RETURNS===
    lon,lat,dep,val - coordinats and the 3-D model (val) for the specified variable.
    """
    ds=nc.Dataset(dfile)
    var=ds.variables

    if metadata:
        md=ds.__dict__
        return var,md
    else:
        return var
def read_ncmodel3d(dfile,var,metadata=False):
    """
    Read 3D seismic model from netCDF file that follows IRIS EMC format.

    dfile - Data file name.
    var - variable name.
    metadata - get metadata or not. Default False.

    ===RETURNS===
    lon,lat,dep,val - coordinats and the 3-D model (val) for the specified variable.
    """
    ds=nc.Dataset(dfile)
    lon=np.array(ds['longitude'][:])
    lat=np.array(ds['latitude'][:])
    dep=np.array(ds['depth'][:])
    val=np.array(ds[var][:])

    if metadata:
        md=ds.__dict__
        return dep,lat,lon,val,md
    else:
        return dep,lat,lon,val
def read_ncmodel2d(dfile,var,metadata=False):
    """
    Read 2D seismic surface models from netCDF file that follows IRIS EMC format.

    dfile - Data file name.
    var - variable name.
    metadata - get metadata or not. Default False.

    ===RETURNS===
    lat,lon,val - coordinats and the 3-D model (val) for the specified variable.
    md - Only returns this when metadata is True.
    """
    ds=nc.Dataset(dfile)
    lon=np.array(ds['longitude'][:])
    lat=np.array(ds['latitude'][:])
    val=np.array(ds[var][:])

    if metadata:
        md=ds.__dict__
        return lat,lon,val,md
    else:
        return lat,lon,val
#
#
def ncmodel_in_polygon(dfile,var,outlines,vmax=9000,stats=False,surface=False,
                        lon_correction=0.0):
    """
    Extract seismic model within polygons from 3d or 2d model in netCDF format.

    ===PARAMETERS===
    dfile - Data file name.
    var - variable name.
    vmax - maximum value, above which will be set to numpy nan.
    stats - If True, returns all statistics (mean, median, min, max, std) of
                the model within the polygon. If False, only returns the mean 1d model.
                Default False.
    lon_correction - add correction to model longitude. Default 0.0.
    ===RETURNS===
    dep - Depth grid. Returns only when surface is False.
    val_mean - Average model value (1d profile in case of 3d ncmodel). Returns in all cases.
    val_median,val_min,val_max,val_std - Only returns these when stats is True.
    """
    if surface: #read in 2d surfaces
        lat,lon,val=read_ncmodel2d(dfile,var)
        lon += lon_correction
        val[val>=vmax]=np.nan

        val_mean=np.ndarray((len(outlines)))
        if stats:
            val_median=np.ndarray((len(outlines)))
            val_min=np.ndarray((len(outlines)))
            val_max=np.ndarray((len(outlines)))
            val_std=np.ndarray((len(outlines)))
        for idx,d in enumerate(outlines):
            ix,iy=points_in_polygon(d,lon,lat)
            val_mean[idx]=np.nanmean(np.nanmean(val[iy,ix]))
            if stats:
                val_median[idx]=np.nanmedian(np.nanmedian(val[iy,ix]))
                val_min[idx]=np.nanmin(np.nanmin(val[iy,ix]))
                val_max[idx]=np.nanmax(np.nanmax(val[iy,ix]))
                val_std[idx]=np.nanstd(val[iy,ix])
        #
        if stats:
            return val_mean,val_median,val_min,val_max,val_std
        else:
            return val_mean
    else:
        dep,lat,lon,val=read_ncmodel3d(dfile,var)
        lon += lon_correction
        val[val>=vmax]=np.nan

        val_mean=np.ndarray((len(outlines),val.shape[0]))
        if stats:
            val_median=np.ndarray((len(outlines),val.shape[0]))
            val_min=np.ndarray((len(outlines),val.shape[0]))
            val_max=np.ndarray((len(outlines),val.shape[0]))
            val_std=np.ndarray((len(outlines),val.shape[0]))
        for idx,d in enumerate(outlines):
            ix,iy=points_in_polygon(d,lon,lat)
            for k in range(val_mean.shape[1]):
                val_mean[idx,k]=np.nanmean(np.nanmean(val[k,iy,ix]))
                if stats:
                    val_median[idx,k]=np.nanmedian(np.nanmedian(val[k,iy,ix]))
                    val_min[idx,k]=np.nanmin(np.nanmin(val[k,iy,ix]))
                    val_max[idx,k]=np.nanmax(np.nanmax(val[k,iy,ix]))
                    val_std[idx,k]=np.nanstd(val[k,iy,ix])
        #
        if stats:
            return dep,val_mean,val_median,val_min,val_max,val_std
        else:
            return dep,val_mean
#
def matrix_in_polygon(x,y,z,val,outlines,vmax=9000.0,stats=False,correction=[0,0,0]):
    """
    Extract matrix values within (x,y) polygons from 3d or 2d data.

    ===PARAMETERS===
    x,y,z: coordinate vectors of the data. set z to None for 2d data.
    vmax - maximum value, above which will be set to numpy nan.
    stats - If True, returns all statistics (mean, median, min, max, std) of
                the model within the polygon. If False, only returns the mean 1d model.
                Default False.
    correction - add correction to model coordinates. Default [0,0,0].
    ===RETURNS===
    z - Z grid. Returns only when surface is False.
    val_mean - Average model value (1d profile in case of 3d ncmodel). Returns in all cases.
    val_median,val_min,val_max,val_std - Only returns these when stats is True.
    """
    if z is None: surface=True
    else: surface=False

    if surface: #read in 2d surfaces
        x += correction[0]
        y += correction[1]
        val[val>=vmax]=np.nan

        val_mean=np.ndarray((len(outlines)))
        val_mean.fill(np.nan)
        if stats:
            val_median=np.ndarray((len(outlines)))
            val_min=np.ndarray((len(outlines)))
            val_max=np.ndarray((len(outlines)))
            val_std=np.ndarray((len(outlines)))
            val_median.fill(np.nan)
            val_min.fill(np.nan)
            val_max.fill(np.nan)
            val_std.fill(np.nan)
        for idx,d in enumerate(outlines):
            ix,iy=points_in_polygon(d,x,y)
            if len(ix) >0:
                dtemp=val[ix,iy]
                if not np.isnan(dtemp).all():
                    val_mean[idx]=np.nanmean(dtemp)
                    if stats:
                        val_median[idx]=np.nanmedian(dtemp)
                        val_min[idx]=np.nanmin(dtemp)
                        val_max[idx]=np.nanmax(dtemp)
                        val_std[idx]=np.nanstd(dtemp)
        #
        if stats:
            return val_mean,val_median,val_min,val_max,val_std
        else:
            return val_mean
    else:
        x += correction[0]
        y += correction[1]
        z += correction[2]
        val[val>=vmax]=np.nan

        val_mean=np.ndarray((len(outlines),val.shape[0]))
        val_mean.fill(np.nan)
        if stats:
            val_median=np.ndarray((len(outlines),val.shape[0]))
            val_min=np.ndarray((len(outlines),val.shape[0]))
            val_max=np.ndarray((len(outlines),val.shape[0]))
            val_std=np.ndarray((len(outlines),val.shape[0]))
            val_median.fill(np.nan)
            val_min.fill(np.nan)
            val_max.fill(np.nan)
            val_std.fill(np.nan)
        for idx,d in enumerate(outlines):
            ix,iy=points_in_polygon(d,x,y)
            if len(ix) >0:
                for k in range(val_mean.shape[1]):
                    dtemp=val[k,iy,ix]
                    if not np.isnan(dtemp).all():
                        val_mean[idx,k]=np.nanmean(dtemp)
                        if stats:
                            val_median[idx,k]=np.nanmedian(dtemp)
                            val_min[idx,k]=np.nanmin(dtemp)
                            val_max[idx,k]=np.nanmax(dtemp)
                            val_std[idx,k]=np.nanstd(dtemp)
        #
        if stats:
            return z,val_mean,val_median,val_min,val_max,val_std
        else:
            return z,val_mean
# ##################### qml_to_event_list #####################################
def qml_to_event_list(events_QML,to_pd=False):
    print("WARNING: this function has been renamed to qml2list. This warning will be removed in v0.7.x.")
    return qml2list(events_QML,to_pd=to_pd)

# modified from qml_to_event_list in obspyDMT.utils.event_handler.py
def qml2list(events_QML,to_pd=False,location_only=False):
    """
    convert QML to event list

    ===PARAMETERS===
    events_QML: event qml (OBSPY CATALOG object)
    to_pd: convert to Pandas DataFrame object. Default: False.
    location_only: only extract lat and longitude. Default: False. Date and time will still be extracted.

    ====return====
    events: a list of event information or a pandas dataframe object.
    """
    events = []
    for i in range(len(events_QML)):
        if location_only: 
            try:
                event_time = events_QML.events[i].preferred_origin().time or \
                            events_QML.events[i].origins[0].time
                event_time_month = '%02i' % int(event_time.month)
                event_time_day = '%02i' % int(event_time.day)
                event_time_hour = '%02i' % int(event_time.hour)
                event_time_minute = '%02i' % int(event_time.minute)
                event_time_second = '%02i' % int(event_time.second)
            except Exception as error:
                print(error)
                continue

            try:
                events.append(OrderedDict(
                    [('number', i+1),
                    ('latitude',
                    events_QML.events[i].preferred_origin().latitude or
                    events_QML.events[i].origins[0].latitude),
                    ('longitude',
                    events_QML.events[i].preferred_origin().longitude or
                    events_QML.events[i].origins[0].longitude),
                    ('depth',np.nan),
                    ('datetime', event_time),
                    ('magnitude',np.nan),
                    ('magnitude_type',"UD"),
                    ('author',"UD"),
                    ('event_id', str(event_time.year) +
                    event_time_month + event_time_day + '_' +
                    event_time_hour + event_time_minute +
                    event_time_second + '.a'),
                    ('origin_id', events_QML.events[i].preferred_origin_id or
                    events_QML.events[i].origins[0].resource_id.resource_id),
                    ('flynn_region', 'NAN'),
                    ]))
            except Exception as error:
                print(error)
                continue
        else:
            try:
                event_time = events_QML.events[i].preferred_origin().time or \
                            events_QML.events[i].origins[0].time
                event_time_month = '%02i' % int(event_time.month)
                event_time_day = '%02i' % int(event_time.day)
                event_time_hour = '%02i' % int(event_time.hour)
                event_time_minute = '%02i' % int(event_time.minute)
                event_time_second = '%02i' % int(event_time.second)

                if not hasattr(events_QML.events[i], 'preferred_mag'):
                    events_QML.events[i].preferred_mag = \
                        events_QML.events[i].magnitudes[0].mag
                    events_QML.events[i].preferred_mag_type = \
                        events_QML.events[i].magnitudes[0].magnitude_type
                    events_QML.events[i].preferred_author = 'None'
                else:
                    if not hasattr(events_QML.events[i], 'preferred_author'):
                        if events_QML.events[i].preferred_magnitude().creation_info:
                            events_QML.events[i].preferred_author = \
                                events_QML.events[i].preferred_magnitude().creation_info.author
                        elif events_QML.events[i].magnitudes[0].creation_info:
                            events_QML.events[i].preferred_author = \
                                events_QML.events[i].magnitudes[0].creation_info.author
            except Exception as error:
                print(error)
                continue
            try:
                if not events_QML.events[i].focal_mechanisms == []:
                    if events_QML.events[i].preferred_focal_mechanism()['moment_tensor']['tensor']:
                        focal_mechanism = [
                            events_QML.events[i].preferred_focal_mechanism()
                            ['moment_tensor']['tensor']['m_rr'],
                            events_QML.events[i].preferred_focal_mechanism()
                            ['moment_tensor']['tensor']['m_tt'],
                            events_QML.events[i].preferred_focal_mechanism()
                            ['moment_tensor']['tensor']['m_pp'],
                            events_QML.events[i].preferred_focal_mechanism()
                            ['moment_tensor']['tensor']['m_rt'],
                            events_QML.events[i].preferred_focal_mechanism()
                            ['moment_tensor']['tensor']['m_rp'],
                            events_QML.events[i].preferred_focal_mechanism()
                            ['moment_tensor']['tensor']['m_tp']]
                    else:
                        found_foc_mech = False
                        for foc_mech_qml in events_QML.events[i].focal_mechanisms:
                            if foc_mech_qml['moment_tensor']['tensor']:
                                focal_mechanism = [
                                    foc_mech_qml['moment_tensor']['tensor']['m_rr'],
                                    foc_mech_qml['moment_tensor']['tensor']['m_tt'],
                                    foc_mech_qml['moment_tensor']['tensor']['m_pp'],
                                    foc_mech_qml['moment_tensor']['tensor']['m_rt'],
                                    foc_mech_qml['moment_tensor']['tensor']['m_rp'],
                                    foc_mech_qml['moment_tensor']['tensor']['m_tp']
                                ]
                                found_foc_mech = True
                                break
                        if not found_foc_mech:
                            focal_mechanism = False
                else:
                    focal_mechanism = False
            except AttributeError:
                print("[WARNING] focal_mechanism does not exist for " \
                    "event: %s -- set to False" % (i+1))
                focal_mechanism = False
            except TypeError:
                focal_mechanism = False
            except Exception as error:
                print(error)
                focal_mechanism = False

            try:
                if not events_QML.events[i].focal_mechanisms == []:
                    source_duration = [
                        events_QML.events[i].preferred_focal_mechanism()
                        ['moment_tensor']['source_time_function']['type'],
                        events_QML.events[i].preferred_focal_mechanism()
                        ['moment_tensor']['source_time_function']
                        ['duration']]
                    if not source_duration[1]:
                        source_duration = mag_duration(
                            mag=events_QML.events[i].preferred_mag)
                else:
                    source_duration = mag_duration(
                        mag=events_QML.events[i].preferred_mag)
            except AttributeError:
                print("[WARNING] source duration does not exist for " \
                    "event: %s -- set to False" % (i+1))
                source_duration = False
            except TypeError:
                source_duration = False
            except Exception as error:
                print(error)
                source_duration = False

            try:
                events.append(OrderedDict(
                    [('number', i+1),
                    ('latitude',
                    events_QML.events[i].preferred_origin().latitude or
                    events_QML.events[i].origins[0].latitude),
                    ('longitude',
                    events_QML.events[i].preferred_origin().longitude or
                    events_QML.events[i].origins[0].longitude),
                    ('depth',
                    events_QML.events[i].preferred_origin().depth/1000. or
                    events_QML.events[i].origins[0].depth/1000.),
                    ('datetime', event_time),
                    ('magnitude',
                    events_QML.events[i].preferred_mag),
                    ('magnitude_type',
                    events_QML.events[i].preferred_mag_type),
                    ('author',
                    events_QML.events[i].preferred_author),
                    ('event_id', str(event_time.year) +
                    event_time_month + event_time_day + '_' +
                    event_time_hour + event_time_minute +
                    event_time_second + '.a'),
                    ('origin_id', events_QML.events[i].preferred_origin_id or
                    events_QML.events[i].origins[0].resource_id.resource_id),
                    ('focal_mechanism', focal_mechanism),
                    ('source_duration', source_duration),
                    ('flynn_region', 'NAN'),
                    ]))
            except Exception as error:
                print(error)
                continue
    if to_pd:
        return pd.DataFrame(events)
    else:
        return events

# ##################### mag_duration ###################################
# modified from the same function in obspyDMT.utils.event_handler.py
def mag_duration(mag, type_curve=1):
    """
    calculate the source duration out of magnitude
    type_curve can be 1, 2, 3:
    1: 2005-2014
    2: 1976-1990
    3: 1976-2014
    :param mag:
    :param type_curve:
    :return:
    """
    if type_curve == 1:
        half_duration = 0.00272*np.exp(1.134*mag)
    elif type_curve == 2:
        half_duration = 0.00804*np.exp(1.025*mag)
    elif type_curve == 3:
        half_duration = 0.00392*np.exp(1.101*mag)
    else:
        sys.exit('%s Type for magnitude to source duration conversion is not '
                 'implemented' % type_curve)
    source_duration = round(half_duration, 3)*2
    return ['triangle', source_duration]
# sta_info_from_inv(inv) is modified from noise_module (with the same name)
#Check NoisePy: https://github.com/mdenolle/NoisePy
# added functionality to process an array of inventory
def sta_info_from_inv(inv,mode='single'):
    '''
    this function outputs station info from the obspy inventory object.
    PARAMETERS:
    ----------------------
    inv: obspy inventory object
    mode: "single" (one inv object only contains one station) or 
          "array" (one inv object contains one network with multiple stations).
          default is "single".
          
    RETURNS:
    ----------------------
    sta: station name
    net: netowrk name
    lon: longitude of the station
    lat: latitude of the station
    elv: elevation of the station
    location: location code of the station
    '''
    # load from station inventory
    sta=[]
    net=[]
    lon=[]
    lat=[]
    elv=[]
    location=[]

    if mode.lower() == "single":
        for i in range(len(inv[0])):
            sta.append(inv[0][i].code)
            net.append(inv[0].code)
            lon.append(inv[0][i].longitude)
            lat.append(inv[0][i].latitude)
            if inv[0][i][0].elevation:
                elv.append(inv[0][i][0].elevation)
            else:
                elv.append(0.)
    
            # print(inv[0][i])
            # print(inv[0][i].location_code)
            if len(inv[0][i][0].location_code)>0:
                location.append(inv[0][i][0].location_code)
            else:
                location.append('00')
    
        if len(inv[0])==1:
            sta=sta[0]
            net=net[0]
            lon=lon[0]
            lat=lat[0]
            elv=elv[0]
            location=location[0]
    elif mode.lower() == "array":
        for i in range(len(inv)):
            for j in range(len(inv[i])):
                sta.append(inv[i][j].code)
                net.append(inv[i].code)
                lon.append(inv[i][j].longitude)
                lat.append(inv[i][j].latitude)
                if inv[i][j].elevation:
                    elv.append(inv[i][j].elevation)
                else:
                    elv.append(0.)
            
                # print(inv[0][i])
                # print(inv[0][i].location_code)
                try:
                    location.append(inv[i][j].location_code)
                except:
                    location.append('00')
    # print(sta,net,lon,lat,elv,location)
    return sta,net,lon,lat,elv,location

def get_tt(event_lat, event_long, sta_lat, sta_long, depth_km,model="iasp91",type='first'):
    """
    Get the seismic phase arrival time of the specified earthquake at the station.
    """
    sta_t = locations2degrees(event_lat, event_long, sta_lat, sta_long)
    taup = TauPyModel(model=model)
    arrivals = taup.get_travel_times(source_depth_in_km=depth_km,distance_in_degree=sta_t)
    if type == 'first':
        tt = arrivals[0].time
        ph = arrivals[0].phase
    else: #get specific phase
        phase_found=False
        phaseall=[]
        for i in range(len(arrivals)):
            phaseall.append(arrivals[i].phase.name)
            if arrivals[i].phase.name == type:
                tt = arrivals[i].time
                ph = type
                phase_found=True
                break
        if not phase_found:
            raise ValueError('phase <'+type+' > not found in '+str(phaseall))
    # del arrivals

    return tt,ph

#modified from NoisePy function
def stats2inv(stats,locs=None,format=None):
    '''
    this function creates inventory given the stats parameters in an obspy stream or a station list.

    PARAMETERS:
    ------------------------
    stats: obspy trace stats object containing all station header info
    locs:  panda data frame of the station list. it is needed for converting miniseed files into ASDF
    format: format of the original data that the obspy trace was built from. if not specified, it will
            read the format by the Trace._format attribute. 'sac' format will be used if there is a sac
            dictionary in stats.
    RETURNS:
    ------------------------
    inv: obspy inventory object of all station info to be used later
    '''
    staxml    = False
    respdir   = "."
    if format is None:
        input_fmt = stats._format.lower()
        if 'sac' in list(stats.keys()):
            input_fmt = 'sac'
    else:
        input_fmt = format

    if staxml:
        if not respdir:
            raise ValueError('Abort! staxml is selected but no directory is given to access the files')
        else:
            invfile = glob.glob(os.path.join(respdir,'*'+stats.station+'*'))
            if os.path.isfile(str(invfile)):
                inv = obspy.read_inventory(invfile)
                return inv

    inv = Inventory(networks=[],source="homegrown")

    if input_fmt=='sac':
        if 'sac' not in list(stats.keys()):
            raise ValueError('Abort! sac key is not in stats for input format: sac.')
        else:
            net = Network(
                # This is the network code according to the SEED standard.
                code=stats.network,
                stations=[],
                description="created from SAC and resp files",
                start_date=stats.starttime)

            sta = Station(
                # This is the station code according to the SEED standard.
                code=stats.station,
                latitude=stats.sac["stla"],
                longitude=stats.sac["stlo"],
                elevation=stats.sac["stel"],
                creation_date=stats.starttime,
                site=Site(name="First station"))

            cha = Channel(
                # This is the channel code according to the SEED standard.
                code=stats.channel,
                # This is the location code according to the SEED standard.
                location_code=stats.location,
                # Note that these coordinates can differ from the station coordinates.
                latitude=stats.sac["stla"],
                longitude=stats.sac["stlo"],
                elevation=stats.sac["stel"],
                depth=-stats.sac["stel"],
                azimuth=stats.sac["cmpaz"],
                dip=stats.sac["cmpinc"],
                sample_rate=stats.sampling_rate)

    else:# input_fmt == 'mseed':
        if locs is not None:
            ista=locs[locs['station']==stats.station].index.values.astype('int64')[0]

            net = Network(
                # This is the network code according to the SEED standard.
                code=locs.iloc[ista]["network"],
                stations=[],
                description="created from SAC and resp files",
                start_date=stats.starttime)

            sta = Station(
                # This is the station code according to the SEED standard.
                code=locs.iloc[ista]["station"],
                latitude=locs.iloc[ista]["latitude"],
                longitude=locs.iloc[ista]["longitude"],
                elevation=locs.iloc[ista]["elevation"],
                creation_date=stats.starttime,
                site=Site(name="First station"))

            cha = Channel(
                code=stats.channel,
                location_code=stats.location,
                latitude=locs.iloc[ista]["latitude"],
                longitude=locs.iloc[ista]["longitude"],
                elevation=locs.iloc[ista]["elevation"],
                depth=-locs.iloc[ista]["elevation"],
                azimuth=0,
                dip=0,
                sample_rate=stats.sampling_rate)
        else:
            raise ValueError('locs has to be specified for miniseed data and other formats.')

    response = obspy.core.inventory.response.Response()

    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)

    return inv

# split_datetimestr(inv) is modified from NoisePy.noise_module.get_event_list()
#Check NoisePy: https://github.com/mdenolle/NoisePy
def split_datetimestr(dtstr1,dtstr2,inc_hours):
    '''
    this function calculates the datetime list between datetime1 and datetime2 by
    increment of inc_hours in the formate of %Y_%m_%d_%H_%M_%S

    PARAMETERS:
    ----------------
    dtstr1: string of the starting time -> 2010_01_01_0_0
    dtstr2: string of the ending time -> 2010_10_11_0_0
    inc_hours: integer of incremental hours
    RETURNS:
    ----------------
    dtlist: a numpy character list
    '''
    date1=dtstr1.split('_')
    date2=dtstr2.split('_')
    y1=int(date1[0]);m1=int(date1[1]);d1=int(date1[2])
    h1=int(date1[3]);mm1=int(date1[4]);mn1=int(date1[5])
    y2=int(date2[0]);m2=int(date2[1]);d2=int(date2[2])
    h2=int(date2[3]);mm2=int(date2[4]);mn2=int(date2[5])

    d1=datetime.datetime(y1,m1,d1,h1,mm1,mn1)
    d2=datetime.datetime(y2,m2,d2,h2,mm2,mn2)
    dt=datetime.timedelta(hours=inc_hours)

    dtlist = []
    while(d1<d2):
        dtlist.append(d1.strftime('%Y_%m_%d_%H_%M_%S'))
        d1+=dt
    dtlist.append(d2.strftime('%Y_%m_%d_%H_%M_%S'))

    return dtlist

#Adapted from NoisePy function with the same name.
@jit('float32[:](float32[:],float32)')
def segment_interpolate(sig1,nfric):
    '''
    this function interpolates the data to ensure all points located on interger times of the
    sampling rate (e.g., starttime = 00:00:00.015, delta = 0.05.)
    PARAMETERS:
    ----------------------
    sig1:  seismic recordings in a 1D array
    nfric: the amount of time difference between the point and the adjacent assumed samples
    RETURNS:
    ----------------------
    sig2:  interpolated seismic recordings on the sampling points
    '''
    npts = len(sig1)
    sig2 = np.zeros(npts,dtype=np.float32)

    #----instead of shifting, do a interpolation------
    for ii in range(npts):

        #----deal with edges-----
        if ii==0 or ii==npts-1:
            sig2[ii]=sig1[ii]
        else:
            #------interpolate using a hat function------
            sig2[ii]=(1-nfric)*sig1[ii+1]+nfric*sig1[ii]

    return sig2

#
def get_tracetag(tr):
    """
    Returns the standard OBSPY format tag for seismic trace.

    Parameter
    ----------
    tr::class:`~obspy.core.Trace`
        Seismic trace.

    Return
    ----------
    tag::String
        Tag for the input trace.
    """
    tag=''
    if not isinstance(tr, Trace):
        raise(Exception("Error get_tracetag() - "
                        + str(tr)+" is not a Trace object"))
    if len(tr.stats.location) == 0:
        tlocation='00'
    else:
        tlocation=tr.stats.location

    tag=tr.stats.channel.lower()+'_'+tlocation.lower()

    return tag

# Modified from Zhitu Ma. Modified by Xiaotao to get filter frequencies from the arguments
# 1. added titles for multiple plots
# 2. determine freqmax as the Nyquist frequency, if not specified
# 3. Added mode with option to plot overlapping figures.
def plot_trace(tr_list,freq=[],size=(10,9),ylabels=[],datalabels=[],\
               title=[],outfile='trace.png',xlimit=[],subplotpar=[],              \
               mode="subplot",spacing=2.0,colors=[],verbose=False,scale=1,
               savefig=False):
    """
    mode: subplot, overlap, or gather. In gather mode, traces will be offset and normalized.
    """
    tr_list=list(tr_list)
    plt.figure(figsize=size)
    ntr=len(tr_list)
    if len(subplotpar)==0 and mode.lower()=="subplot":
        subplotpar=(ntr,1)

    myymin=[]
    myymax=[]
    for itr,tr in enumerate(tr_list,1):
        if isinstance(tr,obspy.core.stream.Stream) or isinstance(tr,list):
            if len(tr) >0:
                tc=tr[0].copy()
            else:
                continue
        else:
            tc=tr.copy()
        tt=tc.times()
        if len(xlimit)==0:
            xlimit=[np.min(tt),np.max(tt)]

        imin = np.searchsorted(tt,xlimit[0],side="left")
        imax = np.searchsorted(tt,xlimit[1],side="left")

        if len(freq)>0:
            if verbose:print("station %s.%s, filtered at [%6.3f, %6.3f]" % (tc.stats.network,
                                                            tc.stats.station,freq[0],freq[1]))
            tc.filter('bandpass',freqmin=freq[0],freqmax=freq[1],zerophase=True)
        else:
            if verbose:print("station %s.%s" % (tc.stats.network,tc.stats.station))

        if mode=="subplot":
            ax=plt.subplot(subplotpar[0],subplotpar[1],itr)

            if len(colors)==0:
                plt.plot(tt,tc.data)
            elif len(colors)==1:
                plt.plot(tt,tc.data,colors[0])
            else:
                plt.plot(tt,tc.data,colors[itr-1])
            plt.xlabel("time (s)")
            ax.ticklabel_format(axis='x',style='plain')
            if np.max(np.abs(tc.data[imin:imax])) >= 1e+4 or \
                            np.max(np.abs(tc.data[imin:imax])) <= 1e-4:
                ax.ticklabel_format(axis='both',style='sci')
            if len(ylabels)>0:
                plt.ylabel(ylabels[itr-1])
            if len(title)>0:
                plt.title(title[itr-1])
            if len(xlimit)>0:
                plt.xlim(xlimit)
            plt.ylim(0.9*np.min(tc.data[imin:imax]),1.1*np.max(tc.data[imin:imax]))
            if len(freq)>0:
                plt.text(np.mean(xlimit),0.9*np.max(tc.data[imin:imax]),\
                        "["+str(freq[0])+", "+str(freq[1])+"] Hz", \
                         horizontalalignment='center',verticalalignment='center',fontsize=12)
            plt.tight_layout(pad=spacing)
        elif mode=="overlap":
            if itr==1:ax=plt.subplot(1,1,1)
            if len(colors)==0:
                plt.plot(tt,tc.data)
            elif len(colors)==1:
                plt.plot(tt,tc.data,colors[0])
            else:
                plt.plot(tt,tc.data,colors[itr-1])
            plt.xlabel("time (s)")
            myymin.append(0.9*np.min(tc.data[imin:imax]))
            myymax.append(1.1*np.max(tc.data[imin:imax]))
            if itr==ntr:
                ax.ticklabel_format(axis='x',style='plain')
                if len(datalabels)>0: ax.legend(datalabels)
                if len(ylabels)>0:
                    plt.ylabel(ylabels[0])
                if len(title)>0:
                    plt.title(title)
                if len(xlimit)>0:
                    plt.xlim(xlimit)
                plt.ylim(np.min(myymin),np.max(myymax))
                if len(freq)>0:
                    plt.text(np.mean(xlimit),0.85*np.max(myymax),"["+str(freq[0])+", "+str(freq[1])+"] Hz",\
                             horizontalalignment='center',verticalalignment='center',fontsize=14)
        elif mode=="gather":
            if itr==1:ax=plt.subplot(1,1,1)
            if len(xlimit)>0:
                # get maximum for normalization.
                tindx=np.where((tt>=xlimit[0] & tt<=xlimit[1]))[0]
                trace_max=np.max(np.abs(tc.data[tindx]))
            else:
                trace_max=np.max(np.abs(tc.data))
            if len(colors)==0:
                plt.plot(tt,itr-1+0.5*scale*tc.data/trace_max)
            elif len(colors)==1:
                plt.plot(tt,itr-1+0.5*scale*tc.data/trace_max,colors[0])
            else:
                plt.plot(tt,itr-1+0.5*scale*tc.data/trace_max,colors[itr-1])
            plt.xlabel("time (s)")
            plt.text(xlimit[0]+10,itr-1+0.2,tc.stats.network+"."+tc.stats.station,
                    horizontalalignment='left',verticalalignment='center',fontsize=11)
            if itr==ntr:
                ax.ticklabel_format(axis='x',style='plain')
                if len(datalabels)>0: ax.legend(datalabels)
                if len(ylabels)>0:
                    plt.ylabel(ylabels[0])
                if len(title)>0:
                    if len(freq)>0:
                        plt.title(title+": ["+str(freq[0])+", "+str(freq[1])+"] Hz")
                    else:
                        plt.title(title)
                else:
                    if len(freq)>0:
                        plt.title("gather: ["+str(freq[0])+", "+str(freq[1])+"] Hz")
                    else:
                        plt.title("gather")
                if len(xlimit)>0:
                    plt.xlim(xlimit)
                plt.ylim([-0.7,ntr-0.3])
                if len(freq)>0:
                    plt.text(np.mean(xlimit),0.95*ntr,"["+str(freq[0])+", "+str(freq[1])+"] Hz",\
                             horizontalalignment='center',verticalalignment='center',fontsize=12)
        else:
            raise ValueError("mode: %s is not recoganized. Can ONLY be: subplot, overlap, or gather."%(mode))

    if savefig:
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()

def check_overlap(t1,t2,error=0):
    """
    check the common
    t1,t2: list or numpy arrays.
    error: measurement error in comparison. default is 0
    """
    ind1=[]
    ind2=[]
    if isinstance(t1,list):t1=np.array(t1)
    if isinstance(t2,list):t2=np.array(t2)

    for i in range(len(t1)):
        f1=t1[i]
        ind_temp=np.where(np.abs(t2-f1)<=error)[0]

        if len(ind_temp)>0:
            ind1.append(i)
            ind2.append(ind_temp[0])

    return ind1,ind2
#Modified from noisepy function cut_trace_make_statis().
def slicing_trace(source,win_len_secs,step_secs=None,taper_frac=0.02):
    '''
    This function cuts continous noise data into user-defined segments, estimate the statistics of
    each segment and keep timestamp of each segment for later use.
    PARAMETERS:
    ----------------------
    source: obspy stream object
    exp_len_hours: expected length of the data (source) in hours
    win_len_secs: length of the slicing segments in seconds
    step_secs: step of slicing in seconds. When None (default) or 0.0, only returns one window.

    RETURNS:
    ----------------------
    trace_stdS: standard deviation of the noise amplitude of each segment
    dataS_t:    timestamps of each segment
    dataS:      2D matrix of the segmented data
    '''
    # statistic to detect segments that may be associated with earthquakes
    #demean and detrend the whole trace first:
    trace_data=source[0].data.copy()
    trace_data=detrend(demean(trace_data))
    all_madS = mad(np.abs(trace_data))	            # median absolute deviation over all noise window
    all_stdS = np.std(np.abs(trace_data))	        # standard deviation over all noise window
    if all_madS==0 or all_stdS==0 or np.isnan(all_madS) or np.isnan(all_stdS):
        print("return empty! madS or stdS equals to 0 for %s" % source)
        return [],[],[]

    if isinstance(source,Trace):source=Stream([source])
    # useful parameters for trace sliding
    sps  = source[0].stats.sampling_rate
    starttime = source[0].stats.starttime-obspy.UTCDateTime(1970,1,1)
    duration = source[0].stats.endtime-obspy.UTCDateTime(1970,1,1) - starttime

    if duration < win_len_secs:
        print("return empty! data duration is < slice length." % source)
        return [],[],[]
    if step_secs is None or step_secs == 0.0:
        nseg=1
        npts_step = 0
    else:
        nseg = int(np.floor((duration-win_len_secs)/step_secs))+1
        npts_step = int(step_secs*sps)

    # initialize variables
    #print(duration,win_len_secs,step_secs)
    npts = int(win_len_secs*sps)
    trace_stdS = np.zeros(nseg,dtype=np.float32)
    dataS    = np.zeros(shape=(nseg,npts),dtype=np.float32)
    dataS_t  = np.zeros(nseg,dtype=np.float32)

    print('slicing trace into ['+str(nseg)+'] segments.')

    indx1 = 0
    for iseg in range(nseg):
        indx2 = indx1+npts
        dataS[iseg] = trace_data[indx1:indx2]
        dataS_t[iseg]    = starttime+step_secs*iseg
        indx1 += npts_step

    # 2D array processing
    dataS = detrend(demean(dataS))
    dataS = taper(dataS,fraction=taper_frac)
    for iseg in range(nseg):
        trace_stdS[iseg] = (np.max(np.abs(dataS[iseg]))/all_stdS)

    return trace_stdS,dataS_t,dataS

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
# Modified by Xiaotao to return window starting indices and the option of forcing to slide through full length.
def sliding_window(a, ws, ss=None, wind=None, getindex=False,full_length=False,verbose=False):
    """
    Function to split a data array into overlapping, possibly tapered sub-windows.

    Parameters
    ----------
    a : :class:`~numpy.ndarray`
        1D array of data to split
    ws : int
        Window size in samples
    ss : int
        Step size in samples. If not provided, window and step size
         are equal.
    wind : :class:`~numpy.ndarray`
        1d array to specify the window used to apply taper or None (default).
    getindex : bool
        Save/return the start index for each window if True.
    full_length : bool
        Add an extra window to include the leftover samples to make sure sliding
        through the entire trace with full-length. This is done by measuring one
        window starting from the end backward. When False, this function skips the
        tailing samples if less than the window size.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        1D array of windowed data
    nd : int
        Number of windows
    idx : :class:`~numpy.ndarray`
        (Optional) The starting indices of the windows, with the size of [nd,1]

    """

    if full_length and verbose:
        print("WARNING: Force slide to the full length, the last window measures backward from the end.")
    if ws > len(a):
        raise(Exception("Error slicing() - window size is bigger than data length."))

    if ss is None:
        # no step size was provided. Return non-overlapping windows
        ss = ws

    # Calculate the number of windows to return, ignoring leftover samples, and
    # allocate memory to contain the samples
    nd = len(a) // ss

    tailcare=False

    if (nd-1)*ss + ws > len(a):
        if full_length:
            tailcare = True
        else:
            nd = nd - 1
    elif (nd-1)*ss + ws < len(a) and full_length:
        tailcare = True
        nd = nd + 1

    out = np.ndarray((nd, ws), dtype=a.dtype)
    idx = np.ndarray((nd,),dtype=int)

    if nd==0:
        if wind is not None:
            out = a * wind
        else:
            out = a
        idx=0
    else:
        for i in range(nd):
            # "slide" the window along the samples
            start = i * ss
            stop = start + ws
            # print(i,start,stop,len(a))
            # print(i,nd)
            if stop > len(a) and tailcare:
                    stop = len(a)
                    start = stop - ws
                    # print(i,start,stop)

            if stop <= len(a):
                if wind is not None:
                    out[i] = a[start: stop] * wind
                else:
                    out[i] = a[start: stop]

                idx[i] = start
            # idx[i][1] = stop

    if getindex:
        return out,nd,idx
    else:
        return out, nd

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def calculate_windowed_fft(a, fs, ws, ss=None, wind=None,getindex=False,full_length=False):
    """
    Calculates windowed Fourier transform

    Parameters
    ----------
    a : :class:`~numpy.ndarray`
        1d array
    fs : int
        sampling rate (samples per second)
    ws : int
        Window size, in number of samples
    ss : int
        Step size, or number of samples until next window
    wind : :class:`~numpy.ndarray`
        1d array to specify the window used to apply taper or None (default).
    getindex : bool
        Save/return the start index for each window if True.
    full_length : bool
        Add an extra window to include the leftover samples to make sure sliding
        through the entire trace with full-length. This is done by measuring one
        window starting from the end backward. When False, this function skips the
        tailing samples if less than the window size.

    Returns
    -------
    ft : :class:`~numpy.ndarray`
        Fourier transform of trace
    f : :class:`~numpy.ndarray`
        Frequency axis in Hz
    idx : :class:`~numpy.ndarray`
        (Optional) The starting indices of the windows, with the size of [nd,1]
    """

    n2 = _npow2(ws)
    f = np.fft.rfftfreq(n2,1/fs)

    # Extract sliding windows
    if getindex:
        tr, nd,idx = sliding_window(a, ws, ss, wind=wind,getindex=True,
                                    full_length=full_length)
    else:
        tr, nd = sliding_window(a, ws, ss,wind=wind,
                                    full_length=full_length)

    # Fourier transform
    ft = np.fft.fft(tr, n=n2)
    if getindex:
        return ft,f,idx
    else:
        return ft, f

def psd(d,s,axis=-1,db=False):
    """
    Compute power spectral density. The power spectrum is normalized by
    frequency resolution.

    ====PARAMETERS====
    d: numpy ndarray containing the data.
    s: sampling frequency (samples per second)
    axis: axis to computer PSD. default is the last dimension (-1).

    ====RETURNS=======
    f: frequency array
    psd: power spectral density
    """
    if isinstance(d,list):d=np.array(d)
    if d.ndim >2:
        print('data has >2 dimension. skip demean.')
    else:
        d=detrend(demean(d))
    if d.ndim == 1:
        axis = 0
    elif d.ndim == 2:
        axis = 1
    Nfft = int(next_fast_len(int(d.shape[axis])))
    Nfft2 = int(Nfft//2)
    ft=fft(d,Nfft,axis=axis)
    psd=np.square(np.abs(ft))/s
    f=np.linspace(0, s/2, Nfft2)
    if d.ndim ==1:
        psd=psd[:Nfft2]
    elif d.ndim==2:
        psd=psd[:,:Nfft2]
    if db:
        psd=10*np.log10(np.abs(psd))
    return f,psd

def plot_slidingwindows(duration=3600*6,fs=20,window=7200,
                        overlaps=[None,0.1,0.1,0.2,0.2,0.3],
                        tapers=[None,None,0.05,0.05,0.1,0.1],
                        full_length=True,size=(12,12),save=False,
                        format='png'):
    """
    This function plots tapered sliding windows for illustration purpose.

    Parameters
    ----------
    duration: length of the demo data in seconds.
    fs: sampling rate of the data, used to get time information
    window: the window length you want to test.
    overlaps: an array specifying the series of window overlaps (0.0-1.0) for test.
    tapers: window ends will be tapered.
    """
    data=np.zeros((duration*fs,))
    t=np.arange(len(data))/fs
    ws=int(window*fs)
    plt.figure(figsize=size)
    print("connecting locations")
    print("start  end")
    colorlist=['k','b','g','c','y','r','m']
    for i in range(len(overlaps)):
        if overlaps[i] is None:
            ol=0
        else:
            ol=overlaps[i]

        if tapers[i] is None:
            tp=0
        else:
            tp=tapers[i]

        tps = int(0.5*window*ol*fs) #location where the two windows connect.
        step = int(window*(1-ol)*fs)
        wind=tukey(ws,2*tp)
        print(tps/fs,window - tps/fs)

        dout,nd,idx=sliding_window(data,ws,ss=step,getindex=True,
                                        full_length=full_length,verbose=False)
        ax=plt.subplot(len(tapers),1,i+1)
        if len(idx) > len(colorlist):
            windcolor=colorlist*(len(idx)//len(colorlist) + 1)
        else:
            windcolor=colorlist

        for j in range(len(idx)):
            plt.tight_layout(pad=1)
            plt.plot(t[np.arange(ws)+idx[j]],wind,windcolor[j])

            if j >0 and j+1 < len(idx):
                plt.plot(t[tps+j*step],1,'og')
                plt.plot(t[int(ol*window*fs)+j*step],1,'^r')
            plt.title("overlap: "+str(ol)+", one-side taper: "+str(tp))

        plt.xlim((np.min(t),np.max(t)))
        ax.legend(['tukey','tukey','connection','overlap'])

    if save:
        plt.savefig("slidingwindows_illustration."+format)
    else:
        plt.show()

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def smooth(data, nd, axis=0):
    """
    Function to smooth power spectral density functions from the convolution
    of a boxcar function with the PSD

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Real-valued array to smooth (PSD)
    nd : int
        Number of samples over which to smooth
    axis : int
        axis over which to perform the smoothing

    Returns
    -------
    filt : :class:`~numpy.ndarray`, optional
        Filtered data

    """
    if np.any(data):
        if data.ndim > 1:
            filt = np.zeros(data.shape)
            for i in range(data.shape[::-1][axis]):
                if axis == 0:
                    filt[:, i] = np.convolve(
                        data[:, i], np.ones((nd,))/nd, mode='same')
                elif axis == 1:
                    filt[i, :] = np.convolve(
                        data[i, :], np.ones((nd,))/nd, mode='same')
        else:
            filt = np.convolve(data, np.ones((nd,))/nd, mode='same')
        return filt
    else:
        return None
def smooth1(data, size=3,verbose=False):
    """
    Function to smooth 1-D vector.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Real-valued 1-d array to smooth
    size : int sequence
        Number of samples over which to smooth for all dimensions. Default 3

    Returns
    -------
    filt : :class:`~numpy.ndarray`
        Filtered data

    """

    if data.ndim > 1:
        raise ValueError('smooth1 works only for 1-d array.')
    a1=data.shape[0]
    if np.any(data):
        filt = np.convolve(data, np.ones((size,))/size, mode='same')

        return filt
    else:
        return None
def smooth2(data, size=[3,3],verbose=False):
    """
    Function to smooth 2-D matrix. This is implemented as 1d convolusion sequentially at
    2 dimensions.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Real-valued 2-d array to smooth
    size : int sequence
        Number of samples over which to smooth for all dimensions. Default [3,3]

    Returns
    -------
    filt : :class:`~numpy.ndarray`
        Filtered data

    """
    if isinstance(size,int): size=[size]
    if len(size) ==1:
        size=2*size
    elif len(size) >2:
        if verbose:print('size has > 3 elements. only the first 3 will be used.')
    if data.ndim < 2 or data.ndim >2:
        raise ValueError('smooth2 works only for 2-d array.')
    a1,a2=data.shape
    if np.any(data):
        filt = np.zeros(data.shape)
        for i in range(a1):
            filt[i,: ] = np.convolve(
                data[i,:], np.ones((size[1],))/size[1], mode='same')

        for i in range(a2):
            filt[:,i] = np.convolve(
                filt[:,i], np.ones((size[0],))/size[0], mode='same')

        return filt
    else:
        return None
def smooth3(data, size=[3,3,3],verbose=False):
    """
    Function to smooth 3-D matrix. This is implemented as 1d convolusion sequentially at
    3 dimensions.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Real-valued 3-d array to smooth
    size : int sequence
        Number of samples over which to smooth for all dimensions. Default [3,3,3]

    Returns
    -------
    filt : :class:`~numpy.ndarray`
        Filtered data

    """
    if isinstance(size,int): size=[size]
    if len(size) ==1:
        size=3*size
    elif len(size) ==2:
        if verbose:print('size has two elements, expand to 3 using the first value.')
        size=3*size[0]
    elif len(size) > 3:
        if verbose:print('size has > 3 elements. only the first 3 will be used.')
    if data.ndim < 3  or data.ndim >3:
        raise ValueError('smooth3 works only for 3-d array.')
    a1,a2,a3=data.shape
    if np.any(data):
        filt = np.zeros(data.shape)
        for i in range(a1):
            for j in range(a2):
                filt[i,j,: ] = np.convolve(
                    data[i,j,:], np.ones((size[2],))/size[2], mode='same')
        for i in range(a1):
            for j in range(a3):
                filt[i,:,j] = np.convolve(
                    filt[i,:,j], np.ones((size[1],))/size[1], mode='same')
        for i in range(a2):
            for j in range(a3):
                filt[:,i,j] = np.convolve(
                    filt[:,i,j], np.ones((size[0],))/size[0], mode='same')

        return filt
    else:
        return None
def _npow2(x):
    return 1 if x == 0 else 2**(x-1).bit_length()

def nextpow2(x):
    """
    Returns the next power of 2 of x.
    """
    return int(np.ceil(np.log2(np.abs(x))))
#save trace to files.
def save2asdf(fname,data,tag,sta_inv=None,group='waveforms',para=None,event=None):
    """
    A wrapper to save obspy stream to asdf file.

    Parameters
    ----------
    fname : string
        Output ASDF file name, with *.h5 extension.
    data :: class `~obspy.core.Stream` or class `~numpy.ndarray`
        Obspy Stream or numpy.ndarray object. For stream, all traces should belong to one single station,
        particularly when sta_inv is provided.
    tag :: string list
        List of tags for each trace in the `data` object.
    sta_inv : station inventory. Same length as the data and tag.
        Staion xml (obspy station inventory).
    group : string
        Group to save the data. Available options include 'waveforms', 'auxiliary'
    para : dictionary
        A dictionary to store saving parameters.
    """
    if group == 'waveforms':
        if len(data) != len(tag) or len(data) != len(sta_inv):
            raise(Exception('save2asdf: the stream, tag list, and sta_inv list should have the same length.'))

    if not os.path.isfile(fname):
        ds=pyasdf.ASDFDataSet(fname,mpi=False,compression="gzip-3",mode='w')
        if event is not None:
            ds.add_quakeml(event)
    else:
        ds=pyasdf.ASDFDataSet(fname,mpi=False,compression="gzip-3",mode='a')

    #save
    if sta_inv is not None:
        ds.add_stationxml(sta_inv)

    if group == 'waveforms':
        for i in range(len(data)):
            ds.add_waveforms(data[i],tag=tag[i])
    elif group == 'auxiliary':
        try:
            data_type=para['data_type']
            data_path=para['data_path']
            parameters = para['parameters']
        except Exception as e:
            raise(Exception('save2adsf: '+e))

        try:
            provenance_id=para['provenance_id']
        except Exception as e:
            provenance_id=None

        ds.add_auxiliary_data(data,data_type,data_path,parameters=parameters,
                            provenance_id=provenance_id)
#
def boundary_points(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                    # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add( (i, j) )

    tri = Delaunay(points)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.simplices: #vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        if area >0:
            circum_r = a*b*c/(4.0*area)

            # Here's the radius filter.
            #print circum_r
            if circum_r < 1.0/alpha:
                add_edge(edges, edge_points, points, ia, ib)
                add_edge(edges, edge_points, points, ib, ic)
                add_edge(edges, edge_points, points, ic, ia)

    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points
def box_smooth(d, box,mode='same'):
    """
    d: 1-d array
    box: number of box points.
    mode: smooth/convolve mode. same option as for numpy.convolve:
     https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    """
    b = np.ones(box)/box

    d_smooth = np.convolve(d, b, mode=mode)

    return d_smooth
#modified from NoisePy function
@jit(nopython = True)
def moving_ave(A,N):
    '''
    this Numba compiled function does running smooth average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the half window length to smooth

    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    '''
    A = np.concatenate((A[:N],A,A[-N:]),axis=0)
    B = np.zeros(A.shape,A.dtype)

    tmp=0.
    for pos in range(N,A.size-N):
        # do summing only once
        if pos==N:
            for i in range(-N,N+1):
                tmp+=A[pos+i]
        else:
            tmp=tmp-A[pos-N-1]+A[pos+N]
        B[pos]=tmp/(2*N+1)
        if B[pos]==0:
            B[pos]=1
    return B[N:-N]

def ftn(data,dt,fl,fh,df=None,taper_frac=None,taper_maxlen=20,max_abs=2,
            inc_type='linear',nf=100):
    """
    Conduct frequency-time normalization, based on the method in Shen, BSSA, 2012. This function
    was wrote based on the MATLAB version, obtained from Dr. Haiying Gao at UMass Amhert.

    ============PARAMETERS===============
    data: Numpy ndarray of the data, maximum dimension=2.
    dt: sample interval in time (s)
    fl: lowest frequency.
    fh: highest frequency.
    df: frequency interval in narrow band filtering, default is df=fl/4
    taper_frac: fraction 0-1 for tapering. ignore tapering if None.
    taper_maxlen: maxlength in number of points for tapering (ignore if taper_frac is None). Defatul 20.
    max_abs: maximum absolute value of the data after FTN. Default 2.
    inc_type: frequency increment type, 'linear' [default] or 'log'. when 'linear', df will be used
                and when 'log', nf will be used.
    nf: number of frequencies for 'log' type increment. default 100.
    ============RETURNS================
    dftn: data after FTN.

    ====================================
    Ref: Shen et al. (2012) An Improved Method to Extract Very-Broadband Empirical Green’s
        Functions from Ambient Seismic Noise, BSSA, doi: 10.1785/0120120023
    """
    if fh>0.5/dt:
        raise ValueError('upper bound of frequency CANNOT be larger than Nyquist frequency.')
    if inc_type=="log":
        dinc=1 - 1/np.geomspace(1,100,nf)
        dinc=np.append(dinc,1)
        freqs=fl + dinc*(fh-fl)
    elif inc_type=="linear":
        if df is None: df=fl/4
        freqs=np.arange(fl,fh+0.5*df,df)

    if freqs[-1]>0.5/dt:freqs[-1]=0.5/dt

    ncorners=4
    if taper_frac is None:
        d=data
    else:
        d=taper(data,fraction=taper_frac,maxlen=taper_maxlen)

    dftn=np.zeros(d.shape,dtype=d.dtype)
    if d.ndim == 1:
        for i in range(len(freqs)-1):
            dfilter=bandpass(d,freqs[i],freqs[i+1],1/dt,corners=ncorners, zerophase=True)
            env=np.abs(hilbert(dfilter))
            dftn += np.divide(dfilter,env)
        dftn /= np.sqrt(len(freqs)-1)

        #normalization
        idx=np.where(np.abs(dftn)>max_abs)[0]
        if len(idx)>0: dftn[idx]=0.0
    elif d.ndim==2:
        for k in range(d.shape[0]):
            for i in range(len(freqs)-1):
                dfilter=bandpass(d[k,:],freqs[i],freqs[i+1],1/dt,corners=ncorners, zerophase=True)
                env=np.abs(hilbert(dfilter))
                dftn[k,:] += np.divide(dfilter,env)
            dftn[k,:] /= np.sqrt(len(freqs)-1)

            #normalization
            idx=np.where(np.abs(dftn[k,:])>max_abs)[0]
            if len(idx)>0: dftn[k,idx]=0.0
    else:
        raise ValueError('Dimension %d is higher than allowed 2.'%(d.ndim))
    #taper
    if taper_frac is not None:
        dftn=taper(dftn,fraction=taper_frac,maxlen=taper_maxlen)

    return dftn

#modified from a NoisePy function
def whiten(data,dt,fmin,fmax,method='phase_only',smooth=20,pad=100):
    """
    Spectral whitening.

    ====PARAMETERS====
    data: time series data, can be 1 or 2 dimensions.
    dt: time sampling interval.
    fmin, fmax: frequency range to whiten.
    method: whitening method. Could be 'phase_only', which is pure whitening to flat spectrum,
            and 'rma', which is a running mean average smoothing. Default: phase_only
    smooth: smoothing length, only needed for 'rma'. Default: 20.
    pad: taper of the whitening extending the frequency range (number of samples in frequency).
            Default: 100.

    =========RETURNS====
    outdata: time-domain data after spectral whitening.
    """

    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1
    Nfft = int(next_fast_len(int(data.shape[axis])))
    Nfft2 = int(Nfft//2)
    FFTRawSign = fft(data, Nfft, axis=axis) # return FFT
    freqVec = fftfreq(Nfft, d=dt)[:Nfft2]
    J = np.where((freqVec >= fmin) & (freqVec <= fmax))[0]
    low = J[0] - pad
    if low <= 0:
        low = 1

    left = J[0]
    right = J[-1]
    high = J[-1] + pad
    if high > Nfft/2:
        high = Nfft2

    # Left tapering:
    if axis == 1:
        FFTRawSign[:,0:low] *= 0
        FFTRawSign[:,low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,low:left]))
        # Pass band:
        if method == 'phase_only':
            FFTRawSign[:,left:right] = np.exp(1j * np.angle(FFTRawSign[:,left:right]))
        elif method == 'rma':
            for ii in range(self.data.shape[0]):
                tave = moving_ave(np.abs(FFTRawSign[ii,left:right]),smooth)
                FFTRawSign[ii,left:right] = FFTRawSign[ii,left:right]/tave
        # Right tapering:
        FFTRawSign[:,right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,right:high]))
        FFTRawSign[:,high:Nfft2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[:,-Nfft2+1:] = np.flip(np.conj(FFTRawSign[:,1:Nfft2]),axis=axis)
        ##re-assign back to data.
        outdata=np.real(ifft(FFTRawSign, Nfft,axis=axis))[:,:data.shape[axis]]
    else:
        FFTRawSign[0:low] *= 0
        FFTRawSign[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[low:left]))
        # Pass band:
        if method == 'phase_only':
            FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        elif method == 'rma':
            tave = moving_ave(np.abs(FFTRawSign[left:right]),smooth)
            FFTRawSign[left:right] = FFTRawSign[left:right]/tave
        # Right tapering:
        FFTRawSign[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[right:high]))
        FFTRawSign[high:Nfft2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[-Nfft2+1:] = FFTRawSign[1:Nfft2].conjugate()[::-1]

        ##re-assign back to data.
        outdata=np.real(ifft(FFTRawSign, Nfft,axis=axis))[:data.shape[axis]]
    ##
    return outdata

#modified from NoisePy function
def mad(arr):
    """
    Median Absolute Deviation: MAD = median(|Xi- median(X)|)
    PARAMETERS:
    -------------------
    arr: numpy.ndarray, seismic trace data array
    RETURNS:
    data: Median Absolute Deviation of data
    """
    if not np.ma.is_masked(arr):
        med = np.median(arr)
        data = np.median(np.abs(arr - med))
    else:
        med = np.ma.median(arr)
        data = np.ma.median(np.ma.abs(arr-med))
    return data

#modified from NoisePy function
def detrend(data):
    '''
    this function removes the signal trend based on QR decomposion
    NOTE: QR is a lot faster than the least square inversion used by
    scipy (also in obspy).
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with trend removed
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        X = np.ones((npts,2))
        X[:,0] = np.arange(0,npts)/npts
        Q,R = np.linalg.qr(X)
        rq  = np.dot(np.linalg.inv(R),Q.transpose())
        coeff = np.dot(rq,data)
        data = data-np.dot(X,coeff)
    elif data.ndim == 2:
        npts = data.shape[1]
        X = np.ones((npts,2))
        X[:,0] = np.arange(0,npts)/npts
        Q,R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R),Q.transpose())
        for ii in range(data.shape[0]):
            coeff = np.dot(rq,data[ii])
            data[ii] = data[ii] - np.dot(X,coeff)
    return data
#modified from NoisePy function
def demean(data,axis=-1):
    '''
    this function remove the mean of the signal
    PARAMETERS:
    ---------------------
    data: input data matrix
    axis: axis to operate.

    RETURNS:
    ---------------------
    data: data matrix with mean removed
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        data = data-np.mean(data)
    elif data.ndim == 2:
        m=np.mean(data,axis=axis)
        for ii in range(data.shape[0]):
            if axis==-1:
                data[ii] = data[ii]-m[ii]
            else:
                data[:,ii] = data[:,ii]-m[ii]

    return data
#modified from NoisePy function
def taper(data,fraction=0.05,maxlen=20):
    '''
    this function applies a cosine taper using obspy functions
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with taper applied
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        # window length
        wlen = int(npts*fraction)
        if wlen>maxlen:wlen = maxlen

        # taper values
        func = hann #_get_function_from_entry_point('taper', 'hann')
        if 2*wlen == npts:
            taper_sides = func(2*wlen)
        else:
            taper_sides = func(2*wlen+1)
        # taper window
        win  = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),taper_sides[len(taper_sides) - wlen:]))
        data *= win
    elif data.ndim == 2:
        npts = data.shape[1]
        # window length
        wlen = int(npts*fraction)
        if wlen>maxlen:wlen = maxlen
        # taper values
        func = hann #_get_function_from_entry_point('taper', 'hann')
        if 2*wlen == npts:
            taper_sides = func(2*wlen)
        else:
            taper_sides = func(2*wlen + 1)
        # taper window
        win  = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),taper_sides[len(taper_sides) - wlen:]))
        for ii in range(data.shape[0]):
            data[ii] *= win
    return data
#modified from NoisePy function
def check_sample_gaps(stream,date_info):
    """
    this function checks sampling rate and find gaps of all traces in stream.
    PARAMETERS:
    -----------------
    stream: obspy stream object.
    date_info: dict of starting and ending time of the stream

    RETURENS:
    -----------------
    stream: List of good traces in the stream
    """
    # remove empty/big traces
    if len(stream)==0 or len(stream)>100:
        stream = []
        return stream

    # remove traces with big gaps
    if portion_gaps(stream,date_info)>0.3:
        stream = []
        return stream

    freqs = []
    for tr in stream:
        freqs.append(int(tr.stats.sampling_rate))
    freq = max(freqs)
    for tr in stream:
        if int(tr.stats.sampling_rate) != freq:
            stream.remove(tr)
        if tr.stats.npts < 10:
            stream.remove(tr)

    return stream

#modified from NoisePy function
def portion_gaps(stream,date_info):
    '''
    this function tracks the gaps (npts) from the accumulated difference between starttime and endtime
    of each stream trace. it removes trace with gap length > 30% of trace size.
    PARAMETERS:
    -------------------
    stream: obspy stream object
    date_info: dict of starting and ending time of the stream

    RETURNS:
    -----------------
    pgaps: proportion of gaps/all_pts in stream
    '''
    # ideal duration of data
    starttime = date_info['starttime']
    endtime   = date_info['endtime']
    npts      = (endtime-starttime)*stream[0].stats.sampling_rate

    pgaps=0
    #loop through all trace to accumulate gaps
    for ii in range(len(stream)-1):
        pgaps += (stream[ii+1].stats.starttime-stream[ii].stats.endtime)*stream[ii].stats.sampling_rate
    if npts!=0:pgaps=pgaps/npts
    if npts==0:pgaps=1
    return pgaps

#modified from NoisePy function
def resp_spectrum(source,resp_file,downsamp_freq,pre_filt=None):
    '''
    this function removes the instrument response using response spectrum from evalresp.
    the response spectrum is evaluated based on RESP/PZ files before inverted using the obspy
    function of invert_spectrum. a module of create_resp.py is provided in directory of 'additional_modules'
    to create the response spectrum
    PARAMETERS:
    ----------------------
    source: obspy stream object of targeted noise data
    resp_file: numpy data file of response spectrum
    downsamp_freq: sampling rate of the source data
    pre_filt: pre-defined filter parameters
    RETURNS:
    ----------------------
    source: obspy stream object of noise data with instrument response removed
    '''
    #--------resp_file is the inverted spectrum response---------
    respz = np.load(resp_file)
    nrespz= respz[1][:]
    spec_freq = max(respz[0])

    #-------on current trace----------
    nfft = _npts2nfft(source[0].stats.npts)
    sps  = int(source[0].stats.sampling_rate)

    #---------do the interpolation if needed--------
    if spec_freq < 0.5*sps:
        raise ValueError('spectrum file has peak freq smaller than the data, abort!')
    else:
        indx = np.where(respz[0]<=0.5*sps)
        nfreq = np.linspace(0,0.5*sps,nfft//2+1)
        nrespz= np.interp(nfreq,np.real(respz[0][indx]),respz[1][indx])

    #----do interpolation if necessary-----
    source_spect = np.fft.rfft(source[0].data,n=nfft)

    #-----nrespz is inversed (water-leveled) spectrum-----
    source_spect *= nrespz
    source[0].data = np.fft.irfft(source_spect)[0:source[0].stats.npts]

    if pre_filt is not None:
        source[0].data = np.float32(bandpass(source[0].data,pre_filt[0],pre_filt[-1],df=sps,corners=4,zerophase=True))

    return source

#extract waveform (raw) from ASDF file.
def extract_waveform(sfile,net=None,sta=None,comp=None,get_stainv=False):
    '''
    extract the downloaded waveform for station A
    PARAMETERS:
    -----------------------
    sfile: containing all wavefrom data for a time-chunck in ASDF format
    net,sta,comp: network, station name and component
    USAGE:
    -----------------------
    extract_waveform('temp.h5','CI','BLC')
    '''
    # open pyasdf file to read
    try:
        ds = pyasdf.ASDFDataSet(sfile,mode='r')
        sta_list = ds.waveforms.list()
    except Exception:
        print("exit! cannot open %s to read"%sfile);sys.exit()

    # check whether station exists
    if net is not None and sta is not None:
        tsta = net+'.'+sta
        if tsta not in sta_list:
            raise ValueError('no data for %s in %s'%(tsta,sfile))
        netstalist=[tsta]
    else:
        netstalist=sta_list
    trout=[]
    invout=[]
    if isinstance(comp, str): comp = [comp]
    for netsta in netstalist:
        tcomp = ds.waveforms[netsta].get_waveform_tags()
        ncomp = len(tcomp)
        if ncomp > 0:
            inv=[]
            if get_stainv:
                try:
                    inv = ds.waveforms[netsta]['StationXML']
                except Exception as e:
                    print('abort! no stationxml for %s in file %s'%(netsta,sfile))

            if ncomp == 1:
                tr=ds.waveforms[netsta][tcomp[0]]
                if comp is not None:
                    chan=tr[0].stats.channel
                    if chan not in comp:
                        raise ValueError('no data for comp %s for %s in %s'%(chan, netsta,sfile))
            elif ncomp>1:
                tr=[]
                for ii in range(ncomp):
                    tr_temp=ds.waveforms[netsta][tcomp[ii]]
                    if comp is not None:
                        chan=tr_temp[0].stats.channel
                        if chan in comp:tr.append(tr_temp[0])
                    else:
                        tr.append(tr_temp[0])
            if len(tr)==0:
                raise ValueError('no data for comp %s for %s in %s'%(comp, netsta,sfile))

            if len(tr)==1:tr=tr[0]

            trout.append(tr)
            invout.append(inv)

    # squeeze list.
    if len(trout)==1:
        trout=trout[0]
        invout=invout[0]
    if get_stainv:
        return trout,invout
    else:
        return trout

def xcorr(x, y, maxlags=10):
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    c = np.correlate(x, y, mode=2)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly positive < %d' % Nx)

    c = c[Nx - 1 - maxlags:Nx + maxlags]

    return c
def cart2pol(x, y):
    """
    cart2pol -- Transform Cartesian to polar coordinates

    =====
    PARAMETERS:
    x -- x coordinate
    y -- y coordinate
    =====
    RETURNS:
    theta -- Angle in radians
    rho -- Distance from origin
    =====

    Source: https://github.com/numpy/numpy/issues/5228 by User espdev.
    """
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    """
    pol2cart -- Transform polar to Cartesian coordinates

    =====
    PARAMETERS:
    theta -- Angle in radians
    rho -- Distance from origin
    =====
    RETURNS:
    x -- x coordinate
    y -- y coordinate
    =====
    
    Source: https://github.com/numpy/numpy/issues/5228 by User espdev.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def cart2compass(x,y):
    """
    CART2COMPASS convert cartesian coordinates into
    speed and direction data (degN).

    THETA,RHO = CART2COMPASS convert the vectors x and y
      from a cartesian reference system into rho (e.g. speed) with
      direction theta (degree North).
     
    Modified from the MATLAB function (same name) by Arnaud Laurent

    ===PARAMETER===
    x: x component of the vector.
    y: y component of the vector.
    ===RETURN===
    theta: direction of the vector in degrees (0-360) from north.
    rho: magnitude of the vector.
    ===REFERENCE===
    https://www.mathworks.com/matlabcentral/fileexchange/24432-cart2compass
    """
    theta,rho = cart2pol(x,y) #theta in radians, rho in x and y unit.
    #convert theta to degrees.
    theta = np.rad2deg(theta)
    
    #convert theta to degrees from north.
    if theta < 0:
        theta += 360
    elif theta>=0 and theta<90:
        theta = np.abs(theta - 90)
    elif theta>=90 and theta<=360:
        theta = np.abs(450 - theta)
    else:
        pass
    
    return theta,rho