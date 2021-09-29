#!/usr/bin/env python
# coding: utf-8
############################################
##Utility functions used in processing seismic data.
############################################
#import needed packages.
import sys
# from warnings import warn
import time
import scipy
import obspy
import pyasdf
import datetime
import os, glob
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot  as plt
from collections import OrderedDict
from scipy.signal import tukey,hilbert
from obspy.clients.fdsn import Client
from obspy.core import Stream, Trace, read
from obspy.core.util.base import _get_function_from_entry_point
from obspy.signal.util import _npts2nfft
from obspy.signal.filter import bandpass
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel
from shapely.geometry import Polygon, Point
import netCDF4 as nc

####
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
def get_filelist(dir,extension,sort=True):
    """
    Get list of files with absolute path, by specifying the format extension.

    ===========PARAMETERS=============
    dir: directory containing the files.
    extension: file extension (the ending format tag), for example "h5" for asdf file.
    sort: (optional) to sort the list, default is True.

    ============RETURN=============
    flist: the list of file names with paths.
    """
    flist=[os.path.join(dir,f) for f in os.listdir(dir) if f[-len(extension):].lower()==extension]
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
                if line.find("\t"):
                    cols = line.split("\t")
                else:
                    cols = line.split(" ")
                data.append([float(i) for i in cols])
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
    dep=np.array(ds['depth'][:])
    val=np.array(ds[var][:])

    if metadata:
        md=ds.__dict__
        return lat,lon,val,md
    else:
        return lat,lon,val
#
#
def ncmodel_in_polygon(dfile,var,outlines,vmax=9000,allstats=False,surface=False,
                        lon_correction=0.0):
    """
    Extract seismic model within polygons from 3d or 2d model in netCDF format.

    ===PARAMETERS===
    dfile - Data file name.
    var - variable name.
    vmax - maximum value, above which will be set to numpy nan.
    allstats - If True, returns all statistics (mean, median, min, max, std) of
                the model within the polygon. If False, only returns the mean 1d model.
                Default False.
    lon_correction - add correction to model longitude. Default 0.0.
    ===RETURNS===
    dep - Depth grid. Returns only when surface is False.
    val_mean - Average model value (1d profile in case of 3d ncmodel). Returns in all cases.
    val_median,val_min,val_max,val_std - Only returns these when allstats is True.
    """
    if surface: #read in 2d surfaces
        lat,lon,val=read_ncmodel2d(dfile,var)
        lon += lon_correction
        val[val>=vmax]=np.nan

        val_mean=np.ndarray((len(outlines)))
        if allstats:
            val_median=np.ndarray((len(outlines)))
            val_min=np.ndarray((len(outlines)))
            val_max=np.ndarray((len(outlines)))
            val_std=np.ndarray((len(outlines)))
        for idx,d in enumerate(outlines):
            ix,iy=points_in_polygon(d,lon,lat)
            val_mean[idx]=np.nanmean(np.nanmean(val[iy,ix]))
            if allstats:
                val_median[idx]=np.nanmedian(np.nanmedian(val[iy,ix]))
                val_min[idx]=np.nanmin(np.nanmin(val[iy,ix]))
                val_max[idx]=np.nanmax(np.nanmax(val[iy,ix]))
                val_std[idx]=np.nanstd(val[iy,ix])
        #
        if allstats:
            return val_mean,val_median,val_min,val_max,val_std
        else:
            return val_mean
    else:
        dep,lat,lon,val=read_ncmodel3d(dfile,var)
        lon += lon_correction
        val[val>=vmax]=np.nan

        val_mean=np.ndarray((len(outlines),val.shape[0]))
        if allstats:
            val_median=np.ndarray((len(outlines),val.shape[0]))
            val_min=np.ndarray((len(outlines),val.shape[0]))
            val_max=np.ndarray((len(outlines),val.shape[0]))
            val_std=np.ndarray((len(outlines),val.shape[0]))
        for idx,d in enumerate(outlines):
            ix,iy=points_in_polygon(d,lon,lat)
            for k in range(val_mean.shape[1]):
                val_mean[idx,k]=np.nanmean(np.nanmean(val[k,iy,ix]))
                if allstats:
                    val_median[idx,k]=np.nanmedian(np.nanmedian(val[k,iy,ix]))
                    val_min[idx,k]=np.nanmin(np.nanmin(val[k,iy,ix]))
                    val_max[idx,k]=np.nanmax(np.nanmax(val[k,iy,ix]))
                    val_std[idx,k]=np.nanstd(val[k,iy,ix])
        #
        if allstats:
            return dep,val_mean,val_median,val_min,val_max,val_std
        else:
            return dep,val_mean
# ##################### qml_to_event_list #####################################
def qml_to_event_list(events_QML,to_pd=False):
    print("WARNING: this function has been renamed to qml2list. This warning will be removed in v0.7.x.")
    return qml2list(events_QML,to_pd=to_pd)

# modified from qml_to_event_list in obspyDMT.utils.event_handler.py
def qml2list(events_QML,to_pd=False):
    """
    convert QML to event list

    ===PARAMETERS===
    events_QML: event qml (OBSPY CATALOG object)
    to_pd: convert to Pandas DataFrame object. Default: False.

    ====return====
    events: a list of event information or a pandas dataframe object.
    """
    events = []
    for i in range(len(events_QML)):
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
def sta_info_from_inv(inv):
    '''
    this function outputs station info from the obspy inventory object.
    PARAMETERS:
    ----------------------
    inv: obspy inventory object
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
    # print(sta,net,lon,lat,elv,location)
    return sta,net,lon,lat,elv,location

def get_tt(event_lat, event_long, sta_lat, sta_long, depth_km,model="iasp91",type='first'):
    # get the seismic phase arrival time of the specified earthquake at the station.
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

#Stolen from NoisePy
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
               title=[],outfile='test.ps',xlimit=[],subplotpar=[],              \
               mode="subplot",spacing=2.0,colors=[],verbose=False):
    """
    mode: subplot, overlap, or gather. In gather mode, traces will be offset and normalized.
    """
    plt.figure(figsize=size)
    ntr=len(tr_list)
    if len(subplotpar)==0 and mode=="subplot":
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
            plt.tight_layout(pad=spacing)
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
            if len(colors)==0:
                plt.plot(tt,itr-1+0.5*tc.data/np.max(np.abs(tc.data)))
            elif len(colors)==1:
                plt.plot(tt,itr-1+0.5*tc.data/np.max(np.abs(tc.data)),colors[0])
            else:
                plt.plot(tt,itr-1+0.5*tc.data/np.max(np.abs(tc.data)),colors[itr-1])
            plt.xlabel("time (s)")
            plt.text(xlimit[0]+10,itr-1+0.2,tc.stats.network+"."+tc.stats.station,
                    horizontalalignment='left',verticalalignment='center',fontsize=11)
            if itr==ntr:
                ax.ticklabel_format(axis='x',style='plain')
                if len(datalabels)>0: ax.legend(datalabels)
                if len(ylabels)>0:
                    plt.ylabel(ylabels[0])
                if len(title)>0:
                    plt.title(title)
                if len(xlimit)>0:
                    plt.xlim(xlimit)
                plt.ylim([-0.7,ntr-0.3])
                if len(freq)>0:
                    plt.text(np.mean(xlimit),0.85*np.max(myymax),"["+str(freq[0])+", "+str(freq[1])+"] Hz",\
                             horizontalalignment='center',verticalalignment='center',fontsize=14)
        else:
            raise ValueError("mode: %s is not recoganized. Can ONLY be: subplot, overlap, or gather."%(mode))

    plt.savefig(outfile,orientation='landscape')
    plt.show()
    plt.close()

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
    this function cuts continous noise data into user-defined segments, estimate the statistics of
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
    all_madS = mad(source[0].data)	            # median absolute deviation over all noise window
    all_stdS = np.std(source[0].data)	        # standard deviation over all noise window
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
        nseg = int(np.floor((duration-win_len_secs)/step_secs))
        npts_step = int(step_secs*sps)

    # initialize variables
    npts = int(win_len_secs*sps)
    trace_stdS = np.zeros(nseg,dtype=np.float32)
    dataS    = np.zeros(shape=(nseg,npts),dtype=np.float32)
    dataS_t  = np.zeros(nseg,dtype=np.float)

    print('slicing trace into ['+str(nseg)+'] segments.')

    indx1 = 0
    for iseg in range(nseg):
        indx2 = indx1+npts
        dataS[iseg] = source[0].data[indx1:indx2]
        trace_stdS[iseg] = (np.max(np.abs(dataS[iseg]))/all_stdS)
        dataS_t[iseg]    = starttime+step_secs*iseg
        indx1 += npts_step

    # 2D array processing
    dataS = demean(dataS)
    dataS = detrend(dataS)
    dataS = taper(dataS,fraction=taper_frac)

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

# def smooth(data, np, poly=0, axis=0):
#     return savgol_filter(data, np, poly, axis=axis, mode='wrap')

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

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def admittance(Gxy, Gxx):
    """
    Calculates admittance between two components

    Parameters
    ---------
    Gxy : :class:`~numpy.ndarray`
        Cross spectral density function of `x` and `y`
    Gxx : :class:`~numpy.ndarray`
        Power spectral density function of `x`

    Returns
    -------
    : :class:`~numpy.ndarray`, optional
        Admittance between `x` and `y`

    """

    if np.any(Gxy) and np.any(Gxx):
        return np.abs(Gxy)/Gxx
    else:
        return None

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def coherence(Gxy, Gxx, Gyy):
    """
    Calculates coherence between two components

    Parameters
    ---------
    Gxy : :class:`~numpy.ndarray`
        Cross spectral density function of `x` and `y`
    Gxx : :class:`~numpy.ndarray`
        Power spectral density function of `x`
    Gyy : :class:`~numpy.ndarray`
        Power spectral density function of `y`

    Returns
    -------
    : :class:`~numpy.ndarray`, optional
        Coherence between `x` and `y`

    """

    if np.any(Gxy) and np.any(Gxx) and np.any(Gxx):
        return np.abs(Gxy)**2/(Gxx*Gyy)
    else:
        return None

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def phase(Gxy):
    """
    Calculates phase angle between two components

    Parameters
    ---------
    Gxy : :class:`~numpy.ndarray`
        Cross spectral density function of `x` and `y`

    Returns
    -------
    : :class:`~numpy.ndarray`, optional
        Phase angle between `x` and `y`

    """

    if np.any(Gxy):
        return np.angle(Gxy)
    else:
        return None

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def rotate_dir(tr1, tr2, direc):

    d = -direc*np.pi/180.+np.pi/2.
    rot_mat = np.array([[np.cos(d), -np.sin(d)],
                        [np.sin(d), np.cos(d)]])

    v12 = np.array([tr2, tr1])
    vxy = np.tensordot(rot_mat, v12, axes=1)
    tr_2 = vxy[0, :]
    tr_1 = vxy[1, :]

    return tr_1

# modified from the same functions as in: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
def ftest(res1, pars1, res2, pars2):

    from scipy.stats import f as f_dist

    N1 = len(res1)
    N2 = len(res2)

    dof1 = N1 - pars1
    dof2 = N2 - pars2

    Ea_1 = np.sum(res1**2)
    Ea_2 = np.sum(res2**2)

    Fobs = (Ea_1/dof1)/(Ea_2/dof2)

    P = 1. - (f_dist.cdf(Fobs, dof1, dof2) - f_dist.cdf(1./Fobs, dof1, dof2))

    return P

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
    sta_inv : station inventory
        Staion xml (obspy station inventory).
    group : string
        Group to save the data. Available options include 'waveforms', 'auxiliary'
    para : dictionary
        A dictionary to store saving parameters.
    """
    if group == 'waveforms':
        if len(data) != len(tag):
            raise(Exception('save2asdf: the stream and tag list should have the same length.'))

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

def get_cc(s1,s_ref):
    # returns the correlation coefficient between waveforms in s1 against reference
    # waveform s_ref.
    #
    cc=np.zeros(s1.shape[0])
    s_ref_norm = np.linalg.norm(s_ref)
    for i in range(s1.shape[0]):
        cc[i]=np.sum(np.multiply(s1[i,:],s_ref))/np.linalg.norm(s1[i,:])/s_ref_norm
    return cc

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

def demean(data):
    '''
    this function remove the mean of the signal
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with mean removed
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        data = data-np.mean(data)
    elif data.ndim == 2:
        for ii in range(data.shape[0]):
            data[ii] = data[ii]-np.mean(data[ii])
    return data

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
        func = _get_function_from_entry_point('taper', 'hann')
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
        func = _get_function_from_entry_point('taper', 'hann')
        if 2*wlen == npts:
            taper_sides = func(2*wlen)
        else:
            taper_sides = func(2*wlen + 1)
        # taper window
        win  = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),taper_sides[len(taper_sides) - wlen:]))
        for ii in range(data.shape[0]):
            data[ii] *= win
    return data


def whiten(data, fft_para):
    '''
    This function takes 1-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: dict containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    '''

    # load parameters
    delta   = fft_para['dt']
    freqmin = fft_para['freqmin']
    freqmax = fft_para['freqmax']
    smooth_N  = fft_para['smooth_N']
    freq_norm = fft_para['freq_norm']

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    Nfft = int(next_fast_len(int(data.shape[axis])))

    Napod = 100
    Nfft = int(Nfft)
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1

    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft/2:
        high = int(Nfft//2)

    FFTRawSign = scipy.fftpack.fft(data, Nfft,axis=axis)
    # Left tapering:
    if axis == 1:
        FFTRawSign[:,0:low] *= 0
        FFTRawSign[:,low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[:,left:right] = np.exp(1j * np.angle(FFTRawSign[:,left:right]))
        elif freq_norm == 'rma':
            for ii in range(data.shape[0]):
                tave = moving_ave(np.abs(FFTRawSign[ii,left:right]),smooth_N)
                FFTRawSign[ii,left:right] = FFTRawSign[ii,left:right]/tave
        # Right tapering:
        FFTRawSign[:,right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,right:high]))
        FFTRawSign[:,high:Nfft//2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[:,-(Nfft//2)+1:] = np.flip(np.conj(FFTRawSign[:,1:(Nfft//2)]),axis=axis)
    else:
        FFTRawSign[0:low] *= 0
        FFTRawSign[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        elif freq_norm == 'rma':
            tave = moving_ave(np.abs(FFTRawSign[left:right]),smooth_N)
            FFTRawSign[left:right] = FFTRawSign[left:right]/tave
        # Right tapering:
        FFTRawSign[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[right:high]))
        FFTRawSign[high:Nfft//2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[-(Nfft//2)+1:] = FFTRawSign[1:(Nfft//2)].conjugate()[::-1]

    return FFTRawSign

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
def extract_waveform(sfile,net,sta,comp=None,get_stainv=False):
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
    tsta = net+'.'+sta
    if tsta not in sta_list:
        raise ValueError('no data for %s in %s'%(tsta,sfile))

    if isinstance(comp, str): comp = [comp]

    tcomp = ds.waveforms[tsta].get_waveform_tags()
    ncomp = len(tcomp)
    if get_stainv:
        try:
            inv = ds.waveforms[tsta]['StationXML']
        except Exception as e:
            print('abort! no stationxml for %s in file %s'%(tsta,sfile))
            inv=[]

    if ncomp == 1:
        tr=ds.waveforms[tsta][tcomp[0]]
        if comp is not None:
            chan=tr[0].stats.channel
            if chan not in comp:
                raise ValueError('no data for comp %s for %s in %s'%(chan, tsta,sfile))
    elif ncomp>1:
        tr=[]
        for ii in range(ncomp):
            tr_temp=ds.waveforms[tsta][tcomp[ii]]
            if comp is not None:
                chan=tr_temp[0].stats.channel
                if chan in comp:tr.append(tr_temp[0])
    if len(tr)==0:
        raise ValueError('no data for comp %s for %s in %s'%(c, tsta,sfile))

    if len(tr)==1:tr=tr[0]

    if get_stainv:
        return tr,inv
    else:
        return tr


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
