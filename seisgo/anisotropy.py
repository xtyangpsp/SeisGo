import obspy,os,utm
import numpy as np
import pandas as pd
from seisgo import noise,utils
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pygmt as gmt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# mpl.rc('font',family='Helvetica')

"""
This module contains functions to compute anisotropy. 

As of now, it contains functions to compute the azimuthal anisotropy using BANX method (Beamforming Azimuthal Anisotropy 
using Noise Cross-correlation). The main worklow was converted from the original MATLAB codes by Jorge C. Castellanos-Martinez, 
provided by the author. See the related publication for more details.
Castellanos, J. C., Perry-Houts, J., Clayton, R. W., Kim, Y., Stanciu, A. C., Niday, B., & Humphreys, E. (2020). Seismic anisotropy 
reveals crustal flow driven by mantle vertical loading in the Pacific NW. Science Advances, 6(28), 1-10. https://doi.org/10.1126/sciadv.abb0476

Noted by Xiaotao Yang, March 11, 2025.
"""

def get_ArrayAttributes(lat,lon,get_utm=False):
    """
    Get the attributes of the station array. 

    ===PARAMETER==
    lat, lon: latitudes and longitudes.
    ===RETURN==
    radius: radius of the circle enclosing the stations/array. Unit: km
    center: center coordinates of the stations [lat, lon].

    ==PACKAGES==
    utm: to convert to and from the geographical coordinates.
    """

    #convert to lat lon to UTM to get the center.
    Ueast,Unorth,Zone,ZoneStr = utm.from_latlon(lat,lon)
    center = utm.to_latlon(np.mean(Ueast),np.mean(Unorth),Zone,ZoneStr)

    #compute distance of each station to the center.
    dist=[]
    for i in range(len(lat)):
        dist.append(obspy.geodetics.base.gps2dist_azimuth(center[0],center[1],
                                                    lat[i],
                                                    lon[i])[0]/1000)
    #
    radius = np.max(dist)

    #
    if get_utm: #return UTM outputs for all stations.
        return radius,center,[Ueast,Unorth,Zone,ZoneStr]
    else:
        return radius,center
    
def compute_anisotropy(x, a, b, c):
    """
    Compute anisotropic velocity. Only use the first three terms (the 2*theta terms).

    $v = a + b*cos(2*\theta) + c*sin(2*\theta) $
    theta: direction/azimuth. Independent variable: x.
    a: isotropic velocity.
    b and c: coefficients of the anisotropic directions.
    v: anisotropic velocity.
    ===PARAMETER===
    x: direction/azimuth in degrees. Independent variable.
    a: isotropic velocity.
    b: coefficient of the anisotropic direction.
    c: coefficient of the anisotropic direction.
    ===RETURN===
    v: anisotropic velocity.
    ===REFERENCE===
    M. L. Smith, F. A. Dahlen, The azimuthal dependence of Love and Rayleigh wave
        propagation in a slightly anisotropic medium. J. Geophys. Res. 78, 3321–3333 (1973).
    """
    return a + b*np.cos(2*np.deg2rad(x)) + c*np.sin(2*np.deg2rad(x))

#################### BANX method ####################
def do_BANX(stationdict_all, reference_site, period_band, reference_velocity, datadir,outdir_root='.',sampling_rate=None,
            min_stations=10, min_snr=5, min_radius_scaling=1,max_radius_scaling=1.5, min_distance_scaling=2.5, 
            signal_window_velocity=None,signal_extent_scaling=3,max_slowness=0.5,slowness_step=0.005,velocity_perturbation=0.4,
            trace_start_time=0,taper_length_scaling=5,azimuth_step=6,min_baz_measurements=3,min_good_bazbin=5,min_beam_sharpness=0,
            doublesided=True, cc_comp ='ZZ', show_fig=True, plot_moveout=True, moveout_scaling = 4, plot_clustermap=True, 
            map_region=None,map_engine='cartopy',receiver_box=None,plot_beampower=True, plot_station_result=True, 
            verbose=False,map_region_precision=0):
    """
    Perform BANX method to compute azimuthal anisotropy.
    ===PARAMETER===
    reference_site: reference station or master station used to search for the station cluster. Contains name.
    period_band: period band for the analysis. Unit: seconds.
    reference_velocity: reference phase velocity. Unit: km/s.
    datadir: directory containing the cross-correlation data. The data should be in asdf format named as 'source_receiver*.h5'.
    outdir_root: root directory to save the results. Default is current directory.
    sampling_rate: target sampling rate. Needs to be integer times (above or below) the data sampling rate to avoid resampling error. 
                    The data will be resampled to sampling_rate (samples per second).
    min_stations: minimum number of stations in the cluster. Default is 10.
    min_snr: minimum SNR for the data. Default is 5.
    min_radius_scaling: minimum radius scaling factor. Default is 1.
    max_radius_scaling: maximum radius scaling factor. Default is 1.5.
    min_distance_scaling: minimum distance scaling factor. Default is 2.5.
    signal_extent_scaling: scaling factor for the signal extent (before and after the predicted arrival). Default is 3. (3 times the longest period)
    signal_window_velocity: group velocity to calculate the predicted arrival time to decide the signal window. 
                    If None [default], will use 80% of the reference_velocity.

    max_slowness: maximum slowness. Default is 0.5.
    slowness_step: slowness step. Default is 0.005.
    velocity_perturbation: velocity perturbation. Default is 0.4. (40%)
    trace_start_time: start time of the xcorr data. Default is 0.
    taper_length_scaling: taper length scaling factor relative to the longest period. Default is 5.

    azimuth_step: azimuthal step used in QC. Default is 6.
    min_baz_measurements: minimum number of measurements in each azimuthal bin used in QC. Should be >=3. Default is 3.
    min_good_bazbin: minimum number of good bins with >= min_baz_measurements in each azimuthal bin used in QC. Should be >=5 (recommended). Default is 5.
    min_beam_sharpness: minimum beam sharpness to pass the QC. Default is 0 [no QC by sharpness].
    doublesided: data contains both negative and positive sides of the cross-correlation data. Default is True.
    cc_comp: cross-correlation component. Default is 'ZZ'.
        
    show_fig: show figures. Default is True. Figures will be plotted if the switches are on.
    plot_moveout: plot moveout of the cluster traces to the source. Default is True.
    moveout_scaling: scaling factor for the moveout plot. Default is 4.
    plot_clustermap: plot map view of the cluster stations and the source. Default is True.
    map_engine: map plotting engine. 'cartopy' [default] or 'gmt'.
    receiver_box: receiver box for the map view. Default is None. Format [lon_min, lon_max, lat_min, lat_max].
    plot_beampower: plot beam power after beamforming with the picked max power slowness positions/values. Default is True.
    plot_station_result: plot station results with all sources after QC. Default is True.

    verbose: print verbose information. Default is False.
    ===RETURN===
    """
    if receiver_box is not None:
        if len(receiver_box) != 4:
            raise ValueError('receiver_box should be in format [lon_min, lon_max, lat_min, lat_max]')
    #check if the reference site is in the station dictionary
    if reference_site not in stationdict_all.keys():
        raise ValueError('Reference site is not in the station dictionary.')
    #check if the period band is in the correct format
    if not isinstance(period_band, (list, np.ndarray)):
        raise ValueError('Period band should be in list or numpy array format.')
    #Create directories for this period band
    outdir=os.path.join(outdir_root,str(period_band[0])+'_'+str(period_band[1]))
    if not os.path.isdir(outdir):os.makedirs(outdir, exist_ok=True)
    figdir=os.path.join(outdir,'figures')
    if not os.path.isdir(figdir):os.makedirs(figdir, exist_ok=True)
    figdir_refsite=os.path.join(figdir,reference_site)
    if not os.path.isdir(figdir_refsite):os.makedirs(figdir_refsite, exist_ok=True)

    if doublesided:
        N_Sides = 2
    else:
        N_Sides = 1

    #
    if signal_window_velocity is None:
        signal_window_velocity = 0.8*reference_velocity
    #
    SourceList_Sites = list(stationdict_all.keys())
    if sampling_rate is None:
        resample = False
    else:
        resample = True
        SamplingRate_Target = sampling_rate
        Sampling_Delta_Target = 1/SamplingRate_Target

    # Period band
    if isinstance(period_band, list):
        period_band = np.array(period_band) #seconds.
    
    # Reference period:
    if verbose: print('Reference period: ',period_band)
    Min_Period = np.min(period_band)
    Max_Period = np.max(period_band)
    # Distance constraints:
    Min_Radius = min_radius_scaling * (Min_Period * reference_velocity)
    Max_Radius = max_radius_scaling * (Max_Period * reference_velocity)
    Min_Distance = min_distance_scaling * (Max_Period * reference_velocity)

    # get source and receiver cluster information/list
    Ref_Coord = stationdict_all[reference_site] #coordinates: lat, lon in order.
    
    ReceiverCluster_Sites = [] #receivers within a certain distance from the reference site
    #compute distance from the reference site to all other sites
    ReceiverCluster_Coord = []
    SourceList_Coord =[]
    for j in range(len(SourceList_Sites)):
        SourceList_Coord.append(stationdict_all[SourceList_Sites[j]])
        dist = obspy.geodetics.base.gps2dist_azimuth(Ref_Coord[0],Ref_Coord[1],
                                                    stationdict_all[SourceList_Sites[j]][0],
                                                    stationdict_all[SourceList_Sites[j]][1])[0]/1000
        if dist <= Max_Radius:
            ReceiverCluster_Sites.append(SourceList_Sites[j])
            ReceiverCluster_Coord.append(stationdict_all[SourceList_Sites[j]])
    #
    #check for minimum number of stations
    
    if len(ReceiverCluster_Sites) < min_stations:
        return None, None
    #use sites other than the receiver cluster as the source sites.
    Source_Sites = sorted(list(set(SourceList_Sites) - set(ReceiverCluster_Sites)))
    
    print(reference_site+' -> '+str(len(ReceiverCluster_Sites))+' stations in the cluster -> '+str(len(Source_Sites))+' sources')

    #######################################
    #### Derive parameters and constants ##
    #######################################
    # Data Processing:
    AZIBIN_EDGES = np.arange(0, 180+0.5*azimuth_step, azimuth_step) #azimuthal bin egesm used in QC step.

    # Search range of local phase velocities
    Phase_Velocity_Limits = [round(reference_velocity * (1 - velocity_perturbation),2),
                            round(reference_velocity * (1 + velocity_perturbation),2)]
    print('Phase Velocity Range: ',Phase_Velocity_Limits)

    # Beamforming space:
    Slowness_Space = [-np.floor(max_slowness / slowness_step) * slowness_step, np.floor(max_slowness / slowness_step) * slowness_step];       
    Ux = np.arange(Slowness_Space[0],Slowness_Space[1]+0.5*slowness_step,slowness_step)
    Uy = np.arange(Slowness_Space[0],Slowness_Space[1]+0.5*slowness_step,slowness_step)
    # [UxUx, UyUy] = np.meshgrid(Ux, Uy)
    SlownessSamples = len(Ux)

    # gmt plotting parameters
    source_coord_array=np.array(SourceList_Coord)
    lon_all,lat_all=source_coord_array[:,1],source_coord_array[:,0]

    if receiver_box is not None:
        lon_box=[receiver_box[0],receiver_box[1],receiver_box[1],receiver_box[0],receiver_box[0]]
        lat_box=[receiver_box[2],receiver_box[2],receiver_box[3],receiver_box[3],receiver_box[2]]
    if map_region is None:
        map_region = np.round([np.min(lon_all),np.max(lon_all),np.min(lat_all),np.max(lat_all)],map_region_precision)

    if plot_clustermap:
        if map_engine.lower() == 'gmt':
            marker_style="i0.17c"
            map_style="plain"
            frame="af"
            GMT_FONT_TITLE="14p,Helvetica-Bold"
            projection = 'M3.5i'
        elif map_engine.lower() == 'cartopy':
            projection=ccrs.LambertAzimuthalEqualArea(central_longitude=np.mean([np.min(lon_all),np.max(lon_all)]), 
                                    central_latitude=np.mean([np.min(lat_all),np.max(lat_all)]), 
                                    false_easting=0.0, false_northing=0.0,globe=None)
            scale = '10m'
            states10 = cfeature.NaturalEarthFeature(
                        category='cultural',
                        name='admin_1_states_provinces_lines',
                        scale=scale,
                        facecolor='none',
                        edgecolor='k')
            country10 = cfeature.NaturalEarthFeature(
                        category='cultural',
                        name = 'admin_0_boundary_lines_land',
                        scale = scale,
                        facecolor='none',
                        edgecolor='k')
        else:
            raise ValueError('Wrong map_engine. Should be cartopy or gmt')
    ################################
    ##### Main beamforming block ###
    ################################

    """
    START BEAMFORMING LOOP
    """
    if verbose: print('  --> Beamforming loop')
    Beam_Local=[]

    for j in range(len(Source_Sites)):
        Source_Name = Source_Sites[j]
        Source_Coord = stationdict_all[Source_Name]
        flist_all=[]
        for jr in range(len(ReceiverCluster_Sites)):
            flist=[] 
            dir_temp1=os.path.join(datadir,Source_Sites[j])
            if os.path.isdir(dir_temp1):
                flist=utils.get_filelist(dir_temp1,extension='h5',pattern=ReceiverCluster_Sites[jr])
            
            dir_temp2=os.path.join(datadir,ReceiverCluster_Sites[jr])
            flist2=[]
            if os.path.isdir(dir_temp2):
                flist2=utils.get_filelist(dir_temp2,extension='h5',pattern=Source_Sites[j])
            #merge two lists
            flist.extend(flist2)
            #
            flist_all.extend(flist)
        #
        if len(flist_all) < N_Sides*min_stations:
            continue
            
        print('  Source: '+Source_Sites[j]+' -> '+str(int(len(flist_all)/N_Sides))+' pairs')

        #######################################################
        # subset receiver cluster sites to find only those with data.
        # Assemble all correlation data for the receiver cluster.
        #######################################################
        site_temp=[]
        coord_temp=[]
        ReceiverCluster_CorrData =[]
        ReceiverCluster_Dist = [] #stores distance from each station in the cluster to the source.
        for jr in range(len(ReceiverCluster_Sites)):
            sitename=ReceiverCluster_Sites[jr]
            filtered_list = [item for item in flist_all if sitename in item]
            if len(filtered_list)==N_Sides: #
                site_temp.append(sitename)
                coord_temp.append(ReceiverCluster_Coord[jr])

                cdata_temp1 = noise.extract_corrdata(filtered_list[0])
                cdata1 = cdata_temp1[list(cdata_temp1.keys())[0]][cc_comp] 
                if N_Sides == 2: #average two sides
                    cdata_temp2 = noise.extract_corrdata(filtered_list[1])
                    cdata2 = cdata_temp2[list(cdata_temp2.keys())[0]][cc_comp] 
                    if len(cdata1.data) > 0 and len(cdata2.data) > 0:
                        #average the negative and positive sides.
                        cdata1.data = (cdata1.data + cdata2.data)/2
                        cdata1.side = None
                    elif len(cdata2.data) > 1:
                        cdata1.data = cdata2.data
                    elif len(cdata1.data) > 1:
                        pass
                    else:
                        continue

                #filter the data
                cdata1.filter(fmin=1/period_band[1],fmax=1/period_band[0],corners=4,zerophase=True)
                #normalize the amplitudes.
                cdata1.data = cdata1.data/np.max(np.abs(cdata1.data))
                
                #print(len(cdata1.data))
                #collect the corrdata object.
                ReceiverCluster_CorrData.append(cdata1)
                ReceiverCluster_Dist.append(cdata1.dist)
                
        #
        if len(ReceiverCluster_CorrData) < min_stations:
            print('  Not enough stations in the cluster. Skip!')
            continue
        ReceiverCluster_Sites = site_temp
        ReceiverCluster_Coord = coord_temp

        # average positive and negative lags when reading data
        ReceiverCluster_Radius,ReceiverCluster_Center, ReceiverCluster_UTM = get_ArrayAttributes(np.array(ReceiverCluster_Coord)[:,0],
                                                                            np.array(ReceiverCluster_Coord)[:,1],get_utm=True)
        Dist2Source,_,ReceiverCluster_BAZ=obspy.geodetics.base.gps2dist_azimuth(\
                                                ReceiverCluster_Center[0],ReceiverCluster_Center[1],
                                                Source_Coord[0],Source_Coord[1])
        Dist2Source = Dist2Source/1000
        if verbose:
            print('  Cluster radius,center,dist2source,baz2source: \n    ',
              str(ReceiverCluster_Radius),str(ReceiverCluster_Center),
              str(Dist2Source),str(ReceiverCluster_BAZ))
        if ReceiverCluster_Radius < Min_Radius or Dist2Source < Min_Distance:
            print('SKIP: small cluster or too close to the source!')
            continue
        #

        """
        # Calculating the theoretical arrival times
        """
        ReceiverCluster_TravelTimes = np.array(ReceiverCluster_Dist) / signal_window_velocity
        # print(ReceiverCluster_TravelTimes)

        # Interpolate or resample if the target sampling rate is different from the data sampling rate
        Sampling_Delta_Data = ReceiverCluster_CorrData[0].dt
        Sampling_Rate_Data = int(1/Sampling_Delta_Data)
        if SamplingRate_Target == Sampling_Rate_Data:
            resample = False
        if resample:
            #max time is the maximum travel time + taper length
            MaxTime = np.round(np.max(ReceiverCluster_TravelTimes) + (signal_extent_scaling+taper_length_scaling) * Max_Period,2)
            if MaxTime > ReceiverCluster_CorrData[0].lag:
                MaxTime = ReceiverCluster_CorrData[0].lag
            # Creating a homogeneous time vector:
            TimeVector = np.arange(trace_start_time , MaxTime + 0.5*Sampling_Delta_Target, Sampling_Delta_Target)
            if (SamplingRate_Target > Sampling_Rate_Data and np.remainder(SamplingRate_Target,Sampling_Rate_Data) > 0 ) or \
                (SamplingRate_Target < Sampling_Rate_Data and np.remainder(Sampling_Rate_Data,SamplingRate_Target) > 0):
                raise ValueError('Resampling fraction (target/data) has to be an integer to avoid errors.')
            if np.abs(Sampling_Rate_Data - SamplingRate_Target) > 1:
                if verbose: print('  Resampling to (sps): ',SamplingRate_Target)
                TimeVectorData = np.arange(trace_start_time , ReceiverCluster_CorrData[0].lag + 0.5*Sampling_Delta_Data, 
                                        Sampling_Delta_Data)
                for cdata in ReceiverCluster_CorrData:
                    cdata.dt = Sampling_Delta_Target #replace the sampling interval with the target value
                    cdata.data = np.interp(TimeVector,TimeVectorData,cdata.data)
                    cdata.lag = MaxTime
        else: #keep the data sampling rate
            SamplingRate_Target = Sampling_Rate_Data
            Sampling_Delta_Target = 1/SamplingRate_Target
            #max time is the maximum travel time + taper length
            MaxTime = np.round(np.max(ReceiverCluster_TravelTimes) + (taper_length_scaling * Max_Period),2)
            # set the max time to the nearest multiple of the data sampling interval.
            if np.remainder(MaxTime,Sampling_Delta_Data) > 0:
                MaxTime = np.round(MaxTime/Sampling_Delta_Data)*Sampling_Delta_Data
            if MaxTime > ReceiverCluster_CorrData[0].lag:
                MaxTime = ReceiverCluster_CorrData[0].lag
            TimeVector = np.arange(trace_start_time , MaxTime + 0.5*Sampling_Delta_Target, Sampling_Delta_Target)

            # cut the data to the max time in corrdata.
            for cdata in ReceiverCluster_CorrData:
                cdata.data = cdata.data[:len(TimeVector)]
                cdata.lag = MaxTime
                
        # Quality control with SNR cutoff
        #1. set a time window around the predicted travel time.
        #2. get Max of the signal.
        #3. Mute the signal and get the rms of the noise outside the signal window
        #4. Compute SNR and check for QC
        GoodMatrix=[]
        GoodDist=[]
        GoodCoord=[]
        x_km=[]
        y_km=[]
        # UTM demean
        x_km_all = (ReceiverCluster_UTM[0] - np.min(ReceiverCluster_UTM[0]))/1000
        y_km_all = (ReceiverCluster_UTM[1] - np.min(ReceiverCluster_UTM[1]))/1000
        if plot_moveout:
            plt.figure(figsize=[5,4])
        for ic,cdata in enumerate(ReceiverCluster_CorrData):
            data_temp=cdata.data.copy()
            tt = ReceiverCluster_TravelTimes[ic]
            #signal time window is half longest period away from the predicted time.
            signal_twin = [tt - signal_extent_scaling*np.max(period_band), tt + signal_extent_scaling*np.max(period_band)]
            if signal_twin[0] < trace_start_time: signal_twin[0]=trace_start_time
            if signal_twin[1] > MaxTime: signal_twin[1]=MaxTime
            signal_idx = [np.argmin(np.abs(TimeVector - signal_twin[0])), np.argmin(np.abs(TimeVector - signal_twin[1]))]
            signal_absmax = np.max(np.abs(data_temp[signal_idx[0]:signal_idx[1]]))
            
            #mute signal window
            data_temp[signal_idx[0]:signal_idx[1]]=0.0
            noise_absrms = utils.rms(data_temp)
            snr = signal_absmax / noise_absrms
            # print('    SNR: ',snr)
            if snr >= min_snr and cdata.dist >= Min_Distance:
                data_mutenoise=cdata.data.copy()
                ################################################################################
                # !!!!!!!!!!!!!!!! mute noise window to zeros !!!!!!!!!!!!!!!!
                ################################################################################
                data_mutenoise[:signal_idx[0]]=0.0
                data_mutenoise[signal_idx[1]:]=0.0
                
                GoodMatrix.append(data_mutenoise)
                GoodDist.append(cdata.dist)
                GoodCoord.append(ReceiverCluster_Coord[ic])
                #x and y coordinates of the stations
                x_km.append(x_km_all[ic])
                y_km.append(y_km_all[ic])
                
                if plot_moveout:
                    plt.plot(TimeVector,moveout_scaling * cdata.data + cdata.dist,'k-',lw=0.5)
                    plt.plot(tt,cdata.dist,'r|',markerfacecolor='none',markersize=5)
                    plt.plot([signal_twin[0],signal_twin[1]],[cdata.dist,cdata.dist],'b|',lw=1)
            #
        #
        
        x_km=np.array(x_km)
        y_km=np.array(y_km)
        if len(GoodMatrix) < min_stations:
            print('  Not enough stations after SNR cutoff. Skip!')
            plt.close()
            continue
        if plot_moveout:
            plt.xlim([trace_start_time,MaxTime])
            plt.ylim([np.min(GoodDist) - 5*moveout_scaling, np.max(GoodDist) + 5*moveout_scaling])
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (km)')
            plt.title('Ref: '+reference_site+', Source: '+Source_Name+': '+str(len(GoodDist)))
            plt.savefig(os.path.join(figdir_refsite,reference_site+'_'+Source_Name+'_moveout.pdf'))
            if show_fig: plt.show()
            else: plt.close()
        GoodMatrix = np.array(GoodMatrix)
        GoodCoord = np.array(GoodCoord)
        GoodDist = np.array(GoodDist)
        if verbose: print('  Num. good trace: ',GoodMatrix.shape[0])

        #free some memory
        del ReceiverCluster_CorrData
        #
        # Plot map view of good stations and the source.
        if plot_clustermap:
            if map_engine.lower() == 'cartopy':
                plt.figure(figsize=[5,4],facecolor='w')
                ax=plt.axes(projection=projection)
                #plot receiver cluster
                plt.plot(GoodCoord[:,1],GoodCoord[:,0],'r^',markersize=4,markerfacecolor='none',transform=ccrs.PlateCarree())
                #plot receiver box
                if receiver_box is not None:
                    plt.plot(lon_box,lat_box,'b-',lw=1,transform=ccrs.PlateCarree())
                #plot source
                plt.plot(Source_Coord[1],Source_Coord[0],'r*',markersize=10, markerfacecolor='none',transform=ccrs.PlateCarree())
                #plot line from cluster center to source
                plt.plot([ReceiverCluster_Center[1],Source_Coord[1]],[ReceiverCluster_Center[0],Source_Coord[0]],'k-',
                        lw=1.5,transform=ccrs.PlateCarree())
                ax.set_extent((map_region[0],map_region[1],map_region[2]*0.95,map_region[3]*1.0))
                ax.coastlines(resolution='10m',color='k', linewidth=.5)
                gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True, dms=True, x_inline=False, y_inline=False,
                                color='gray', alpha=0.5, linestyle='--',linewidth=0.5)
                ax.add_feature(country10,lw=1)
                ax.add_feature(states10,lw=0.5)
                gl.xlocator = mticker.FixedLocator(np.arange(map_region[0],map_region[1]+2.5,5))
                gl.ylocator = mticker.FixedLocator(np.arange(map_region[2],map_region[3]+2.5,5))
                gl.xlabels_top = False
                plt.title(reference_site+' -> '+Source_Name,pad=20)
                
                plt.savefig(os.path.join(figdir_refsite,reference_site+'_'+Source_Name+'_stationmap.pdf'),format='pdf',dpi=300)
                if show_fig: plt.show()
                else: plt.close()
            elif map_engine.lower() == 'gmt':
                if verbose: print('  Plotting cluster map ...')
                fig = gmt.Figure()
                gmt.config(MAP_FRAME_TYPE=map_style, FONT_TITLE=GMT_FONT_TITLE)
                fig.coast(region=map_region, resolution="f",projection=map_projection, 
                        water="0/180/255",frame=frame,land="240",borders=["1/1p,black", "2/0.5p,100"])
                fig.basemap(frame='+t'+reference_site+' -> '+Source_Name)
                #plot receiver cluster
                fig.plot(x=GoodCoord[:,1],y=GoodCoord[:,0],style=marker_style,pen="0.5p,red",)
                #plot receiver box
                if receiver_box is not None:
                    fig.plot(x=lon_box,y=lat_box,pen="1p,blue",)
                #plot source
                fig.plot(x=Source_Coord[1],y=Source_Coord[0],style="a0.3c",pen="1p,red",)
                #plot line from cluster center to source
                fig.plot(x=[ReceiverCluster_Center[1],Source_Coord[1]],
                         y=[ReceiverCluster_Center[0],Source_Coord[0]],
                         pen="1p,black",)
                fig.savefig(os.path.join(figdir_refsite,reference_site+'_'+Source_Name+'_stationmap.pdf'))
                if show_fig: fig.show()
                else: gmt.set_display('none')
       
        # Beamforming
        if verbose: print('  Performing beamforming ...')
        Slowness_Image = np.zeros((SlownessSamples, SlownessSamples))
        Slowness_Image.fill(np.nan)
        for k in range(Slowness_Image.shape[0]):
            for kk in range(Slowness_Image.shape[1]):
                # Backazimuth and velocity:
                Slowness_Temp = utils.cart2compass(Ux[k], Uy[kk])[1]
                Velocity_Temp = 1 / Slowness_Temp
                
                if Velocity_Temp < Phase_Velocity_Limits[0] or Velocity_Temp > Phase_Velocity_Limits[1]: 
                    continue
                #
                # Time shifts:
                TimeShifts = x_km * Ux[k] + y_km * Uy[kk]
                SampleShifts = np.round(TimeShifts * SamplingRate_Target)

                Shifted_Traces = np.zeros(GoodMatrix.shape)
                for l in range(GoodMatrix.shape[0]):
                    shift = int(SampleShifts[l])
                    Trace_Temp = np.roll(GoodMatrix[l], shift)
                    
                    if shift < 0:
                        Trace_Temp[-shift:] = 0
                    elif shift > 0:
                        Trace_Temp[1:shift] = 0

                    # Allocating the shifted traces:
                    Shifted_Traces[l] = Trace_Temp
                #
                # Measuring power
                Beam = np.sum(Shifted_Traces, axis=0) / Shifted_Traces.shape[0]
                Slowness_Image[kk,k] = np.sum(np.power(Beam,2))
        ##
        if verbose: print('  ... Done beamforming')

        #Free some memory
        del GoodMatrix
        
        # Extracting the velocity and azimuth:
        Power = np.nanmax(Slowness_Image)
        Max_idx=np.nanargmax(Slowness_Image)
        Max_idx_sub=np.unravel_index(Max_idx,Slowness_Image.shape)
        
        Best_Ux = Ux[Max_idx_sub[1]]
        Best_Uy = Uy[Max_idx_sub[0]]
        Beam_BAZ, Beam_U = utils.cart2compass(Best_Ux, Best_Uy)
        Beam_Velocity = 1 / Beam_U
        Beam_Sharpness = np.nanmax(Slowness_Image) / np.nanmedian(Slowness_Image) #sharpness of the beam image.
        if verbose: print('  Beam results (Max power, sharpness, Best Ux, Best Uy, BAZ, velocity): %.4f, %.2f, %.4f, %.4f, %.4f, %.4f'
              %(Power,Beam_Sharpness,Best_Ux,Best_Uy,Beam_BAZ,Beam_Velocity))
        # Allocating the output:
        Beam_Local_temp = [ReceiverCluster_Center[0],
                           ReceiverCluster_Center[1],
                           Beam_Sharpness,
                           Beam_Velocity,
                           Beam_BAZ,
                           Power,
                           ReceiverCluster_Radius,
                           len(GoodDist),
                           reference_velocity,
                           ReceiverCluster_BAZ]
        #
        if np.isnan(np.array(Beam_Local_temp)).any():
            continue
        else:
            Beam_Local.append(Beam_Local_temp)
        #
        if plot_beampower:
            plt.figure(figsize=[5,4])
            plt.imshow(Slowness_Image/Power,extent=[Ux[0], Ux[-1], Uy[-1], Uy[0]],vmin=0,vmax=1)
            plt.plot(Best_Ux,Best_Uy,'ro',markerfacecolor='none')
            plt.xlim([Ux[0], Ux[-1]])
            plt.ylim([Uy[0], Uy[-1]])
            plt.xlabel('Ux')
            plt.ylabel('Uy')
            plt.colorbar(label='Normalized Power')
            plt.title('Ref: '+reference_site+', Source: '+Source_Name+'\nSharpness: %.1f, Velocity: %.2f km/s'%(Beam_Sharpness,Beam_Velocity))
            plt.savefig(os.path.join(figdir_refsite,reference_site+'_'+Source_Name+'_beam.pdf'))
            if show_fig: plt.show()
            else: plt.close()

        #
        del Slowness_Image
    # End of loop for one reference site to all available sources.
    if len(Beam_Local) < 1:
        return None, None
    #
    
    Beam_Local = pd.DataFrame(np.array(Beam_Local),columns=['lat','lon','sharpness','velocity','baz','power',
                                                            'radius','num','velref','baz_cluster'])
    if verbose: print('  Fitting anisotropy parameters ...')
    #reuse good_idx to filter the data
    good_idx = np.where((Beam_Local['sharpness'] >= min_beam_sharpness) & (Beam_Local['velocity'] >= Phase_Velocity_Limits[0]) & \
                        (Beam_Local['velocity'] <= Phase_Velocity_Limits[1]))[0]
    Beam_Local_subset = Beam_Local.iloc[good_idx]
    
    # following are needed for QC of the BAZ coverage.
    BAZ_check = Beam_Local_subset['baz'].copy()
    BAZ_check[BAZ_check<0] += 180
    BAZ_check[BAZ_check>180] -= 180
    histcount=np.histogram(BAZ_check,bins=AZIBIN_EDGES)[0]
    goodazbin_idx = np.where(histcount >= min_baz_measurements)[0]
    if len(goodazbin_idx) < min_good_bazbin:
        print('  Not enough azimuthal coverage of sources after QC with sharpness and velocity value. Skip!')
        return None, None
    
    # Estimate anisotropy parameters by curve fitting.
    # Only use the bins with good measurements. Need to add weight in the future.
    fitcoef=curve_fit(compute_anisotropy,Beam_Local_subset['baz'],Beam_Local_subset['velocity'],p0=[3.5,0.1,0.1])[0]
    A0 = fitcoef[0]
    A1 = fitcoef[1]
    A2 = fitcoef[2]

    # Converting the anisotropy to percentage:
    RHO = np.sqrt(A1**2 + A2**2) / A0 * 100
    THETA = np.degrees(0.5 * np.arctan2(A2, A1))
    if THETA < 0:
        THETA += 180
    elif THETA >= 0 and THETA < 90:
        THETA = np.abs(THETA - 90)
    elif THETA >= 90 and THETA <= 180:
        THETA = np.abs(180 - THETA)
    else:
        pass
    print('  Anisotropy: %.2f%%, Fast direction (degrees from north): %.2f'%(RHO,THETA))
    ReceiverCluster_Center_Final = np.round(get_ArrayAttributes(Beam_Local_subset['lat'].values,Beam_Local_subset['lon'].values)[1],4)

    if verbose: print('  Final cluster center: %.4f, %.4f'%(ReceiverCluster_Center_Final[0],ReceiverCluster_Center_Final[1]))

    # Save the anisotropy results
    beam_outfile=os.path.join(outdir,reference_site+'_beam.csv')
    Beam_Local_subset.to_csv(beam_outfile,index=False)
    print('  Beam results saved to: ',beam_outfile)

    anisotropy=[period_band[0],period_band[1],Max_Radius,min_stations,
                ReceiverCluster_Center_Final[0],ReceiverCluster_Center_Final[1],
                os.path.split(beam_outfile)[1],A0,A1,A2,RHO,THETA]
    anisotropy_outfile=os.path.join(outdir,reference_site+'_anisotropy.csv')
    anisotropy = pd.DataFrame([anisotropy],columns=['Period_min','Period_max','Max_Radius','min_stations',
                                                    'lat','lon','beamfile','A0','A1','A2','RHO','THETA'])
    anisotropy.to_csv(anisotropy_outfile,index=False)
    print('  Anisotropy results saved to: ',anisotropy_outfile)

    # plot phase velocity of the reference station
    if plot_station_result:
        fig=plt.figure(figsize=[7,5])
        ax = fig.add_subplot(6,1,(1,2))
        ax.hist(Beam_Local_subset['baz'],bins=180,range=[0,360],color='tab:blue',label='BAZ')
        # plt.xlabel('BAZ')
        ax.set_ylabel('Source count')
        ax.set_xlim([0,360])
        ax.grid(lw=0.5)
        ax.set_title('Ref site: '+reference_site)

        ax2 = fig.add_subplot(6,1,(3,6))
        #plt beam baz v.s. velocity
        sc=ax2.scatter(Beam_Local_subset['baz'],Beam_Local_subset['velocity'],s=5*Beam_Local_subset['sharpness'],marker='o',linewidths=0.5,
                    facecolor='none',c=Beam_Local_subset['sharpness'],edgecolor='k',label='Data')
        #plot the fitted curve
        x_fit=np.linspace(0,360,100)
        y_fit=compute_anisotropy(x_fit,*fitcoef)
        ax2.plot(x_fit,y_fit,'r-',label='Fitted curve')
        ax2.plot(x_fit,np.ones_like(x_fit)*fitcoef[0],'k--',label='Isotropic velocity')
        ax2.set_xlabel('BAZ')
        ax2.set_ylabel('Velocity')
        # plt.title('Ref site: '+reference_site+')
        ax2.set_xlim([0,360])
        ax2.set_ylim([Phase_Velocity_Limits[0],Phase_Velocity_Limits[1]])
        ax2.grid(lw=0.5)
        ax2.legend(ncol=3,loc='upper center')
        fig.colorbar(sc,label='Beam sharpness',location='bottom',pad=0.2)
        fig.tight_layout()
        plt.savefig(os.path.join(figdir_refsite,reference_site+'_station_result.pdf'))
        if show_fig: plt.show()
        else: plt.close()
#
    return Beam_Local, anisotropy
######################### End of BANX method ##############################


