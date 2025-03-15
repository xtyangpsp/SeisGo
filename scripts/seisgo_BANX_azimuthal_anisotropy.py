#!/usr/bin/env python
# coding: utf-8
import os,sys,time
import numpy as np
import pandas as pd
from seisgo import noise,utils
from multiprocessing import Pool
import pygmt as gmt
from seisgo.anisotropy import do_BANX
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# ## BANX wrapper with control parameters
def BANX_wrapper(stationdict_all, reference_site, datadir, outdir_root, receiver_box):
    """
    Wrapper function for do_BANX function in anisotropy module.
    This function is called by the main script mainly for parallelization purpose.
    """
    XCorrComp = 'ZZ'
    # Subarray parameters:
    Min_Stations = 10   

    # SNR:
    Min_SNR = 10

    # Scaling factors
    Min_Radius_scaling = 1
    Max_Radius_scaling = 1.25
    Min_Distance_scaling = 3
    Signal_Extent_Scaling = 2.5 # used to set the signal window when computing the signal to noise ratio. 
    # The window would be [predicted arrival time - Signal_Extent_Scaling*Max_Period, predicted arrival time + Signal_Extent_Scaling*Max_Period] 

    Vel_Signal_Window = 3.2
    # Beamforming space:
    Max_Slowness = 0.5   # [s/km] Maximum slowness in the beamforming
    Slowness_Step = 0.005  #[s/km] Slowness interval

    # Beamforming limit:
    Vel_Reference = 3.5  # [km/s]
    Vel_Perturbation = 0.4 # [Percentage fraction] 0.5 = %50                                                                                              

    Taper_Length_Scaling= 7 # Taper length scaling factor. The taper length is Taper_Length_Scaling*Max_Period.

    AZIBIN_STEP = 6 # azimuthal bin step size in degrees used in the QC step after beamforming of all sources.
    # QC baz coverage
    Min_BAZ_measurements = 1 #minimum number of measurements in each azimuthal bin. Should be >=3. Use 1 for testing here.
    Min_Good_BAZBIN = 5 #minimum number of good bins with >= Min_BAZ_measurements in each azimuthal bin. Should be >=5 (recommended).

    Min_Beam_Sharpness = 10 #minimum beam sharpness to be considered as a good measurement.

    MinTime = 0.0 #start time of the xcorr data.

    Sampling_Rate_Target = 5 #target sampling rate. Needs to be integer times the data sampling rate to avoid resampling error.
    #The data will be resampled to Sampling_Rate_Target (samples per second).

    Period_Band = [15,30]

    DoubleSided = False

    ####################################################
    #### plotting controls ##############################
    ####################################################
    show_fig = False #figures will be plotted by not shown.
    # plot moveout of good traces
    plot_moveout = True
    moveout_scaling = 4

    # plot cluster map and the source
    plot_clustermap = True #not plot if on HPC cluster. some display issue may happend if plotting.
    map_engine = 'cartopy'
    #slowness image
    plot_beampower =True

    # plot phase velocity of the reference station
    plot_station_result = True

    ########################################################
    #### Calling do_BANX function in anisotropy module. ####
    ########################################################
    Beam_Local, anisotropy=do_BANX(stationdict_all, reference_site, Period_Band, Vel_Reference,datadir,
                                    outdir_root,sampling_rate=Sampling_Rate_Target,min_stations=Min_Stations, 
                                    min_snr=Min_SNR, min_radius_scaling=Min_Radius_scaling,
                                    max_radius_scaling=Max_Radius_scaling, min_distance_scaling=Min_Distance_scaling, 
                                    signal_window_velocity=Vel_Signal_Window,
                                    signal_extent_scaling=Signal_Extent_Scaling,max_slowness=Max_Slowness,slowness_step=Slowness_Step,
                                    velocity_perturbation=Vel_Perturbation, trace_start_time=MinTime,taper_length_scaling=Taper_Length_Scaling,
                                    azimuth_step=AZIBIN_STEP,min_baz_measurements=Min_BAZ_measurements,min_good_bazbin=Min_Good_BAZBIN,
                                    min_beam_sharpness=Min_Beam_Sharpness,doublesided=DoubleSided, cc_comp =XCorrComp,
                                    receiver_box=receiver_box,show_fig=show_fig,plot_moveout=plot_moveout, moveout_scaling=moveout_scaling,
                                    plot_clustermap=plot_clustermap, map_engine = map_engine, plot_beampower=plot_beampower,
                                    plot_station_result=plot_station_result,verbose=True)
    
    return Beam_Local, anisotropy

## Main script
# This is the main script for BANX processing.
def main():
    """
    Main script for BANX processing.
    """
    # Read number of processors from command line.
    narg = len(sys.argv)
    if narg == 1:
        nproc=1
    else:
        nproc=int(sys.argv[1])
    #
    if nproc > 1:
        # Create a pool of workers
        pool = Pool(processes=nproc)
    """
    Most of the time, users only need to change the following parameters:
    """
    # set root directory
    rootdir='.'
    datadir=os.path.join(rootdir,'data_craton/PAIRS_AVERAGEDSIDES_stack_robust_egf') #'data_craton/PAIRS_TWOSIDES_stack_robust')
    outdir_root=os.path.join(rootdir,'BANX_results')
    if not os.path.isdir(outdir_root):os.makedirs(outdir_root, exist_ok=True)
    ReceiverBox_lat=[36,39.5]
    ReceiverBox_lon=[-92,-84]
    stationinfo_file='station_info.csv'
    use_stationfile=True
    plot_station_map = True
    ######
    ##########################################################
    ####### End of user parameters. ############################
    ##########################################################

    # ## Extract the netsta list and their coordinates from the xcorr data
    # The coordinates for each net.sta are stored in dictionaries.
    # load data
    if use_stationfile:
        station_df=pd.read_csv(stationinfo_file)
        coord_all=dict()
        for i in range(len(station_df)):
            coord_all[station_df['net.sta'].iloc[i]] = [station_df['lat'].iloc[i],station_df['lon'].iloc[i],station_df['ele'].iloc[i]]
        # remove duplicates
        netsta_all=list(coord_all.keys()) #sorted(set(netsta_all))
        print('Read %d station from %s'%(len(netsta_all),stationinfo_file))
    else:
        sourcelist=utils.get_filelist(datadir)
        t1 = time.time()
        # netsta_all=[]
        coord_all=dict()
        if nproc < 2:
            for src in sourcelist:
                # srcdir=os.path.join(datadir,src)
                ccfiles=utils.get_filelist(src,'h5',pattern='P_stack')
                _,_,coord=noise.get_stationpairs(ccfiles,getcoord=True,verbose=True)
                # netsta_all.extend(netsta)
                coord_all = coord_all | coord
        else:
            #parallelization
            print('Using %d processes to process %d source files'%(nproc,len(sourcelist)))
            results=pool.starmap(noise.get_stationpairs, [(utils.get_filelist(src,'h5','P_stack'),
                                                        False,False,True) for src in sourcelist])
            # If running interactively, change the above line to: 
            # results = pool.startmap(noise.get_stationpairs, [(src,True) for src in sourcelist])
            # unpack results. Needed when running interactively. Otherwise, the results are not unpacked and have been saved to files.
            _, _, coord_all = zip(*results)
            # netsta_all = [item for sublist in netsta_all for item in sublist]
            coord_all = {k: v for d in coord_all for k, v in d.items()}
        #
        # remove duplicates
        netsta_all=list(coord_all.keys()) #sorted(set(netsta_all))
        print('Extracted %d net.sta from %d source files in %.2f seconds.'%(len(netsta_all),len(sourcelist),time.time()-t1))
        
        stationfile = os.path.join(rootdir,stationinfo_file)
        fout = open(stationfile,'w')
        fout.write('net.sta,lat,lon,ele\n')
        for i in range(len(netsta_all)):
            coord0 = coord_all[netsta_all[i]]
            fout.write('%s,%f,%f,%f\n'%(netsta_all[i],coord0[0],coord0[1],coord0[2]))
        fout.close()
        print('Station information saved to %s'%stationfile)

    # ## Subset the station list for the receiver box region
    #set receiver region box
    #this is usually a smaller region than the entire dataset. 
    # Stations within this box region are used as receivers while all
    # stations may be used as the sources.
    # set receiver box for do_BANX function.
    ReceiverBox = [ReceiverBox_lon[0],ReceiverBox_lon[1],ReceiverBox_lat[0],ReceiverBox_lat[1]]

    ReceiverList_Sites=[] #net.sta strings.
    ReceiverList_Coord=[] #lat, lon

    SourceList_Sites=[]
    SourceList_Coord=[]
    for i in range(len(netsta_all)):
        #Master site
        coord0 = coord_all[netsta_all[i]][:2] #coordinates: lat, lon in order.
        SourceList_Sites.append(netsta_all[i])
        SourceList_Coord.append(coord0)
        if coord0[0] >= ReceiverBox_lat[0] and coord0[0] <= ReceiverBox_lat[1] and \
                    coord0[1] >= ReceiverBox_lon[0] and coord0[1] <= ReceiverBox_lon[1]:
            ReceiverList_Sites.append(netsta_all[i])
            ReceiverList_Coord.append(coord0)
    #
    """
    Plot the station map.
    """
    if plot_station_map:
        #plot station map.
        source_coord_array=np.array(SourceList_Coord)
        marker_style="i0.17c"
        map_style="plain"
        projection="M3.i"
        frame="af"
        title="station map"
        GMT_FONT_TITLE="14p,Helvetica-Bold"
        lon_all,lat_all=source_coord_array[:,1],source_coord_array[:,0]

        region="%6.2f/%6.2f/%5.2f/%5.2f"%(np.min(lon_all),np.max(lon_all),np.min(lat_all),np.max(lat_all))
        fig = gmt.Figure()
        gmt.config(MAP_FRAME_TYPE=map_style, FONT_TITLE=GMT_FONT_TITLE)
        fig.coast(region=region, resolution="f",projection=projection, 
                water="0/180/255",frame=frame,land="240",
                borders=["1/1p,black", "2/0.5p,100"])
        fig.basemap(frame='+t'+title+'')
        fig.plot(
            x=lon_all,
            y=lat_all,
            style=marker_style,
            pen="0.5p,red",
        )
        #plot receiver box
        lon_box=[ReceiverBox_lon[0],ReceiverBox_lon[1],ReceiverBox_lon[1],ReceiverBox_lon[0],ReceiverBox_lon[0]]
        lat_box=[ReceiverBox_lat[0],ReceiverBox_lat[0],ReceiverBox_lat[1],ReceiverBox_lat[1],ReceiverBox_lat[0]]
        fig.plot(
            x=lon_box,
            y=lat_box,
            pen="1p,blue",
        )
        fig.savefig(os.path.join(rootdir,'station_map.pdf'))
        gmt.set_display('none')
        fig.show()
    #

    """"
    Start the main loop
    """
    #######################################
    #### Loop over the reference sites ####
    #######################################
    if nproc <2:
        # Beam_Local_all, anisotropy_all = [],[]
        for i in range(len(ReceiverList_Sites)):
            #Master site
            Ref_Site = ReceiverList_Sites[i]
            print('Processing reference site %s --- %d/%d'%(Ref_Site,i+1,len(ReceiverList_Sites)))
            
            # Beam_Local, anisotropy=BANX_wrapper(coord_all,Ref_Site, datadir, outdir_root, ReceiverBox)
            _,_=BANX_wrapper(coord_all,Ref_Site, datadir, outdir_root, ReceiverBox)
            #end here for debug/test.
            # Beam_Local_all.append(Beam_Local)
            # anisotropy_all.append(anisotropy)
            # break
        #
    else:
        #parallelization
        print('Using %d processes to process %d receiver sites'%(nproc,len(ReceiverList_Sites)))
        ############
        
        pool.starmap(BANX_wrapper, [(coord_all,Ref_Site, datadir, outdir_root, ReceiverBox) for Ref_Site in ReceiverList_Sites])
        # If running interactively, change the above line to: 
        # results = pool.startmap(BANX_wrapper, [(coord_all,Ref_Site, datadir, outdir_root, ReceiverBox) for Ref_Site in ReceiverList_Sites])
        pool.close()

        # unpack results. Needed when running interactively. Otherwise, the results are not unpacked and have been saved to files.
        
        # Beam_Local_all, anisotropy_all = zip(*results)
##end of main

########### 
if __name__ == "__main__":
    main()
    



