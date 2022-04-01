##
import pickle,os,math
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from seisgo import utils
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from minisom import MiniSom
from kneed import KneeLocator
######
def vpcluster_evaluate_kmean(ts,nrange,smooth=False,smooth_n=3,plot=True,njob=1,
                        metric='euclidean',max_iter_barycenter=100, random_state=0):
    """

    """
    distortion=[]
    for n in nrange:
        distortion.append(TimeSeriesKMeans(n_clusters=int(n), n_jobs=njob,\
                metric=metric, max_iter_barycenter=max_iter_barycenter, \
                random_state=random_state,verbose=False).fit(ts).inertia_)

    if smooth:
        ys=utils.box_smooth(distortion,smooth_n)
        ys[0]=distortion[0]
        ys[-1]=distortion[-1]
    else:
        ys = distortion


    nbest=list(KneeLocator(nrange, ys, S=1, curve="convex", direction="decreasing").all_knees)[0]
    if plot:
        plt.figure(figsize=(8,4),facecolor='w')
        plt.plot(nrange,distortion,'o',label='data')
        if smooth:
            plt.plot(nrange,ys,'r-',label='smoothed')
        else:
            plt.plot(nrange,ys,'r-')
        plt.vlines(nbest,np.min(ys),np.max(ys),label='knee:'+str(nbest))
        plt.xlabel('number of clusters',fontsize=12)
        plt.ylabel('sum of distance to center',fontsize=12)
        plt.xticks(nrange,fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.show()

    return nbest,ys
def vpcluster_kmean(lat, lon, dep,vmodel,ncluster=None,nrange=None,spacing=1,njob=1,zrange=None,dz=None,
                         verbose=False,plot=True,savefig=True,figbase='kmean',
                      metric='euclidean',max_iter_barycenter=100, random_state=0,save=True,
                      source='vmodel',tag='v',figsize=None,evaluate_smooth=False,evaluate_plot=True):
    """
    zrange: target depth range for clustering. Default None, will use full range.
    dz: depth grid interval. If given, will interpolate the depth profiles.
    """
    if zrange is None:
        zrange=[np.min(dep),np.max(dep)]

    didx=np.where((dep>= zrange[0]) & (dep<= zrange[1]))[0]
    v=vmodel[didx]
    depth=dep[didx]

    if dz is not None: #do interpolation at depth direction.
        print("interpolation of depth at the interval of: %6.2f km"%(dz))
        zvec=np.arange(np.min(depth),np.max(depth)+0.5*dz,dz)
    else:
        zvec = depth

    all_v = []
    lat_subidx=[int(x) for x in np.arange(0,len(lat),spacing)]
    lon_subidx=[int(x) for x in np.arange(0,len(lon),spacing)]
    lat0=[]
    lon0=[]
    count=0
    for i in lat_subidx:
        for j in lon_subidx:
            v0=np.ndarray((v.shape[0]))
            for pp in range(v.shape[0]):
                v0[pp]=v[pp,i,j]
            if not np.isnan(v0).any() :
                if dz is not None:
                    vtemp=np.interp(zvec,depth,v0)
                    all_v.append(vtemp)
                else:
                    all_v.append(v0)
                lat0.append(lat[i])
                lon0.append(lon[j])
                count += 1

    ts = to_time_series_dataset(all_v)

    # determine the best number of clusters if ncluster is None.
    ss=[]
    if ncluster is None:
        print('ncluster is None. Determine the best. This may take a few minutes.')
        if nrange is None:
            nrange=np.arange(2,21,1)
        ncluster,ss = vpcluster_evaluate_kmean(ts,nrange,smooth=evaluate_smooth,smooth_n=3,plot=evaluate_plot,njob=njob,
                                metric=metric,max_iter_barycenter=max_iter_barycenter,
                                random_state=random_state)
    km = TimeSeriesKMeans(n_clusters=ncluster, n_jobs=njob,metric=metric, verbose=verbose,
                          max_iter_barycenter=max_iter_barycenter, random_state=random_state)
    y_pred = km.fit_predict(ts)


    rows = []
    for c in range(count):
        cluster = km.labels_[c]
        rows.append([lat0[c], lon0[c], cluster+1])

    df = pd.DataFrame(rows, columns=['lat', 'lon', 'cluster'])
    cdata=[]
    for yi in range(ncluster):
        cdata.append(ts[y_pred == yi].T)

    outdict=dict()
    outdict['method']="k-means"
    outdict['source']=source
    outdict['tag']=tag
    outdict['depth']=zvec
    outdict['model']=km
    outdict['pred']=cdata
    outdict['para']={'n_clusters':ncluster,'n_jobs':njob,'metric':metric,
                        'max_iter_barycenter':max_iter_barycenter,
                        'random_state':random_state,'nrange':nrange,'sum_square':ss}
    outdict['cluster_map']=df

    outfile=figbase+"_clusters_k"+str(ncluster)+"_results.pk"
    if save:
        with open(outfile,'wb') as f:
            pickle.dump(outdict,f)

    if plot:
        ###########
        #### plotting clustered data/time series.
        ###########
        if figsize is None:
            if ncluster<4:
                plt.figure(figsize=(13, 4),facecolor='w')
            else:
                plt.figure(figsize=(13, 9),facecolor='w')
        else:
            plt.figure(figsize=figsize,facecolor='w')
        for yi in range(ncluster):
            if ncluster<4:
                plt.subplot(1, ncluster, yi + 1)
            elif ncluster<9:
                plt.subplot(int(np.ceil(ncluster/2)), 2, yi + 1)
            elif ncluster < 16:
                plt.subplot(int(np.ceil(ncluster/3)), 3, yi + 1)
            elif ncluster < 21:
                plt.subplot(int(np.ceil(ncluster/4)), 4, yi + 1)
            elif ncluster < 26:
                plt.subplot(int(np.ceil(ncluster/5)), 5, yi + 1)
            else:
                plt.subplot(int(np.ceil(ncluster/6)), 6, yi + 1)
            for xx in cdata[yi]:
                plt.plot(zvec,xx, "k-", alpha=.2)
            plt.plot(zvec,km.cluster_centers_[yi].ravel(), "r-")
            plt.text(0.65, 0.15, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            plt.title(f"Cluster {yi+1}")
            plt.xlabel('depth (km)')
            plt.ylabel('Vs (km/s)')
        plt.tight_layout()

        if savefig:
            plt.savefig(figbase+"_clusters_k"+str(ncluster)+".png",format="png")
            plt.close()
        else:
            plt.show()

        #####################
        ######## plot map view of clusters.
        ####################
        # Create map using plotly
        fig = px.scatter_mapbox(
            df,lat="lat",lon='lon',color='cluster',size_max=13,zoom=3,width=900,height=800,
        )

        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "source": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"]
                }
            ]
        )
        if savefig:
            fig.write_image(figbase+"_clustermap_k"+str(ncluster)+".png",format="png")
        else:
            fig.show()
    #
    if not save:
        return outdict

#
def vpcluster_som(lat, lon, depth,v,grid_size=None,spacing=1,niteration=50000,sigma=0.3,
                     rate=0.1,verbose=False,plot=True,savefig=True,figbase='som',
                      save=True,source='vmodel',tag='v',figsize=None):
    all_v = []
    lat_subidx=[int(x) for x in np.arange(0,len(lat),spacing)]
    lon_subidx=[int(x) for x in np.arange(0,len(lon),spacing)]
    lat0=[]
    lon0=[]
    count=0
    for i in lat_subidx:
        for j in lon_subidx:
            v0=np.ndarray((v.shape[0]))
            for pp in range(v.shape[0]):
                v0[pp]=v[pp,i,j]
            if not np.isnan(v0).any() :
                all_v.append(v0)
                lat0.append(lat[i])
                lon0.append(lon[j])
                count += 1

    if grid_size is None:
        som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(all_v))))
    else:
        som_x=grid_size[0]
        som_y=grid_size[1]
    som = MiniSom(som_x, som_y,len(all_v[0]), sigma=sigma, learning_rate = rate)
    som.random_weights_init(all_v)
    som.train(all_v, niteration)

    win_map = som.win_map(all_v)
    # Returns the mapping of the winner nodes and inputs

    rows = []
    for c in range(count):
        wm=som.win_map([all_v[c]])
        x,y=list(wm.keys())[0]
        cluster=x*som_y + y + 1
        rows.append([lat0[c], lon0[c], cluster])
    df = pd.DataFrame(rows, columns=['lat', 'lon', 'cluster'])

    #clustered data
    cdata=[]

    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                cdata.append(win_map[cluster])

    outdict=dict()
    outdict['method']="som"
    outdict['source']=source
    outdict['tag']=tag
    outdict['depth']=depth
    outdict['model']=som
    outdict['pred']=cdata
    outdict['para']={'nx':som_x,'ny':som_y,'sigma':sigma,
                        'niteration':niteration, 'learning_rate':rate}
    outdict['cluster_map']=df

    outfile=figbase+"_clusters_k"+str(som_x)+"x"+str(som_y)+"_results.pk"
    if save:
        with open(outfile,'wb') as f:
            pickle.dump(outdict,f)

    ###########
    #######
    if plot:
        ncluster=len(cdata)
        if figsize is None:
            if ncluster<4:
                plt.figure(figsize=(13, 4),facecolor='w')
            else:
                plt.figure(figsize=(13, 9),facecolor='w')
        else:
            plt.figure(figsize=figsize,facecolor='w')
        for i in range(len(cdata)):
            if ncluster<4:
                plt.subplot(1, ncluster, i + 1)
            elif ncluster<9:
                plt.subplot(int(np.ceil(ncluster/2)), 2, i + 1)
            elif ncluster < 16:
                plt.subplot(int(np.ceil(ncluster/3)), 3, i + 1)
            elif ncluster < 21:
                plt.subplot(int(np.ceil(ncluster/4)), 4, i + 1)
            elif ncluster < 26:
                plt.subplot(int(np.ceil(ncluster/5)), 5, i + 1)
            else:
                plt.subplot(int(np.ceil(ncluster/6)), 6, i + 1)

            for series in cdata[i]:
                plt.plot(depth,series,c="gray",alpha=0.3)
            plt.plot(depth,np.average(np.vstack(cdata[i]),axis=0),c="red")
            plt.title(f"Cluster {i+1}")
            plt.xlabel('depth (km)')
            plt.ylabel('Vs (km/s)')
        plt.tight_layout()
        if savefig:
            plt.savefig(figbase+"_clusters_k"+str(som_x)+"x"+str(som_y)+".png",format="png")
            plt.close()
        else:
            plt.show()

        #####################
        ######## plot map view of clusters.
        ####################
        # Create map using plotly
        fig = px.scatter_mapbox(
            df,lat="lat",lon='lon',color='cluster',size_max=13,zoom=3,width=900,height=800,
        )

        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "source": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"]
                }
            ]
        )
        if savefig:
            fig.write_image(figbase+"_clustermap_k"+str(som_x)+"x"+str(som_y)+".png",format="png")
        else:
            fig.show()
    if not save:
        return outdict
