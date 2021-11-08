##
import pickle,os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from seisgo import utils
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

def vmodel_kmean_depth(lat, lon, depth,v,ncluster,spacing=1,njob=1,
                         verbose=False,plot=True,savefig=True,figbase='kmean',
                      metric='dtw',max_iter_barycenter=10, random_state=0,save=True,
                      source='vmodel',tag='v'):
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
            if not np.isnan(v0).any():
                all_v.append(v0)
                lat0.append(lat[i])
                lon0.append(lon[j])
                count += 1

    ts = to_time_series_dataset(all_v)
    km = TimeSeriesKMeans(n_clusters=ncluster, n_jobs=njob,metric=metric, verbose=verbose,
                          max_iter_barycenter=max_iter_barycenter, random_state=random_state)
    y_pred = km.fit_predict(ts)


    rows = []
    for c in range(count):
        cluster = km.labels_[c]
        rows.append([lat0[c], lon0[c], cluster+1])

    df = pd.DataFrame(rows, columns=['lat', 'lon', 'cluster'])

    outdict=dict()
    outdict['ts']=ts
    outdict['source']=source
    outdict['tag']=tag
    outdict['depth']=depth
    outdict['km_model']=km
    outdict['km_pred']=y_pred
    outdict['km_para']={'n_clusters':ncluster,'n_jobs':njob,'metric':metric,
                        'max_iter_barycenter':max_iter_barycenter, 'random_state':random_state}
    outdict['cluster_location']=df

    outfile=figbase+"_clusters_k"+str(ncluster)+"_results.pk"
    if save:
        with open(outfile,'wb') as f:
            pickle.dump(outdict,f)

    if plot:
        ###########
        #### plotting clustered data/time series.
        ###########
        if ncluster<4:
            plt.figure(figsize=(13, 4))
        else:
            plt.figure(figsize=(13, 9))
        for yi in range(ncluster):
            if ncluster<4:
                plt.subplot(1, ncluster, yi + 1)
            else:
                plt.subplot(np.ceil(ncluster/2), 2, yi + 1)
            for xx in ts[y_pred == yi]:
                plt.plot(depth,xx.ravel(), "k-", alpha=.2)
            plt.plot(depth,km.cluster_centers_[yi].ravel(), "r-")
            plt.text(0.65, 0.15, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("Euclidean $k$-means")
            plt.xlabel('depth (km)')
            plt.ylabel('Vs (km/s)')
        plt.tight_layout()

        if savefig:
            plt.savefig(figbase+"_clusters_k"+str(ncluster)+".png",format="png")
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
