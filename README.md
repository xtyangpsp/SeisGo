# SeisGo
*A ready-to-go Python toolbox for seismic data analysis*

![plot1](/figs/seisgo_logo.png)

## Introduction
This package is currently heavily dependent on **obspy** (www.obspy.org) to handle seismic data (download, read, and write, etc). Users are referred to **obspy** toolbox for related functions.

## Available modules
This package is under active development. The currently available modules are listed here.

1.  `utils`: This module contains frequently used utility functions not readily available in `obspy`.

2. `downloaders`: This module contains functions used to download earthquake waveforms, earthquake catalogs, station information, continous waveforms, and read data from local files.

3. `obsmaster`: This module contains functions to get and processing Ocean Bottom Seismometer (OBS) data. The functions and main processing modules for removing the tilt and compliance noises are inspired and modified from `OBStools` (https://github.com/nfsi-canada/OBStools) developed by Pascal Audet & Helen Janiszewski. The main tilt and compliance removal method is based on Janiszewski et al. (2019).

4. `noise`: This module contains functions used in ambient noise processing, including cross-correlations and monitoring. The key functions were converted from `NoisePy` (https://github.com/mdenolle/NoisePy) with heavy modifications. Inspired by `SeisNoise.jl` (https://github.com/tclements/SeisNoise.jl), We modified the cross-correlation workflow with FFTData and CorrData (defined in `types` module) objects. The original NoisePy script for cross-correlations have been disassembled and wrapped in functions, primarily in this module. We also changed the way NoisePy handles timestamps when cross-correlating. This change results in more data, even with gaps. The xcorr functionality in SeisGo also has the minimum requirement on knowledge about the downloading step. We try to optimize and minimize inputs from the user. We added functionality to better manipulate the temporal resolution of xcorr results.

5. `plotting`: This module contains major plotting functions for raw waveforms, cross-correlation results, and station maps.

6. `monitoring`: This module contains functions for ambient noise seismic monitoring, adapted from functions by Yuan et al. (2021).

7. `clustering`: Clustering functions for seismic data and velocity models.

8. `stacking`: stacking of seismic data.

9. `types`: This module contains the definition of major data types and classes.

## Installation
**SeisGo** is available on PyPi (https://pypi.org/project/seisgo/). You can install it as a regular package `pip install seisgo`. The following instruction shows how to install seisgo with a virtual environment with github repository.

1. Create and activate the **conda** `seisgo` environment

Make sure you have a working Anaconda installed. This step is required to have all dependencies installed for the package. You can also manually install the listed packages **without** creating the `seisgo` environment OR if you already have these packages installed. **The order of the following commands MATTERS.**

```
$ conda create -n seisgo -c conda-forge jupyter numpy scipy pandas numba pycwt python obspy mpi4py
$ conda activate seisgo
```

The `jupyter` package is currently not required, **unless** you plan to run the accompanied Jupyter notebooks in **<notebooks>** directory. `mip4py` is **required** to run parallel scripts stored in **scripts** directory. The modules have been fully tested on python 3.7.x but versions >= 3.6 also seem to work from a few tests.

**Install PyGMT plotting funcitons**

Map views with geographical projections are plotted using **PyGMT** (https://www.pygmt.org/latest/). It seems that only PIP install could get the latest version [personal experience]. The following are steps to install PyGMT package (please refer to PyGMT webpage for trouble shooting and testing):

Install GMT through conda first into the `SeisGo` environment:

```
conda activate seisgo
conda config --prepend channels conda-forge
conda install  python pip numpy pandas xarray netcdf4 packaging gmt
```

**You may need to specify the python version available on your environment.** In ~/.bash_profile, add this line: `export GMT_LIBRARY_PATH=$SEISGOROOT/lib`, where `$SEISGOROOT` is the root directory of the `seisgo` environment. Then, run:

```
conda install pygmt
```

Test your installation by running:
```
python
> import pygmt
```

2. Download `SeisGo`

`cd` to the directory you want to save the package files. Then,
```
$ git clone https://github.com/xtyangpsp/SeisGo.git
```

3. Install `seisgo` package functions using `pip`

This step will install the **SeisGo** modules under `seisgo` environment. The modules would then be imported under any working directory. Remember to rerun this command if you modified the functions/modules.

```
$ pip install .
```

3. Test the installation

Run the following commands to test your installation.
```
$ python
>>> from seisgo import obsmaster as obs
>>> tflist=obs.gettflist(help=True)
------------------------------------------------------------------
| Key    | Default  | Note                                       |
------------------------------------------------------------------
| ZP     | True     | Vertical and pressure                      |
| Z1     | True     | Vertical and horizontal-1                  |
| Z2-1   | True     | Vertical and horizontals (1 and 2)         |
| ZP-21  | True     | Vertical, pressure, and two horizontals    |
| ZH     | True     | Vertical and rotated horizontal            |
| ZP-H   | True     | Vertical, pressure, and rotated horizontal |
------------------------------------------------------------------
```

## Update SeisGo
If you installed SeisGo through github, run the following lines to update to the latest version (may not be a release on pip yet):

```Python
git pull
pip install .   #note there is a period "." sign here, indicating the current directory
```

If you installed SeisGo through pip, you can get the latest release (GitHub always has the most recent commits) though running these lines:

```Python
pip install seisgo --upgrade
```


## Structure of the package
1. **seisgo**: This directory contains the main modules.

2. **notebooks**: This directory contains the jupyter notebooks that provide tutorials for all modules.

3. **data**: Data for testing or running the tutorials is saved under this folder.

4. **figs**: Here we put figures in tutorials and other places.

5. **scripts**: This directory contains example scripts for data processing using `seisgo`. Users are welcome to modify from the provided example scripts to work on their own data.

## Tutorials on key functionalities
1. Download continuous waveforms for large-scale job (see item 3 for small jobs processing in memory). Example script using MPI is here: `scripts/seisgo_download_MPI.py`. The following lines show an example of the structure without MPI (so that you can easily test run it in Jupyter Notebook).

```Python
import os,glob
from seisgo.utils import split_datetimestr,extract_waveform,plot_trace
from seisgo import downloaders

rootpath = "data_test" # roothpath for the project
DATADIR  = os.path.join(rootpath,'Raw')          # where to store the downloaded data
down_list  = os.path.join(DATADIR,'station.txt') # CSV file for station location info

# download parameters
source='IRIS'
samp_freq = 10                      # targeted sampling rate at X samples per seconds
rmresp   = True
rmresp_out = 'DISP'

# targeted region/station information
lamin,lamax,lomin,lomax= 39,41,-88,-86           # regional box:
net_list  = ["TA"] #                              # network list
chan_list = ["BHZ"]
sta_list  = ["O45A","SFIN"]                       # station
start_date = "2012_01_01_0_0_0"                   # start date of download
end_date   = "2012_01_02_1_0_0"                   # end date of download
inc_hours  = 12                                   # length of data for each request (in hour)
maxseischan = 1                                   # the maximum number of seismic channels
ncomp      = maxseischan #len(chan_list)

downlist_kwargs = {"source":source, 'net_list':net_list, "sta_list":sta_list, "chan_list":chan_list, \
                   "starttime":start_date, "endtime":end_date, "maxseischan":maxseischan, "lamin":lamin, \
                   "lamax":lamax,"lomin":lomin, "lomax":lomax, "fname":down_list}

stalist=downloaders.get_sta_list(**downlist_kwargs) #
#this is a critical step for long duration downloading, as a demo here.
all_chunk = split_datetimestr(start_date,end_date,inc_hours)

#################DOWNLOAD SECTION#######################
for ick in range(len(all_chunk)-1):
   s1= all_chunk[ick];s2=all_chunk[ick+1]
   print('time segment:'+s1+' to '+s2)
   downloaders.download(source=source,rawdatadir=DATADIR,starttime=s1,endtime=s2,\
                      stationinfo=stalist,samp_freq=samp_freq)

print('downloading finished.')

#extrace waveforms
tr=extract_waveform(glob.glob(os.path.join(DATADIR,"*.h5"))[0],net_list[0],sta_list[0],comp=chan_list[0])
plot_trace([tr],size=(10,4),ylabels=['displacement'],title=[net_list[0]+'.'+sta_list[0]+'.'+chan_list[0]])
```

You should see the following image showing the waveform for TA.O45A.
![plot1](/figs/download_continuous_example.png)

2. Download earthquake catalog and waveforms with given window length relative to phase arrivals
TBA.

3. Ambient noise cross-correlations
* Minimum lines version for processing small data sets in memory. Another example is in `notebooks/seisgo_download_xcorr_demo.ipynb`.

```Python
from seisgo import downloaders
from seisgo.noise import compute_fft,correlate

# download parameters
source='IRIS'                                 # client/data center. see https://docs.obspy.org/packages/obspy.clients.fdsn.html for a list
samp_freq = 10                                                  # targeted sampling rate at X samples per seconds

chan_list = ["BHZ","BHZ"]
net_list  = ["TA","TA"] #                                             # network list
sta_list  = ["O45A","SFIN"]                                               # station (using a station list is way either compared to specifying stations one by one)
start_date = "2012_01_01_0_0_0"                               # start date of download
end_date   = "2012_01_02_1_0_0"                               # end date of download

# Download
print('downloading ...')
trall,stainv_all=downloaders.download(source=source,starttime=start_date,endtime=end_date,\
                                  network=net_list,station=sta_list,channel=chan_list,samp_freq=samp_freq)

print('cross-correlation ...')
cc_len    = 1800                                                            # basic unit of data length for fft (sec)
cc_step      = 900                                                             # overlapping between each cc_len (sec)
maxlag         = 100                                                        # lags of cross-correlation to save (sec)

#get FFT
fftdata1=compute_fft(trall[0],cc_len,cc_step,stainv=stainv_all[0])
fftdata2=compute_fft(trall[1],cc_len,cc_step,stainv=stainv_all[1])

#do correlation
corrdata=correlate(fftdata1,fftdata2,maxlag,substack=True)

#plot correlation results
corrdata.plot(freqmin=0.1,freqmax=1,lag=100)
```

You should get the following figure:
![plot1](/figs/noise_xcorr_example.png)

* Run large-scale jobs through MPI. For processing of large datasets, the downloaded and xcorr data will be saved to disk. Example script here: `scripts/seisgo_xcorr_MPI.py`

## Contribute
Any bugs and ideas are welcome. Please file an issue through GitHub.


## References
* Bell, S. W., D. W. Forsyth, & Y. Ruan (2015), Removing Noise from the Vertical Component Records of Ocean-Bottom Seismometers: Results from Year One of the Cascadia Initiative, Bull. Seismol. Soc. Am., 105(1), 300-313, doi:10.1785/0120140054.
* Janiszewski, H A, J B Gaherty, G A Abers, H Gao, Z C Eilon, Amphibious surface-wave phase-velocity measurements of the Cascadia subduction zone, Geophysical Journal International, Volume 217, Issue 3, June 2019, Pages 1929-1948, https://doi.org/10.1093/gji/ggz051
* Tian, Y., & M. H. Ritzwoller (2017), Improving ambient noise cross-correlations in the noisy ocean bottom environment of the Juan de Fuca plate, Geophys. J. Int., 210(3), 1787-1805, doi:10.1093/gji/ggx281.
* Jiang, C., & Denolle, M. A. (2020). NoisePy: A New High-Performance Python Tool for Ambient-Noise Seismology. Seismological Research Letters. https://doi.org/10.1785/0220190364
* Clements, T., & Denolle, M. A. (2020). SeisNoise.jl: Ambient Seismic Noise Cross Correlation on the CPU and GPU in Julia. Seismological Research Letters. https://doi.org/10.1785/0220200192
* Yuan, C., Bryan, J., & Denolle, M. (2021). Numerical comparison of time-, frequency-, and wavelet-domain methods for coda wave interferometry. Geophysical Journal International, 828–846. https://doi.org/10.1093/gji/ggab140
