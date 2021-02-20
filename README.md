# SeisPy
*Python modules for seismic data analysis*

### Author: Xiaotao Yang (stcyang@gmail.com)

## Introduction
This package is currently heavily dependent on **obspy** (www.obspy.org) to handle seismic data (download, read, and write, etc). Users are referred to **obspy** toolbox for related functions.

## Available modules
This package is under active development. The currently available modules are listed here.
1. `utils`

This module contains frequently used utility functions not readily available in `obspy`.

2. `downloaders`

This module contains functions used to downloading earthquake waveforms and earthquake catalogs.

3. `obsmaster`

This module contains functions to get and processing Ocean Bottom Seismometer (OBS) data. The functions and main processing modules for removing the tilt and compliance noises are inspired and modified from `OBStools` (https://github.com/nfsi-canada/OBStools) developed by Pascal Audet & Helen Janiszewski. The main tilt and compliance removal method is based on Janiszewski et al. (2019).

4. `noise`

This module contains functions used in ambient noise processing, including cross-correlations and monitoring. The key functions were converted from `NoisePy` (https://github.com/mdenolle/NoisePy) with heavy modifications. Inspired by `SeisNoise.jl` (https://github.com/tclements/SeisNoise.jl), I modified the cross-correlation workflow with FFTData and CorrData (defined in `types` module) objects. The original NoisePy script for cross-correlations have been disassembled and wrapped in functions, primarily in this module. Examples are in `notebooks/seispy_download_xcorr_demo.ipynb`.

```Python
from seispy import downloaders
from seispy.noise import compute_fft,correlate

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

5. `plotting`

This module contains major plotting functions for raw waveforms, cross-correlation results, and station maps.

## Installation
1. Create and activate the **conda** `seispy` environment

Make sure you have a working Anaconda installed. This step is required to have all dependencies installed for the package. You can also manually install the listed packages **without** creating the `seispy` environment OR if you already have these packages installed. **The order of the following commands MATTERS.**

```
$ conda create -n seispy -c conda-forge jupyter numpy scipy pandas numba pycwt python obspy mpi4py
$ conda activate seispy
```

The `jupyter` package is currently not required, **unless** you plan to run the accompanied Jupyter notebooks in **<notebooks>** directory. `mip4py` is **required** to run parallel scripts stored in **scripts** directory. The modules have been fully tested on python 3.7.x but versions >= 3.6 also seem to work from a few tests.

**Install PyGMT plotting funcitons**

Map views with geographical projections are plotted using **PyGMT** (https://www.pygmt.org/latest/). The following are steps to install PyGMT package (please refer to PyGMT webpage for trouble shooting and testing):

Install GMT through conda first into the `SeisPy` environment:

```
conda activate seispy
conda config --prepend channels conda-forge
conda install  python pip numpy pandas xarray netcdf4 packaging gmt
```

**You may need to specify the python version available on your environment.** In ~/.bash_profile, add this line: `export GMT_LIBRARY_PATH=$SEISPYROOT/lib`, where `$SEISPYROOT` is the root directory of the `seispy` environment. Then, run:

```
conda install pygmt
```

Test your installation by running:
```
python
> import pygmt
```

2. Download `seispy`

`cd` to the directory you want to save the package files. Then,
```
$ git clone https://github.com/xtyangpsp/SeisPy.git
```

3. Install `seispy` package functions using `pip`

This step will install the **SeisPy** modules under `seispy` environment. The modules would then be imported under any working directory. Remember to rerun this command if you modified the functions/modules.

```
$ pip install .
```

3. Test the installation

Run the following commands to test your installation.
```
$ python
>>> from seispy import obsmaster as obs
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

## Structure of the package
1. **seispy**: This directory contains the main modules.

2. **notebooks**: This directory contains the jupyter notebooks that provide tutorials for all modules.

3. **data**: Data for testing or running the tutorials is saved under this folder.

4. **figs**: Here we put figures in tutorials and other places.

5. **scripts**: This directory contains example scripts for data processing using `seispy`. Users are welcome to modify from the provided example scripts to work on their own data.

## Tutorials
To-be-added.

## Contribute
Any bugs and ideas are welcome. Please contact me.


## References
* Bell, S. W., D. W. Forsyth, & Y. Ruan (2015), Removing Noise from the Vertical Component Records of Ocean-Bottom Seismometers: Results from Year One of the Cascadia Initiative, Bull. Seismol. Soc. Am., 105(1), 300-313, doi:10.1785/0120140054.
* Janiszewski, H A, J B Gaherty, G A Abers, H Gao, Z C Eilon, Amphibious surface-wave phase-velocity measurements of the Cascadia subduction zone, Geophysical Journal International, Volume 217, Issue 3, June 2019, Pages 1929-1948, https://doi.org/10.1093/gji/ggz051
* Tian, Y., & M. H. Ritzwoller (2017), Improving ambient noise cross-correlations in the noisy ocean bottom environment of the Juan de Fuca plate, Geophys. J. Int., 210(3), 1787-1805, doi:10.1093/gji/ggx281.
