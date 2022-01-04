# SeisGo
*A ready-to-go Python toolbox for seismic data analysis*

### Author: Xiaotao Yang

## Introduction
This package is currently dependent on **obspy** (www.obspy.org) to basic handling of seismic data (download, read, and write, etc). Users are referred to **obspy** toolbox for related functions.

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
1. Create and activate the **conda** `seisgo` environment

Make sure you have a working Anaconda installed. This step is required to have all dependencies installed for the package. You can also manually install the listed packages **without** creating the `seisgo` environment OR if you already have these packages installed. **The order of the following commands MATTERS.**

```
$ conda create -n seisgo -c conda-forge jupyter numpy scipy pandas numba pycwt python obspy mpi4py
$ conda activate seisgo
```

The `jupyter` package is currently not required, **unless** you plan to run the accompanied Jupyter notebooks in **<notebooks>** directory. `mip4py` is **required** to run parallel scripts stored in **scripts** directory. The modules have been fully tested on python 3.7.x but versions >= 3.6 also seem to work from a few tests.

**Install PyGMT plotting funcitons**

Map views with geographical projections are plotted using **PyGMT** (https://www.pygmt.org/latest/). The following are steps to install PyGMT package (please refer to PyGMT webpage for trouble shooting and testing):

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

2. Install `seisgo` package functions using `pip`

`cd` to the directory you want to save the package files. Then,
```
$ conda activate seisgo
$ pip install seisgo
```

This step will install the **SeisGo** modules under `seisgo` environment. The modules would then be imported under any working directory. Remember to rerun this command if you modified the functions/modules.

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


## Tutorials on key functionalities
See https://github.com/xtyangpsp/SeisGo for tutorials and more detailed descriptions.


## Contribute
Any bugs and ideas are welcome. Please file an issue through GitHub https://github.com/xtyangpsp/SeisGo.


## References
* Bell, S. W., D. W. Forsyth, & Y. Ruan (2015), Removing Noise from the Vertical Component Records of Ocean-Bottom Seismometers: Results from Year One of the Cascadia Initiative, Bull. Seismol. Soc. Am., 105(1), 300-313, doi:10.1785/0120140054.
* Clements, T., & Denolle, M. A. (2020). SeisNoise.jl: Ambient Seismic Noise Cross Correlation on the CPU and GPU in Julia. Seismological Research Letters. https://doi.org/10.1785/0220200192
* Janiszewski, H A, J B Gaherty, G A Abers, H Gao, Z C Eilon, Amphibious surface-wave phase-velocity measurements of the Cascadia subduction zone, Geophysical Journal International, Volume 217, Issue 3, June 2019, Pages 1929-1948, https://doi.org/10.1093/gji/ggz051
* Jiang, C., & Denolle, M. A. (2020). NoisePy: A New High-Performance Python Tool for Ambient-Noise Seismology. Seismological Research Letters. https://doi.org/10.1785/0220190364
* Tian, Y., & M. H. Ritzwoller (2017), Improving ambient noise cross-correlations in the noisy ocean bottom environment of the Juan de Fuca plate, Geophys. J. Int., 210(3), 1787-1805, doi:10.1093/gji/ggx281.
* Yuan, C., Bryan, J., & Denolle, M. (2021). Numerical comparison of time-, frequency-, and wavelet-domain methods for coda wave interferometry. Geophysical Journal International, 828–846. https://doi.org/10.1093/gji/ggab140
