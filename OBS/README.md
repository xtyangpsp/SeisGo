# Collection of scripts to process OBS data (original authors if not me are credited and noted clearly within the codes).
## By Xiaotao Yang @ Harvard University

## Setup
To run the script, which uses ObsPy for downloading and some processing, the user need to go through the following setup steps (if not already done so):

1. Install ObsPy and recommended packages following the [instructions here](https://github.com/obspy/obspy/wiki/Installation-via-Anaconda). Here are the main steps for reference:

    * `conda config --add channels conda-forge`
    
    * `conda create -n obspy python=3.7`
    
    * `conda activate obspy`
    
    * `conda install obspy`
    
    * `conda install cartopy`
    
    * `conda install jupyter`
2. `cd` to the directory you save the scripts.
3. Open `Terminal` and make sure the shell script is set to python environment for obspy. This is usually done by:`conda activate obspy`
4. Test the installation of `ObsPy` by:
```
$python
>import obspy
```
5. In terminal: `$jupyter-notebook`

