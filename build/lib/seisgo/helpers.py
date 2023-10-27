#SeisGo helper functions.
#

"""
This module contains functionsn that help the users understand and use SeisGo.
It has similar role as a tutorial, though it can be accessed within codes. The
purpose is to reduce redundance and make it easier to maintain and update.
"""

def xcorr_methods():
    """
    Returns available xcorr methods.
    """
    o=["xcorr", "deconv", "coherency"]

    return o

def stack_methods():
    """
    Returns available stacking methods.
    """
    o=["linear","pws","tf-pws","robust","acf","nroot","selective","cluster"]

    return o
def dvv_methods():
    """
    Returns available dv/v measuring methods.
    """
    o=['wts','ts']

    return o

def wavelet_labels():
    """
    Returns the available wavelets.
    """
    o=["gaussian","ricker"]

    return o
#
def xcorr_norm_methods(mode="tf"):
    """
    Normalization methods for cross-correlations.
    """

    fnorm=["rma","phase_only"]
    tnorm=["rma","one_bit","ftn"]

    if mode=="t": return tnorm
    elif mode=="f": return fnorm
    else: return tnorm,fnorm

def xcorr_output_structure():
    """
    Options to organize xcorr output files. These options determine the subdirectory
    under the root data directory.

    Available options:
    raw: same as raw data, normally by time chunks for all pairs.
    source: organized by subfolder named with virtual source, with all receiver pairs in the same time chunk file.
    station-pair: subfolder named by station-pair. all components will be saved in the same chunk file.
    station-component-pair: subfolder named by station-pair, with lower level folder named by component pair.
    """
    o=["raw","source","station-pair","station-component-pair"]
    o_short=["r","s","sp","scp"]

    return o,o_short

def xcorr_sides():
    """
    Side options/labels for xcorr data.
    """
    o=["a","n","p"]

    return o

def outdatafile_formats():
    """
    Formats when saving data files.
    """

    o=["asdf","pickle"]

    return o

def datafile_extension():
    """
    File extensions for input and output data.
    """

    o=["h5","pk"]

    return o
