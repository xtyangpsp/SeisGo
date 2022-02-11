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

def xcorr_norm_methods(mode="tf"):
    """
    Normalization methods for cross-correlations.
    """

    fnorm=["rma","phase_only"]
    tnorm=["rma","one_bit","ftn"]

    if mode=="t": return tnorm
    elif mode=="f": return fnorm
    else: return tnorm,fnorm

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
