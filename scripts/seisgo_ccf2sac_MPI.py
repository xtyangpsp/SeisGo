import sys,time,os,glob
from mpi4py import MPI
from seisgo.noise import save_corrfile_to_sac

"""
Saves CCF data to SAC.
"""
# absolute path parameters
rootpath  = 'data_injection'                                 # root path for this data processing
CCFDIR    = os.path.join(rootpath,'CCF_test')                            # dir where CC data is stored
SACDIR  = os.path.join(rootpath,'CCF_sac')                          # dir where stacked data is going to

#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # cross-correlation files
    ccfiles   = sorted(glob.glob(os.path.join(CCFDIR,'*.h5')))
    splits  = len(ccfiles)
    if splits==0:raise IOError('Abort! no available CCF data for converting')
else:
    splits,ccfiles = [None for _ in range(2)]

# broadcast the variables
splits    = comm.bcast(splits,root=0)
ccfiles   = comm.bcast(ccfiles,root=0)
#--------End of setting up MPI parameters---------

# MPI loop: loop through each user-defined time chunck
for ifile in range(rank,splits,size):
    save_corrfile_to_sac(ccfiles[ifile],rootdir=SACDIR,v=False)

comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()
