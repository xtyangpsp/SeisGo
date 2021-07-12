import sys,time,os, glob
import pandas as pd
from mpi4py import MPI
from seisgo import noise
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
'''
Stacking script of SeisGo to:
    1) load cross-correlation data for sub-stacking (if needed) and all-time average;
    2) stack data with either linear or phase weighted stacking (pws) methods (or both);
    3) save outputs in ASDF;
    4) rotate from a E-N-Z to R-T-Z system if needed.

Modified from NoisePy
'''
tt0=time.time()
########################################
#########PARAMETER SECTION##############
########################################
# absolute path parameters
rootpath  = 'data_injection'                                 # root path for this data processing
CCFDIR    = os.path.join(rootpath,'CCF')                            # dir where CC data is stored
STACKDIR  = os.path.join(rootpath,'STACK')                          # dir where stacked data is going to

# define new stacking para
flag         = True                                                # output intermediate args for debugging
stack_method = ['linear','robust']                                                # linear, pws, robust or all

# new rotation para
rotation     = False #True                                                 # rotation from E-N-Z to R-T-Z
correction   = False                                                # angle correction due to mis-orientation
if rotation and correction:
    corrfile = os.path.join(rootpath,'meso_angles.txt')             # csv file containing angle info to be corrected
    locs     = pd.read_csv(corrfile)
else: locs = None

# make a dictionary to store all variables: also for later cc
stack_para={'rootpath':rootpath,'STACKDIR':STACKDIR,\
    'stack_method':stack_method,'rotation':rotation,'correction':correction}
#######################################
###########PROCESSING SECTION##########
#######################################
#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if not os.path.isdir(STACKDIR):os.mkdir(STACKDIR)
    # save fft metadata for future reference
    stack_metadata  = os.path.join(STACKDIR,'stack_data.txt')
    fout = open(stack_metadata,'w');fout.write(str(stack_para));fout.close()

    # cross-correlation files
    ccfiles   = sorted(glob.glob(os.path.join(CCFDIR,'*.h5')))
    pairs_all,netsta_all=noise.get_stationpairs(ccfiles,False)
    splits  = len(pairs_all)
    if len(ccfiles)==0 or splits==0:
        raise IOError('Abort! no available CCF data for stacking')

    for s in netsta_all:
        tmp = os.path.join(STACKDIR,s)
        if not os.path.isdir(tmp):os.mkdir(tmp)
else:
    splits,ccfiles,pairs_all,ccomp_all = [None for _ in range(4)]

# broadcast the variables
splits    = comm.bcast(splits,root=0)
ccfiles   = comm.bcast(ccfiles,root=0)
pairs_all = comm.bcast(pairs_all,root=0)
# MPI loop: loop through each user-defined time chunck
for ipair in range (rank,splits,size):
    pair=pairs_all[ipair]
    if flag:print('station-pair %s'%(pair))
    noise.do_stacking(ccfiles,pair,outdir=STACKDIR,method=stack_method,rotation=rotation,correctionfile=locs,flag=flag)

tt1 = time.time()
print('it takes %6.2fs to stack in total' % (tt1-tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()
