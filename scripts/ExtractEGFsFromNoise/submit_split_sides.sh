#!/bin/bash
#SBATCH -J split  #job name to remember
#SBATCH -n 40  #number of CPU cores you request for the job
#SBATCH -A xtyang  #queue to submit the job
#SBATCH --mem-per-cpu 2000  #requested memory per CPU
#SBATCH -t 2-0:00   #requested time day-hour:minute
#SBATCH -o %x.out  #path and name to save the output file
#SBATCH -e %x.err  #path to save the error file

module purge			#clean up the modules
module load rcac		#reload rcac modules.
module use /depot/xtyang/etc/modules
module load conda-env/seisgo-py3.7.6

python 4_split_sides_bysources_MPI.py $SLURM_NTASKS
