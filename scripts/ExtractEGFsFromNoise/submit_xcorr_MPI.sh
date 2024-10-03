#!/bin/bash
#SBATCH -J xc  #job name to remember
#SBATCH -n 30  #number of CPU cores you request for the job
#SBATCH -A xtyang  #queue to submit the job
#SBATCH --mem-per-cpu 16000  #requested memory per CPU
#SBATCH -t 5-0:00   #requested time day-hour:minute
#SBATCH -o %x.out  #path and name to save the output file
#SBATCH -e %x.err  #path to save the error file
#module --force purge

module load rcac
module use /depot/xtyang/etc/modules
module load conda-env/seisgo-py3.7.6

mpirun -n $SLURM_NTASKS python 2_xcorr_MPI.py
