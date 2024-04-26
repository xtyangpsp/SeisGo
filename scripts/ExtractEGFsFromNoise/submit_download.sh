#!/bin/bash
#SBATCH -J dld        #job name to remember
#SBATCH -n 5	#number of CPU cores you request for the job
#SBATCH -A xtyang  #queue to submit the job, our lab queue.
#SBATCH --mem-per-cpu 5000 	#requested memory per CPU
#SBATCH -t 7-10:00			#requested time day-hour:minute
#SBATCH -o %x.out  #path and name to save the output file.
#SBATCH -e %x.err 	#path to save the error file.

module purge			#clean up the modules
module load rcac		#reload rcac modules.
module use /depot/xtyang/etc/modules		#load conda module
module load conda-env/seisgo-py3.7.6  	#let every core activate the environment before running the job

mpirun -n $SLURM_NTASKS python 1_download_MPI.py
 					#run line, change file name
