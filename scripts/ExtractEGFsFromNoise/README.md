This folder contains scripts to extract EGFs with SeisGo. 

Steps to extract EGFs
1. Download data. Script: 1_download_MPI.py
2. Compute xcorr. Script: 2_xcorr_MPI.py
3. Merge station pairs. Script: 3_merge_pairs_bysources_MPI.py
4. Split two sides (negative and positive). Script: 4_split_sides_bysources_MPI.py

The above four steps produce the base of the data for FWANT. The following step creates the shaped data with specific wavelet. This step should be run after finalize simulation parameters (particularly the source time function). 

Step 3, when merging, has the option of splitting. Then step 4 is not needed if the data is already splitted.

All of these scripts can be run on local computer or the cluster. The tag "MPI" means the scripts are written for paralle run with "mpirun". The syntax can be found in the shell script. When running on the cluster, there are shell scripts that are used to submit the jobs. The file names are straightforward enough to use.

The tag "bysources" means the files (input or output) are organized by folders named with the virtual source (i.e., net.station). This is necessary to process large dataset.

Please change the parameters for each step for each project. 