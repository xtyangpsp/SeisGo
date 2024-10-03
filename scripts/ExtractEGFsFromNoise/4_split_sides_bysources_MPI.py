import sys,time,os
from multiprocessing import Pool
from seisgo import noise,utils
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
Stacking script of SeisGo to:
    1) split the negative and positive sides after merging station pairs.

'''
########################################
#########PARAMETER SECTION##############
########################################
#get arguments on the number of processors

# absolute path parameters
def split_sides_wrapper(ccfile,outdir):
    taper = True
    taper_frac=0.01
    taper_maxlen=10
    flag   = True 
    print(ccfile)
    noise.split_sides(ccfile,outdir=outdir,taper=taper,taper_frac=taper_frac,
                taper_maxlen=taper_maxlen,verbose=flag)
    return 0

def main():
    narg=len(sys.argv)
    if narg == 1:
        nproc=1
    else:
        nproc=int(sys.argv[1]) 

    ## Global parameters
    rootpath  = "data_craton"                                 # root path for this data processing
    MERGEDIR  = os.path.join(rootpath,'MERGED_PAIRS')                          # dir where stacked data is going to
    SPLITDIR    = os.path.join(rootpath,'PAIRS_SPLIT_SIDES')                            # dir where CC data is stored

    if not os.path.isdir(SPLITDIR):os.makedirs(SPLITDIR)
    #######################################
    ###########PROCESSING SECTION##########
    #######################################

    #loop through resources, for each source MPI through station pairs.
    sources_temp=utils.get_filelist(MERGEDIR)
    #exclude non-directory item in the list
    sources=[]
    for src in sources_temp:
        if os.path.isdir(src): sources.append(src)
    if nproc >=2:
        p=Pool(int(nproc))
    for src in sources:
        tt0=time.time()

        # cross-correlation files
        ccfiles = utils.get_filelist(src,"h5")
        print("assembled %d files"%(len(ccfiles)))
        outdir=os.path.join(SPLITDIR,os.path.split(src)[1])
        #loop for each station pair
        print("working on all pairs with %d processors."%(nproc))
        if nproc < 2:
            for j in range(len(ccfiles)):
                results=split_sides_wrapper(ccfiles[j],SPLITDIR)
        else: 
            results=p.starmap(split_sides_wrapper,[(ccfile,outdir) for ccfile in ccfiles])    
        del results

        print('it takes %6.2fs to merge %s' % (time.time()-tt0,src))
        # 
    if nproc >=2:
        p.close()
if __name__ == "__main__":
    main()
