import numpy as np
from seisgo import utils
def fd1d_dx4dt4(x,dt,tmax,vmodel,rho,xsrc,xrcv,stf_freq=1,stf_shift=None,stf_type='ricker',t_interval=1):
    """
    Modified from Florian Wittkamp

    Finite-Difference acoustic seismic wave simulation
    Discretization of the first-order acoustic wave equation

    Temporal second-order accuracy $O(\Delta T^4)$

    Spatial fourth-order accuracy  $O(\Delta X^4)$

    Temporal discretization is based on the Adams-Basforth method is available in:

    Bohlen, T., & Wittkamp, F. (2016).
    Three-dimensional viscoelastic time-domain finite-difference seismic modelling using the staggered Adams-Bashforth time integrator.

    Geophysical Journal International, 204(3), 1781-1788.

    =====PARAMETERS=====
    x: spatial vector
    dt: time step for simulation.
    tmax: maximum time for simulation.
    vmodel: velocity model for each spatial grid.
    rho: density model for each spatial grid.
    xsrc: src grid index.
    xrcv: receiver grid index.
    stf_freq: source time function frequency parameter. Gaussian: width; Ricker: central frequency.
    stf_shift: source time function shift. default is 3*stf_freq.
    stf_type: "ricker" or "gaussian". Currently only support ricker.
    t_inverval: time interval of the output waveform. default: 1.

    ======RETURNS======
    tout,seisout: time and seismogram output.
    """
    c2=0.5  # CFL-Number. Stability condition.
    cmax=max(vmodel.flatten())
    dt_max=np.max(np.diff(x))/(cmax)*c2 #maximum time interval/step.
    if stf_shift is None:
        stf_shift = 3*stf_width
    nx=len(x)
    dx=np.abs(x[1]-x[0])

    if dt > dt_max:
        raise ValueError('dt %f is larger than allowable %f. '%(dt,dt_max))
    t=np.arange(0,tmax+stf_shift+0.5*dt,dt)     # Time vector
    nt=len(t)
    #wavelet information.
    # q0=1
    wavelet = np.zeros((len(t)))

    if stf_type.lower() == 'ricker':
        wlet0=utils.ricker(dt,stf_freq,stf_shift)[1]
        # tau=np.pi*stf_width*(t-1.5/stf_width)
        # wavelet=q0*(1.0-2.0*tau**2.0)*np.exp(-tau**2)
    elif stf_type.lower() == 'gaussian' or stf_type.lower() == 'gauss':
        wlet0=utils.gaussian(dt,stf_freq,stf_shift)[1]
    else:
        raise ValueError(stf_type+" not recoganized.")
    #
    wavelet[:len(wlet0)]=wlet0

    # Plotting source signal
#     plt.figure(figsize=(10,3))
#     plt.plot(t,wavelet)
#     plt.title('Source signal Ricker-Wavelet')
#     plt.ylabel('Amplitude')
#     plt.xlabel('Time in s')
#     plt.xlim(0,10)
#     plt.draw()

    # Init wavefields
    vx=np.zeros(nx)
    p=np.zeros(nx)
    vx_x=np.zeros(nx)
    p_x=np.zeros(nx)
    vx_x2=np.zeros(nx)
    p_x2=np.zeros(nx)
    vx_x3=np.zeros(nx)
    p_x3=np.zeros(nx)
    vx_x4=np.zeros(nx)
    p_x4=np.zeros(nx)

    # Calculate first Lame-Paramter
    l=rho * vmodel * vmodel

    ## Time stepping

    # Init Seismograms
    seisout=np.zeros((nt)); # Three seismograms

    # Calculation of some coefficients
    i_dx=1.0/(dx)
    kx=np.arange(0,nx-4)

    print("Starting time stepping...")
    ## Time stepping
    for n in range(2,nt):

        # Inject source wavelet
        p[xsrc]=p[xsrc]+wavelet[n]

        # Calculating spatial derivative
        p_x[kx]=i_dx*9.0/8.0*(p[kx+1]-p[kx])-i_dx*1.0/24.0*(p[kx+2]-p[kx-1])

        # Update velocity
        vx[kx]=vx[kx]-dt/rho[kx]*(13.0/12.0*p_x[kx]-5.0/24.0*p_x2[kx]+1.0/6.0*p_x3[kx]-1.0/24.0*p_x4[kx])

        # Save old spatial derivations for Adam-Bashforth method
        np.copyto(p_x4,p_x3)
        np.copyto(p_x3,p_x2)
        np.copyto(p_x2,p_x)

        # Calculating spatial derivative
        vx_x[kx]= i_dx*9.0/8.0*(vx[kx]-vx[kx-1])-i_dx*1.0/24.0*(vx[kx+1]-vx[kx-2])

        # Update pressure
        p[kx]=p[kx]-l[kx]*dt*(13.0/12.0*vx_x[kx]-5.0/24.0*vx_x2[kx]+1.0/6.0*vx_x3[kx]-1.0/24.0*vx_x4[kx])

        # Save old spatial derivations for Adam-Bashforth method
        np.copyto(vx_x4,vx_x3)
        np.copyto(vx_x3,vx_x2)
        np.copyto(vx_x2,vx_x)

        # Save seismograms
        seisout[n]=p[xrcv]

    print("Finished time stepping!")
    #shift to account for the stf_shift.

    if t_interval >1: #downsample data in time.
        tout=np.arange(0,tmax+0.5*t_interval*dt,t_interval*dt)
        seisout=np.interp(tout,t-stf_shift,seisout)
    else:
        tout=t[int(stf_shift/dt):]
        seisout=seisout[int(stf_shift/dt):]

    return tout,seisout

###
def build_vmodel(zmax,dz,nlayer,vmin,vmax,rhomin,rhomax,zmin=0,layer_dv=None):
    """
    Build layered velocity model with linearly increasing velocity, with the option of specifying anomalous layers.
    =========
    zmax: maximum depth of the model.
    dz: number of model grids, not the velocity layers. this is to create a fine grid layered model.
        with multiple grids within each layer.
    nlayer: number of velocity layers.
    vmin, vmax: velocity range.
    rhomin,rhomax: density range.
    zmin=0: minimum depth. default is 0.
    layer_dv: velocity perturbation for each layer.
    """
    layerv=np.linspace(vmin,vmax,nlayer)
    if layer_dv is None:
        layer_dv = np.zeros((nlayer))
    layerv = np.multiply(layerv,1+layer_dv)
    layerrho=np.linspace(rhomin,rhomax,nlayer)
    
    z=np.arange(zmin,zmax+-.5*dz,dz)
    
    zlayer=np.linspace(zmin,zmax,nlayer)
    v=np.zeros((len(z)))
    rho=np.zeros((len(z)))
    for i in range(len(z)):
        zidx_all=np.where((zlayer<=z[i]))[0]
        zidx=np.argmax(zlayer[zidx_all])
        
        v[i]=layerv[zidx]
        rho[i]=layerrho[zidx]
    #
    return z,v,rho
