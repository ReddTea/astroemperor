# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

# Mostly inspired in Agathav2
hip_epoch = 1991.25
gdr1_epoch = 2015
gdr2_epoch = 2015.5
gdr3_epoch = 2016
BJD_, PSI_, PARF_, PARX_ = [], [], [], []
starmass = 1
kplan = 1

def astrometry_kepler(theta):

    ## GOST
    tmp_barycenter = astrometry_bary()
    #rel = astrometry_rel()
    reflex = astrometry_epoch()

    obs0 = tmp_barycenter['iref']

    ##
    bary = obs_lin_prop(obs0, t=gost['BJD'] - out_astrometry)

    EPOCH_AM = 1  # epoch
    for jj in EPOCH_AM:
        i = EPOCH_FLAG
        ins = 'hip'  # or 'gaia'
        band = 'Hp' if ins == 'hip' else 'G'
        if ins2 == 'TYC':
            ra = 'ra'
            dec = 'dec'
            pass



    pass
    

def astrometry_bary(theta, data):
    '''
    IN : - tt, a ref time


    OUT: pmra, pmde, parallax, TSB
    # notes: 
    globals: X_AM, X_[0]
    '''
    tt = astrometry[['epoch_ra', 'epoch_de']]  # first two rows of astrometry

    
    data_astrometry = data
    iref = [True]
    X_AM_ = data_astrometry[iref][:, 0]  # global

    dt = X_AM_[0] - X_[0]  # epoch offset between astrometry ref point and RV
    DT = tt - X_AM_[0]  # relative to astrometry reference point

    if astrotype == 'hgca':
        pass

    
    # retrieve theta

    dra0, ddec0, dpmra0, dpmdec0 = 0, 0, 0, 0
    dplx0, drv0 = 0, 0

    dra, ddec, dplx, dpmra, dpmdec, drv = theta

    dra0 += dra  # from theta
    ddec0 += ddec
    dpmra0 += dpmra
    dpmdec0 += dpmdec

    dplx0 += dplx
    drv0 += drv

    # MODEL PARALLAX AND MEAN PROPER MOTION, ref gaia epoch

    # obs iref
    
    RA_ = data_astrometry[iref][:, 1]  # global
    DEC_ = data_astrometry[iref][:, 2]  # global
    PLX_ = data_astrometry[iref][:, 3]  # global
    PMRA_ = data_astrometry[iref][:, 4]  # global
    PMDEC_ = data_astrometry[iref][:, 5]  # global
    RV_ = data_astrometry[iref][:, 6]  # global

    RA = RA_ - dra/3.6e6/np.cos(obs['dec']/(180*np.pi))  # mas to deg
    DEC = DEC_ - ddec/3.6e6
    PLX = PLX_ - dplx
    PMRA = PMRA_ - dpmra
    PMDEC = PMDEC_ - dpmdec
    RV = RV_ - drv

    obs = [RA, DEC, PLX,
           PMRA, PMDEC, RV]
    ## propagate barycentric observable
    return obs_lin_prop(obs, DT)


def astrometry_epoch(extras={}):
    ra_p, dec_p, pmra_p, pmdec_p = 0, 0, 0, 0

    plx = 0  # from out!!

    # dplx pars.kep
    if dplx:
        plx = plx - dplx
    
    dra_reflex = 0
    ddec_reflex = 0
    drv_reflex = 0
    
    dpmra_reflex = 0
    dpmdec_reflex = 0
    dplx_reflex = 0
    
    for k in range(kplan):
        ms = M0 + 2*np.pi*(X_- t0)/P  # phase
        E = kepler(ms, e)
        tmp = calc_astro()

        dra_reflex += tmp['dra_reflex']
        ddec_reflex += tmp['ddec_reflex']
        drv_reflex += tmp['drv_reflex']
        
        dpmra_reflex += tmp['dpmra_reflex']
        dpmdec_reflex += tmp['dpmdec_reflex']
        dplx_reflex += tmp['dplx_reflex']


    return all_reflex


def calc_astro():
    '''
    if(state){
        BT <- cbind(raP/plx,decP/plx,beta0*(C*X+H*Y),pmraP/plx,pmdecP/plx,rv)#au,au/yr
        return(BT)
    }else{
        return(cbind(ra=raP,dec=decP,plx=plxP,pmra=pmraP,pmdec=pmdecP,rv=rv*4.74047))
    }
    '''
    alpha0 = K/np.sin(Inc)/1e3/4.74047  # au/yr
    beta0 <- P/365.25*(K*1e-3/4.74047)*np.sqrt(1-e**2)/(2*np.pi)/np.sin(Inc)  # au

    A, B, F, G, C, H = thiele_innes(omega, Omega, Inc)

    X = np.cos(E) - e
    Y = np.sqrt(1-e**2) * np.sin(E)

    beta = beta0*plx
    raP = beta * (B*X + G*Y)
    decP = beta * (A*X + F*Y)
    plxP = -beta * (C*X + H*Y)*plx/206265e3#parallax change
    pmraP = alpha * (B*Vx + G*Vy)
    pmdecP = alpha * (A*Vx + F*Vy)

    mp = 1 # mass_planet, k2m

    eta = calc_eta(starmass, mp)
    xi = 1 / (eta+1)
    '''
    '''


    pass


def obs_lin_prop(obs, t, PA=True):
    # PA considers perspective acceleration
    # globals
    kpcmyr2auyr = 1e3*206265/1e6
    pc2au = 206265
    kpcmyr2kms = kpcmyr2auyr*auyr2kms
    
    RA, DEC, PLX, PMRA, PMDEC, RV = obs
    ra = RA / 180 * np.pi
    dec = DEC / 180 * np.pi
    plx = PLX
    pmra = PMRA
    pmdec = PMDEC
    rv = RV
    if PA:
        pass

    decs = dec + pmdec*t/365.25/206265e3  # rad
    ras = ra + pmra*t/365.25/np.cos(decs)/206265e3  # rad

    out = [ras*180/np.pi, decs*180/np.pi, np.ones(len(t))*plx,
           np.ones(len(t))*pmra, np.ones(len(t))*pmdec, np.ones(len(t))*rv]
    
    return out


    
    pass


def thiele_innes(omega, Omega, I):
    A = (np.cos(omega) * np.cos(Omega)
         - np.sin(omega) * np.sin(Omega) * np.cos(I))
    B = (np.cos(omega) * np.sin(Omega)
         + np.sin(omega) * np.cos(Omega) * np.cos(I))

    F = (-np.sin(omega) * np.cos(Omega)
         - np.cos(omega) * np.sin(Omega) * np.cos(I))
    G = (-np.sin(omega) * np.sin(Omega)
         + np.cos(omega) * np.cos(Omega) * np.cos(I))

    # these are negative in Catanzarite 2010    
    C = np.sin(omega) * np.sin(I)
    H = np.cos(omega) * np.sin(I)
    return A, B, F, G, C, H


def keplerian_motion(ar, e, E, n):
    '''
    # m1(target mass, Msun); 
    # m2(companion/planet mass, Msun); 
    # m <- m1+m2#Msun
    # E <- solveKepler(Mt,e)%%(2*pi)

    n <- 2*pi/Py#1/yr
    Py <- pars['pb']#yr
    arr <- (m*Py^2)^{1/3}#au
    ar <- arr*m2/m
    '''
    ar = 1 # cte
    x = ar * (np.cos(E) - e)  # au
    y = ar * (np.sqrt(1 - e**2) * np.sin(E))
    vx = -ar*n*np.sin(E) / (1 - e*np.cos(E))  # au/yr
    vy = ar*n*np.sqrt(1 - e**2)*np.cos(E) / (1 - e*np.cos(E))

    return x, y, vx, vy


def sky_plane_coords(P, ar, e, omega, Omega, I, T0, t):
    '''
    E(e, M)
    M(time, P, T0)

    '''
    # DEFINED ELSEWHERE
    E = 0

    # calculate thiele innes constants
    A, B, F, G, C, H = thiele_innes(omega, Omega, I)
 
    # calculate elliptical rectangular coords from E and e
    x, y, vx, vy = keplerian_motion(ar, e, E, n)
    # with thiele, X,Y, we can calculate the final position x,y for a time t
    # sky plane coordinates [pb, qb, ub]
    X = B*x + G*y
    Y = A*x + F*y
    Z = C*x + H*y

    VX = B*vx + G*vy
    VY = A*vx + F*vy
    VZ = C*vx + H*vy
    
    return X, Y, Z, VX, VY, VZ


def barycor():
    '''
    This code is to calculate the observer's barycentric velocity 
    at each epoch of the RV data.
     1. get the parameters and astrometry information needed
     ast <- out$astrometry[nrow(out$astrometry),
     c('ref_epoch','ra','dec','parallax','pmra','pmdec','radial_velocity')]
    '''


    pass


def astrometry_rel():
    pass


def RV_kepler(theta):
    # X_ to 2450000 fmt
    # import mstar
    # import planet masses
    # import planet smas
    
    Eastro = []
    
    Eb, rvg, rvs = [], [], []
    rvm, rvc = [], []
    drvT, rvgP, rvsP = np.zeros(len(tt)), np.zeros(len(tt)), np.zeros(len(tt))
    
    pass

    
def full_kepler(theta, B1, B2, extras={}):
    '''
    global  : data, starmass, mass_func
    hardcode: slices!
    input   : theta
    output  : RVs
    '''
    X_, Y_, YERR_ = np.zeros(10), np.zeros(10), np.zeros(10)
    X_AM_ = np.zeros(10)
    starmass = extras['starmass']
	######################################
    
    theta_rv = theta[B1.slice]
    theta_am = theta[B2.slice]
    
    per, K, tp, ecc, w = theta_rv
    Inc, Omega = theta_am  # RA, DEC, MURA, MUDEC

	######################################
    
    freq = 2. * np.pi / per
    M = freq * (X_ - tp)  # mean anomaly
    E = np.array([kepler.solve(m, ecc) for m in M])
    f = (np.arctan(np.sqrt((1. + ecc)/ (1. - ecc)) * np.tan(E / 2.)) * 2.)  # true anomaly

	######################################
    
	#  other params
    # check def k2m(K, P, e, starmass, Inc)
    sma, mm = cps(per, K, ecc, starmass)  # consider Inc
    mps = get_abs_mass(sma, mm)  

    eta = (starmass + mps) / mps

    # this one compares with the RV data
    model0 += K * (np.cos(f + w) + ecc * np.cos(w)) * eta # np.sin(Inc)


    # see calc.astro

    plx = 1  # get from elsewhere
    sup_const = 1/1e3/4.74047
    alpha0 = K / np.sin(Inc) * sup_const  # au/yr
    beta0 = per/365.25*(K*sup_const)*sqrt(1-e**2)/(2*np.pi)/np.sin(Inc)  # au

    alpha = alpha0 * plx


    
	# calculate thiele innes constants
    A, B, F, G, C, H = thiele_innes(w, Omega, I)
	
    Vx = -np.sin(f)
    Vy = np.cos(f) + e

    rX = np.cos(E) - e
    rY = np.sqrt(1 - e**2) * np.sin(E)
    
    beta = beta0 * plx  # mas

    raP = beta*(B*rX + G*rY)
    deP = beta*(A*rX + F*rY)
    plxP = -beta*(C*rX + H*rY)*plx/206265e3  # parallax change
    pmraP = alpha*(B*Vx + G*Vy)
    pmdeP = alpha*(A*Vx + F*Vy)

    xi = 1/(eta+1)
    # if k==2, xi = something
    # if(comp==2) xi <- -Mstar/mp
    
    raP *= xi
    deP *= xi
    plxP *= xi
    pmraP *= xi
    pmdeP *= xi

    rv = alpha0 * (C*Vx + H*Vy)  # km/s
    if extras['BT']:
        pass

    calc_astro = raP, deP, plxP, pmraP, pmdeP, rv*4.74047

    pass


def full_astro(theta, B2, extras={}):
    '''
    global  : data, starmass, mass_func, epochs
    hardcode: slices!
    input   : theta
    output  : tmp
    '''
    barycenter = astrometry_bary()  # just needed for hip?
    rel = astrometry_rel()

    # assume data and masks global
    X_AM_ = np.zeros(10)
    RA_, DE_ = np.zeros(10), np.zeros(10)

    masks = [1, 2, 3]

    tmp = {'epoch':[]}

    for mask in masks:  # runs through ins
        tmp.epoch[mask] = astrometry_epoch(mask)
        ra, de = RA_[mask], DE_[mask]


        # this calculates ra/dec diff relative to gaia dr3 epoch
        if mask == 'hip' or mask == 'TYC':
            dt = tt-epoch[mask] / 365.25  # year
            dra = (barycenter.pmra*dt
                   + ra*barycenter.parallax
                   + dra_)
            dde = (barycenter.pmdec*dt
                   + de*barycenter.parallax
                   + dde_)
            
            # REWRITE EPOCH
            tmp.epoch[mask].dra += dra
            tmp.epoch[mask].dde += dde

            # tmp$epoch[[i]] <- data.frame(dra=tmp$epoch[[i]][,'dra']+dra,ddec=tmp$epoch[[i]][,'ddec']+ddec)
    if gost:
        pass
        
    return epoch


def calc_astro():
    '''
    if(state){
        BT <- cbind(raP/plx,decP/plx,beta0*(C*X+H*Y),pmraP/plx,pmdecP/plx,rv)#au,au/yr
        return(BT)
    }else{
        return(cbind(ra=raP,dec=decP,plx=plxP,pmra=pmraP,pmdec=pmdecP,rv=rv*4.74047))
    }
    '''
    pass

# pars.kep[paste0('dra_',i)]
# pars.kep[paste0('ddec_',i)]
            

class astro_data():
     def init(file):
          pass
          


def astrometry_epoch(extras={}):
    ra_p, dec_p, pmra_p, pmdec_p = 0, 0, 0, 0

    plx = 0  # from out!!

    # dplx pars.kep
    if dplx:
        plx = plx - dplx
    
    dra_reflex = 0
    ddec_reflex = 0
    drv_reflex = 0
    
    dpmra_reflex = 0
    dpmdec_reflex = 0
    dplx_reflex = 0
    
    for k in range(kplan):
        ms = M0 + 2*np.pi*(X_- t0)/P  # phase
        E = kepler(ms, e)
        tmp = calc_astro()

        dra_reflex += tmp['dra_reflex']
        ddec_reflex += tmp['ddec_reflex']
        drv_reflex += tmp['drv_reflex']
        
        dpmra_reflex += tmp['dpmra_reflex']
        dpmdec_reflex += tmp['dpmdec_reflex']
        dplx_reflex += tmp['dplx_reflex']


    return all_reflex


def Keplerian_Model():
    pass

def cps():
    pass

def get_abs_mass(sma, mm):
    pass

#





'''
general
- pmra, not divided by cos(dec)?


astrometry.kepler

- why M0 over T_p?


# AstroDiff, if hip, cal offsets for diff catalogues
'''


















'''
options(repos='https://cran.rstudio.com')

install.packages(c("shiny","fields","magicaxis","minpack.lm",
"fields","magicaxis","foreach","doMC","parallel","MASS","doParallel","e1071",
"kernlab","glasso","JPEN","mvtnorm","rootSolve","ramify","utils","ggplot2"),
repos='https://cran.rstudio.com')



'''