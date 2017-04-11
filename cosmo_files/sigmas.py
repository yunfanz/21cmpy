import itertools
import numpy as np
import matplotlib.pyplot as plt
import cosmolopy.perturbation as pb
import cosmolopy.density as cd
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d
import os
from tempfile import mkdtemp
cachedir = mkdtemp()
from joblib import Parallel, delayed, Memory
memory = Memory(cachedir=cachedir, verbose=0)
import multiprocessing
num_cores = multiprocessing.cpu_count()
# try:
#     from functools import lru_cache
# except ImportError:
#     from backports.functools_lru_cache import lru_cache
from ..Parameter_files import COSMO
cosmo = COSMO
NINT = 1000

def RG(RL): return 0.46*RL
def W(y): return 3.*(np.sin(y)-y*np.cos(y))/y**3.
def WG(y): return np.exp(-y**2/2)
# @lru_cache(maxsize=1000)
# def _Del2k(k):
#     Pk = pb.power_spectrum(k,0.0,**cosmo)
#     Del2 = k**3*Pk/2./np.pi**2.
#     return Del2
# _Del2k_vec = np.vectorize(_Del2k)
# def Del2k(k):
#     if np.isscalar(k):
#         return _Del2k(k)
#     else:
#         return _Del2k_vec(k)
#         #return np.array([_Del2k(K) for K in k])
# @memory.cache
# def Del2k(k):
#     Pk = pb.power_spectrum(k,0.0,**cosmo)
#     Del2 = k**3*Pk/2./np.pi**2.
#     return Del2

def _Del2k(k):
    print 'computing non-cached Del2k'
    Pk = pb.power_spectrum(k,0.0,**cosmo)
    Del2 = k**3*Pk/2./np.pi**2.
    np.savez('Del2k', k=k, Del2=Del2)
    return Del2
k = np.exp(np.arange(-21., 21., 0.1))
Del2 = _Del2k(k)
Del2k = interp1d(k, Del2)

def _klims(integrand, factor=1.e-6):
    """Integration limits used internally by the sigma_r functionp."""
    logk = np.arange(-20., 20., 0.1)
    maxintegrand = np.max(np.abs(integrand))
    highmask = np.abs(integrand) > maxintegrand * factor
    try:
        while highmask.ndim > logk.ndim:
            highmask = np.logical_or.reduce(highmask)
        mink = np.min(logk[highmask])
        maxk = np.max(logk[highmask])
    except:
        import IPython; IPython.embed()
    return mink, maxk
def _SX_integrand_log(logk, RL, R0):
    k = np.exp(logk)
    return Del2k(k)*W(RL*k)*W(R0*k)
def _sig1mX_integrand_log(logk, RL, R0):
    k = np.exp(logk)
    return Del2k(k)*(k**2)*WG(RG(RL)*k)*W(R0*k)
def _sigG_integrand_log(logk,RL,j):
    k = np.exp(logk)
    return Del2k(k)*(k**(2*j))*WG(RG(RL)*k)**2
def _sig0_integrand_log(logk,RL):
    k = np.exp(logk)
    return Del2k(k)*W(RL*k)**2
def _sig1m_integrand_log(logk,RL):
    k = np.exp(logk)
    #return Del2k(k)*WG(RG(RL)*k)*W(RL*k)
    return Del2k(k)*(k**2)*WG(RG(RL)*k)*W(RL*k)
def _SX_klims(RL, R0):
    logk = np.arange(-20., 20., 0.1)
    integrand = _SX_integrand_log(logk, RL, R0)
    return _klims(integrand)
def _sig1mX_klims(RL, R0):
    logk = np.arange(-20., 20., 0.1)
    integrand = _sig1mX_integrand_log(logk, RL, R0)
    return _klims(integrand)
def _sig1m_klims(RL):
    logk = np.arange(-20., 20., 0.1)
    integrand = _sig1m_integrand_log(logk, RL)
    return _klims(integrand)
def _sigG_klims(RL, j):
    logk = np.arange(-20., 20., 0.1)
    integrand = _sigG_integrand_log(logk, RL, j)
    return _klims(integrand)
def _sig0_klims(RL):
    logk = np.arange(-20., 20., 0.1)
    integrand = _sig0_integrand_log(logk, RL)
    return _klims(integrand)
def _SXlog_scalar(RL,R0, method='trapz'): 
    logk_lim = _SX_klims(RL, R0)
    #print "Integrating from logk = %.1f to %.1f." % logk_lim

    # Integrate over logk from -infinity to infinity.
    if method == 'trapz':
        logk = np.linspace(logk_lim[0], logk_lim[1], 100)
        integral = np.trapz(y=_SX_integrand_log(logk, RL, R0), x=logk)
        return integral, 0.
    else:
        integral, error = quad(_SX_integrand_log,
                                  logk_lim[0],
                                  logk_lim[1],
                                  args=(RL, R0),
                                  limit=10000)#, epsabs=1e-9, epsrel=1e-9)
        return  integral, error
_SXlog_vec = np.vectorize(_SXlog_scalar)

def _sig1mXlog_scalar(RL,R0, method='trapz'): 
    logk_lim = _sig1mX_klims(RL, R0)
    #print "Integrating from logk = %.1f to %.1f." % logk_lim

    if method == 'trapz':
        logk = np.linspace(logk_lim[0], logk_lim[1], NINT)
        integral = np.trapz(y=_sig1mX_integrand_log(logk, RL, R0), x=logk)
        return integral, 0.
    else:
        # Integrate over logk from -infinity to infinity.
        integral, error = quad(_sig1mX_integrand_log,
                                  logk_lim[0],
                                  logk_lim[1],
                                  args=(RL, R0),
                                  limit=10000)#, epsabs=1e-9, epsrel=1e-9)
        return integral, error
_sig1mXlog_vec = np.vectorize(_sig1mXlog_scalar)
#@memory.cache
#@lru_cache(maxsize=1000)
def _sig1mlog_scalar(RL, method='trapz'): 
    logk_lim = _sig1m_klims(RL)
    #print "Integrating from logk = %.1f to %.1f." % logk_lim
    if method == 'trapz':
        logk = np.linspace(logk_lim[0], logk_lim[1], NINT)
        integral = np.trapz(y=_sig1m_integrand_log(logk, RL), x=logk)
        return integral, 0.
    else:
        # Integrate over logk from -infinity to infinity.
        integral, error = quad(_sig1m_integrand_log,
                                  logk_lim[0],
                                  logk_lim[1],
                                  args=(RL),
                                  limit=10000)#, epsabs=1e-9, epsrel=1e-9)
        return integral, error
_sig1mlog_vec = np.vectorize(_sig1mlog_scalar)
#@memory.cache
#@lru_cache(maxsize=1000)
def _sigG_scalar(RL, j, method='trapz'): 
    logk_lim = _sigG_klims(RL, j)
    #print "Integrating from logk = %.1f to %.1f." % logk_lim
    if method == 'trapz':
        logk = np.linspace(logk_lim[0], logk_lim[1], NINT)
        integral = np.trapz(y=_sigG_integrand_log(logk, RL, j), x=logk)
        return integral, 0.
    else:
        # Integrate over logk from -infinity to infinity.
        integral, error = quad(_sigG_integrand_log,
                                  logk_lim[0],
                                  logk_lim[1],
                                  args=(RL, j),
                                  limit=10000)#, epsabs=1e-9, epsrel=1e-9)
        return integral, error
_sigG_vec = np.vectorize(_sigG_scalar)
#@memory.cache
#@lru_cache(maxsize=1000)
def _sig0_scalar(RL, method='trapz'): 
    logk_lim = _sig0_klims(RL)
    #print "Integrating from logk = %.1f to %.1f." % logk_lim
    if method == 'trapz':
        logk = np.linspace(logk_lim[0], logk_lim[1], NINT)
        integral = np.trapz(y=_sig0_integrand_log(logk, RL), x=logk)
        return integral, 0.
    else:
        # Integrate over logk from -infinity to infinity.
        integral, error = quad(_sig0_integrand_log,
                                  logk_lim[0],
                                  logk_lim[1],
                                  args=(RL),
                                  limit=10000)#, epsabs=1e-9, epsrel=1e-9)
        return integral, error
_sig0_vec = np.vectorize(_sig0_scalar)

def SX(RL,R0, method='trapz', ret_err=False): 
    SX, SXerr = _SXlog_scalar(RL, R0, method=method)
    SXerr = SXerr/2/SX
    if not ret_err:
        return SX
    return SX, SXerr
def sig1mX(RL,R0, method='trapz', ret_err=False):
    sig1mX, err = _sig1mXlog_scalar(RL, R0, method=method)
    err = err/2/sig1mX
    if not ret_err:
        return sig1mX
    return sig1mX, err
def sig1m(RL, method='trapz',ret_err=False):
    if np.isscalar(RL):
        sig1m, err = _sig1mlog_scalar(RL, method=method)
    else:
        sig1m, err = _sig1mlog_vec(RL, method=method)
    err = err/2/sig1m
    if not ret_err:
        return sig1m
    return sig1m, err
def sigG(RL, j,  method='trapz', ret_err=False):
    if np.isscalar(RL):
        sigG, err = _sigG_scalar(RL, j, method=method)
    else:
        sigG, err = _sigG_vec(RL, j, method=method)
    err = err/2/sigG
    if not ret_err:
        return sigG
    return sigG, err

def sig0(RL, method='trapz', ret_err=False):
    if np.isscalar(RL):
        sig0, err = _sig0_scalar(RL, method=method)
    else:
        sig0, err = _sig0_vec(RL, method=method)
    err = err/2/sig0
    if not ret_err:
        return sig0
    return sig0, err
def sig0_pb(RL,  ret_err=False):
    sig, err = pb.sigma_r(RL, 0, **cosmo)
    if not ret_err:
        return sig
    return sig, err

if __name__=='__main__':

    lrl = np.logspace(np.log10(0.04), np.log10(50), num=1000)
    lml = pb.radius_to_mass(lrl, **cosmo)
    lsig = Parallel(n_jobs=num_cores)(delayed(sig0)(rl,method='quad') for rl in lrl)
    #print lsig
    memory.clear()
    #lsig = sig0(lrl)
    #import IPython; IPython.embed()
    np.savez('sig0',radius=lrl,mass=lml,sig0=lsig)

# lR0min, lR0max = np.log(0.2),np.log(40.)
# lrlmin = np.log(0.04)
# lR0 = np.linspace(lR0min, lR0max,50)
# lRl = np.linspace(lrlmin,lR0max,50)
# xx,yy = np.meshgrid(lRl,lR0)
# arr = np.ones_like(xx)
# for i,lrl in enumerate(lRl):
#     for j,lr0 in enumerate(lR0):
#         rl,r0 = np.exp(lrl),np.exp(lr0)
#         arr[i,j] = sig1mX(rl,r0)
#         print rl,r0,arr[i,j]

# # plt.figure()
# # plt.imshow(arr)
# # plt.colorbar()
# # plt.show()
# np.savez('logsig1mX',xx,yy,arr)
# fsig1mX = interp2d(xx,yy,arr,kind='cubic')
# print sig1mX(0.1,0.2), fsig1mX(np.log(0.1),np.log(0.2))
# print sig1mX(0.7,10.), fsig1mX(np.log(0.7),np.log(10.))