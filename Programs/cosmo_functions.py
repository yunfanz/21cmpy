import cosmolopy.perturbation as pb
import cosmolopy
import numpy as np
from ..Parameter_files import *

# redshift derivative of the growth function at z 
def dDdz(z):
	dz = 1e-10
	return ( pb.fgrowth(z+dz, COSMO['omega_M_0'])-pb.fgrowth(z, COSMO['omega_M_0']) )/dz

def dtdz(z):
	dz = 1e-10
	return (cosmolopy.cd.age(z+dz, **COSMO)-cosmolopy.cd.age(z, **COSMO))/dz

def dDdt(z):
	dz = 1e-10
	tiny = 1e-4
	return dDdz(z)/dtdz(z)

def dDdtoverD(z):
	return dDdt(z)/pb.fgrowth(z, COSMO['omega_M_0'])
