from sigmas import *
import sigmas
import scipy
COMMONDIR = os.path.dirname(sigmas.__file__)
print COMMONDIR
rhobar = cd.cosmo_densities(**cosmo)[1]  #msun/Mpc

def m2R(m):
	RL = (3*m/4/np.pi/rhobar)**(1./3)
	return RL
def m2V(m):
	return m/rhobar
def R2m(RL):
	m = 4*np.pi/3*rhobar*RL**3
	return m

dmS = np.load(COMMONDIR+'/sig0.npz')
RLtemp, MLtemp,SLtemp = dmS['radius'], dmS['mass'],dmS['sig0']
fs2m = interp1d(SLtemp,MLtemp)
# fsig0 = interp1d(RLtemp,SLtemp)
# def sig0(RL):
# 	return fsig0(RL)
print 'generated fs2m'
def S2M(S):
	return fs2m(S)
def m2S(m):
	return sig0(m2R(m))
def mmin(z,Tvir=1.E4,cosmo=cosmo):
	return pb.virial_mass(Tvir,z,**cosmo)
def Deltac(z):
	fgrowth = pb.fgrowth(z, cosmo['omega_M_0'])    # = D(z)/D(0)
	return 1.686/fgrowth
	#return 1.686*fgrowth  
	
def fcoll_FZH(del0,M0,z,debug=False):
	# Eq. (6)
	print del0
	mm = mmin(z)
	R0 = m2R(M0)
	smin, S0 = sig0(m2R(mm)), sig0(R0)
	return erfc((Deltac(z)-del0)/np.sqrt(2*(smin-S0)))
def fFZH(S,zeta,B0,B1):
	res = B0/np.sqrt(2*np.pi*S**3)*np.exp(-B0**2/2/S-B0*B1-B1**2*S/2)
	return res
def BFZH(S0,deltac,smin,K):
	return deltac-np.sqrt(2*(smin-S0))*K
def BFZHlin(S0,deltac,smin,K):
	b0 = deltac-K*np.sqrt(2*smin)
	b1 = K/np.sqrt(2*smin)
	return b0+b1*S0
def dlnBFdlnS0(S0,deltac,smin,K,d=0.001):
	Bp,Bo,Bm = BFZH(S0+d,deltac,smin,K), BFZH(S0,deltac,smin,K), BFZH(S0-d,deltac,smin,K)
	return S0/Bo*(Bp-Bm)/2/d
def dlnBFlindlnS0(S0,deltac,smin,K,d=0.001):
	Bp,Bo,Bm = BFZHlin(S0+d,deltac,smin,K), BFZHlin(S0,deltac,smin,K), BFZHlin(S0-d,deltac,smin,K)
	return S0/Bo*(Bp-Bm)/2/d

dDoZ = np.load(COMMONDIR+'/theta.npz')
thetal,DoZl = dDoZ['arr_0'],dDoZ['arr_1']
ftheta = interp1d(DoZl,thetal,kind='cubic')
def theta(z,del0):
	return ftheta(del0/(1+z))
def RphysoR0(del0,z):
	th = theta(z,del0)
	return 3./10/del0*(1-np.cos(th))
def RcovEul(del0,z):
	return RphysoR0(del0,z)*(1+z)
def dlinSdlnR(lnR,d=0.001):
	res = (np.log(sig0(np.exp(lnR+d)))-np.log(sig0(np.exp(lnR-d))))/d/2
	return np.abs(res)

Rfile = np.load(COMMONDIR+'/radius_z12.npz')
R0l,REl = Rfile['arr_0'],Rfile['arr_1']
fR = interp1d(REl,R0l,kind='cubic')
def R0Lag(RE, z=12.): 
	return fR(RE)

class ESets:
	def __init__(self, cosmo=cosmo, z=12., zeta=40., Tvir=1.E4):
		self.cosmo = cosmo
		self.zeta = zeta
		self.Tvir = Tvir
		self.z = float(z)
		self.K = scipy.special.erfinv(1-1./self.zeta)
		self.update_z(self.z)
	def update_z(self, z):
		self.z = z
		self.deltac = Deltac(self.z)
		self.mm = mmin(self.z, self.Tvir, self.cosmo) #minimum mass of ionizing source
		self.M0min = self.mm*self.zeta  # corresponding minimum mass of ionized region
		self.RLmin = m2R(self.mm) 
		self.R0min = m2R(self.M0min)
		self.smin = sig0(self.RLmin)
		self.S0min = sig0(self.R0min)
		self.fgrowth = pb.fgrowth(z, self.cosmo['omega_M_0'])
	def BFZH(self, R0):
		S0 = sig0(R0)
		return BFZH(S0,self.deltac,self.smin,self.K)
	def RcovEul(self, R0):
		return R0*RcovEul(del0=self.BFZH(R0), z=self.z)
	def R0(self, RE):
		return R0Lag(RE, self.z)