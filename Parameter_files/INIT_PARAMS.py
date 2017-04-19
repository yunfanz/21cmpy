import numpy as np

IO_DIR = "/home/yunfanz/Projects/21cm/Py21cm"

DIM = np.int32(512) #dimension of large box in pixels
HII_DIM = np.int32(128)
PIXEL_FACTOR = np.float32(DIM)/np.float32(HII_DIM)
MASS_FACTOR = np.float32(PIXEL_FACTOR**3.)
BOX_LEN = np.float32(128)
DELTA_K = np.float32(2*np.pi/BOX_LEN)
VOLUME = (BOX_LEN*BOX_LEN*BOX_LEN)
PIX_VOLUME = np.float32(DIM*DIM*DIM)
NBYTES = PIX_VOLUME*PIX_VOLUME.dtype.itemsize
HII_TOT_NUM_PIXELS = HII_DIM*HII_DIM*HII_DIM
HII_shape = (HII_DIM, HII_DIM, HII_DIM)

INITIAL_REDSHIFT = 300.
EVOLVE_DENSITY_LINEARLY = False
ION_Tvir_MIN = 1.e4
ZETA = 40.
R_BUBBLE_MAX = 30. #Mpc
R_BUBBLE_MIN = 0.5
FIND_BUBBLE_ALGORITHM = 0
DELTA_R_FACTOR = 1.1

# Filter for the Halo or density field used to generate ionization field
# 0 = use real space top hat filter
# 1 = use k-space top hat filter
# 2 = use gaussian filter
HII_FILTER = 1
L_FACTOR = np.float32(0.620350491)

PLANCK13 = {'baryonic_effects':False,
			'omega_k_0':0,
			'omega_M_0':0.315, 
			'omega_b_0':0.0487, 
			'n':0.96, 
			'N_nu':0, 
			'omega_lambda_0':0.685,
			'omega_n_0':0., 
			'sigma_8':0.829,
			'h':0.673}

COSMO = PLANCK13

ZERO = np.int32(0)
ONE = np.int32(1)
TWO = np.int32(2)

#GPU block and grid sizes
block_size =  (8,8,8)
BLOCK_VOLUME = 8*8*8
grid_size =   (DIM/(block_size[0]),
				DIM/(block_size[0]),
				DIM/(block_size[0]))
HII_grid_size =   (HII_DIM/(block_size[0]),
			HII_DIM/(block_size[0]),
			HII_DIM/(block_size[0]))



  # Efficiency parameter corresponding to the number of x-ray photons per
  # solar mass in stars.
ZETA_X  = 2.0e56 #2e56 ~ 0.3 X-ray photons per stellar baryon
DIMENSIONAL_T_POWER_SPEC = False
T_USE_VELOCITIES = False
NUM_BINS = 100
USE_TS_IN_21CM = 0