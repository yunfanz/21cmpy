import numpy as np

DIM = np.int32(64) #dimension of large box in pixels
HII_DIM = np.int32(16)
PIXEL_FACTOR = np.float32(DIM)/np.float32(HII_DIM)
MASS_FACTOR = np.float32(PIXEL_FACTOR**3.)
BOX_LEN = np.float32(128)
DELTA_K = np.float32(2*np.pi/BOX_LEN)
VOLUME = (BOX_LEN*BOX_LEN*BOX_LEN)
PIX_VOLUME = np.float32(DIM*DIM*DIM)
HII_PIX_VOLUME = HII_DIM*HII_DIM*HII_DIM
HII_shape = (HII_DIM, HII_DIM, HII_DIM)

INITIAL_REDSHIFT = 300.
EVOLVE_DENSITY_LINEARLY = True
ION_Tvir_MIN = 1.e4
ZETA = 40.
R_BUBBLE_MAX = 30. #Mpc
R_BUBBLE_MIN = 1.
FIND_BUBBLE_ALGORITHM = 1

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
grid_size =   (DIM/(block_size[0]),
				DIM/(block_size[0]),
				DIM/(block_size[0]))
small_grid_size =   (HII_DIM/(block_size[0]),
			HII_DIM/(block_size[0]),
			HII_DIM/(block_size[0]))