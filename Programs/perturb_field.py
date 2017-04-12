"""
PROGRAM PERTURB_FIELD uses the first-order Langragian displacement field
to move the masses in the cells of the density field.
The high-res density field is extrapolated to some high-redshift
(INITIAL_REDSHIFT in ANAL_PARAMS.H), then uses the zeldovich approximation
to move the grid "particles" onto the lower-res grid we use for the
maps.  Then we recalculate the velocity fields on the perturbed grid.


Output files:

"../Boxes/updated_smoothed_deltax_z%06.2f_%i_%.0fMpc", REDSHIFT, HII_DIM, BOX_LEN
-- This file contains the perturbed overdensity field, \delta, at <REDSHIFT>. The binary box has FFT padding

"../Boxes/updated_vx_z%06.2f_%i_%.0fMpc", REDSHIFT, HII_DIM, BOX_LEN
"../Boxes/updated_vy_z%06.2f_%i_%.0fMpc", REDSHIFT, HII_DIM, BOX_LEN
"../Boxes/updated_vz_z%06.2f_%i_%.0fMpc", REDSHIFT, HII_DIM, BOX_LEN
-- These files contain the velocity fields recalculated using the perturbed velocity fields at <REDSHIFT>.  The units are cMpc/s.  The boxes have FFT padding.

Author: Yunfan G. Zhang
        07/2017

"""
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cu
import pycuda.autoinit
from pycuda.curandom import *
import pycuda.cumath as cumath
from pyfft.cuda import Plan
from pycuda.tools import make_default_context
from IO_utils import *
#print cmd_folder
from ..cosmo_files import *
from ..Parameter_files import *

def evolve_linear(z, IN='numpy'):
	"""
	Input type IN must be numpy or 21cmfast
	"""
	kernel_source = open(cmd_folder+"/perturb_field.cu").read()
	kernel_code = kernel_source % {

		'DELTAK': DELTA_K,
		'HII_DIM': HII_DIM,
		'DIM': DIM, 
		'PIXEL_FACTOR': PIXEL_FACTOR
	}
	main_module = nvcc.SourceModule(kernel_code)
	move_mass = main_module.get_function("move_mass")
	velocity_kernel = main_module.get_function("set_velocity")
	fgrowth = np.float32(1./pb.fgrowth(z, cosmo['omega_M_0'])) #normalized to 1 at z=0
	primordial_fgrowth = np.float32(1./pb.fgrowth(INITIAL_REDSHIFT, cosmo['omega_M_0'])) #normalized to 1 at z=0
	if IN == 'numpy':
		deltax =  np.load(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc.npy".format(HII_DIM, BOX_LEN))
	elif IN == '21cmfast':
		deltax =  boxio.readbox(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc".format(HII_DIM, BOX_LEN)).box_data
	deltax_d = gpuarray.to_gpu(deltax)
	updated_d = deltax_d.copy()/fgrowth
		
	np.save(parent_folder+"/Boxes/updated_smoothed_deltax_z{0:.00f}_{1:d}_{2:.0f}Mpc".format(z, HII_DIM, BOX_LEN), updated_d.get())


	if False:
		plan = Plan(HII_shape, dtype=np.complex64)
		deltak_d = deltax_d.astype(np.complex64)
		vbox_d = gpuarray.zeros_like(deltak_d)
		plan.execute(deltak_d)
		dDdt_D = np.float32(1)
		for num, mode in enumerate(['x', 'y', 'z']):
			velocity_kernel(deltak_d, vbox_d, dDdt_D, DIM, np.int32(num), block=block_size, grid=grid_size)
			np.save(parent_folder+"/Boxes/updated_v{0}overddot_{1:d}_{2:.0f}Mpc".format(mode, HII_DIM, BOX_LEN), smallvbox_d.get())

	return

def evolve_zeldovich(z, IN='numpy'):
	if BOX_LEN > DIM:
		print "perturb_field: WARNING: Resolution is likely too low for accurate evolved density fields"
	#move_mass(updated_d, deltax_d, vx_d, vy_d, vz_d, np.float32(1./primordial_fgrowth))
	raise Exception("Not yet implemented")

	return

def run(z, IN='numpy'):
	if EVOLVE_DENSITY_LINEARLY:
		evolve_linear(z,IN)
	else:
		evolve_zeldovich(z,IN)

if __name__=='__main__':
	z = 12.
	run(z)