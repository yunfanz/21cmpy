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
#from ..cosmo_files import *
from cosmo_functions import *
from ..Parameter_files import *

def evolve_linear(z, deltax):
	"""
	Input type IN must be numpy or 21cmfast
	"""
	
	fgrowth = pb.fgrowth(z, cosmo['omega_M_0']) #normalized to 1 at z=0
	#primordial_fgrowth = pb.fgrowth(INITIAL_REDSHIFT, cosmo['omega_M_0']) #normalized to 1 at z=0
	

	updated = deltax*fgrowth
		
	np.save(parent_folder+"/Boxes/updated_smoothed_deltax_z{0:.00f}_{1:d}_{2:.0f}Mpc".format(z, HII_DIM, BOX_LEN), updated)


	if False: #velocity information may not be useful for linear field
		plan = Plan(HII_shape, dtype=np.complex64)
		deltak_d = deltax_d.astype(np.complex64)
		vbox_d = gpuarray.zeros_like(deltak_d)
		plan.execute(deltak_d)
		dDdt_D = np.float32(dDdt_D(z))
		for num, mode in enumerate(['x', 'y', 'z']):
			velocity_kernel(deltak_d, vbox_d, dDdt_D, DIM, np.int32(num), block=block_size, grid=grid_size)
			np.save(parent_folder+"/Boxes/updated_v{0}overddot_{1:d}_{2:.0f}Mpc".format(mode, HII_DIM, BOX_LEN), smallvbox_d.get())

	return

def evolve_zeldovich(z, deltax):
	"""First order Zeldovich approximation. """
	if BOX_LEN > DIM:
		print "perturb_field: WARNING: Resolution is likely too low for accurate evolved density fields"
	#move_mass(updated_d, deltax_d, vx_d, vy_d, vz_d, np.float32(1./primordial_fgrowth))
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
	filter_kernel = main_module.get_function("filter")
	subsample_kernel = main_module.get_function("subsample")

	fgrowth = np.float32(pb.fgrowth(z, COSMO['omega_M_0'])) #normalized to 1 at z=0
	primordial_fgrowth = np.float32(pb.fgrowth(INITIAL_REDSHIFT, COSMO['omega_M_0'])) #normalized to 1 at z=0

	vx = np.load(parent_folder+"/Boxes/vxoverddot_{0:d}_{1:.0f}Mpc.npy".format(HII_DIM, BOX_LEN))
	vy = np.load(parent_folder+"/Boxes/vyoverddot_{0:d}_{1:.0f}Mpc.npy".format(HII_DIM, BOX_LEN))
	vz = np.load(parent_folder+"/Boxes/vzoverddot_{0:d}_{1:.0f}Mpc.npy".format(HII_DIM, BOX_LEN))
	vx_d = gpuarray.to_gpu(vx)
	vy_d = gpuarray.to_gpu(vy)
	vz_d = gpuarray.to_gpu(vz)
	vx_d *= ((fgrowth-primordial_fgrowth) / BOX_LEN)
	vy_d *= ((fgrowth-primordial_fgrowth) / BOX_LEN)
	vz_d *= ((fgrowth-primordial_fgrowth) / BOX_LEN)

	updated_d = gpuarray.zeros_like(vx_d)
	delta_d = gpuarray.to_gpu(deltax)

	move_mass(updated_d, delta_d, vx_d, vy_d, vz_d, primordial_fgrowth, block=block_size, grid=grid_size)
	updated_d /= MASS_FACTOR
	updated_d -= np.float32(1.) #renormalize to the new pixel size, and make into delta
	updated = updated_d.get_async()
	np.save(parent_folder+"/Boxes/updated_smoothed_deltax_z{0:.2f}_{1:d}_{2:.0f}Mpc".format(z, HII_DIM, BOX_LEN), updated)


	plan = Plan(HII_shape, dtype=np.complex64)
	delta_d = delta_d.astype(np.complex64)
	vbox_d = gpuarray.zeros_like(delta_d)
	smallvbox_d = gpuarray.zeros(HII_shape, dtype=np.float32)
	plan.execute(delta_d) #now deltak
	dDdt_D = np.float32(dDdtoverD(z))
	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	for num, mode in enumerate(['x', 'y', 'z']):
		velocity_kernel(delta_d, vbox_d, dDdt_D, DIM, np.int32(num), block=block_size, grid=grid_size)
		filter_kernel(vbox_d, DIM, ZERO, smoothR, block=block_size, grid=grid_size)
		plan.execute(vbox_d, inverse=True)
		subsample_kernel(vbox_d.real, smallvbox_d, DIM, HII_DIM,PIXEL_FACTOR, block=block_size, grid=HII_grid_size)
		np.save(parent_folder+"/Boxes/updated_v{0}overddot_{1:d}_{2:.0f}Mpc".format(mode, HII_DIM, BOX_LEN), smallvbox_d.get())

	return

def run(z):
	
	
	if EVOLVE_DENSITY_LINEARLY:
		try:
			deltax =  np.load(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc.npy".format(HII_DIM, BOX_LEN))
		except:
			deltax =  boxio.readbox(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc".format(HII_DIM, BOX_LEN)).box_data
		evolve_linear(z,deltax)
	else:
		try:
			deltax =  np.load(parent_folder+"/Boxes/deltax_z0.00_{0:d}_{1:.0f}Mpc.npy".format(DIM, BOX_LEN))
		except:
			deltax =  boxio.readbox(parent_folder+"/Boxes/deltax_z0.00_{0:d}_{1:.0f}Mpc".format(DIM, BOX_LEN)).box_data
		evolve_zeldovich(z,deltax)

if __name__=='__main__':
	z = 12.
	run(z)