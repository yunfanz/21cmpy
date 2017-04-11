import cosmolopy.perturbation as pb
import cosmolopy.density as cd
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cu
import pycuda.autoinit
from pycuda.curandom import *
import pycuda.cumath as cumath
from pyfft.cuda import Plan
from pycuda.tools import make_default_context

from ..cosmo_files import *
from ..Parameter_files import *

"""
Generates the initial conditions:
  gaussian random density field (DIM^3)
  as well as the equal or lower resolution
  velocity fields, and smoothed density field (HII_DIM^3).
  See Parameter_files

  Author: Yunfan G. Zhang
  04/2017
  """

def step1():
	"""outputs the high resolution k-box, and the smoothed r box"""
	N = np.int32(DIM)
	#HII_DIM = np.int32(HII_DIM)
	f_pixel_factor = N/HII_DIM;
	scale = np.float32(BOX_LEN)/DIM
	HII_scale = np.float32(BOX_LEN)/HII_DIM
	shape = (N,N,N)
	#ratio of large box to small size
	
	MRGgen = MRG32k3aRandomNumberGenerator(seed_getter=seed_getter_uniform, offset=0)

	kernel_source = open("./initialize.cu").read()
	kernel_code = kernel_source % {
        
        'DELTAK': DELTA_K
    }
	main_module = nvcc.SourceModule(kernel_code)
	initpk_kernel = main_module.get_function("init_pk")
	HII_filter = main_module.get_function("HII_filter")
	subsample_kernel = main_module.get_function("subsample")

	block_size =  (8,8,8)
	grid_size =   (N/(block_size[0]),
				N/(block_size[0]),
				N/(block_size[0]))
	small_grid_size =   (HII_DIM/(block_size[0]),
				HII_DIM/(block_size[0]),
				HII_DIM/(block_size[0]))


	largebox_d = gpuarray.zeros((N,N,N), dtype=np.float32)
	initpk_kernel(largebox_d, np.int32(N), block=block_size, grid=grid_size)
	powk = largebox_d.get()
	powk = pb.power_spectrum(powk.astype(np.float64), 0.0, **COSMO)  #host code, slow, TODO: at least replace with interpolation
	largebox_d = gpuarray.to_gpu(powk.astype(np.float32))
	largebox_d = cumath.sqrt(N*N*N*largebox_d)
	largebox_d_imag = largebox_d.copy()
	largebox_d *= MRGgen.gen_normal((N,N,N), dtype=np.float32)
	largebox_d_imag *= MRGgen.gen_normal((N,N,N), dtype=np.float32)
	largebox_d = largebox_d + 1j * largebox_d_imag
	np.save("../Boxes/deltak_z0.00_{%i}_{%.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get())


	HII_filter(largebox_d, N, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0), block=block_size, grid=grid_size);
	plan = Plan((N,N,N), dtype=np.complex64)
	plan.execute(largebox_d, inverse=True)  #FFT to real space of smoothed box

	smallbox_d = gpuarray.zeros(HII_shape, dtype=np.complex64)
	subsample_kernel(largebox_d, smallbox_d, N, HII_DIM, block=block_size, grid=small_grid_size) #subsample in real space
	np.save("../Boxes/smoothed_deltax_z0.00_{%i}_{%.0f}Mpc".format(HII_DIM, BOX_LEN), smallbox_d.get())

	return

def step2():

	largebox = np.load("../Boxes/deltak_z0.00_{%i}_{%.0f}Mpc".format(DIM, BOX_LEN))
	largebox_d = gpuarray.to_gpu(largebox)

	kernel_source = open("./initialize.cu").read()
	kernel_code = kernel_source % {
        
        'DELTAK': DELTA_K
    }
	main_module = nvcc.SourceModule(kernel_code)
	HII_filter = main_module.get_function("HII_filter")
	subsample_kernel = main_module.get_function("subsample")
	velocity_kernel = main_module.get_function("set_velocity")
	block_size =  (8,8,8)
	grid_size =   (N/(block_size[0]),
				N/(block_size[0]),
				N/(block_size[0]))
	small_grid_size =   (HII_DIM/(block_size[0]),
				HII_DIM/(block_size[0]),
				HII_DIM/(block_size[0]))


	plan = Plan((DIM,DIM,DIM), dtype=np.complex64)
	# This saves a large resolution deltax
	# plan.execute(largebox_d, inverse=True)  #FFT to real space of unsmoothed box
	# np.save("../Boxes/deltax_z0.00_{%i}_{%.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get())

	largevbox_d = np.zeros((DIM,DIM,DIM), dtype=np.complex64)
	smallvbox_d = gpuarray.zeros(HII_shape, dtype=np.complex64)
	for num, mode in enumerate(['x', 'y', 'z']):
		velocity_kernel(largebox_d, largevbox_d, DIM, num)
		HII_filter(largevbox_d, N, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0), block=block_size, grid=grid_size)
		plan.execute(largevbox_d, inverse=True)
		subsample_kernel(largevbox_d, smallvbox_d, DIM, HII_DIM, block=block_size, grid=small_grid_size)
		np.save("../Boxes/v{}overddot_{%i}_{%.0f}Mpc".format(mode, HII_DIM, BOX_LEN))
	return






def run():
	if not os.path.exists('../Boxes'):
		os.makedirs("../Boxes")
	step1()
	step2()

if __name__=="__main__":
	run()
	import IPython; IPython.embed()
