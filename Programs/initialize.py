import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.curandom import *
import pycuda.cumath as cumath
from pyfft.cuda import Plan
from pycuda.tools import make_default_context
from IO_utils import *
#print cmd_folder
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
def init_pspec():
	K = np.logspace(np.log10(DELTA_K/10), np.log10(DELTA_K*np.sqrt(3.)*DIM), BLOCK_VOLUME)
	print K[0], K[-1]
	pspec = pb.power_spectrum(K, 0.0, **COSMO)
	return np.vstack((K, pspec)).astype(np.float32), K.size

def step1():
	"""outputs the high resolution k-box, and the smoothed r box"""
	N = np.int32(DIM) #prepare for stitching
	#HII_DIM = np.int32(HII_DIM)
	f_pixel_factor = DIM/HII_DIM;
	scale = np.float32(BOX_LEN)/DIM
	HII_scale = np.float32(BOX_LEN)/HII_DIM
	shape = (N,N,N)
	#ratio of large box to small size
	
	MRGgen = MRG32k3aRandomNumberGenerator(seed_getter=seed_getter_uniform, offset=0)

	kernel_source = open(cmd_folder+"/initialize.cu").read()
	kernel_code = kernel_source % {

		'DELTAK': DELTA_K,
		'PIX_VOLUME': PIX_VOLUME
	}
	main_module = nvcc.SourceModule(kernel_code)
	init_kernel = main_module.get_function("init_kernel")
	HII_filter = main_module.get_function("HII_filter")
	subsample_kernel = main_module.get_function("subsample")
	velocity_kernel = main_module.get_function("set_velocity")
	pspec_texture = main_module.get_texref("pspec")

	interpPspec, interpSize = init_pspec() #interpPspec contains both k array and P array
	interp_cu = cuda.matrix_to_array(interpPspec, order='C')
	cuda.bind_array_to_texref(interp_cu, pspec_texture)

	largebox_d = gpuarray.zeros(shape, dtype=np.float32)
	init_kernel(largebox_d, np.int32(DIM), block=block_size, grid=grid_size)
	# initpk_kernel(largebox_d, np.int32(DIM), block=block_size, grid=grid_size)
	# powk = largebox_d.get()
	# powk = pb.power_spectrum(powk.astype(np.float64), 0.0, **COSMO)  #host code, slow, TODO: at least replace with interpolation
	# largebox_d = gpuarray.to_gpu(powk.astype(np.float32))
	# largebox_d = cumath.sqrt(PIX_VOLUME*largebox_d)

	largebox_d_imag = largebox_d.copy()
	largebox_d *= MRGgen.gen_normal(shape, dtype=np.float32)
	largebox_d_imag *= MRGgen.gen_normal(shape, dtype=np.float32)
	largebox_d = largebox_d + np.complex64(1.j) * largebox_d_imag
	np.save(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get())

	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	HII_filter(largebox_d, N, ZERO, smoothR, block=block_size, grid=grid_size);
	plan = Plan(shape, dtype=np.complex64)
	plan.execute(largebox_d, inverse=True)  #FFT to real space of smoothed box

	smallbox_d = gpuarray.zeros(HII_shape, dtype=np.float32)
	subsample_kernel(largebox_d.real, smallbox_d, N, HII_DIM, PIXEL_FACTOR, block=block_size, grid=small_grid_size) #subsample in real space
	np.save(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc".format(HII_DIM, BOX_LEN), smallbox_d.get())

	# This saves a large resolution deltax
	# plan.execute(largebox_d, inverse=True)  #FFT to real space of unsmoothed box
	# np.save(parent_folder+"/Boxes/deltax_z0.00_{%i}_{%.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get())
	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	largevbox_d = gpuarray.zeros((DIM,DIM,DIM), dtype=np.complex64)
	smallvbox_d = gpuarray.zeros(HII_shape, dtype=np.float32)
	for num, mode in enumerate(['x', 'y', 'z']):
		velocity_kernel(largebox_d, largevbox_d, DIM, np.int32(num), block=block_size, grid=grid_size)
		HII_filter(largevbox_d, DIM, ZERO, smoothR, block=block_size, grid=grid_size)
		plan.execute(largevbox_d, inverse=True)
		subsample_kernel(largevbox_d.real, smallvbox_d, DIM, HII_DIM,PIXEL_FACTOR, block=block_size, grid=small_grid_size)
		np.save(parent_folder+"/Boxes/v{0}overddot_{1:d}_{2:.0f}Mpc".format(mode, HII_DIM, BOX_LEN), smallvbox_d.get())

	return

# def step2():

# 	largebox = np.load(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc.npy".format(DIM, BOX_LEN))
# 	largebox_d = gpuarray.to_gpu(largebox)

# 	kernel_source = open(cmd_folder+"/initialize.cu").read()
# 	kernel_code = kernel_source % {
        
#         'DELTAK': DELTA_K,
#         'PIX_VOLUME': PIX_VOLUME
#     }
# 	main_module = nvcc.SourceModule(kernel_code)
# 	HII_filter = main_module.get_function("HII_filter")
# 	subsample_kernel = main_module.get_function("subsample")
# 	velocity_kernel = main_module.get_function("set_velocity")


# 	plan = Plan((DIM,DIM,DIM), dtype=np.complex64)
# 	# This saves a large resolution deltax
# 	# plan.execute(largebox_d, inverse=True)  #FFT to real space of unsmoothed box
# 	# np.save(parent_folder+"/Boxes/deltax_z0.00_{%i}_{%.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get())
# 	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
# 	largevbox_d = gpuarray.zeros((DIM,DIM,DIM), dtype=np.complex64)
# 	smallvbox_d = gpuarray.zeros(HII_shape, dtype=np.float32)
# 	for num, mode in enumerate(['x', 'y', 'z']):
# 		velocity_kernel(largebox_d, largevbox_d, DIM, np.int32(num), block=block_size, grid=grid_size)
# 		HII_filter(largevbox_d, DIM, ZERO, smoothR, block=block_size, grid=grid_size)
# 		plan.execute(largevbox_d, inverse=True)
# 		subsample_kernel(largevbox_d.real, smallvbox_d, DIM, HII_DIM,PIXEL_FACTOR, block=block_size, grid=small_grid_size)
# 		np.save(parent_folder+"/Boxes/v{0}overddot_{1:d}_{2:.0f}Mpc".format(mode, HII_DIM, BOX_LEN), smallvbox_d.get())
# 	return






def run():
	(free,total) = cuda.mem_get_info()
	print "Device global memory {0:.2f}GB total, {1:0.2f}GB free".format(total/1.e9, free/1.e9)
	print "Roughly {0:.2f}GB required for large box".format(float(NBYTES*2)/1.e9)
	if not os.path.exists(parent_folder+'/Boxes'):
		os.makedirs(parent_folder+"/Boxes")
	if NBYTES*2 < free:
		print "Congratulations, your GPU has enough memory, running without stitching"
		step1()
	else:
		print "Stitching with {} meta_grid size".format(DIM/HII_DIM)
	#step2()

if __name__=="__main__":
	run()
	import IPython; IPython.embed()
