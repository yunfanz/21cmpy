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
import pyfftw
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

def init():
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
		'VOLUME': VOLUME
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

	largebox_d_imag = largebox_d.copy()
	largebox_d *= MRGgen.gen_normal(shape, dtype=np.float32)
	largebox_d_imag *= MRGgen.gen_normal(shape, dtype=np.float32)
	largebox_d = largebox_d + np.complex64(1.j) * largebox_d_imag
	np.save(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get())

	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	HII_filter(largebox_d, N, ZERO, smoothR, block=block_size, grid=grid_size);
	plan = Plan(shape, dtype=np.complex64)
	plan.execute(largebox_d, inverse=True)  #FFT to real space of smoothed box
	#largebox_d /=  VOLUME  #divide by VOLUME if using fft (vs ifft)
	largebox_d *= scale**3

	# This saves a large resolution deltax
	# np.save(parent_folder+"/Boxes/deltax_z0.00_{%i}_{%.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get())
	smallbox_d = gpuarray.zeros(HII_shape, dtype=np.float32)
	subsample_kernel(largebox_d.real, smallbox_d, N, HII_DIM, PIXEL_FACTOR, block=block_size, grid=small_grid_size) #subsample in real space
	np.save(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc".format(HII_DIM, BOX_LEN), smallbox_d.get())

	# reload the k-space box
	largebox = np.load(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc.npy".format(DIM, BOX_LEN))
	largebox_d = gpuarray.to_gpu(largebox)
	largebox_d *= scale**3
	#largebox_d /=  VOLUME  #divide by VOLUME if using fft (vs ifft)
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

def init_stitch():
	"""outputs the high resolution k-box, and the smoothed r box"""
	N = np.int32(HII_DIM) #prepare for stitching
	META_GRID_SIZE = DIM/N
	M = np.int32(HII_DIM/META_GRID_SIZE)
	#HII_DIM = np.int32(HII_DIM)
	f_pixel_factor = DIM/HII_DIM;
	scale = np.float32(BOX_LEN/DIM)
	HII_scale = np.float32(BOX_LEN)/HII_DIM
	shape = (N,N,N)
	#ratio of large box to small size
	kernel_source = open(cmd_folder+"/initialize_stitch.cu").read()
	kernel_code = kernel_source % {

		'DELTAK': DELTA_K,
		'DIM': DIM, 
		'VOLUME': VOLUME,
		'META_BLOCKDIM': N
	}
	main_module = nvcc.SourceModule(kernel_code)
	init_stitch = main_module.get_function("init_kernel")
	HII_filter = main_module.get_function("HII_filter")
	subsample_kernel = main_module.get_function("subsample")
	velocity_kernel = main_module.get_function("set_velocity")
	pspec_texture = main_module.get_texref("pspec")

	interpPspec, interpSize = init_pspec() #interpPspec contains both k array and P array
	interp_cu = cuda.matrix_to_array(interpPspec, order='C')
	cuda.bind_array_to_texref(interp_cu, pspec_texture)
	hbox_large = pyfftw.empty_aligned((DIM, DIM, DIM), dtype='complex64')
	#hbox_large = np.zeros((DIM, DIM, DIM), dtype=np.complex64)
	hbox_small = np.zeros(HII_shape, dtype=np.float32)
	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	for meta_x in xrange(META_GRID_SIZE):
		for meta_y in xrange(META_GRID_SIZE):
			for meta_z in xrange(META_GRID_SIZE):
				MRGgen = MRG32k3aRandomNumberGenerator(seed_getter=seed_getter_uniform, offset=meta_x*N**3)
				largebox_d = gpuarray.zeros(shape, dtype=np.float32)
				init_stitch(largebox_d, N, np.int32(meta_x), np.int32(meta_y), np.int32(meta_z),block=block_size, grid=grid_size)
				largebox_d_imag = largebox_d.copy()
				largebox_d *= MRGgen.gen_normal(shape, dtype=np.float32)
				largebox_d_imag *= MRGgen.gen_normal(shape, dtype=np.float32)
				largebox_d = largebox_d + np.complex64(1.j) * largebox_d_imag
	#if want to get velocity need to use this
	#np.save(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc".format(DIM, BOX_LEN), hbox_large)
				HII_filter(largebox_d, N, np.int32(meta_x), np.int32(meta_y), np.int32(meta_z), ZERO, smoothR, block=block_size, grid=grid_size);
				hbox_large[meta_x*N:(meta_x+1)*N, meta_y*N:(meta_y+1)*N, meta_z*N:(meta_z+1)*N] = largebox_d.get()
	print "Executing FFT on host"
	#hbox_large = hifft(hbox_large).astype(np.complex64).real
	hbox_large = pyfftw.interfaces.numpy_fft.fftn(hbox_large).real
	hbox_large *= scale**3
	print "Finished FFT on host"
	

	# for meta_x in xrange(META_GRID_SIZE):
	# 	for meta_y in xrange(META_GRID_SIZE):
	# 		for meta_z in xrange(META_GRID_SIZE):
	# 			largebox_d = gpuarray.to_gpu(hbox_large[meta_x*N:(meta_x+1)*N, meta_y*N:(meta_y+1)*N, meta_z*N:(meta_z+1)*N])
	# 			HII_filter(largebox_d, N, np.int32(meta_x), np.int32(meta_y), np.int32(meta_z), ZERO, smoothR, block=block_size, grid=grid_size);
	# 			hbox_large[meta_x*N:(meta_x+1)*N, meta_y*N:(meta_y+1)*N, meta_z*N:(meta_z+1)*N] = largebox_d.get()
	#plan = Plan(shape, dtype=np.complex64)
	#plan.execute(largebox_d, inverse=True)  #FFT to real space of smoothed box
	#largebox_d /=  VOLUME  #divide by VOLUME if using fft (vs ifft)


	# This saves a large resolution deltax
	# np.save(parent_folder+"/Boxes/deltax_z0.00_{%i}_{%.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get())

	smallbox_d = gpuarray.zeros((M,M,M), dtype=np.float32)
	for meta_x in xrange(META_GRID_SIZE):
		for meta_y in xrange(META_GRID_SIZE):
			for meta_z in xrange(META_GRID_SIZE):
				temp = hbox_large[meta_x*N:(meta_x+1)*N, meta_y*N:(meta_y+1)*N, meta_z*N:(meta_z+1)*N].copy()
				largebox_d = gpuarray.to_gpu(temp)
				subsample_kernel(largebox_d, smallbox_d, N, M, PIXEL_FACTOR, block=block_size, grid=small_grid_size) #subsample in real space
				hbox_small[meta_x*M:(meta_x+1)*M, meta_y*M:(meta_y+1)*M, meta_z*M:(meta_z+1)*M] = smallbox_d.get()
	np.save(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc".format(HII_DIM, BOX_LEN), hbox_small)
	import IPython; IPython.embed()
	# To get velocities: reload the k-space box
	# hbox_large = np.load(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc.npy".format(DIM, BOX_LEN))
	# largebox_d = gpuarray.to_gpu(largebox)
	# #largebox_d /=  VOLUME  #divide by VOLUME if using fft (vs ifft)
	# smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	# largevbox_d = gpuarray.zeros((DIM,DIM,DIM), dtype=np.complex64)
	# smallvbox_d = gpuarray.zeros(HII_shape, dtype=np.float32)
	# for num, mode in enumerate(['x', 'y', 'z']):
	# 	velocity_kernel(largebox_d, largevbox_d, DIM, np.int32(num), block=block_size, grid=grid_size)
	# 	HII_filter(largevbox_d, DIM, ZERO, smoothR, block=block_size, grid=grid_size)
	# 	plan.execute(largevbox_d, inverse=True)
	# 	subsample_kernel(largevbox_d.real, smallvbox_d, DIM, HII_DIM,PIXEL_FACTOR, block=block_size, grid=small_grid_size)
	# 	np.save(parent_folder+"/Boxes/v{0}overddot_{1:d}_{2:.0f}Mpc".format(mode, HII_DIM, BOX_LEN), smallvbox_d.get())

	return
def init_stitch_2():
	"""this version first subsamples, and then compute fft on the HII
	probably doesn't make sense. 
	"""
	N = np.int32(HII_DIM) #prepare for stitching
	META_GRID_SIZE = DIM/N
	M = np.int32(HII_DIM/META_GRID_SIZE)
	#HII_DIM = np.int32(HII_DIM)
	f_pixel_factor = DIM/HII_DIM;
	scale = np.float32(BOX_LEN/DIM)
	HII_scale = np.float32(BOX_LEN)/HII_DIM
	shape = (N,N,N)
	#ratio of large box to small size
	kernel_source = open(cmd_folder+"/initialize_stitch.cu").read()
	kernel_code = kernel_source % {

		'DELTAK': DELTA_K,
		'DIM': DIM, 
		'VOLUME': VOLUME,
		'META_BLOCKDIM': N
	}
	main_module = nvcc.SourceModule(kernel_code)
	init_stitch = main_module.get_function("init_kernel")
	HII_filter = main_module.get_function("HII_filter")
	subsample_kernel = main_module.get_function("subsample_kspace")
	velocity_kernel = main_module.get_function("set_velocity")
	pspec_texture = main_module.get_texref("pspec")

	interpPspec, interpSize = init_pspec() #interpPspec contains both k array and P array
	interp_cu = cuda.matrix_to_array(interpPspec, order='C')
	cuda.bind_array_to_texref(interp_cu, pspec_texture)

	hbox_small = np.zeros(HII_shape, dtype=np.float32)
	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	for meta_x in xrange(META_GRID_SIZE):
		for meta_y in xrange(META_GRID_SIZE):
			for meta_z in xrange(META_GRID_SIZE):
				MRGgen = MRG32k3aRandomNumberGenerator(seed_getter=seed_getter_uniform, offset=meta_x*N**3)
				largebox_d = gpuarray.zeros(shape, dtype=np.float32)
				smallbox_d = gpuarray.zeros((M,M,M), dtype=np.complex64)
				init_stitch(largebox_d, N, np.int32(meta_x), np.int32(meta_y), np.int32(meta_z),block=block_size, grid=grid_size)
				largebox_d_imag = largebox_d.copy()
				largebox_d *= MRGgen.gen_normal(shape, dtype=np.float32)
				largebox_d_imag *= MRGgen.gen_normal(shape, dtype=np.float32)
				largebox_d = largebox_d + np.complex64(1.j) * largebox_d_imag
	#if want to get velocity need to use this
	#np.save(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc".format(DIM, BOX_LEN), hbox_large)
				HII_filter(largebox_d, N, np.int32(meta_x), np.int32(meta_y), np.int32(meta_z), ZERO, smoothR, block=block_size, grid=grid_size);
				#TODO make sense if we can subsample with antialiasing window
				subsample_kernel(largebox_d, smallbox_d, N, M, PIXEL_FACTOR, block=block_size, grid=small_grid_size) #subsample in real space
				hbox_small[meta_x*M:(meta_x+1)*M, meta_y*M:(meta_y+1)*M, meta_z*M:(meta_z+1)*M] = smallbox_d.get()

	plan = Plan(shape, dtype=np.complex64)
	largebox_d = gpuarray.to_gpu(hbox_small)
	plan.execute(largebox_d, inverse=True)  #FFT to real space of smoothed box
	#largebox_d /=  VOLUME  #divide by VOLUME if using fft (vs ifft)
	largebox_d *= scale**3

	np.save(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc".format(HII_DIM, BOX_LEN), hbox_small)
	import IPython; IPython.embed()
	

	return



def run():
	(free,total) = cuda.mem_get_info()
	print "Device global memory {0:.2f}GB total, {1:0.2f}GB free".format(total/1.e9, free/1.e9)
	print "Roughly {0:.2f}GB required for large box".format(float(NBYTES*8)/1.e9)
	if not os.path.exists(parent_folder+'/Boxes'):
		os.makedirs(parent_folder+"/Boxes")
	if NBYTES*8 < free:
		print "Congratulations, your GPU has enough memory, running without stitching"
		init()
	else:
		print "Stitching with {} meta_grid size".format(DIM/HII_DIM)
		init_stitch()
	#step2()

if __name__=="__main__":
	run()
	import IPython; IPython.embed()
