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
#import pyfftw

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
	#pspec from Eisenstein & Hu (1999 ApJ 511 5)
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
	
	MRGgen = MRG32k3aRandomNumberGenerator(seed_getter=seed_getter_uniform, offset=0)

	kernel_source = open(cmd_folder+"/initialize.cu").read()
	kernel_code = kernel_source % {

		'DELTAK': DELTA_K,
		'VOLUME': VOLUME,
		'DIM': DIM
	}
	main_module = nvcc.SourceModule(kernel_code)
	init_kernel = main_module.get_function("init_kernel")
	HII_filter = main_module.get_function("HII_filter")
	adj_complex_conj = main_module.get_function("adj_complex_conj")
	subsample_kernel = main_module.get_function("subsample")
	velocity_kernel = main_module.get_function("set_velocity")
	pspec_texture = main_module.get_texref("pspec")

	interpPspec, interpSize = init_pspec() #interpPspec contains both k array and P array
	interp_cu = cuda.matrix_to_array(interpPspec, order='C')
	cuda.bind_array_to_texref(interp_cu, pspec_texture)

	largebox_d = gpuarray.zeros(shape, dtype=np.float32)
	init_kernel(largebox_d, np.int32(DIM), block=block_size, grid=grid_size)
	largebox_d_imag = gpuarray.zeros(shape, dtype=np.float32)
	init_kernel(largebox_d_imag, np.int32(DIM), block=block_size, grid=grid_size)

	largebox_d *= MRGgen.gen_normal(shape, dtype=np.float32)
	largebox_d_imag *= MRGgen.gen_normal(shape, dtype=np.float32)
	largebox_d = largebox_d + np.complex64(1.j) * largebox_d_imag

	#adj_complex_conj(largebox_d, DIM, block=block_size, grid=grid_size)
	largebox = largebox_d.get_async()
	#np.save(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc".format(DIM, BOX_LEN), largebox)

	#save real space box before smoothing
	plan = Plan(shape, dtype=np.complex64)
	plan.execute(largebox_d, inverse=True)  #FFT to real space of smoothed box
	np.save(parent_folder+"/Boxes/deltax_z0.00_{0:d}_{1:.0f}Mpc".format(DIM, BOX_LEN), largebox_d.get_async())

	#save real space box after smoothing and subsampling
	# host largebox is still in k space, no need to reload from disk
	largebox_d = gpuarray.to_gpu(largebox)
	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	HII_filter(largebox_d, N, ZERO, smoothR, block=block_size, grid=grid_size);
	plan.execute(largebox_d, inverse=True)  #FFT to real space of smoothed box
	largebox_d /= scale**3
	smallbox_d = gpuarray.zeros(HII_shape, dtype=np.float32)
	subsample_kernel(largebox_d.real, smallbox_d, N, HII_DIM, PIXEL_FACTOR, block=block_size, grid=HII_grid_size) #subsample in real space
	np.save(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc".format(HII_DIM, BOX_LEN), smallbox_d.get_async())

	# reload the k-space box for velocity boxes
	largebox_d = gpuarray.to_gpu(largebox)
	largebox_d /= scale**3
	#largebox_d /=  VOLUME  #divide by VOLUME if using fft (vs ifft)
	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	largevbox_d = gpuarray.zeros((DIM,DIM,DIM), dtype=np.complex64)
	smallbox_d = gpuarray.zeros(HII_shape, dtype=np.float32)
	for num, mode in enumerate(['x', 'y', 'z']):
		velocity_kernel(largebox_d, largevbox_d, DIM, np.int32(num), block=block_size, grid=grid_size)
		HII_filter(largevbox_d, DIM, ZERO, smoothR, block=block_size, grid=grid_size)
		plan.execute(largevbox_d, inverse=True)
		subsample_kernel(largevbox_d.real, smallbox_d, DIM, HII_DIM,PIXEL_FACTOR, block=block_size, grid=HII_grid_size)
		np.save(parent_folder+"/Boxes/v{0}overddot_{1:d}_{2:.0f}Mpc".format(mode, HII_DIM, BOX_LEN), smallbox_d.get_async())

	return

def fft_stitch(N, plan2d, plan1d, hostarr, largebox_d):
	w = hostarr.shape[0]
	META_GRID_SIZE = w/N
	fftbatch = 2
	for meta_z in xrange(META_GRID_SIZE): #fft along x
		largebox_d = gpuarray.to_gpu_async(hostarr[:, :, meta_z*N:(meta_z+1)*N].transpose(1,2,0))
		#print largebox_d.shape
		plan1d.execute(largebox_d, batch=fftbatch)
		hostarr[:, :, meta_z*N:(meta_z+1)*N] = largebox_d.real.get_async().transpose(2,0,1)
	for meta_x in xrange(META_GRID_SIZE): #fft along y, z
		largebox_d = gpuarray.to_gpu_async(hostarr[meta_x*N:(meta_x+1)*N, :, :].copy())
		plan2d.execute(largebox_d, batch=fftbatch)
		hostarr[meta_x*N:(meta_x+1)*N, :, :] = largebox_d.real.get_async()
	return hostarr



def init_stitch(N):
	"""outputs the high resolution k-box, and the smoothed r box

	Input
	-----------
	N:  int32
		size of box to load onto the GPU, should be related to DIM by powers of 2

	"""
	if N is None:
		N = np.int32(HII_DIM) #prepare for stitching
	META_GRID_SIZE = DIM/N
	M = np.int32(HII_DIM/META_GRID_SIZE)
	#HII_DIM = np.int32(HII_DIM)
	f_pixel_factor = DIM/HII_DIM;
	scale = np.float32(BOX_LEN/DIM)
	print 'scale', scale
	HII_scale = np.float32(BOX_LEN/HII_DIM)
	shape = (DIM,DIM,N)
	stitch_grid_size = (DIM/(block_size[0]),
						DIM/(block_size[0]),
						N/(block_size[0]))
	HII_stitch_grid_size = (HII_DIM/(block_size[0]),
						HII_DIM/(block_size[0]),
						M/(block_size[0]))
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
	MRGgen = MRG32k3aRandomNumberGenerator(seed_getter=seed_getter_uniform, offset=0)
	plan2d = Plan((np.int64(DIM), np.int64(DIM)), dtype=np.complex64)
	plan1d = Plan((np.int64(DIM)), dtype=np.complex64)
	print "init pspec"
	interpPspec, interpSize = init_pspec() #interpPspec contains both k array and P array
	interp_cu = cuda.matrix_to_array(interpPspec, order='C')
	cuda.bind_array_to_texref(interp_cu, pspec_texture)
	#hbox_large = pyfftw.empty_aligned((DIM, DIM, DIM), dtype='complex64')
	hbox_large = np.zeros((DIM, DIM, DIM), dtype=np.complex64)
	hbox_small = np.zeros(HII_shape, dtype=np.float32)
	smoothR = np.float32(L_FACTOR*BOX_LEN/HII_DIM)
	largebox_d = gpuarray.zeros(shape, dtype=np.float32)
	largebox_d_imag = gpuarray.zeros(shape, dtype=np.float32)
	print "init boxes"
	for meta_z in xrange(META_GRID_SIZE):
		# MRGgen = MRG32k3aRandomNumberGenerator(seed_getter=seed_getter_uniform, offset=meta_x*N**3)
		init_stitch(largebox_d, DIM, np.int32(meta_z),block=block_size, grid=stitch_grid_size)
		init_stitch(largebox_d_imag, DIM, np.int32(meta_z),block=block_size, grid=stitch_grid_size)
		largebox_d *= MRGgen.gen_normal(shape, dtype=np.float32)
		largebox_d_imag *= MRGgen.gen_normal(shape, dtype=np.float32)
		largebox_d = largebox_d + np.complex64(1.j) * largebox_d_imag
		hbox_large[:, :, meta_z*N:(meta_z+1)*N] = largebox_d.get_async()
	#if want to get velocity need to use this
	np.save(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc.npy".format(DIM, BOX_LEN), hbox_large)

	print "Executing FFT on device"
	#hbox_large = pyfftw.interfaces.numpy_fft.ifftn(hbox_large).real
	hbox_large = fft_stitch(N, plan2d, plan1d, hbox_large, largebox_d).real
	print hbox_large.dtype
	print "Finished FFT on device"
	np.save(parent_folder+"/Boxes/deltax_z0.00_{0:d}_{1:.0f}Mpc.npy".format(DIM, BOX_LEN), hbox_large)
	return

	hbox_large = np.load(parent_folder+"/Boxes/deltak_z0.00_{0:d}_{1:.0f}Mpc.npy".format(DIM, BOX_LEN))
	for meta_z in xrange(META_GRID_SIZE):
		largebox_d = gpuarray.to_gpu_async(hbox_large[:, :, meta_z*N:(meta_z+1)*N].copy())
		HII_filter(largebox_d, DIM, np.int32(meta_z), ZERO, smoothR, block=block_size, grid=stitch_grid_size);
		hbox_large[:, :, meta_z*N:(meta_z+1)*N] = largebox_d.get_async()
	#import IPython; IPython.embed()
	print "Executing FFT on host"
	#hbox_large = hifft(hbox_large).astype(np.complex64).real
	#hbox_large = pyfftw.interfaces.numpy_fft.ifftn(hbox_large).real
	hbox_large = fft_stitch(N, plan2d, plan1d, hbox_large, largebox_d).real
	print "Finished FFT on host"
	#import IPython; IPython.embed()

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

	
	print "downsampling"
	smallbox_d = gpuarray.zeros((HII_DIM,HII_DIM,M), dtype=np.float32)
	for meta_z in xrange(META_GRID_SIZE):
		largebox_d = gpuarray.to_gpu_async(hbox_large[:, :, meta_z*N:(meta_z+1)*N].copy())
		largebox_d /= scale**3 #
		subsample_kernel(largebox_d, smallbox_d, DIM, HII_DIM, PIXEL_FACTOR, block=block_size, grid=HII_stitch_grid_size) #subsample in real space
		hbox_small[:, :, meta_z*M:(meta_z+1)*M] = smallbox_d.get_async()
	np.save(parent_folder+"/Boxes/smoothed_deltax_z0.00_{0:d}_{1:.0f}Mpc".format(HII_DIM, BOX_LEN), hbox_small)
	#import IPython; IPython.embed()
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




def run():
	(free,total) = cuda.mem_get_info()
	print "Device global memory {0:.2f}GB total, {1:0.2f}GB free".format(total/1.e9, free/1.e9)
	print "Roughly {0:.2f}GB required for large box".format(float(NBYTES*4)/1.e9)
	if not os.path.exists(parent_folder+'/Boxes'):
		os.makedirs(parent_folder+"/Boxes")
	if NBYTES*4 < free:
		print "Congratulations, your GPU has enough memory, running without stitching"
		init()
	else:
		N = DIM
		while float(N)/DIM*NBYTES*32 > free:
			N /= 2
		print "Stitching with {} meta block size".format(N)
		init_stitch(np.int32(128))
	#step2()

if __name__=="__main__":
	run()
	#import IPython; IPython.embed()
