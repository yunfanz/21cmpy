from Choud14 import *
from tocmfastpy import *
from IO_utils import *
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cu
import pycuda.autoinit
from pyfft.cuda import Plan
from pycuda.tools import make_default_context

def find_bubbles(I, scale=1., fil='kspace'):
	"""brute force method"""
	zeta = 40.
	Z = 12.
	RMAX = 30.
	RMIN = 1.
	mm = mmin(Z)
	smin = sig0(m2R(mm))
	deltac = Deltac(Z)
	fgrowth = deltac/1.686
	"""find bubbbles for deltax box I"""
	kernel_source = open("find_bubbles.cu").read()
	kernel_code = kernel_source % {
        'DELTAC': deltac,
        'RMIN': RMIN,
        'SMIN': smin, 
        'ZETA': zeta
    }
	main_module = nvcc.SourceModule(kernel_code)
	if fil == 'rspace':
		kernel = main_module.get_function("real_tophat_kernel")
	elif fil == 'kspace':
		kernel = main_module.get_function("k_tophat_kernel")
	image_texture = main_module.get_texref("img")

	# Get contiguous image + shape.
	height, width, depth = I.shape
	I = np.float32(I.copy()*fgrowth)

	# Get block/grid size for steps 1-3.
	block_size =  (8,8,8)
	grid_size =   (width/(block_size[0]-2)+1,
				height/(block_size[0]-2)+1,
				depth/(block_size[0]-2)+1)
	 # Initialize variables.
	ionized       = np.zeros([height,width,depth]) 
	ionized       = np.float32(ionized)
	width         = np.int32(width)

	# Transfer labels asynchronously.
	ionized_d = gpuarray.to_gpu_async(ionized)
	I_cu = cu.np_to_array(I, order='C')
	cu.bind_array_to_texref(I_cu, image_texture)

	
	R = RMAX
	while R > RMIN:
		print R
		Rpix = np.float32(R/scale)
		S0 = np.float32(sig0(R))
		start = cu.Event()
		end = cu.Event()
		start.record()
		kernel(ionized_d, width, Rpix, S0, block=block_size, grid=grid_size)
		end.record()
		end.synchronize()
		R *= (1./1.5)

	ionized = ionized_d.get()
	return ionized

def conv(delta_d, filt_d, shape, fil):
	smoothI = np.zeros(shape, dtype=np.complex64)
	smoothed_d = gpuarray.to_gpu(smoothI)
	plan = Plan(shape, dtype=np.complex64)
	plan.execute(delta_d)
	if fil == 'rspace':
		plan.execute(filt_d)
	smoothed_d = delta_d * filt_d.conj()
	plan.execute(smoothed_d, inverse=True)
	return smoothed_d.real

def conv_bubbles(I, param_dict, scale=None, fil=1, update=0, visualize=False):
	"""uses fft convolution"""
	zeta = 40.
	Lfactor = 0.620350491
	Z = param_dict['z']
	DELTA_R_FACTOR = 1.1
	print "Using filter_type {}".format(fil)	
	if scale is None:
		scale = param_dict['BoxeSize']/param_dict['HIIdim']
	dk = 2*np.pi/I.shape[0]*scale#param_dict['BoxSize'] #delta k in inverse Mpc
	RMAX = np.float32(30) #in Mpc
	RMIN = np.float32(0.5)
	mm = mmin(Z, Tvir=1.e4)
	smin = sig0(m2R(mm))
	#smin = pb.sigma_r(m2R(mm), Z, **cosmo)[0]
	deltac = Deltac(Z)
	fgrowth = deltac/1.686
	fc_mean_ps = pb.collapse_fraction(np.sqrt(smin), deltac).astype(np.float32)  #mean collapse fraction of universe
	print fc_mean_ps
	"""find bubbbles for deltax box I"""
	kernel_source = open("find_bubbles.cu").read()
	kernel_code = kernel_source % {
        'DELTAC': deltac,
        'RMIN': RMIN,
        'SMIN': smin, 
        'ZETA': zeta,
        'DELTAK': dk
    }
	main_module = nvcc.SourceModule(kernel_code)
	fcoll_kernel = main_module.get_function("fcoll_kernel")
	update_kernel = main_module.get_function("update_kernel")
	update_sphere_kernel = main_module.get_function("update_sphere_kernel")
	final_kernel = main_module.get_function("final_kernel")
	HII_filter = main_module.get_function("HII_filter")
	# Get contiguous image + shape.
	height, width, depth = I.shape
	HII_TOT_NUM_PIXELS = height*width*depth
	
	
	# Get block/grid size for steps 1-3.
	block_size =  (8,8,8)
	grid_size =   (width/(block_size[0]-2)+1,
				height/(block_size[0]-2)+1,
				depth/(block_size[0]-2)+1)
	 # Initialize variables.
	ionized       = np.zeros([height,width,depth]) 
	ionized       = np.float32(ionized)
	width         = np.int32(width)
	I             = np.float32(I.copy()*fgrowth) #linearly extrapolate the non-linear density to present?
	#filt          = np.ones_like(I)


	# Transfer labels asynchronously.
	ionized_d = gpuarray.to_gpu_async(ionized)
	delta_d = gpuarray.to_gpu_async(I)
	# I_cu = cu.np_to_array(I, order='C')
	# cu.bind_array_to_texref(I_cu, image_texture)

	fftplan = Plan(I.shape, dtype=np.complex64)
	R = RMAX

	if visualize:
		fig = plt.figure(figsize=(12,12))
		ax = fig.add_subplot(111)
		fig.suptitle(" PyCUDA")
		ax.set_title('title')
		plt.subplot(121)
		mydelta = plt.imshow(delta_d.get().real[width/2])
		plt.colorbar()
		plt.subplot(122)
		myion = plt.imshow(np.ones_like(I)[width/2])
		plt.colorbar()
		plt.pause(.01)
		plt.draw()
		#plt.colorbar()
	final_step = False
	final_denom = -1
	if RMIN < Lfactor*scale:
		temparg = 2*(smin - sig0(Lfactor*scale) )
		if temparg < 0:
			raise(Exception)
		else:
			final_denom = np.sqrt(temparg).astype(np.float32)
	while not final_step:
		print 'R={} Mpc'.format(R)
		if (R/DELTA_R_FACTOR) <= (Lfactor*scale) or ((R/DELTA_R_FACTOR) <= RMIN): #stop if reach either rmin or cell size
			final_step = True
		R = np.float32(R)
		Rpix = np.float32(R/scale)

		S0 = np.float32(sig0(R))
		#S0 = np.float32(pb.sigma_r(R, Z, **cosmo)[0])
		denom = np.sqrt(2*(smin - S0)).astype(np.float32)
		print 'denom', denom

		start = cu.Event()
		step1 = cu.Event()
		step2 = cu.Event()
		step3 = cu.Event()
		step4 = cu.Event()
		end = cu.Event()

		start.record()
		#smoothed_d = conv(delta_d.astype(np.complex64), I.shape, fil=fil)

		delta_d = gpuarray.to_gpu_async(I.astype(np.complex64))
		fcoll_d = gpuarray.zeros(I.shape, dtype=np.float32)
		start.synchronize()
		fftplan.execute(delta_d)
		step1.record(); step1.synchronize()
		
		HII_filter(delta_d, width, np.int32(fil), R, block=block_size, grid=grid_size)
		step2.record(); step2.synchronize()
		#import IPython; IPython.embed()
		fftplan.execute(delta_d, inverse=True)
		step2.synchronize()
		# import IPython; IPython.embed()

		
		if not final_step:
			fcoll_kernel(fcoll_d, delta_d.real, width, denom, block=block_size, grid=grid_size)
			step3.record(); step3.synchronize()
			fcollmean = gpuarray.sum(fcoll_d).get()/float(HII_TOT_NUM_PIXELS)
			fcoll_d *= fc_mean_ps/fcollmean# #normalize since we used non-linear density
			#print fcoll_d.dtype
			step4.record(); step4.synchronize()
			if update == 0:
				update_kernel(ionized_d, fcoll_d, width, block=block_size, grid=grid_size)
			else:
				update_sphere_kernel(ionized_d, fcoll_d, width, R, block=block_size, grid=grid_size)
		else:
			if final_denom < 0: final_denom = denom
			print 'final denom', final_denom
			fcoll_kernel(fcoll_d, delta_d.real, width, denom, block=block_size, grid=grid_size)
			step3.record(); step3.synchronize()
			fcollmean = gpuarray.sum(fcoll_d).get()/float(HII_TOT_NUM_PIXELS)
			fcoll_d *= fc_mean_ps/fcollmean
			step4.record(); step4.synchronize()
			final_kernel(ionized_d, fcoll_d, width, block=block_size, grid=grid_size)
		end.record()
		end.synchronize()
		if visualize:
			mydelta.set_data(delta_d.real.get()[width/2])
			myion.set_data(ionized_d.get()[width/2])
			ax.set_title('R = %f'%(R))
			plt.pause(.01)
			plt.draw()


		R = R/DELTA_R_FACTOR

	ionized = ionized_d.get()
	return ionized

if __name__ == '__main__':
	o = optparse.OptionParser()
	o.add_option('-d','--dir', dest='DIR', default='/home/yunfanz/Data/21cmFast/Boxes/')
	o.add_option('-f','--filt', dest='FILTER_TYPE', default=1) #0: rtophat; 1: ktophat, 2: Gaussian
	o.add_option('-u','--upd', dest='UPDATE_TYPE', default=0) #0: center pixel, 1: sphere painting
	(opts, args) = o.parse_args()
	print opts
	print args
	z = 12.00
	files = find_deltax(opts.DIR, z=z)
	file = files[0]
	b1 = boxio.readbox(file)
	scale = 1
	#d1 = 1 - b1.box_data[::scale, ::scale, ::scale]
	d1 = b1.box_data[:256, :256, :256]
	print d1.shape
	ion_field = conv_bubbles(d1, b1.param_dict, scale=float(scale), fil=opts.FILTER_TYPE, update=opts.UPDATE_TYPE, visualize=False)
	print ion_field.shape
	import IPython; IPython.embed()
