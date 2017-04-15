from ..tocmfastpy import *
from IO_utils import *
#print cmd_folder
from ..cosmo_files import *
from ..Parameter_files import *
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cu
import pycuda.autoinit
from pyfft.cuda import Plan
from pycuda.tools import make_default_context
"""
  Author: Yunfan G. Zhang
  04/2017
"""

def conv_bubbles(deltax, param_dict, Z=None, scale=None, fil=1, update=0, LE=False, visualize=0, quiet=False):
	"""
Excursion-set formalism, or Fast Fourier Radiative-Transform. 
Calculates ionization fields from density field provided. 
For each box pixel, it cycles through various bubble radii
  , until it finds the largest radius such that the enclosed collapsed mass fraction 
  (obtained by summing masses from the halo list file of
  halos whose centers are within the bubble, or by taking 
  the mean collapsed mass from conditional press-schechter)
  is larger than 1/ZETA. 

Parameters
----------
deltax : numpy.float32 array
	Real space density box, must have dimensions powers of 2. 
param_dict: python dictionary
	dictionary of parameters created by boxio.parse_filename
Z: float32
	Required if input density is the present day linear density, program would extrapolate to Z. 
fil: int32
	type of filter for smoothing : 0: rtophat; 1: ktophat, 2: Gaussian
update: int32
	Method to update the ionization field 0: center pixel, 1: sphere painting
visualize: bool
	if True, draw slice of density field and created ionization field 
quiet: bool

Returns
----------

ion_field: numpy array float32

"""
	
	if not quiet: 
		print "Using filter_type {}".format(fil)	
	if scale is None:
		scale = param_dict['BoxSize']/param_dict['HIIdim']
	if Z is None:
		Z = param_dict['Z']
	sigmamin, deltac = pb.sig_del(ION_Tvir_MIN, Z, **COSMO)
	fgrowth = np.float32(deltac/1.686)
	smin = sigmamin**2
	fc_mean_ps = pb.collapse_fraction(sigmamin, deltac).astype(np.float32)  #mean collapse fraction of universe

	"""find bubbbles for deltax box I"""
	kernel_source = open(cmd_folder+"/find_bubbles.cu").read()
	kernel_code = kernel_source % {
        'DELTAC': deltac,
        'RMIN': R_BUBBLE_MIN,
        'ZETA': ZETA,
        'DELTAK': DELTA_K
    }
	main_module = nvcc.SourceModule(kernel_code)
	fcoll_kernel = main_module.get_function("fcoll_kernel")
	update_kernel = main_module.get_function("update_kernel")
	update_sphere_kernel = main_module.get_function("update_sphere_kernel")
	final_kernel = main_module.get_function("final_kernel")
	HII_filter = main_module.get_function("HII_filter")
	# Get contiguous image + shape.
	height, width, depth = deltax.shape
	HII_TOT_NUM_PIXELS = height*width*depth
	
	
	 # Initialize variables.
	width         = np.int32(width)
	deltax        = np.float32(deltax.copy()) 
	if not LE:
		deltax *= fgrowth #linearly extrapolate the non-linear density to present
	# Transfer asynchronously.
	ionized_d = gpuarray.zeros([height,width,depth], dtype=np.float32)
	delta_d = gpuarray.to_gpu_async(deltax)


	fftplan = Plan(deltax.shape, dtype=np.complex64)
	R = R_BUBBLE_MAX; cnt = 0

	if visualize > 0:
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		fig.suptitle(" Smoothed Density and Ionization")
		ax1.set_title('Density')
		mydelta = plt.imshow(delta_d.get().real[width/2])
		plt.colorbar()
		ax2 = fig.add_subplot(122)
		ax2.set_title('Ionization')
		myion = plt.imshow(np.ones_like(deltax)[width/2])
		plt.colorbar()
		if visualize == 1:
			print "HERE"
			plt.pause(.01)
			plt.draw()
		elif visualize == 2:
			plt.savefig('tmp/{0:03d}.png'.format(cnt))

		#plt.colorbar()
	final_step = False
	final_denom = -1
	if R_BUBBLE_MIN < L_FACTOR*scale:
		temparg = 2*(smin - sig0(L_FACTOR*scale) )
		if temparg < 0:
			raise(Exception)
		else:
			final_denom = np.sqrt(temparg).astype(np.float32)
	while not final_step:
		
		if (R/DELTA_R_FACTOR) <= (L_FACTOR*scale) or ((R/DELTA_R_FACTOR) <= R_BUBBLE_MIN): #stop if reach either rmin or cell size
			final_step = True
		R = np.float32(R)
		S0 = np.float32(sig0(R))
		#S0 = np.float32(pb.sigma_r(R, Z, **cosmo)[0])
		denom = np.sqrt(2*(smin - S0)).astype(np.float32)
		if not quiet:
			print 'R={} Mpc'.format(R)
			print 'denom', denom

		start = cu.Event()
		step1 = cu.Event()
		step2 = cu.Event()
		step3 = cu.Event()
		step4 = cu.Event()
		end = cu.Event()

		start.record()
		delta_d = gpuarray.to_gpu_async(deltax.astype(np.complex64))
		fcoll_d = gpuarray.zeros(deltax.shape, dtype=np.float32)
		start.synchronize()
		fftplan.execute(delta_d)
		step1.record(); step1.synchronize()
		
		HII_filter(delta_d, width, np.int32(fil), R, block=block_size, grid=grid_size)
		step2.record(); step2.synchronize()
		#import IPython; IPython.embed()
		fftplan.execute(delta_d, inverse=True)
		step2.synchronize()
		

		# if not the final step, get ionized regions, if final step paint partial ionizations
		if not final_step:
			fcoll_kernel(fcoll_d, delta_d.real, width, denom, block=block_size, grid=grid_size)
			step3.record(); step3.synchronize()
			if not LE:
				fcollmean = gpuarray.sum(fcoll_d).get()/float(HII_TOT_NUM_PIXELS)
				fcoll_d *= fc_mean_ps/fcollmean# #normalize since we used non-linear density
				step4.record(); step4.synchronize()
			if update == 0:
				update_kernel(ionized_d, fcoll_d, width, block=block_size, grid=grid_size)
			else:
				update_sphere_kernel(ionized_d, fcoll_d, width, R, block=block_size, grid=grid_size)
		else:
			if final_denom < 0: final_denom = denom
			# print 'final denom', final_denom
			fcoll_kernel(fcoll_d, delta_d.real, width, denom, block=block_size, grid=grid_size)
			step3.record(); step3.synchronize()
			if not LE:
				fcollmean = gpuarray.sum(fcoll_d).get()/float(HII_TOT_NUM_PIXELS)
				fcoll_d *= fc_mean_ps/fcollmean
				step4.record(); step4.synchronize()
			final_kernel(ionized_d, fcoll_d, width, block=block_size, grid=grid_size)
		end.record()
		end.synchronize()
		if visualize > 0:
			mydelta.set_data(delta_d.real.get()[width/2])
			myion.set_data(ionized_d.get()[width/2])
			ax1.set_title('R = %f'%(R))
			if visualize == 1:
				plt.pause(.01)
				plt.draw()
			elif visualize == 2:
				plt.savefig('tmp/{0:03d}.png'.format(cnt))


		R = R/DELTA_R_FACTOR
		cnt +=1 

	ionized = ionized_d.get()
	return ionized

def run(opts, args):

	print "=============================== find_HII_bubbles ================================="

	if len(args) > 0:
		file = args[0]
		print file
	else:
		raise Exception("Must specify density box input!")
	

	if file.endswith('npy'):
		param_dict = boxio.parse_filename(file)
		d1 = np.load(file)
	else:
		b1 = boxio.readbox(file)
		d1 = b1.box_data[:256,:256,:256]
		param_dict = b1.param_dict

	if not param_dict['filename'].startswith('updated'):
		print "input is initial density field, will linearly extrapolate to z={}".format(opts.Z)
		Z = opts.Z
		opts.LIN = True
	else:
		print "input is nonlinear density field"
		Z = param_dict['z']
		opts.LIN = False

	ion_field = conv_bubbles(d1, param_dict, Z, fil=opts.FILTER_TYPE, update=opts.UPDATE_TYPE, 
		LE=opts.LIN, visualize=int(opts.VISUAL), quiet=opts.QUIET)
	nf = 1 - np.mean(ion_field)
	if not opts.INTERACTIVE:
		outname = "xH_nohalos_z{0:.2f}_nf{1:f}_eff{2:.1f}_HIIfilter{3:d}_Tvirmin{4:.1f}_RHIImax{5:d}_{6:d}_{7:d}Mpc.npy".format(
			Z,nf, ZETA, opts.FILTER_TYPE, ION_Tvir_MIN, int(R_BUBBLE_MAX), ion_field.shape[0], int(param_dict['BoxSize']))
		print "saving", outname
		np.save(opts.DIR+outname, 1-ion_field)
		return
	else:
		return ion_field


if __name__ == '__main__':
	o = optparse.OptionParser()
	o.add_option('-d','--dir', dest='DIR', default=IO_DIR+'/Boxes/')
	o.add_option('-f','--filt', dest='FILTER_TYPE', default=HII_FILTER, help="0: rtophat; 1: ktophat, 2: Gaussian")
	o.add_option('-u','--upd', dest='UPDATE_TYPE', default=FIND_BUBBLE_ALGORITHM, help="0: center pixel, 1: sphere painting")
	o.add_option('-z','--reds', dest='Z', default=12., help="redshift, this is needed if using linear density field at z=0")
	o.add_option('-v','--vis', dest='VISUAL', default=0, help="0: no visualization, 1: draw, 2:save pngs")
	o.add_option('-q','--qui', dest='QUIET', action="store_true")
	o.add_option('-i','--inter', dest='INTERACTIVE', action="store_true")
	(opts, args) = o.parse_args()

	_ = run(opts, args)