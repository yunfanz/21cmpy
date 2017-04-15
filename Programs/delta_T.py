import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.curandom import *
import pycuda.cumath as cumath
from pyfft.cuda import Plan
from pycuda.tools import make_default_context
from ..tocmfastpy import *
from IO_utils import *
#print cmd_folder
from ..cosmo_files import *
from ..Parameter_files import *


"""

  generates the 21-cm temperature offset from the CMB field and power spectrum 
  the spin temperature filename is optional

  Author: Yunfan G. Zhang
  04/2017

"""

def init_pspec():
	K = np.logspace(np.log10(DELTA_K/10), np.log10(DELTA_K*np.sqrt(3.)*DIM), BLOCK_VOLUME)
	#pspec from Eisenstein & Hu (1999 ApJ 511 5)
	pspec = pb.power_spectrum(K, 0.0, **COSMO)
	return np.vstack((K, pspec)).astype(np.float32), K.size

def run(xHfile=None, IO_DIR=None):

	if IO_DIR is None: 
		IO_DIR = parent_folder
	if not os.path.exists(IO_DIR+"/Outfiles"):
		os.makedirs(IO_DIR+"/Outfiles")
	if xHfile is None:
		xHfile = find_files(IO_DIR+"/Boxes/", pattern="xH*{0:06.2f}_{1:i}_{2:.0f}*".format(Z, HII_DIM, BOX_LEN))[0]
	if xHfile.endswith('.npy'):
		xH = np.load(xHfile)
		p_dict = boxio.parse_filename( os.path.splitext(xHfile)[0])
	else:
		b = boxio.readbox(xHfile)
		xH = b.box_data
		p_dict = b.param_dict
	Z = p_dict['z']
	#growth_factor = pb.fgrowth(Z, COSMO['omega_M_0'], unnormed=True)
	HII_DIM = p_dict['dim']
	BOX_LEN = np.float32(p_dict['BoxSize'])
	DELTA_K = np.float32(2*np.pi/BOX_LEN)
	VOLUME = (BOX_LEN*BOX_LEN*BOX_LEN)
	try:
		deltax = np.load(IO_DIR+"/Boxes/updated_smoothed_deltax_z0{0:.2f}_{1:d}_{2:.0f}Mpc.npy".format(Z, HII_DIM, BOX_LEN))
	except:
		#deltax = boxio.readbox(IO_DIR+"/Boxes/updated_smoothed_deltax_z{0:.2f}_{1:d}_{2:.0f}Mpc".format(Z, HII_DIM, BOX_LEN)).box_data
		deltax = boxio.readbox(IO_DIR+"/Boxes/updated_smoothed_deltax_z0{0:.2f}_{1:d}_{2:.0f}Mpc".format(Z, HII_DIM*2, BOX_LEN)).box_data[:HII_DIM,:HII_DIM,:HII_DIM]

	kernel_source = open(cmd_folder+"/delta_T.cu").read()
	kernel_code = kernel_source % {

		'DELTAK': DELTA_K,
		'VOLUME': VOLUME,
		'NUM_BINS': NUM_BINS
	}
	main_module = nvcc.SourceModule(kernel_code)
	pbox_kernel = main_module.get_function("pbox_kernel")
	#pixel_deltax_d = gpuarray.to_gpu(deltax)
	#pixel_xH_d = gpuarray.to_gpu(xH)

	_const_factor = np.float32(27 * (COSMO['omega_b_0']*COSMO['h']*COSMO['h']/0.023) * 
		np.sqrt( (0.15/COSMO['omega_M_0']/COSMO['h']/COSMO['h']) * (1+Z)/10.0 ))
	delta_T = np.float32(_const_factor*xH*(1.0+deltax))
	ave = np.mean(delta_T)
	np.save(IO_DIR+"/Boxes/delta_T_no_halos_z{0:.2f}_nf{1:f}_useTs{2:d}_zetaX{3:.1e}_TvirminX{4:.1e}_aveTb{5:.2f}_{6:d}_{7:d}Mpc.npy".format(Z, 
		p_dict['nf'], USE_TS_IN_21CM, p_dict['eff'], ION_Tvir_MIN, ave, HII_DIM, int(BOX_LEN)), delta_T)

	deldel_T = (delta_T/ave-1)*VOLUME/HII_TOT_NUM_PIXELS
	if DIMENSIONAL_T_POWER_SPEC:
		deldel_T *= ave

	plan = Plan(HII_shape, dtype=np.complex64)
	deldel_T_d = gpuarray.to_gpu(deldel_T.astype(np.complex64))
	plan.execute(deldel_T_d)
	K = np.float32(np.logspace(np.log10(DELTA_K), np.log10(DELTA_K*np.sqrt(3.)*DIM), NUM_BINS))
	K_d = gpuarray.to_gpu(K)
	k_ave_d = gpuarray.zeros_like(K_d)
	in_bin_ct_d = gpuarray.zeros_like(K_d)
	ps_d = gpuarray.zeros_like(K_d)

	pbox_kernel(deldel_T_d, DIM, ps_d,  k_ave_d, in_bin_ct_d, K_d, block=block_size, grid=HII_grid_size)
	ps = ps_d.get()
	in_bin_ct = in_bin_ct_d.get()
	k_ave = k_ave_d.get()
	k_ave = np.where(in_bin_ct>0, k_ave/in_bin_ct, 0.)
	ps_ave = np.where(in_bin_ct>0, ps/in_bin_ct, 0.)
	#ps_fname = "/ps_nov_no_halos_z{0:.2f}_nf{1:f}_useTs{2:d}_zetaX{3:.1e}_TvirminX{4:.1e}_aveTb{5:.2f}_{6:d}_{7:d}Mpc".format(Z, p_dict['nf'], USE_TS_IN_21CM, p_dict['eff'], ION_Tvir_MIN, ave, HII_DIM, np.int32(BOX_LEN))
	#np.savez(IO_DIR+ps_fname, k_ave=k_ave, ps_ave=ps_ave)


	return K, k_ave, ps_ave


if __name__ == "__main__":
	o = optparse.OptionParser()
	o.add_option('-d','--dir', dest='DIR', default=IO_DIR)
	o.add_option('-i','--inter', dest='INTERACTIVE', action="store_true")
	(opts, args) = o.parse_args()
	k, kav, ps = run(args[0], opts.DIR)
	delsq = k**3*ps/2/np.pi**2*1.e6
	plt.plot(k, delsq)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$k Mpc^{-1}$')
	plt.ylabel(r'$\Delta^2 mK^{2}$')
	plt.xlim([0.1, 10])
	import IPython; IPython.embed()