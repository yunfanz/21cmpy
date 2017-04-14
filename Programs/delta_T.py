import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.curandom import *
import pycuda.cumath as cumath
from pyfft.cuda import Plan
from pycuda.tools import make_default_context
from IO_utils import *


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

def run(xHfile=None):
	  #************  BEGIN INITIALIZATION **************************** 
	if IO_DIR is None: 
		IO_DIR = parent_folder
	if not os.path.exists(IO_DIR+"/Outfiles"):
		os.makedirs(IO_DIR+"/Outfiles")
	if xHfile is None:
		xHfile = find_files(IO_DIR+"/Boxes/", pattern="xH*{0:06.2f}_{1:i}_{2:.0f}*".format(Z, HII_DIM, BOX_LEN))[0]
	if xHfile.endswith('.npy'):
		xH = np.load(filename)
		p_dict = boxio.parse_filename(filename.split['.'][1])
	else:
		b = boxio.readbox(filename)
		xH = b.box_data
		p_dict = b.param_dict
	Z = param_dict['Z']
	#growth_factor = pb.fgrowth(Z, COSMO['omega_M_0'], unnormed=True)
	try:
		deltax = np.load(IO_DIR+"/Boxes/updated_smoothed_deltax_z{0:06.2f}_{1:i}_{2:.0f}Mpc.npy".format(Z, HII_DIM, BOX_LEN))
	except:
		deltax = boxio.readbox(IO_DIR+"/Boxes/updated_smoothed_deltax_z{0:06.2f}_{1:i}_{2:.0f}Mpc".format(Z, HII_DIM, BOX_LEN)).box_data

	kernel_source = open(cmd_folder+"/delta_T.cu").read()
	kernel_code = kernel_source % {

		'DELTAK': DELTA_K,
		'VOLUME': VOLUME,
		'DIM': DIM, 
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
	np.save(parent_folder+"/Boxes/delta_T_no_halos_z%06.2f_nf%f_useTs%i_zetaX%.1e_TvirminX%.1e_aveTb%06.2f_%i_%.0fMpc.npy".format(Z, 
		p_dict['nf'], USE_TS_IN_21CM, p_dict['eff'], ION_Tvir_MIN, ave, HII_DIM, BOX_LEN);)

	deldel_T = (delta_T/ave-1)*VOLUME/HII_TOT_NUM_PIXELS
	if DIMENSIONAL_T_POWER_SPEC:
		deldel_T *= ave

	plan = Plan(HII_shape, dtype=np.complex64)
	deldel_T_d = gpu.to_gpu(deldel_T.astype(np.complex64))
	plan.execute(deldel_T_d)
	K = np.float32(np.logspace(np.log10(DELTA_K/10), np.log10(DELTA_K*np.sqrt(3.)*DIM), NUM_BINS))
	K_d = gpuarray.to_gpu(K)
	k_ave_d = gpuarray.zeros_like(K_d)
	in_bin_ct_d = gpuarray.zeros_like(K_d)
	ps_d = gpuarray.zeros_like(K_d)

	pbox_kernel(deldel_T_d, DIM, ps_d,  k_ave_d, in_bin_ct_d, K_d)
	ps = ps_d.get()
	in_bin_ct = in_bin_ct_d.get()
	k_ave = k_ave_d.get()
	ps_fname = "/ps_nov_no_halos_z%06.2f_nf%f_useTs%i_zetaX%.1e_TvirminX%.1e_aveTb%06.2f_%i_%.0fMpc".format(Z, 
		p_dict['nf'], USE_TS_IN_21CM, p_dict['eff'], ION_Tvir_MIN, ave, HII_DIM, BOX_LEN)
	k_ave = np.where(in_bin_ct>0, k_ave/in_bin_ct, 0.)
	ps_ave = np.where(in_bin_ct>0, ps/in_bin_ct, 0.)
	np.savez(IO_DIR+ps_fname, k_ave=k_ave, ps_ave=ps_ave)


	return k_ave, ps_ave


if __name__ == "__main__":
	k, ps = run()
	import IPython; IPython.embed()