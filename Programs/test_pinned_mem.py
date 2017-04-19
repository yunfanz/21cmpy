import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import resource

from pycuda.compiler import SourceModule

def run():
	# Create the kernel
	mod = SourceModule("""
	__global__ void test_kernel(int * data, int N)
	{
	  int tid = blockIdx.x*blockDim.x + threadIdx.x;
	  if(tid < N) {
	    data[tid] = tid + data[tid];
	  }
	}
	""")

	test_kernel = mod.get_function("test_kernel")

	# Create a stream
	stream = drv.Stream()

	# Problem size is N
	N = np.int64(32*1024*1024)
	shape, dtype = (2*N,), np.int32

	# Create N ints, aligned to the system page size and initialise to 100
	#a = drv.aligned_empty(shape=shape, dtype=dtype,
	#    alignment=resource.getpagesize())
	a_pin = drv.pagelocked_empty(shape=shape, dtype=dtype)
	a_gpu = drv.mem_alloc(a_pin.nbytes)
	#a_pin = drv.register_host_memory(a)
	a_pin[:] = 100

	for i in xrange(100):
		

		# Allocate number of bytes required by A on the GPU
		# Pin the host memory
		
		# Uncomment, this works
		#a_pin = drv.pagelocked_empty(shape=shape, dtype=dtype)
		#a_pin[:] = 100

		#assert np.all(a_pin == a)

		# Asynchronously copy pinned array data to the gpu array
		drv.memcpy_htod_async(a_gpu, a_pin[:N], stream)

		# Set up kernel dimensions
		nthreads, nblocks = 1024, N/1024

		# Call the kernel
		test_kernel(a_gpu, N,
		    block=(nthreads,1,1),
		    grid=(nblocks,1,1),
		    stream=stream)

		# Transfer gpu array data to pinned memory
		drv.memcpy_dtoh_async(a_pin, a_gpu, stream)

	# Synchronize
	#drv.Context.synchronize()

	# Check that results are the same
	#assert np.all(a_pin == np.arange(N) + 100)

if __name__ == "__main__":
	run()