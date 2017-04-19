#define BLOCK_SIZE 8
#include <pycuda-complex.hpp>
#define INDEX(k,j,i,ld) ((k)*ld*ld + (j) * ld + (i))
#define E (float) (2.7182818284)

 __global__ void filter(pycuda::complex<float>* fourierbox, int w, int filter_type, float R)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w);
  if (j >= w || i >= w || k >= w) return;
  float k_x, k_y, k_z, k_mag, kR;
  int hw = w/2; 
  k_z = (k>hw) ? (k-w)*%(DELTAK)s : k*%(DELTAK)s;
  k_y = (j>hw) ? (j-w)*%(DELTAK)s : j*%(DELTAK)s;
  k_x = (i>hw) ? (i-w)*%(DELTAK)s : i*%(DELTAK)s;

  k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
  kR = k_mag*R; 
  switch (filter_type) {
    case 0: // real space top-hat
      if (kR > 1e-4){
        fourierbox[p] *= 3.0 * (sin(kR)/pow(kR, float(3)) - cos(kR)/pow(kR, float(2)));
      }
    case 1: // k-space top hat
      kR *= 0.413566994; // equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
      if (kR > 1){
        fourierbox[p] = 0;
      }
    case 2: // gaussian
      kR *= 0.643; // equates integrated volume to the real space top-hat
      fourierbox[p] *= pow(E, float(-kR*kR/2.0));
  }
}

__global__ void set_velocity(pycuda::complex<float>* fourierbox, pycuda::complex<float>* vbox, 
float dDdt_overD, int w, int comp)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w);
  if (j >= w || i >= w || k >= w) return;
  float k_x, k_y, k_z, k_sq;
  int hw = w/2; 
  k_z = (k>hw) ? (k-w)*%(DELTAK)s : k*%(DELTAK)s;
  k_y = (j>hw) ? (j-w)*%(DELTAK)s : j*%(DELTAK)s;
  k_x = (i>hw) ? (i-w)*%(DELTAK)s : i*%(DELTAK)s;

  k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
  if (k_sq == 0)
  {
    vbox[p] = 0.0;
    return;
  }
  pycuda::complex<float> I = pycuda::complex<float>(0.f, 1.f);
  pycuda::complex<float> factor;
  switch (comp) {
    case 0:
      factor = k_x*dDdt_overD*I/k_sq;
    case 1:
      factor = k_y*dDdt_overD*I/k_sq;
    case 2:
      factor = k_z*dDdt_overD*I/k_sq;
  vbox[p] = factor * fourierbox[p];
  }
}

__global__ void subsample(float* largebox, float* smallbox, int w, int sw, float pixel_factor)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,sw);
  int lk = floor(k*pixel_factor + 0.5);
  int lj = floor(j*pixel_factor + 0.5);
  int li = floor(i*pixel_factor + 0.5);

  if (j >= sw || i >= sw || k >= sw) return;
  smallbox[p] = largebox[INDEX(lk,lj,li,w)];
}

__global__ void move_mass(float* updated, float* deltax, float* vx, float* vy, float* vz, float init_growth_factor)
{
	int w = %(DIM)s;
	int sw = %(HII_DIM)s;

	int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
	int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
	int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
	int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
	int p = INDEX(k,j,i,w);
	float xf = (i+0.5)/(w + 0.0);
	float yf = (j+0.5)/(w + 0.0);
	float zf = (k+0.5)/(w + 0.0);
	int HII_i = int(i/%(PIXEL_FACTOR)s);
	int HII_j = int(j/%(PIXEL_FACTOR)s);
	int HII_k = int(k/%(PIXEL_FACTOR)s);
	xf += vx[INDEX(HII_k, HII_j, HII_i, sw)];
	yf += vy[INDEX(HII_k, HII_j, HII_i, sw)];
	zf += vz[INDEX(HII_k, HII_j, HII_i, sw)];
	xf *= sw; yf *= sw; zf *= sw;

	while (xf >= (float)sw){ xf -= sw;}
	while (xf < 0){ xf += sw;}
	while (yf >= (float)sw){ yf -= sw;}
	while (yf < 0){ yf += sw;}
	while (zf >= (float)sw){ zf -= sw;}
	while (zf < 0){ zf += sw;}
	int xi = xf; 
	int yi = yf; 
	int zi = zf;
	if (xi >= sw){ xi -= sw;}
	if (xi < 0) {xi += sw;}
	if (yi >= sw){ yi -= sw;}
	if (yi < 0) {yi += sw;}
	if (zi >= sw){ zi -= sw;}
	if (zi < 0) {zi += sw;}

	// move the mass
	//updated[INDEX(zi, yi, xi, sw)] += (1 + init_growth_factor*deltax[p]);
	//__syncthreads();
	atomicAdd(&updated[INDEX(zi, yi, xi, sw)], (1 + init_growth_factor*deltax[p]));
	
	}