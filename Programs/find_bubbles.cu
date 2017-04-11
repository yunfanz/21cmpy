#define BLOCK_SIZE 8
#include <pycuda-complex.hpp>
#define E (float) (2.7182818284)
#define L_FACTOR (float) (0.620350491) // factor relating cube length to filter radius = (4PI/3)^(-1/3)
// Convert 3D index to 1D index.
#define INDEX(k,j,i,ld) ((k)*ld*ld + (j) * ld + (i))

// Texture memory for image.
texture<float,3> img;

// brute force real_tophat kernel
__global__ void real_tophat_kernel(float* ionized, const int w, float R, float S0)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w);
  if (j >= w || i >= w || k >= w || ionized[p] == 1.0) return;

  float rsq;
  float deltasum = 0;
  float deltac = %(DELTAC)s;
  float smin = %(SMIN)s;
  int count = 0;
  for (int kk = 0; kk < w; kk++) {
  	for (int jj = 0; jj < w; jj++) {
  		for (int ii = 0; ii < w; ii++){
  			rsq = (ii-i)*(ii-i)+(jj-j)*(jj-j)+(kk-k)*(kk-k);
  			if (rsq < R*R)
  			{
  				deltasum += tex3D(img,i,j,k);
  				count ++;
  			}
  		}
  	}
  }
  float delta0 = deltasum/count;
  float fcoll = 1 - erf((deltac - delta0)/sqrt(2*(smin - S0)));
  //ionized[p] = fcoll* %(ZETA)s;;
  if (fcoll >= 1./%(ZETA)s) ionized[p] = 1.0;
  else { ionized[p] = fcoll * %(ZETA)s; }
 }
 __global__ void k_tophat_kernel(float* ionized, const int w, float R, float S0)
{
	int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
	int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
	int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
	int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
	int p = INDEX(k,j,i,w);
	float ks = pow((9*3.14159/2),1.0/3) / R;
	if (j >= w || i >= w || k >= w || ionized[p] == 1.0) return;

	float rsq, r, y;
	float deltasum = 0;
	float deltac = %(DELTAC)s;
	float smin = %(SMIN)s;
	float count = 0;
	float W;
	for (int kk = 0; kk < w; kk++) {
		for (int jj = 0; jj < w; jj++) {
			for (int ii = 0; ii < w; ii++){
				rsq = (ii-i)*(ii-i)+(jj-j)*(jj-j)+(kk-k)*(kk-k);
				r = pow( rsq, float(0.5) );
				y = ks*r;
				W = (sin(y) - y*cos(y))/( 2*pow(3.14159,2)*pow(r,3) );
				deltasum += W * tex3D(img,i,j,k);
				count += W;
			}
		}
	}
	float delta0 = deltasum/count;
	float fcoll = 1 - erf((deltac - delta0)/sqrt(2*(smin - S0)));
	//ionized[p] = fcoll* %(ZETA)s;;
	if (fcoll >= 1./%(ZETA)s) ionized[p] = 1.0;
	else { ionized[p] = fcoll * %(ZETA)s; }
 }


__global__ void real_tophat(float* filter, int w, float R)
{
	int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
	int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
	int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
	int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
	int p = INDEX(k,j,i,w);
	if (j >= w || i >= w || k >= w) return;
	float rsq = (w/2-i)*(w/2-i)+(w/2-j)*(w/2-j)+(w/2-k)*(w/2-k);
	float vol = 4.0*3.1415926*R*R*R/3.0;
	if (rsq < R*R)
	{
		filter[p] = 1./vol;
	}
	else
	{
		filter[p] = 0;
	}
 }

 __global__ void HII_filter(pycuda::complex<float>* fourierbox, int w, int filter_type, float R)
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
__global__ void fcoll_kernel(float* fcollapse, float* smoothed, const int w, float denom)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w); 
  
  if (j >= w || i >= w || k >= w) return;

  float delta0 = smoothed[p];
  float deltac = %(DELTAC)s;
  float smin = %(SMIN)s;
  float fcoll = erfcf((deltac - delta0)/denom);
  //fcollapse[p] = (fcoll<1.0) ? fcoll : 1.0 ;
  fcollapse[p] = fcoll;
 }
__global__ void update_kernel(float* ionized, float* fcollapse, const int w)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w);
  
  if (j >= w || i >= w || k >= w || ionized[p] == 1) return;

  float fcoll = fcollapse[p];
  if (fcoll >= 1/%(ZETA)s) ionized[p] = 1.0;
 }

 __global__ void update_sphere_kernel(float* ionized, float* fcollapse, const int w, float R)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w);
  float rsq;
  if (j >= w || i >= w || k >= w) return;

  if (fcollapse[p] >= 1/%(ZETA)s) 
  {
  	if (R > BLOCK_SIZE && false)
  	{
  		for (int kk = 0; kk < w; kk++) {
			for (int jj = 0; jj < w; jj++) {
				for (int ii = 0; ii < w; ii++){
					rsq = (ii-i)*(ii-i)+(jj-j)*(jj-j)+(kk-k)*(kk-k);
					if (rsq < R*R) ionized[INDEX(kk,jj,ii,w)] = 1.0;
				}
			}
		}
  	}
  	else
  	{
		__shared__ float s_I[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
		s_I[INDEX(tz,ty,tx,BLOCK_SIZE)] = ionized[p];
		__syncthreads();
		int ttx, tty, ttz;
		int bbx1 = ((i-R)<0) ? 0 : floor(i-R);
		int bby1 = ((j-R)<0) ? 0 : floor(j-R);
		int bbz1 = ((k-R)<0) ? 0 : floor(k-R);
		int bbx2 = ((i+R)>w) ? w : ceil(i+R);
		int bby2 = ((j+R)>w) ? w : ceil(j+R);
		int bbz2 = ((k+R)>w) ? w : ceil(k+R);

		for (int kk = bbz1; kk < bbz2; kk++) {
			for (int jj = bby1; jj < bby2; jj++) {
				for (int ii = bbx1; ii < bbx2; ii++){
					rsq = (ii-i)*(ii-i)+(jj-j)*(jj-j)+(kk-k)*(kk-k);
					if (rsq < R*R) {
						ttx = ii - bdx * bx;  tty = jj - bdy * by; ttz = kk - bdz * bz;
						if (ttx<0 || tty<0 || ttz<0 ||
						    ttx>=bdx || tty>=bdy || ttz>=bdz) {
						    	ionized[INDEX(kk,jj,ii,w)] = 1.0;
						    }
						else {
							s_I[INDEX(ttz,tty,ttx,BLOCK_SIZE)] = 1.0;
						}
					}
				}
			}
		}
		__syncthreads();
		if (s_I[INDEX(tz,ty,tx,BLOCK_SIZE)] > 0) { ionized[p] = 1.0; }

  	}

 }
}

__global__ void final_kernel(float* ionized, float* fcollapse, const int w, float denom)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w); 
  
  if (j >= w || i >= w || k >= w || ionized[p] == 1) return;

  float fcoll = fcollapse[p];
  //ionized[p] = fcoll;
  if (fcoll >= 1/%(ZETA)s) ionized[p] = 1.0;
  else { 
  	ionized[p] = fcoll * %(ZETA)s; 
  }
 }

