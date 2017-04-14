#define INDEX(k,j,i,ld) ((k)*ld*ld + (j) * ld + (i))
#define NUM_BINS %(NUM_BINS)s
#define PI 3.1415926

__global__ void pbox_kernel(float* deldel_T, int w, float* ps, float* k_ave, float* in_bin_ct, float* K)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w); int tp = INDEX(tz,ty,tx, bdx);
  if (j >= w || i >= w || k >= w) return;
  float k_x, k_y, k_z, k_mag, ps;
  int hw = w/2; 
  k_z = (k>hw) ? (k-w)*%(DELTAK)s : k*%(DELTAK)s;
  k_y = (j>hw) ? (j-w)*%(DELTAK)s : j*%(DELTAK)s;
  k_x = (i>hw) ? (i-w)*%(DELTAK)s : i*%(DELTAK)s;

  k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
  __shared__ s_K[NUM_BINS];
  if (tp < NUM_BINS) {s_K[tp] = K[tp]; }

  __syncthreads();

  int ct = 0;
  while (s_K[ct]< k_mag){ ct++; }
  in_bin_ct[ct] += 1.
  ps[ct] += pow(k_mag,float(3))*pow(abs(deldel_T[p]), float(2))/(2.0*PI*PI*%(VOLUME)s);
  k_ave[ct] += k_mag;
}