/*  ALLEN DANIEL
 *  YESA
 *  ALLENDAN
 */

#ifndef A3_HPP
#define A3_HPP

#include<math.h>
#include<cuda_runtime_api.h>

__global__
void calculate_block(int n, float h, float a, float *x, float *y);
__global__
void calculate_over_block(float *x, float *y);
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) 
{
int size = n * sizeof(float);
float *d_x;

cudaMalloc(&d_x, size);

cudaMemcpy(d_x, &x[0], size, cudaMemcpyHostToDevice);

for(int i=0;i<n;i++)
	{
		float *d_y_temp,*d_y;
		cudaMalloc(&d_y_temp,((n+1023)/1024)*sizeof(float));
		cudaMalloc(&d_y,sizeof(float));
		calculate_block<<<(n+1023)/1024,1024,1024*sizeof(float)>>>(n,h,x[i],d_x,d_y_temp);
		cudaThreadSynchronize();
		calculate_over_block<<<1,(n+1023)/1024,((n+1023)/1024)*sizeof(float)>>>(d_y_temp,d_y);
		cudaMemcpy(&y[i],d_y,sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_y);
		cudaFree(d_y_temp);
	}
cudaFree(d_x);
} // gaussian_kde

//Used to calculate the value for different blocks simultaneously
__global__
void calculate_block(int n, float h, float a, float *x, float *y_temp) 
{
 extern __shared__ float sdata[];
 int tid = threadIdx.x;
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 sdata[tid] = (1/(n*h))*(1/pow((2*(22/7)),0.5))*(exp(-(((a-x[i])/h)*((a-x[i])/h))/2));
 __syncthreads();

for (int s = blockDim.x >> 1; s > 0; s >>= 1) 
	{
  		if(tid < s) sdata[tid] += sdata[tid + s];
	 	__syncthreads();
 	}

 if (tid == 0) y_temp[blockIdx.x] = sdata[0];

}

//Bringing the calculated value to a single block and summing it up.
__global__
void calculate_over_block(float *x, float *y)
{
 extern __shared__ float sdata[];
 int tid = threadIdx.x;
 int i = blockIdx.x * blockDim.x  + threadIdx.x;
 sdata[tid] = x[i];
 __syncthreads();

for (int s = blockDim.x >> 1; s > 0; s >>= 1) 
	{
  		if(tid < s) sdata[tid] += sdata[tid + s];
	 	__syncthreads();
 	}

 if (tid == 0) y[0] = sdata[0];
}


#endif // A3_HPP


