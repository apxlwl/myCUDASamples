
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include<E:\CUDA Samples\v10.2\matrixadd\lib\helper.h>
using namespace std;
#define BLOCKSIZE 1024;
int recursiveReduce(int* data, int const size)
{
	// terminate check
	if (size == 1) return data[0];
	// renew the stride
	int const stride = size / 2;
	if (size % 2 == 1)
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
		data[0] += data[size - 1];
	}
	else
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
	}
	// call
	return recursiveReduce(data, stride);
}
__global__ void warmup(int* g_idata, int* g_odata, unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the 
	int* idata = g_idata + blockIdx.x * blockDim.x;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}
__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	// 每块block的起始数据点
	int* idata = g_idata + blockDim.x * blockIdx.x;
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if ((tid % (2 * stride)) == 0) {
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int* idata = g_idata + blockIdx.x * blockDim.x;
	if (idx > n)
		return;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		//convert tid into local array index
		int index = 2 * stride * tid;
		if (index < blockDim.x)
		{
			idata[index] += idata[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int* idata = g_idata + blockIdx.x * blockDim.x;
	if (idx >= n)
		return;
	//in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{

		if (tid < stride)
		{
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved_share(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int* idata = g_idata + blockIdx.x * blockDim.x;
	if (idx >= n)
		return;
	__shared__ int shareidata[1024];
	//load in shared_memory
	shareidata[tid] = idata[tid];
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
	
		if (tid < stride)
		{
			shareidata[tid] += shareidata[tid + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = shareidata[0];
}

__global__ void reduceunroll(int* g_idata, int* g_odata, unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int* idata = g_idata + blockIdx.x * blockDim.x * 2;
	if (idx + blockDim.x < n)
	{
		g_idata[idx] += g_idata[idx + blockDim.x];

	}
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceunroll_threadunroll(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	// convert global data pointer to the local point of this block
	if (tid >= n)
		return;
	int* idata = g_idata + blockIdx.x * blockDim.x * 2;

	if (idx + blockDim.x < n) {
		g_idata[idx] += g_idata[idx + blockDim.x];
	}
	__syncthreads();

	//in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{

		if (tid < stride)
		{
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	// threads unroll
	if (tid < 32)
	{
		volatile int* vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char** argv)
{
	initDevice(0);
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps, 0);
	printf("CUDA device [%s]\n", deviceProps.name);
	std::cout << "SM的数量：" << deviceProps.multiProcessorCount << std::endl;
	std::cout << "每个线程块的共享内存大小：" << deviceProps.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "每个线程块的最大线程数：" << deviceProps.maxThreadsPerBlock << std::endl;
	std::cout << "每个SM的共享内存大小：" << deviceProps.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
	std::cout << "每个SM的最大线程数：" << deviceProps.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "每个SM的最大线程束数：" << deviceProps.maxThreadsPerMultiProcessor / 32 << std::endl;
	printf("[%s] - Starting...\n", argv[0]);
	bool bResult = false;
	//initialization

	int size = 1 << 24;
	int blocksize = BLOCKSIZE;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);
	}
	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	size_t inbytes = size * sizeof(int);
	size_t outbytes = grid.x* sizeof(int);

	int* inp, * outp;
	int* dummy = (int*)malloc(inbytes);
	cudaMallocManaged((void**)&inp, inbytes);
	cudaMallocManaged((void**)&outp, outbytes);
	initialData_int(dummy, size);

	//cpu reduction
	memcpy(inp, dummy, inbytes);
	double iStart, iElaps;
	int gpu_sum = 0;
	int cpu_sum = 0;
	iStart = cpuSecond();
	//cpu_sum = recursiveReduce(tmp, size);
	for (int i = 0; i < size; i++)
		cpu_sum += inp[i];
	printf("cpu sum:%d \n", cpu_sum);
	iElaps = cpuSecond() - iStart;
	printf("cpu reduce                 elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);


	//kernel 0: warmup the gpu
	memcpy(inp, dummy, inbytes);
	iStart = cpuSecond();
	warmup << <grid, block >> > (inp, outp, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += outp[i];
	printf("gpu warmup                 elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);
	////kernel 1:reduceNeighbored
	memcpy(inp, dummy, inbytes);
	iStart = cpuSecond();
	reduceNeighbored << <grid, block >> > (inp, outp, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += outp[i];
	printf("gpu reduceNeighbored       elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);
	////kernel 2:reduceNeighboredLess
	memcpy(inp, dummy, inbytes);
	iStart = cpuSecond();
	reduceNeighboredLess << <grid, block >> > (inp, outp, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += outp[i];
	printf("gpu reduceNeighboredLess   elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	////kernel 3:reduceInterleaved
	memcpy(inp, dummy, inbytes);
	iStart = cpuSecond();
	reduceInterleaved << <grid, block >> > (inp, outp, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += outp[i];
	printf("gpu reduceInterleaved      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);
	////kernel 3:reduceInterleaved_shared
	memcpy(inp, dummy, inbytes);
	iStart = cpuSecond();
	reduceInterleaved_share << <grid, block >> > (inp, outp, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += outp[i];
	printf("gpu reduceInterleavedshared      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);
	//kernel 4: unroll model
	memcpy(inp, dummy, inbytes);
	iStart = cpuSecond();
	reduceunroll << <grid.x / 2, block >> > (inp, outp, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 2; i++)
		gpu_sum += outp[i];
	printf("gpu blockunroll      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x / 2, block.x);

	//kernel 5: unroll thread
	memcpy(inp, dummy, inbytes);
	iStart = cpuSecond();
	reduceunroll_threadunroll << <grid.x / 2, block >> > (inp, outp, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 2; i++)
		gpu_sum += outp[i];
	printf("gpu threadunroll      elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x / 2, block.x);

	// free 
	free(dummy);
	cudaFree(inp);
	cudaFree(outp);

	return EXIT_SUCCESS;

}
// Helper function for using CUDA to add vectors in parallel.
