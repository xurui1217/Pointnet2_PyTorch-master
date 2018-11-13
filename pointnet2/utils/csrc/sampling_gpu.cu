#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "sampling_gpu.h"

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gather_points_kernel(int b, int c, int n, int m,
				     const float *__restrict__ points,
				     const int *__restrict__ idx,
				     float *__restrict__ out) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
	for (int l = blockIdx.y; l < c; l += gridDim.y) {
	    for (int j = threadIdx.x; j < m; j += blockDim.x) {
		int a = idx[i * m + j];
		out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
	    }
	}
    }
}

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
				  const float *points, const int *idx,
				  float *out, cudaStream_t stream) {

    cudaError_t err;
    gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0, stream>>>(
	b, c, n, npoints, points, idx, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
					  const float *__restrict__ grad_out,
					  const int *__restrict__ idx,
					  float *__restrict__ grad_points) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
	for (int l = blockIdx.y; l < c; l += gridDim.y) {
	    for (int j = threadIdx.x; j < m; j += blockDim.x) {
		int a = idx[i * m + j];
		atomicAdd(grad_points + (i * c + l) * n + a,
			  grad_out[(i * c + l) * m + j]);
	    }
	}
    }
}

void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
				       const float *grad_out, const int *idx,
				       float *grad_points,
				       cudaStream_t stream) {

    cudaError_t err;
    gather_points_grad_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
				stream>>>(b, c, n, npoints, grad_out, idx,
					  grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
			 int idx1, int idx2) {
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
    if (m <= 0)
	return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;  //只有0和1 两个batch test的时候
    dataset += batch_index * n * 3;
    temp += batch_index * n; //确定temp的大小
    idxs += batch_index * m; //确定idxs的大小

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
	idxs[0] = old; //让第1个点为初始点，设为idx[0]

    __syncthreads();
    for (int j = 1; j < m; j++) {
	int besti = 0;
	float best = -1;
	float x1 = dataset[old * 3 + 0];
	float y1 = dataset[old * 3 + 1];
	float z1 = dataset[old * 3 + 2];
	for (int k = tid; k < n; k += stride) {  //在dataset中顺序搜索并且计算点和old点之间的距离，k为第几个点
	    float x2, y2, z2;
	    x2 = dataset[k * 3 + 0];
	    y2 = dataset[k * 3 + 1];
	    z2 = dataset[k * 3 + 2];
	    float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
	    if (mag <= 1e-3)
		continue;

	    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
		      (z2 - z1) * (z2 - z1);

	    float d2 = min(d, temp[k]);  //temp[k]初始化无限大
	    temp[k] = d2;
	    besti = d2 > best ? k : besti;  //让besti为距离最大的点
	    best = d2 > best ? d2 : best;  //best为最大的距离
	}
	dists[tid] = best;
	dists_i[tid] = besti;
	__syncthreads();

	if (block_size >= 512) {
	    if (tid < 256) {
		__update(dists, dists_i, tid, tid + 256);
	    }
	    __syncthreads();
	}
	if (block_size >= 256) {
	    if (tid < 128) {
		__update(dists, dists_i, tid, tid + 128);
	    }
	    __syncthreads();
	}
	if (block_size >= 128) {
	    if (tid < 64) {
		__update(dists, dists_i, tid, tid + 64);
	    }
	    __syncthreads();
	}
	if (block_size >= 64) {
	    if (tid < 32) {
		__update(dists, dists_i, tid, tid + 32);
	    }
	    __syncthreads();
	}
	if (block_size >= 32) {
	    if (tid < 16) {
		__update(dists, dists_i, tid, tid + 16);
	    }
	    __syncthreads();
	}
	if (block_size >= 16) {
	    if (tid < 8) {
		__update(dists, dists_i, tid, tid + 8);
	    }
	    __syncthreads();
	}
	if (block_size >= 8) {
	    if (tid < 4) {
		__update(dists, dists_i, tid, tid + 4);
	    }
	    __syncthreads();
	}
	if (block_size >= 4) {
	    if (tid < 2) {
		__update(dists, dists_i, tid, tid + 2);
	    }
	    __syncthreads();
	}
	if (block_size >= 2) {
	    if (tid < 1) {
		__update(dists, dists_i, tid, tid + 1);
	    }
	    __syncthreads();
	}

	old = dists_i[0];
	if (tid == 0)
	    idxs[j] = old;
    }
}

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
					    const float *dataset, float *temp,
					    int *idxs, cudaStream_t stream) {

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
    case 512:
	furthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 256:
	furthest_point_sampling_kernel<256><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 128:
	furthest_point_sampling_kernel<128><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 64:
	furthest_point_sampling_kernel<64><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 32:
	furthest_point_sampling_kernel<32><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 16:
	furthest_point_sampling_kernel<16><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 8:
	furthest_point_sampling_kernel<8><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 4:
	furthest_point_sampling_kernel<4><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 2:
	furthest_point_sampling_kernel<2><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    case 1:
	furthest_point_sampling_kernel<1><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
	break;
    default:
	furthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(
	    b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}
