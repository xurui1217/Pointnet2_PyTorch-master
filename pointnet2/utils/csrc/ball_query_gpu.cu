#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
					int nsample,
					const float *__restrict__ new_xyz,
					const float *__restrict__ xyz,
					int *__restrict__ idx) {
    int batch_index = blockIdx.x;
    //printf('batch_index is:',batch_index);
    xyz += batch_index * n * 3;  //batch_index*2048*3
    new_xyz += batch_index * m * 3;  //batch_index*512*3
    idx += m * nsample * batch_index; //batch_index*15*512

    int index = threadIdx.x;
    int stride = blockDim.x;

    float radius2 = radius * radius; //radii=[0.1, 0.2, 0.4],
    for (int j = index; j < m; j += stride) {
	float new_x = new_xyz[j * 3 + 0];
	float new_y = new_xyz[j * 3 + 1];
	float new_z = new_xyz[j * 3 + 2];

	for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
	    float x = xyz[k * 3 + 0];
	    float y = xyz[k * 3 + 1];
	    float z = xyz[k * 3 + 2];
	    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
		       (new_z - z) * (new_z - z);
	    if (d2 < radius2) {
		if (cnt == 0) {
		    for (int l = 0; l < nsample; ++l) {
			idx[j * nsample + l] = k;
		    }
		}
		idx[j * nsample + cnt] = k;
		++cnt;
	    }
	}
    }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
				     int nsample, const float *new_xyz,
				     const float *xyz, int *idx,
				     cudaStream_t stream) {

    cudaError_t err;
    query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
	b, n, m, radius, nsample, new_xyz, xyz, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}
