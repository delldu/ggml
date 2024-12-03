#include "norm.cuh"

template <int block_size>
static __global__ void norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = (x[row*ncols + col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    int start = blockIdx.x * group_size;
    int end = start + group_size;

    start += threadIdx.x;

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    float variance = tmp / group_size;
    float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size>
static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
}

static void norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

static __device__ void gpu_do_norm_ext_f32(const float *src, float *dst, const int dim, const int64_t ne, 
    const int64_t src_offset, const int64_t src_step, const int64_t dst_offset, const int64_t dst_step, const float eps) {
    ggml_float sum, sum2;
    float *p, *sp, *dp, v, scale, mean, variance;
    int64_t s_offset, d_offset;

    sum = 0.0;
    s_offset = src_offset;
    for (int64_t i = 0; i < ne; i++) {
        p = (float *)((char *)src + s_offset);
        sum += (ggml_float)(*p);
        s_offset += src_step;
    }
    mean = sum/ne;

    sum2 = 0.0;
    s_offset = src_offset;
    d_offset = dst_offset;
    for (int64_t i = 0; i < ne; i++) {
        sp = (float *)((char *)src + s_offset);
        v = *sp - mean;
        dp = (float *)((char *)dst + d_offset);
        *dp = v;
        sum2 += (ggml_float)(v*v);

        s_offset += src_step;
        d_offset += dst_step;
    }

    variance = sum2/ne;
    scale = 1.0f/sqrtf(variance + eps);

    d_offset = dst_offset;
    for (int64_t i = 0; i < ne; i++) {
        dp = (float *)((char *)dst + d_offset);
        *dp = (*dp)*scale;
        d_offset += dst_step;
    }
}


static __global__ void norm_ext_f32(const float * src, float * dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim, const float eps) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == (dim == 0)?s_ne1:s_ne0
        return;
    }

    int64_t d_offset, s_offset;
    if (dim == 0) { // thread is running on axis i1 ..., so i1 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(0 /*i0*/, index, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(0 /*i0*/, index, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_norm_ext_f32(src, dst, dim, s_ne0, s_offset, s_nb0 /*s_step*/, d_offset, d_nb0 /*d_step*/, eps);
            }
        }
        return;
    }

    if (dim == 1) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_norm_ext_f32(src, dst, dim, s_ne1, s_offset, s_nb1 /*s_step*/, d_offset, d_nb1 /*d_step*/, eps);
            }
        }
        return;
    }

    if (dim == 2) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_norm_ext_f32(src, dst, dim, s_ne2, s_offset, s_nb2 /*s_step*/, d_offset, d_nb2 /*d_step*/, eps);
            }
        }
        return;
    }

    if (dim == 3) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i2 = 0; i2 < d_ne2; i2++) {
                s_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_norm_ext_f32(src, dst, dim, s_ne3, s_offset, s_nb3 /*s_step*/, d_offset, d_nb3 /*d_step*/, eps);
            }
        }
        return;
    }
    // GGML_ASSERT(dim >= 0 && dim < 4);
}

// dell_xxxx
#define CUDA_NORM_EXT_BLOCK_SIZE 256
static void norm_ext_f32_cuda(const float *src, float * dst, const int n, 
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim, const float eps, cudaStream_t stream) {

    int num_blocks = (n + CUDA_NORM_EXT_BLOCK_SIZE - 1) / CUDA_NORM_EXT_BLOCK_SIZE;

    norm_ext_f32<<<num_blocks, CUDA_NORM_EXT_BLOCK_SIZE, 0, stream>>>(src, dst, n, 
        s_ne0, s_ne1, s_ne2, s_ne3, s_nb0, s_nb1, s_nb2, s_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3, dim, eps);
}


static void group_norm_f32_cuda(const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
}

// dell_xxxx
void ggml_cuda_op_norm_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    // GGML_ASSERT(ggml_is_contiguous(src));

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(dim >= 0 && dim < 4);

    const int n = (dim == 0)? src->ne[1] : src->ne[0];

    norm_ext_f32_cuda(src_d, dst_d, n, 
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        dim, eps, stream);
}


void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
}
