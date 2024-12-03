#include "common.cuh"
#include "softmax.cuh"

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

template <bool vals_smem, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(const float * x, const T * mask, float * dst, const int ncols_par, const int nrows_y, const float scale, const float max_bias, const float m0, const float m1, uint32_t n_head_log2) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx % nrows_y; // broadcast the mask in the row dimension

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(max_bias, rowx/nrows_y, n_head_log2, m0, m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = vals_smem ? buf_iw + WARP_SIZE : dst + (int64_t)rowx*ncols;

    float max_val = -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        const int64_t iy = (int64_t)rowy*ncols + col;

        const float val = x[ix]*scale + (mask ? slope*t2f32(mask[iy]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        const int64_t idst = (int64_t)rowx*ncols + col;
        dst[idst] = vals[col] * inv_sum;
    }
}

template<typename T>
static void soft_max_f32_cuda(const float * x, const T * mask, float * dst, const int ncols_x, const int nrows_x, const int nrows_y, const float scale, const float max_bias, cudaStream_t stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t shmem = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    const uint32_t n_head      = nrows_x/nrows_y;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    if (shmem < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        switch (ncols_x) {
            case 32:
                soft_max_f32<true, 32, 32><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 64:
                soft_max_f32<true, 64, 64><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 128:
                soft_max_f32<true, 128, 128><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 256:
                soft_max_f32<true, 256, 256><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 512:
                soft_max_f32<true, 512, 512><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 1024:
                soft_max_f32<true, 1024, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 2048:
                soft_max_f32<true, 2048, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 4096:
                soft_max_f32<true, 4096, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            default:
                soft_max_f32<true, 0, 0><<<block_nums, block_dims, shmem, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
        }
    } else {
        const size_t shmem_low = WARP_SIZE*sizeof(float);
        soft_max_f32<false, 0, 0><<<block_nums, block_dims, shmem_low, stream>>>(x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
    }
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const void  * src1_d = src1 ? (const void *)src1->data : nullptr;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t ne00    = src0->ne[0];
    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    if (use_f16) {
        const half * src1_dd = (const half *)src1_d;

        soft_max_f32_cuda(src0_d, src1_dd, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, stream);
    } else {
        const float * src1_dd = (const float *)src1_d;

        soft_max_f32_cuda(src0_d, src1_dd, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, stream);
    }
}


static __device__ void gpu_do_softmax_f32(const float *src, float *dst, int dim, int64_t ne, 
    int64_t src_offset, int64_t src_step, int64_t dst_offset, int64_t dst_step) {

    float m, sum, *p, *sp, *dp;
    int64_t s_offset, d_offset;

    // get max ...
    m = -INFINITY;
    s_offset = src_offset;
    for (int64_t i = 0; i < ne; i++) {
        p = (float *)((char *)src + s_offset);
        if (*p > m) {
            m = *p;
        }
        s_offset += src_step;
    }

    // temp update ...
    sum = 0.0;
    s_offset = src_offset;
    d_offset = dst_offset;
    for (int64_t i = 0; i < ne; i++) {
        sp = (float *)((char *)src + s_offset);
        dp = (float *)((char *)dst + s_offset);

        *dp = expf(*sp - m);
        sum += *dp;

        s_offset += src_step;
        d_offset += dst_step;
    }

    // final update ...
    d_offset = dst_offset;
    for (int64_t i = 0; i < ne; i++) {
        p = (float *)((char *)dst + d_offset);
        *p = (*p)/sum;

        d_offset += dst_step;
    }
}


static __global__ void softmax_f32(const float * src, float * dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim) {

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
                gpu_do_softmax_f32(src, dst, dim, s_ne0, s_offset, s_nb0 /*s_step*/, d_offset, d_nb0 /*d_step*/);
            }
        }
        return;
    }

    if (dim == 1) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_softmax_f32(src, dst, dim, s_ne1, s_offset, s_nb1 /*s_step*/, d_offset, d_nb1 /*d_step*/);
            }
        }
        return;
    }

    if (dim == 2) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_softmax_f32(src, dst, dim, s_ne2, s_offset, s_nb2 /*s_step*/, d_offset, d_nb2 /*d_step*/);
            }
        }
        return;
    }

    if (dim == 3) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i2 = 0; i2 < d_ne2; i2++) {
                s_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_softmax_f32(src, dst, dim, s_ne3, s_offset, s_nb3 /*s_step*/, d_offset, d_nb3 /*d_step*/);
            }
        }
        return;
    }
    // GGML_ASSERT(dim >= 0 && dim < 4);
}

// dell_xxxx
#define CUDA_SOFTMAX_BLOCK_SIZE 256
static void softmax_f32_cuda(const float * src, float * dst, const int n, 
    int const s_ne0, int const s_ne1, int const s_ne2, int const s_ne3,
    int const s_nb0, int const s_nb1, int const s_nb2, int const s_nb3,
    int const d_ne0, int const d_ne1, int const d_ne2, int const d_ne3,
    int const d_nb0, int const d_nb1, int const d_nb2, int const d_nb3,
    const int dim, cudaStream_t stream) {
    int num_blocks = (n + CUDA_SOFTMAX_BLOCK_SIZE - 1) / CUDA_SOFTMAX_BLOCK_SIZE;

    softmax_f32<<<num_blocks, CUDA_SOFTMAX_BLOCK_SIZE, 0, stream>>>(src, dst, n, 
        s_ne0, s_ne1, s_ne2, s_ne3, s_nb0, s_nb1, s_nb2, s_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3, dim);
}


void ggml_cuda_op_softmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int dim = dst->op_params[0];
    GGML_ASSERT(dim >= 0 && dim < 4);

    const int n = (dim == 0)?src->ne[1]:src->ne[0];

    softmax_f32_cuda(src_d, dst_d, n, 
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        dim, stream);
}
