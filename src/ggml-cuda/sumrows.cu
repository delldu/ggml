#include "sumrows.cuh"

static __global__ void k_sum_rows_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum);

    if (col == 0) {
        dst[row] = sum;
    }
}

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    k_sum_rows_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
}


static __global__ void k_mean_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum);

    if (col == 0) {
        dst[row] = sum/(float)ncols;
    }
}

// dell_xxxx
static __device__ void gpu_do_mean_ext_f32(const float *src, float *dst, const int64_t ne, 
    const int64_t src_offset, const int64_t src_step, const int64_t dst_offset) {
    ggml_float sum;
    float *p, mean;
    int64_t s_offset;

    sum = 0.0;
    s_offset = src_offset;
    for (int64_t i = 0; i < ne; i++) {
        p = (float *)((char *)src + s_offset);
        sum += (ggml_float)(*p);
        s_offset += src_step;
    }
    mean = sum/ne;
    *(float *)((char *)dst + dst_offset) = mean;
}


// dell_xxxx
static __global__ void mean_ext_f32(const float *src, float *dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim)
{
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
                
                gpu_do_mean_ext_f32(src, dst, s_ne0, s_offset, s_nb0 /*step*/, d_offset);
            }
        }
        return;
    }

    if (dim == 1) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_mean_ext_f32(src, dst, s_ne1, s_offset, s_nb1 /*step*/, d_offset);
            }
        }
        return;
    }

    if (dim == 2) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_mean_ext_f32(src, dst, s_ne2, s_offset, s_nb2 /*step*/, d_offset);
            }
        }
        return;
    }

    if (dim == 3) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i2 = 0; i2 < d_ne2; i2++) {
                s_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, d_nb0, d_nb1, d_nb2, d_nb3);
                gpu_do_mean_ext_f32(src, dst, s_ne3, s_offset, s_nb3 /*step*/, d_offset);
            }
        }
        return;
    }
    // GGML_ASSERT(dim >= 0 && dim < 4);
}


static void __device__ get_f32_argmax(const float *src, const int64_t ne, int64_t start_offset, int64_t step, int64_t *index)
{
    float m, *p;

    *index = 0;
    m = -INFINITY;
    for (int64_t i = 0; i < ne; i++) {
        p = (float *)((char *)src + start_offset);
        if (*p > m) {
            *index = i;
            m = *p;
        }
        start_offset += step;
    }
}


// dell_xxxx
static __global__ void argmax_ext_f32(const float * src, float * dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == (dim == 0)?s_ne1:s_ne0
        return;
    }

    int64_t d_offset, s_offset, s_index;
    if (dim == 0) { // thread is running on axis i1 ..., so i1 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(0 /*i0*/, index, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(0 /*i0*/, index, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);

                get_f32_argmax(src, s_ne0, s_offset, s_nb0 /*step*/, &s_index);
                *(float *)((char *)dst + d_offset) = (float)s_index;
            }
        }
        return;
    }

    if (dim == 1) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);

                get_f32_argmax(src, s_ne1, s_offset, s_nb1 /*step*/, &s_index);
                *(float *)((char *)dst + d_offset) = (float)s_index;
            }
        }
        return;
    }

    if (dim == 2) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, d_nb0, d_nb1, d_nb2, d_nb3);

                get_f32_argmax(src, s_ne2, s_offset, s_nb2 /*step*/, &s_index);
                *(float *)((char *)dst + d_offset) = (float)s_index;
            }
        }
        return;
    }

    if (dim == 3) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i2 = 0; i2 < d_ne2; i2++) {
                s_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, s_nb0, s_nb1, s_nb2, s_nb3);
                d_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, d_nb0, d_nb1, d_nb2, d_nb3);

                get_f32_argmax(src, s_ne3, s_offset, s_nb3 /*step*/, &s_index);
                *(float *)((char *)dst + d_offset) = (float)s_index;
            }
        }
        return;
    }
    // GGML_ASSERT(dim >= 0 && dim < 4);
}

static void mean_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    k_mean_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
}

// dell_xxxx
#define CUDA_MEAN_EXT_BLOCK_SIZE 256
static void mean_ext_f32_cuda(const float * src, float * dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim, cudaStream_t stream) {
    int num_blocks = (n + CUDA_MEAN_EXT_BLOCK_SIZE - 1) / CUDA_MEAN_EXT_BLOCK_SIZE;
    mean_ext_f32<<<num_blocks, CUDA_MEAN_EXT_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        s_ne0, s_ne1, s_ne2, s_ne3, s_nb0, s_nb1, s_nb2, s_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3,
        dim);
}


#define CUDA_ARGMAX_EXT_BLOCK_SIZE 256
static void argmax_ext_f32_cuda(const float * src, float * dst, const int n, 
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
    const int dim, cudaStream_t stream) {
    int num_blocks = (n + CUDA_ARGMAX_EXT_BLOCK_SIZE - 1) / CUDA_ARGMAX_EXT_BLOCK_SIZE;

    argmax_ext_f32<<<num_blocks, CUDA_ARGMAX_EXT_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        s_ne0, s_ne1, s_ne2, s_ne3, s_nb0, s_nb1, s_nb2, s_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3,
        dim);
}


void ggml_cuda_op_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    mean_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
}

// dell_xxxx
void ggml_cuda_op_mean_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));

    const int dim = dst->op_params[0];
    GGML_ASSERT(dim >= 0 && dim < 4);
    const int n = (dim == 0)? src->ne[1] : src->ne[0];

    mean_ext_f32_cuda(src_d, dst_d, n,
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        dim, stream);
}

// dell_add
static __global__ void max_f32(const float *src, float *dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == (dim == 0)?s_ne1:s_ne0
        return;
    }
    float m, *p;
    int64_t d_offset, s_offset;
    if (dim == 0) { // thread is running on axis i1 ..., so i1 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(0 /*i0*/, index, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                p = (float *)((char *)src + s_offset);
                m = *p;
                for (int64_t i0 = 1; i0 < s_ne0; i0++) {
                    s_offset += s_nb0;
                    p = (float *)((char *)src + s_offset);
                    m = (*p > m)? *p : m;
                }
                d_offset = tensor_full_offset(0 /*i0*/, index, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + d_offset) = m;
            }
        }
        return;
    }

    if (dim == 1) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                p = (float *)((char *)src + s_offset);
                m = *p;
                for (int64_t i1 = 1; i1 < s_ne1; i1++) {
                    s_offset += s_nb1;
                    p = (float *)((char *)src + s_offset);
                    m = (*p > m)? *p : m;
                }
                d_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + d_offset) = m;
            }
        }
        return;
    }

    if (dim == 2) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                p = (float *)((char *)src + s_offset);
                m = *p;
                for (int64_t i2 = 1; i2 < s_ne2; i2++) {
                    s_offset += s_nb2;
                    p = (float *)((char *)src + s_offset);
                    m = (*p > m)? *p : m;
                }
                d_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + d_offset) = m;
            }
        }
        return;
    }

    if (dim == 3) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i2 = 0; i2 < d_ne2; i2++) {
                s_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, s_nb0, s_nb1, s_nb2, s_nb3);
                p = (float *)((char *)src + s_offset);
                m = *p;
                for (int64_t i3 = 1; i3 < s_ne3; i3++) {
                    s_offset += s_nb3;
                    p = (float *)((char *)src + s_offset);
                    m = (*p > m)? *p : m;
                }
                d_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + d_offset) = m;
            }
        }
        return;
    }
    // GGML_ASSERT(dim >= 0 && dim < 4);
}

#define CUDA_MAX_BLOCK_SIZE 256
static void max_f32_cuda(const float * src, float * dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim, cudaStream_t stream) {
    int num_blocks = (n + CUDA_MAX_BLOCK_SIZE - 1) / CUDA_MAX_BLOCK_SIZE;
    max_f32<<<num_blocks, CUDA_MAX_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        s_ne0, s_ne1, s_ne2, s_ne3, s_nb0, s_nb1, s_nb2, s_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3,
        dim);
}

// dell_add
void ggml_cuda_op_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    GGML_ASSERT(dim >= 0 && dim < 4);
    const int n = (dim == 0)? src->ne[1] : src->ne[0];

    max_f32_cuda(src_d, dst_d, n,
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        dim, stream);
}

// dell_add
static __global__ void min_f32(const float *src, float *dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == (dim == 0)?s_ne1:s_ne0
        return;
    }
    float m, *p;
    int64_t d_offset, s_offset;
    if (dim == 0) { // thread is running on axis i1 ..., so i1 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(0 /*i0*/, index, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                p = (float *)((char *)src + s_offset);
                m = *p;
                for (int64_t i0 = 1; i0 < s_ne0; i0++) {
                    s_offset += s_nb0;
                    p = (float *)((char *)src + s_offset);
                    m = (*p < m)? *p : m;
                }
                d_offset = tensor_full_offset(0 /*i0*/, index, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + d_offset) = m;
            }
        }
        return;
    }

    if (dim == 1) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i2 = 0; i2 < d_ne2; i2++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                p = (float *)((char *)src + s_offset);
                m = *p;
                for (int64_t i1 = 1; i1 < s_ne1; i1++) {
                    s_offset += s_nb1;
                    p = (float *)((char *)src + s_offset);
                    m = (*p < m)? *p : m;
                }
                d_offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + d_offset) = m;
            }
        }
        return;
    }

    if (dim == 2) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i3 = 0; i3 < d_ne3; i3++) {
                s_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, s_nb0, s_nb1, s_nb2, s_nb3);
                p = (float *)((char *)src + s_offset);
                m = *p;
                for (int64_t i2 = 1; i2 < s_ne2; i2++) {
                    s_offset += s_nb2;
                    p = (float *)((char *)src + s_offset);
                    m = (*p < m)? *p : m;
                }
                d_offset = tensor_full_offset(index, i1, 0 /*i2*/, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + d_offset) = m;
            }
        }
        return;
    }

    if (dim == 3) { // thread is running on axis i0 ..., so i0 == index
        for (int64_t i1 = 0; i1 < d_ne1; i1++) {
            for (int64_t i2 = 0; i2 < d_ne2; i2++) {
                s_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, s_nb0, s_nb1, s_nb2, s_nb3);
                p = (float *)((char *)src + s_offset);
                m = *p;
                for (int64_t i3 = 1; i3 < s_ne3; i3++) {
                    s_offset += s_nb3;
                    p = (float *)((char *)src + s_offset);
                    m = (*p < m)? *p : m;
                }
                d_offset = tensor_full_offset(index, i1, i2, 0 /*i3*/, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + d_offset) = m;
            }
        }
        return;
    }
    // GGML_ASSERT(dim >= 0 && dim < 4);
}

#define CUDA_MIN_BLOCK_SIZE 256
static void min_f32_cuda(const float * src, float * dst, const int n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int dim, cudaStream_t stream) {
    int num_blocks = (n + CUDA_MIN_BLOCK_SIZE - 1) / CUDA_MIN_BLOCK_SIZE;
    min_f32<<<num_blocks, CUDA_MIN_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        s_ne0, s_ne1, s_ne2, s_ne3, s_nb0, s_nb1, s_nb2, s_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3,
        dim);
}

void ggml_cuda_op_min(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    GGML_ASSERT(dim >= 0 && dim < 4);
    const int n = (dim == 0)? src->ne[1] : src->ne[0];

    min_f32_cuda(src_d, dst_d, n,
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        dim, stream);
}


// dell_xxxx
static __global__ void corr_patch_sum_f32(float *dst, const float *f1, const float *f2, const int n,
    const int f1_ne0, const int f1_ne1, const int f1_ne2, const int f1_ne3,
    const int f1_nb0, const int f1_nb1, const int f1_nb2, const int f1_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int patch_size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // index === channel C
        return;
    }

    float f1_value, f2_value, sum;
    int64_t start_i, start_j, f1_offset, f2_offset, dst_offset;

    start_i = index/patch_size - patch_size/2; // -top_pad
    start_j = index%patch_size - patch_size/2; // -left_pad

    for (int64_t i0 = 0; i0 < f1_ne0; i0++) {
        for (int64_t i1 = 0; i1 < f1_ne1; i1++) {
            for (int64_t i3 = 0; i3 < f1_ne3; i3++) {
                // sum on dim C
                sum = 0.0f;
                for (int64_t i2 = 0; i2 < f1_ne2; i2++) {
                    if (i0 + start_j < 0 || i1 + start_i < 0 || i0 + start_j >= f1_ne0 || i1 + start_i >= f1_ne1)
                        continue;

                    f1_offset = tensor_full_offset(i0, i1, i2, i3, f1_nb0, f1_nb1, f1_nb2, f1_nb3);
                    f2_offset = tensor_full_offset(i0 + start_j, i1 + start_i, i2, i3, f1_nb0, f1_nb1, f1_nb2, f1_nb3);
                    f1_value = *(float *)((char *)f1 + f1_offset);
                    f2_value = *(float *)((char *)f2 + f2_offset);
                    sum += f1_value * f2_value;
                }
                dst_offset = tensor_full_offset(i0, i1, index /*i2*/, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + dst_offset) = sum;
            }
        }
    }
}

// dell_xxxx
#define CUDA_CORR_BLOCK_SIZE 256
static void corr_patch_sum_f32_cuda(float * dst, const float * f1, const float * f2, const int n,
    const int f1_ne0, const int f1_ne1, const int f1_ne2, const int f1_ne3,
    const int f1_nb0, const int f1_nb1, const int f1_nb2, const int f1_nb3,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int patch_size, cudaStream_t stream) {

    int num_blocks = (n + CUDA_CORR_BLOCK_SIZE - 1) / CUDA_CORR_BLOCK_SIZE;
    corr_patch_sum_f32<<<num_blocks, CUDA_CORR_BLOCK_SIZE, 0, stream>>>(dst, f1, f2, n,
        f1_ne0, f1_ne1, f1_ne2, f1_ne3, f1_nb0, f1_nb1, f1_nb2, f1_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3, patch_size);
}

void ggml_cuda_op_corr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * f1 = dst->src[0];
    const ggml_tensor * f2 = dst->src[1];
    const float * f1_d = (const float *)f1->data;
    const float * f2_d = (const float *)f2->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(f1->type == GGML_TYPE_F32);
    GGML_ASSERT(f2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int patch_size = dst->op_params[0];
    const int n = dst->ne[2]; // channel dim

    corr_patch_sum_f32_cuda(dst_d, f1_d, f2_d, n,
        f1->ne[0], f1->ne[1], f1->ne[2], f1->ne[3],
        f1->nb[0], f1->nb[1], f1->nb[2], f2->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        patch_size, stream);
}

static __global__ void global_attn_f32(float *dst, const float *src, const int n,
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int H1, const int W1, const int H2, const int W2, const int D2)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // index === channel C
        return;
    }

    // Torch implement ...
    // # for i in range(H1*W1):
    // #     for j in range(H2 * W2):
    // #         y_t = i//W1 -j//W2 + D2
    // #         x_t = i%W1 - j%W2 + D2
    // #         local_mask[i, j] = abs(y_t) <= D2 and abs(x_t) <= D2
    float *p;
    int64_t dst_offset, local_index, y_t, x_t;

    local_index = 0;
    for (int64_t j = 0; j < H2 * W2; j++) { // i0
        y_t = index / W1 - j / W2 + D2;
        x_t = index % W1 - j % W2 + D2;
        dst_offset = tensor_full_offset(j /*i0*/, index/*i1*/, 0, 0, d_nb0, d_nb1, d_nb2, d_nb3);
        p = (float *)((char *)dst + dst_offset);
        if (y_t >= -D2 && y_t <= D2 && x_t >= -D2 && x_t <= D2) {
            *p = src[local_index++]; // ggml_is_contiguous(src)
        } else {
            *p = 0.0f;
        }         
    }
}


#define CUDA_GLOBAL_ATTN_BLOCK_SIZE 256
static void global_attn_f32_cuda(float * dst, const float *src, const int n,
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
    const int H1, const int W1, const int H2, const int W2, const int D2, cudaStream_t stream) {
    int num_blocks = (n + CUDA_GLOBAL_ATTN_BLOCK_SIZE - 1) / CUDA_GLOBAL_ATTN_BLOCK_SIZE;
    global_attn_f32<<<num_blocks, CUDA_GLOBAL_ATTN_BLOCK_SIZE, 0, stream>>>(dst, src, n,
        d_nb0, d_nb1, d_nb2, d_nb3, H1, W1, H2, W2, D2);
}

// dell_xxxx
void ggml_cuda_op_global_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));

    const int H1 = dst->op_params[0];
    const int W1 = dst->op_params[1];
    const int H2 = dst->op_params[2];
    const int W2 = dst->op_params[3];
    const int D2 = dst->op_params[4];

    const int n = H1 * W1; // thread on H1*W1

    global_attn_f32_cuda(dst_d, src_d, n,
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], H1, W1, H2, W2, D2, stream);
}


// dell_xxxx
void ggml_cuda_op_argmax_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    const int dim = dst->op_params[0];
    GGML_ASSERT(dim >= 0 && dim < 4);

    const int n = (dim == 0)? src->ne[1] : src->ne[0];

    argmax_ext_f32_cuda(src_d, dst_d, n, 
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        dim, stream);
}

// dell_xxxx
static __global__ void cumsum_f32_kernel(const float*src, float* dst, const int n,
        const int ne0, const int ne1, const int ne2, const int ne3, // dst, dst share ...
        const int nb0, const int nb1, const int nb2, const int nb3, // src, dst share ...
        const int dim) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == (dim == 0)?ne[1]:ne[0]
        return;
    }
    int64_t offset;
    float sum;
    if (dim == 0) { // cumsum on i0 -- W
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {// i1 == index
                sum = 0.0;
                offset = tensor_full_offset(0 /*i0*/, index, i2, i3, nb0, nb1, nb2, nb3);
                // i3 * nb3 + i2 * nb2 + index * nb1; // byte offset
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    // offset = i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0;
                    sum += *(float *)((char *)src + offset);
                    *(float *)((char *)dst + offset) = sum; // Save
                    offset += nb0;
                } // i0
            } // i2
        } // i3
        return;
    }

    if (dim == 1) { // csum on i1 -- H
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {// i0 == index
                sum = 0.0;
                offset = tensor_full_offset(index, 0 /*i1*/, i2, i3, nb0, nb1, nb2, nb3);
                // i3 * nb3 + i2 * nb2 + index * nb0; // byte offset
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    // offset = i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0;
                    sum += *(float *)((char *)src + offset);
                    *(float *)((char *)dst + offset) = sum; // Save
                    offset += nb1;
                } // i1
            } // i2
        } // i3
        return;
    }

    // GGML_ASSERT(dim == 0 || dim == 1);
}

#define CUDA_CUMSUM_BLOCK_SIZE 256
static void cumsum_f32_cuda(const float* src, float* dst, const int n,
        const int ne0, const int ne1, const int ne2, const int ne3,
        const int nb0, const int nb1, const int nb2, const int nb3,
        const int dim, cudaStream_t stream) {
    int num_blocks = (n + CUDA_CUMSUM_BLOCK_SIZE - 1) / CUDA_CUMSUM_BLOCK_SIZE;
    cumsum_f32_kernel<<<num_blocks, CUDA_CUMSUM_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3, dim);
}

// dell_xxxx
void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int dim = dst->op_params[0];
    GGML_ASSERT(dim == 0 || dim == 1);

    int n = (dim == 0)? dst->ne[1] : dst->ne[0];
    cumsum_f32_cuda(src_d, dst_d, n, 
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        dim, stream);
}

static __global__ void norm2_f32_kernel(const float*src, float* dst, const int n,
        const int ne0, const int ne1, const int ne2, const int ne3, // src ne ...
        const int nb0, const int nb1, const int nb2, const int nb3, // src nb ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // dst nb ...
        const int dim) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == (dim == 0)?ne[1]:ne[0]
        return;
    }
    int64_t offset;
    float sum, x;
    if (dim == 0) { // norm2 on i0
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {
                // i1 == index
                sum = 0.0;
                offset = tensor_full_offset(0, index /*i1*/, i2, i3, nb0, nb1, nb2, nb3);
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    x = *(float *)((char *)src + offset);
                    sum += x*x;
                    offset += nb0;
                } // i0
                offset = tensor_full_offset(0, index /*i1*/, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + offset) = sqrtf(sum); // Save
            } // i2
        } // i3
        return;
    }

    if (dim == 1) { // norm2 on i1
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) { // i0 == index
                sum = 0.0;
                offset = tensor_full_offset(index /*i0*/, 0 /*i1*/, i2, i3, nb0, nb1, nb2, nb3);
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    x = *(float *)((char *)src + offset);
                    sum += x*x;
                    offset += nb1;
                } // i1
                offset = tensor_full_offset(index /*i0*/, 0 /*i1*/, i2, i3, d_nb0, d_nb1, d_nb2, d_nb3);
                *(float *)((char *)dst + offset) = sqrtf(sum); // Save
            } // i2
        } // i3
        return;
    }
    // GGML_ASSERT(dim == 0 || dim == 1);
}

#define CUDA_NORM2_BLOCK_SIZE 256
static void norm2_f32_cuda(const float* src, float* dst, const int n,
        const int ne0, const int ne1, const int ne2, const int ne3,
        const int nb0, const int nb1, const int nb2, const int nb3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // dst ...
        const int dim, cudaStream_t stream) {
    int num_blocks = (n + CUDA_NORM2_BLOCK_SIZE - 1) / CUDA_NORM2_BLOCK_SIZE;
    norm2_f32_kernel<<<num_blocks, CUDA_NORM2_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3, d_nb0, d_nb1, d_nb2, d_nb3, dim);
}


// dell_xxxx
void ggml_cuda_op_norm2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int dim = dst->op_params[0];
    GGML_ASSERT(dim == 0 || dim == 1);

    int n = (dim == 0)? src->ne[1] : src->ne[0];
    norm2_f32_cuda(src_d, dst_d, n, 
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        dim, stream);
}
