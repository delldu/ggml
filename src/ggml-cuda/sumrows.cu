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
