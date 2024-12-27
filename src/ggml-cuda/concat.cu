#include "concat.cuh"

// contiguous kernels
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// non-contiguous kernel (slow)
static __global__ void concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3,
          int32_t   dim) {
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    const float * x;

    for (int i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const float * src0_d = (const float *)src0->data;
        const float * src1_d = (const float *)src1->data;

        float * dst_d = (float *)dst->data;

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_f32_cuda(
                        src0_d + i3 * (src0->nb[3] / 4),
                        src1_d + i3 * (src1->nb[3] / 4),
                        dst_d + i3 * ( dst->nb[3] / 4),
                        src0->ne[0], src0->ne[1], src0->ne[2],
                        dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d,           src0_d, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst_d + size0/4, src1_d, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        concat_f32_non_cont<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *)src0->data,
                (const char *)src1->data,
                (      char *)dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
                dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3], dim);
    }
}


static __global__ void cat_f32(const float * src, float * dst,
        const int n, const int dim, const int dim_c,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // for src ...
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, // for dst ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3) { // for dst ...
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) {
        return;
    }

    int s_0 = index % s_ne0; // W
    int s_1 = (index / s_ne0) % s_ne1; // H
    int s_2 = (index / (s_ne0 * s_ne1)) % s_ne2; // C
    int s_3 = (index / (s_ne0 * s_ne1 * s_ne2)) % s_ne3; // B
    int64_t s_offset, d_offset;

    s_offset = tensor_full_offset(s_0, s_1, s_2, s_3, s_nb0, s_nb1, s_nb2, s_nb3);
    if (dim == 0) {
        d_offset = tensor_full_offset(s_0 + dim_c, s_1, s_2, s_3, d_nb0, d_nb1, d_nb2, d_nb3);

        *(float *)((char *)dst + d_offset) = *(float *)((char *)src + s_offset);
        return;
    }

    if (dim == 1) {
        d_offset = tensor_full_offset(s_0, s_1 + dim_c, s_2, s_3, d_nb0, d_nb1, d_nb2, d_nb3);

        *(float *)((char *)dst + d_offset) = *(float *)((char *)src + s_offset);
        return;
    }

    if (dim == 2) {
        d_offset = tensor_full_offset(s_0, s_1, s_2 + dim_c, s_3, d_nb0, d_nb1, d_nb2, d_nb3);
        *(float *)((char *)dst + d_offset) = *(float *)((char *)src + s_offset);
        return;
    }

    if (dim == 3) {
        d_offset = tensor_full_offset(s_0, s_1, s_2, s_3 + dim_c, d_nb0, d_nb1, d_nb2, d_nb3);
        *(float *)((char *)dst + d_offset) = *(float *)((char *)src + s_offset);
        return;
    }

    // GGML_ASSERT(dim >= 0 && dim < 4);
}



#define CUDA_CAT_BLOCK_SIZE 256
static void cat_f32_cuda(const float * src, float * dst,
        const int n, const int dim, const int dim_c,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        cudaStream_t stream) {

    int num_blocks = (n + CUDA_CAT_BLOCK_SIZE - 1) / CUDA_CAT_BLOCK_SIZE;

    cat_f32<<<num_blocks, CUDA_CAT_BLOCK_SIZE, 0, stream>>>(src, dst, 
        n, dim, dim_c,
        s_ne0, s_ne1, s_ne2, s_ne3, 
        s_nb0, s_nb1, s_nb2, s_nb3, 
        d_nb0, d_nb1, d_nb2, d_nb3);
}

// dell_xxxx
void ggml_cuda_op_cat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    const int32_t n = ((int32_t *) dst->op_params)[0];
    const int32_t dim = ((int32_t *) dst->op_params)[1];

    int dim_c = 0;
    ggml_tensor *src;

    for (int i = 0; i < n; i++) {
        src = dst->src[i];
        const float * src_d = (const float *)src->data;

        cat_f32_cuda(src_d, dst_d, 
            ggml_nelements(src), dim, dim_c,
            src->ne[0], src->ne[1], src->ne[2], src->ne[3],
            src->nb[0], src->nb[1], src->nb[2], src->nb[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
            stream);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        dim_c += src->ne[dim];
    }
}
