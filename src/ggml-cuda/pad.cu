#include "pad.cuh"

static __global__ void pad_f32(const float * x, float * dst, const int ne0, const int ne00, const int ne01, const int ne02, const int ne03) {
    // blockIdx.z: idx of ne2*ne3, aka ne02*ne03
    // blockIdx.y: idx of ne1
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;
    if (nidx < ne00 && blockIdx.y < ne01 && blockIdx.z < ne02*ne03) {
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = 0.0f;
    }
}

static __global__ void replication_pad2d_f32(const float * x, float * dst, const int ne0, 
        const int ne00, const int ne01, const int ne02, const int ne03,
        const int left, const int right, const int top, const int bottom) {
    // blockIdx.z: idx of ne2*ne3, aka ne02*ne03
    // blockIdx.y: idx of ne1
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_src, nidx_src, blky_src;
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y; // blockIdx.z == C * B, gridDim.y == W * H

    // if (nidx < ne00 && blockIdx.y < ne01 && blockIdx.z < ne02*ne03) {
    //     int offset_src =
    //         nidx +
    //         blockIdx.y * ne00 +
    //         blockIdx.z * ne00 * ne01;
    //     dst[offset_dst] = x[offset_src];
    // } else {
    //     dst[offset_dst] = 0.0f;
    // }
    if ((nidx >= left && nidx < ne00 + left /*W*/) && (blockIdx.y >= top && blockIdx.y < ne01 + top) /*H*/) {
        nidx_src = nidx - left; // W
        blky_src = blockIdx.y - top; // H
    } else {
        nidx_src = (nidx < left)? 0 : ne00 - 1; // W
        blky_src = (blockIdx.y < top)? 0 : ne01 - 1; // H
    }

    offset_src = 
        nidx_src +
        blky_src * ne00 +
        blockIdx.z * ne00 * ne01;
    dst[offset_dst] = x[offset_src];
}

static __global__ void deconv_pad2d_f32(const float * x, float * dst, 
        const int x_ne0, const int x_ne1, const int x_ne2, const int x_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        const int stride) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= x_ne0 * x_ne1 * x_ne2 * x_ne3) {
        return;
    }

    // x index ...
    int x0 = index % x_ne0;
    int x1 = (index / x_ne0) % x_ne1;
    int x2 = (index / (x_ne0 * x_ne1)) % x_ne2;
    int x3 = (index / (x_ne0 * x_ne1 * x_ne2)) % x_ne3;

    // dst index ...
    int d0 = x0 * stride; // W
    int d1 = x1 * stride; // H
    int d2 = x2;
    int d3 = x3;

    *(float *)((char *)dst + d3 * d_nb3 + d2 * d_nb2 + d1 * d_nb1 + d0 * d_nb0) = x[index];
}


static void pad_f32_cuda(const float * x, float * dst,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int ne0, const int ne1, const int ne2, const int ne3, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2*ne3);
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(x, dst, ne0, ne00, ne01, ne02, ne03);
}

static void replication_pad2d_f32_cuda(const float * x, float * dst,
    const int ne00, const int ne01, const int ne02, const int ne03, // src
    const int ne0, const int ne1, const int ne2, const int ne3, // dst
    const int left, const int right, const int top, const int bottom, cudaStream_t stream) {
    GGML_ASSERT(ne0 == ne00 + left + right); // W
    GGML_ASSERT(ne1 == ne01 + top + bottom); // H
    GGML_ASSERT(ne2 == ne02); // C
    GGML_ASSERT(ne3 == ne03); // B

    int num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2*ne3);
    replication_pad2d_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(x, dst, ne0, ne00, ne01, ne02, ne03, left, right, top, bottom);
}

static void deconv_pad2d_f32_cuda(const float * x, float * dst,
    const int x_ne0, const int x_ne1, const int x_ne2, const int x_ne3, // src
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // dst
    const int stride, cudaStream_t stream) {
    int num_blocks = (x_ne0 * x_ne1 * x_ne2 * x_ne2 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    deconv_pad2d_f32<<<num_blocks, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(x, dst, 
        x_ne0, x_ne1, x_ne2, x_ne3, d_nb0, d_nb1, d_nb2, d_nb3, stride);
}



void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    pad_f32_cuda(src0_d, dst_d,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

void ggml_cuda_op_replication_pad2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int left, right, top, bottom;
    memcpy(&left, (int *)dst->op_params + 0, sizeof(int));
    memcpy(&right,  (int *)dst->op_params + 1, sizeof(int));
    memcpy(&top,  (int *)dst->op_params + 2, sizeof(int));
    memcpy(&bottom,  (int *)dst->op_params + 3, sizeof(int));

    replication_pad2d_f32_cuda(src0_d, dst_d,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        left, right, top, bottom, stream);
}

void ggml_cuda_op_deconv_pad2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int stride;
    memcpy(&stride, (int *)dst->op_params + 0, sizeof(int));

    deconv_pad2d_f32_cuda(src0_d, dst_d,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        stride, stream);
}
