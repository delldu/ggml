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

static __global__ void replication_pad2d_f32(const float* src, float *dst, const int n, 
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
        const int left, const int top) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == d_ne0 * d_ne1 * d_ne2 * d_ne3 -- ggml_elements(dst)
        return;
    }

    // dst index ...
    int d0 = index % d_ne0;
    int d1 = (index / d_ne0) % d_ne1;
    int d2 = (index / (d_ne0 * d_ne1)) % d_ne2;
    int d3 = (index / (d_ne0 * d_ne1 * d_ne2)) % d_ne3;

    // src index ...
    int s0, s1, s2, s3;
    // W
    if (d0 >= left && d0 < s_ne0 + left) {
        s0 = d0 - left;
    } else {
        s0 = (d0 < left)? 0 : s_ne0 - 1;
    }
    // H
    if (d1 >= top && d1 < s_ne1 + top) {
        s1 = d1 - top;
    } else {
        s1 = (d1 < top)? 0 : s_ne1 - 1;
    }
    s2 = d2; // C
    s3 = d3; // B

    int64_t s_offset = tensor_full_offset(s0, s1, s2, s3, s_nb0, s_nb1, s_nb2, s_nb3);
    int64_t d_offset = tensor_full_offset(d0, d1, d2, d3, d_nb0, d_nb1, d_nb2, d_nb3);

    *(float *)((char *)dst + d_offset) = *(float *)((char *)src + s_offset);
}

// dell_xxxx
static __global__ void reflection_pad2d_f32(const float* src, float *dst, const int n, 
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
        const int left, const int top) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == d_ne0 * d_ne1 * d_ne2 * d_ne3 -- ggml_elements(dst)
        return;
    }

    // dst index ...
    int d0 = index % d_ne0;
    int d1 = (index / d_ne0) % d_ne1;
    int d2 = (index / (d_ne0 * d_ne1)) % d_ne2;
    int d3 = (index / (d_ne0 * d_ne1 * d_ne2)) % d_ne3;

    // src index ...
    int s0, s1, s2, s3;
    // W
    if (d0 >= left && d0 < s_ne0 + left) {
        s0 = d0 - left;
    } else {
        s0 = (d0 < left)? left - d0 : 2*(s_ne0 - 1) - d0;
    }
    // H
    if (d1 >= top && d1 < s_ne1 + top) {
        s1 = d1 - top;
    } else {
        s1 = (d1 < top)? top - d1 : 2*(s_ne1 - 1) - d1;
    }
    s2 = d2; // C
    s3 = d3; // B

    int64_t s_offset = tensor_full_offset(s0, s1, s2, s3, s_nb0, s_nb1, s_nb2, s_nb3);
    int64_t d_offset = tensor_full_offset(d0, d1, d2, d3, d_nb0, d_nb1, d_nb2, d_nb3);

    *(float *)((char *)dst + d_offset) = *(float *)((char *)src + s_offset);
}


static __global__ void deconv_pad2d_f32_init(float * dst, const int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == x_ne0 * x_ne1 * x_ne2 * x_ne3
        return;
    }
    dst[index] = 0.0;
}

static __global__ void deconv_pad2d_f32(const float* src, float* dst, const int s_n,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        const int stride) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= s_n) { // s_n == s_ne0 * s_ne1 * s_ne2 * s_ne3
        return;
    }

    // src index ...
    int s0 = index % s_ne0;
    int s1 = (index / s_ne0) % s_ne1;
    int s2 = (index / (s_ne0 * s_ne1)) % s_ne2;
    int s3 = (index / (s_ne0 * s_ne1 * s_ne2)) % s_ne3;

    // dst index ...
    int d0 = s0 * stride; // W
    int d1 = s1 * stride; // H
    int d2 = s2;
    int d3 = s3;

    int64_t s_offset = tensor_full_offset(s0, s1, s2, s3, s_nb0, s_nb1, s_nb2, s_nb3);
    int64_t d_offset = tensor_full_offset(d0, d1, d2, d3, d_nb0, d_nb1, d_nb2, d_nb3);

    *(float *)((char *)dst + d_offset) = *(float *)((char *)src + s_offset);
}


static void pad_f32_cuda(const float * x, float * dst,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int ne0, const int ne1, const int ne2, const int ne3, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2*ne3);
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(x, dst, ne0, ne00, ne01, ne02, ne03);
}

static void replication_pad2d_f32_cuda(const float* src, float* dst, const int n,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // dst
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // dst
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // src
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, // src
    const int left, const int top, cudaStream_t stream) {

    int num_blocks = (n + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    replication_pad2d_f32<<<num_blocks, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        d_ne0, d_ne1, d_ne2, d_ne3, 
        d_nb0, d_nb1, d_nb2, d_nb3, 
        s_ne0, s_ne1, s_ne2, s_ne3, 
        s_nb0, s_nb1, s_nb2, s_nb3, 
        left, top);
}

static void reflection_pad2d_f32_cuda(const float* src, float* dst, const int n,
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // dst
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // dst
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // src
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, // src
    const int left, const int top, cudaStream_t stream) {

    int num_blocks = (n + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    reflection_pad2d_f32<<<num_blocks, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        d_ne0, d_ne1, d_ne2, d_ne3, 
        d_nb0, d_nb1, d_nb2, d_nb3, 
        s_ne0, s_ne1, s_ne2, s_ne3, 
        s_nb0, s_nb1, s_nb2, s_nb3, 
        left, top);
}


static void deconv_pad2d_f32_cuda(const float * src, float * dst, const int s_n, const int d_n,
    const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // src
    const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, // src
    const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // dst
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // dst
    const int stride, cudaStream_t stream) {

    int num_blocks = (d_n + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    deconv_pad2d_f32_init<<<num_blocks, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(dst, d_n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    // ------------------------------------------------------------------------------------
    num_blocks = (s_n + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    deconv_pad2d_f32<<<num_blocks, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, dst, s_n,
        s_ne0, s_ne1, s_ne2, s_ne3,
        s_nb0, s_nb1, s_nb2, s_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3,
        d_nb0, d_nb1, d_nb2, d_nb3,
        stride);
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

// dell_xxxx
void ggml_cuda_op_replication_pad2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int left, right, top, bottom;
    memcpy(&left, (int *)dst->op_params + 0, sizeof(int));
    memcpy(&right,  (int *)dst->op_params + 1, sizeof(int));
    memcpy(&top,  (int *)dst->op_params + 2, sizeof(int));
    memcpy(&bottom,  (int *)dst->op_params + 3, sizeof(int));

    GGML_ASSERT(dst->ne[0] == src->ne[0] + left + right); // W
    GGML_ASSERT(dst->ne[1] == src->ne[1] + top + bottom); // H
    GGML_ASSERT(dst->ne[2] == src->ne[2]); // C
    GGML_ASSERT(dst->ne[3] == src->ne[3]); // B

    replication_pad2d_f32_cuda(src_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        left, top, stream);
}

// dell_xxxx
void ggml_cuda_op_reflection_pad2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int left, top, right, bottom;
    memcpy(&left, (int *)dst->op_params + 0, sizeof(int));
    memcpy(&right,  (int *)dst->op_params + 1, sizeof(int));
    memcpy(&top,  (int *)dst->op_params + 2, sizeof(int));
    memcpy(&bottom,  (int *)dst->op_params + 3, sizeof(int));

    GGML_ASSERT(dst->ne[0] == src->ne[0] + left + right); // W
    GGML_ASSERT(dst->ne[1] == src->ne[1] + top + bottom); // H
    GGML_ASSERT(dst->ne[2] == src->ne[2]); // C
    GGML_ASSERT(dst->ne[3] == src->ne[3]); // B

    reflection_pad2d_f32_cuda(src_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        left, top, stream);
}

// dell_xxxx
void ggml_cuda_op_deconv_pad2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int stride;
    memcpy(&stride, (int *)dst->op_params + 0, sizeof(int));
    const int s_n = ggml_nelements(src);
    const int d_n = ggml_nelements(dst);

    deconv_pad2d_f32_cuda(src_d, dst_d, s_n, d_n,
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        stride, stream);
}
