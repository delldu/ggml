#include "upscale.cuh"

static __global__ void upscale_f32(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03, // for src ...
        const int ne10, const int ne11, const int ne12, const int ne13, // for dst ...
        const float sf0, const float sf1, const float sf2, const float sf3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= ne10 * ne11 * ne12 * ne13) {
        return;
    }

    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    int i00 = i10 / sf0;
    int i01 = i11 / sf1;
    int i02 = i12 / sf2;
    int i03 = i13 / sf3;

    dst[index] = *(float *)((char *)x + i03 * nb03 + i02 * nb02 + i01 * nb01 + i00 * nb00);
}

// torch convert x from (B, C*R^2, H, W) to (B, C, H*R, W*R)
static __global__ void shuffle_f32(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03, // for src ...
        const int ne10, const int ne11, const int ne12, const int ne13, // for dst ...
        const int R) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= ne10 * ne11 * ne12 * ne13) {
        return;
    }

    int d0 = index % ne10;
    int d1 = (index / ne10) % ne11;
    int d2 = (index / (ne10 * ne11)) % ne12;
    int d3 = (index / (ne10 * ne11 * ne12)) % ne13;

    // s_c = d_c*R*R + (d_h % R)*R + (d_w % R);    
    int s0 = d0/R; // W
    int s1 = d1/R; // H
    int s2 = d2 * R * R + (d1 % R) * R + (d0 % R); // C
    int s3 = d3; // B

    dst[index] = *(float *)((char *)x + s3 * nb03 + s2 * nb02 + s1 * nb01 + s0 * nb00);
}

static __global__ void scatter_f32(float * dst, const int n,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // for dst ...
        const int dim, const int *index) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= n) { // n -- d_ne0 * d_ne1 * d_ne2 * d_ne3
        return;
    }

    // src index ...
    // int s0 = k % d_ne0;
    // int s1 = (k / d_ne0) % d_ne1;
    // int s2 = (k / (d_ne0 * d_ne1)) % d_ne2;
    // int s3 = (k / (d_ne0 * d_ne1 * d_ne2)) % d_ne3;

    int s0, s1, s2, s3;

    // GGML_ASSERT(dim >= 0 && dim < 2); // only for dim == 0 || dim == 1
    if (dim == 0) { // same as d_ne0 === 1
        s0 = 0;
        s1 = k % d_ne1;
        s2 = (k / d_ne1) % d_ne2;
        s3 = (k / (d_ne1 * d_ne2)) % d_ne3;
    } else { // dim == 1, same as d_ne1 === 1
        s0 = k % d_ne0;
        s1 = 0;
        s2 = (k / d_ne0) % d_ne2;
        s3 = (k / (d_ne0 * d_ne2)) % d_ne3;
    }

    // dst index ...
    int d0 = (dim == 0)? index[s1] : s0;
    int d1 = (dim == 1)? index[s0] : s1;
    int d2 = s2;
    int d3 = s3;

    *(float *)((char *)dst + d3 * d_nb3 + d2 * d_nb2 + d1 * d_nb1 + d0 * d_nb0) = 1.0;
}

static __global__ void slice_scatter_copy_f32(const float *x, float * dst,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3) {
    // ******************************************************************************
    // * x_ne[i] == d_ne[i] for all i
    // ******************************************************************************
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= d_ne0 * d_ne1 * d_ne2 * d_ne3) {
        return;
    }
    dst[index] = x[index]; // copy x0 to dst
}

// embed x to dst
static __global__ void slice_scatter_embed_f32(const float * x, float * dst,
        const int x_ne0, const int x_ne1, const int x_ne2, const int x_ne3, // for src ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // for dst ...
        const int dim, const int start, const int stop, const int step) {

    // ******************************************************************************
    // * 1) x_ne[i] == d_ne[i] if i != dim 
    // * 2) stop = MIN(d_ne[dim], stop)
    // ******************************************************************************
    
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
    int d0 = (dim == 0)? start + x0 * step : x0;
    if (dim == 0 && d0 >= stop)
        return;

    int d1 = (dim == 1)? start + x1 * step : x1;
    if (dim == 1 && d1 >= stop)
        return;

    int d2 = (dim == 2)? start + x2 * step : x2;
    if (dim == 2 && d2 >= stop)
        return;

    int d3 = (dim == 3)? start + x3 * step : x3;
    if (dim == 3 && d3 >= stop)
        return;

    *(float *)((char *)dst + d3 * d_nb3 + d2 * d_nb2 + d1 * d_nb1 + d0 * d_nb0) = x[index];
}


static void upscale_f32_cuda(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const float sf0, const float sf1, const float sf2, const float sf3,
        cudaStream_t stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = (dst_size + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

    upscale_f32<<<num_blocks, CUDA_UPSCALE_BLOCK_SIZE,0,stream>>>(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
}

static void shuffle_f32_cuda(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const int R,
        cudaStream_t stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = (dst_size + CUDA_SHUFFLE_BLOCK_SIZE - 1) / CUDA_SHUFFLE_BLOCK_SIZE;
    shuffle_f32<<<num_blocks, CUDA_SHUFFLE_BLOCK_SIZE,0,stream>>>(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, R);
}


template<typename T>
static __global__ void flip_kernel(const T*src, T* dst, const int n,
        const int ne0, const int ne1, const int ne2, const int ne3, // src, dst share ...
        const int nb0, const int nb1, const int nb2, const int nb3, // src, dst share ...
        const int dim0, const int dim1, const int dim2, const int dim3) {
    // xxxx_temp
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n == ne0 * ne1 * ne2 * ne3
        return;
    }

    // dst index ...
    int d0 = index % ne0;
    int d1 = (index / ne0) % ne1;
    int d2 = (index / (ne0 * ne1)) % ne2;
    int d3 = (index / (ne0 * ne1 * ne2)) % ne3;

    // src index ...
    int s0 = (dim0)? ne0 - 1 - d0 : d0;
    int s1 = (dim1)? ne1 - 1 - d1 : d1;
    int s2 = (dim2)? ne2 - 1 - d2 : d2;
    int s3 = (dim3)? ne3 - 1 - d3 : d3;

    dst[index] = *(T *)((char *)src + s3 * nb3 + s2 * nb2 + s1 * nb1 + s0 * nb0);
}


template<typename T>
static void flip_cuda(const T* src, T* dst, const int n,
        const int ne0, const int ne1, const int ne2, const int ne3,
        const int nb0, const int nb1, const int nb2, const int nb3,
        const int dim0, const int dim1, const int dim2, const int dim3,
        cudaStream_t stream) {
    int num_blocks = (n + CUDA_FLIP_BLOCK_SIZE - 1) / CUDA_FLIP_BLOCK_SIZE;
    flip_kernel<<<num_blocks, CUDA_FLIP_BLOCK_SIZE, 0, stream>>>(src, dst, n,
        ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3, dim0, dim1, dim2, dim3);
}

static void flip_f16_cuda(const half* src, half* dst, const int n,
        const int ne0, const int ne1, const int ne2, const int ne3,
        const int nb0, const int nb1, const int nb2, const int nb3,
        const int dim0, const int dim1, const int dim2, const int dim3,
        cudaStream_t stream) {

    flip_cuda<half>(src, dst, n, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3, dim0, dim1, dim2, dim3, stream);
}


static void flip_f32_cuda(const float * src, float * dst, const int n,
        const int ne0, const int ne1, const int ne2, const int ne3,
        const int nb0, const int nb1, const int nb2, const int nb3,
        const int dim0, const int dim1, const int dim2, const int dim3,
        cudaStream_t stream) {
    flip_cuda<float>(src, dst, n, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3, dim0, dim1, dim2, dim3, stream);
}


static void slice_scatter_f32_cuda(const float * x0, const float * x1, float * dst,
        const int x1_ne0, const int x1_ne1, const int x1_ne2, const int x1_ne3,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        const int dim, const int start, const int stop, const int step,
        cudaStream_t stream) {

    int dst_size = d_ne0 * d_ne1 * d_ne2 * d_ne3;
    int num_blocks = (dst_size + CUDA_SCATTER_BLOCK_SIZE - 1) / CUDA_SCATTER_BLOCK_SIZE;
    slice_scatter_copy_f32<<<num_blocks, CUDA_SCATTER_BLOCK_SIZE, 0, stream>>>(x0, dst, 
        d_ne0, d_ne1, d_ne2, d_ne3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    dst_size = x1_ne0 * x1_ne1 * x1_ne2 * x1_ne3;
    num_blocks = (dst_size + CUDA_SCATTER_BLOCK_SIZE - 1) / CUDA_SCATTER_BLOCK_SIZE;
    slice_scatter_embed_f32<<<num_blocks, CUDA_SCATTER_BLOCK_SIZE, 0, stream>>>(x1, dst, 
        x1_ne0, x1_ne1, x1_ne2, x1_ne3,
        d_ne0, d_ne1, d_ne2, d_ne3,
        d_nb0, d_nb1, d_nb2, d_nb3,
        dim, start, stop, step);
}

static void scatter_f32_cuda(float * dst, const int n,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        const int dim, const int *index, cudaStream_t stream) {
    int num_blocks = (n + CUDA_SCATTER_BLOCK_SIZE - 1) / CUDA_SCATTER_BLOCK_SIZE;

    scatter_f32<<<num_blocks, CUDA_SCATTER_BLOCK_SIZE, 0, stream>>>(dst, n, 
        d_ne0, d_ne1, d_ne2, d_ne3,
        d_nb0, d_nb1, d_nb2, d_nb3,
        dim, index);
}



void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const float sf0 = (float)dst->ne[0]/src0->ne[0];
    const float sf1 = (float)dst->ne[1]/src0->ne[1];
    const float sf2 = (float)dst->ne[2]/src0->ne[2];
    const float sf3 = (float)dst->ne[3]/src0->ne[3];

    upscale_f32_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
}

void ggml_cuda_op_shuffle(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // torch convert from src0: (B, C*r*2, H, W) to dst: (B, C, H*r, W*r)
    const int R = dst->ne[0]/src0->ne[0];
    // const int R = dst->op_params[0];

    shuffle_f32_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], R, stream);
}

// dell_xxxx
void ggml_cuda_op_flip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT( dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 );
    const ggml_tensor *src = dst->src[0];
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT( dst->type == src->type);
    const int dim0 = dst->op_params[0];
    const int dim1 = dst->op_params[1];
    const int dim2 = dst->op_params[2];
    const int dim3 = dst->op_params[3];

    const int n = ggml_nelements(dst);
    if (dst->type == GGML_TYPE_F32) {
        flip_f32_cuda((const float *)src->data, (float *)dst->data, n, 
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
            dim0, dim1, dim2, dim3, stream);
        return;
    }

    if (dst->type == GGML_TYPE_F16) {
        flip_f16_cuda((const half *)src->data, (half *)dst->data, n, 
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
            dim0, dim1, dim2, dim3, stream);
        return;
    }
    // GGML_ASSERT( dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
}

// dell_xxxx
void ggml_cuda_op_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    // const float * src0_d = (const float *)src0->data;
    const ggml_tensor * src1 = dst->src[1];
    const int * src1_d = (const int *)src1->data; // index

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    GGML_ASSERT(dim >= 0 && dim < 2); // only for dim == 0 || dim == 1
    GGML_ASSERT(ggml_is_contiguous(src1));

    int n = (int)ggml_nelements(dst);
    n /= dst->ne[dim]; // skip dim 0 loop

    scatter_f32_cuda(dst_d, n,
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        dim, src1_d /*index*/, stream);
}


// dell_xxxx
void ggml_cuda_op_slice_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int start = dst->op_params[1];
    int stop = dst->op_params[2];
    const int step = dst->op_params[3];

    if (stop > dst->ne[dim])
        stop = dst->ne[dim];

    slice_scatter_f32_cuda(src0_d, src1_d, dst_d, 
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], 
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        dim, start, stop, step, stream);
}


// dell_xxxx
#include <cufft.h>

static __global__ void rfft2_f32_save_output(float* dst, const int n, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
    const int b, const int c, const int H, const int W, const cufftComplex *out_complex) {

    // for (int h = 0; h < H; h++) { // H
    //     for (int w = 0; w < H2W; w++) { // W
    //         // dst [w, h, c, b]
    //         point = (char *) dst->data + w*dst->nb[0] + h*dst->nb[1] + (2*c + 0)*dst->nb[2] + b*dst->nb[3];
    //         ((float *)point)[0] = out_complex[h * H2W + w][0]; // re

    //         point = (char *) dst->data + w*dst->nb[0] + h*dst->nb[1] + (2*c + 1)*dst->nb[2] + b*dst->nb[3];
    //         ((float *)point)[0] = out_complex[h * H2W + w][1]; // im
    //     }
    // }

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n -- H * H2W
        return;
    }

    int H2W = (W/2) + 1;
    int h = index/H2W;
    int w = index % H2W;

    float *point  = (float *)((char *) dst + w*d_nb0 + h*d_nb1 + (2*c + 0)*d_nb2 + b*d_nb3);
    *point = out_complex[h * H2W + w].x; // complex-re
    point  = (float *)((char *) dst + w*d_nb0 + h*d_nb1 + (2*c + 1)*d_nb2 + b*d_nb3);
    *point = out_complex[h * H2W + w].y; // complex-im
}


static __global__ void rfft2_f32_get_input(cufftReal *dst, const float *src, const int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n -- H * W
        return;
    }
    dst[index] = src[index];
}


// Get input_real from src
static void rfft2_f32_cuda_get_input(cufftReal *dst, const float *src, const int n, cudaStream_t stream) {
    int num_blocks = (n + CUDA_RFFT2_BLOCK_SIZE - 1) / CUDA_RFFT2_BLOCK_SIZE;
    rfft2_f32_get_input<<<num_blocks, CUDA_RFFT2_BLOCK_SIZE, 0, stream>>>(dst, src, n);
}

// Save out_complex to dst
static void rfft2_f32_cuda_save_output(float* dst, const int n,
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, 
    const int b, const int c, const int H, const int W, 
    const cufftComplex * out_complex, cudaStream_t stream) {
    int num_blocks = (n + CUDA_RFFT2_BLOCK_SIZE - 1) / CUDA_RFFT2_BLOCK_SIZE;
    rfft2_f32_save_output<<<num_blocks, CUDA_RFFT2_BLOCK_SIZE, 0, stream>>>(dst, n, 
        d_nb0, d_nb1, d_nb2, d_nb3, b, c, H, W, out_complex);
}

// dell_xxxx
void ggml_cuda_op_rfft2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // GGML_TENSOR_UNARY_OP_LOCALS

    int W = (int)src->ne[0];
    int H = (int)src->ne[1];
    int C = (int)src->ne[2];
    int B = (int)src->ne[3];
    int H2W = (W/2) + 1;

    char *point;
    cufftHandle plan;
    cufftReal *input_real; // H, W
    cufftComplex *output_complex; // H, H2W

    cudaMalloc((void**)&input_real, sizeof(cufftReal)*H*W);
    cudaMalloc((void**)&output_complex, sizeof(cufftComplex)*H*H2W);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    /* Create a 2D FFT plan. */
    if (cufftPlan2d(&plan, H, W, CUFFT_R2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Error: Unable to create plan\n");
        return;
    }
    // cufftSetStream(plan, stream);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            // 1) Get input_float from src0 (b, c, H, W)
            point = (char *) src->data + c*src->nb[2] + b*src->nb[3];
            rfft2_f32_cuda_get_input(input_real, (float *)point, H * W, stream);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // 2) Do fft
            if (cufftExecR2C(plan, input_real, output_complex) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
                return;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // 3) Save output_complex to dst
            rfft2_f32_cuda_save_output((float *)dst->data, H * H2W,
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], b, c, H, W, output_complex, stream);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cufftDestroy(plan);
    cudaFree(output_complex);
    cudaFree(input_real);
}

// Get input_complex from src
static __global__ void irfft2_f32_get_input(cufftComplex *input_complex, const int n,
    const float *src, const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, 
    const int b, const int c, const int H, const int W) {
    // for (int h = 0; h < H; h++) { // H
    //     for (int w = 0; w < H2W; w++) { // W
    //         point = (char *) src0->data + w*src0->nb[0] + h*src0->nb[1] + (2*c + 0)*src0->nb[2] + b*src0->nb[3];
    //         input_complex[h * H2W + w][0] = *(float *)point; // 0 -- re

    //         point = (char *) src0->data + w*src0->nb[0] + h*src0->nb[1] + (2*c + 1)*src0->nb[2] + b*src0->nb[3];
    //         input_complex[h * H2W + w][1] = *(float *)point; // 1 --- im
    //     }
    // }

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n -- H * H2W
        return;
    }
    int H2W = (W/2) + 1;
    int h = index/H2W;
    int w = index % H2W;

    void *point_re = (char *) src + w*s_nb0 + h*s_nb1 + (2*c + 0)*s_nb2 + b*s_nb3;
    void *point_im = (char *) src + w*s_nb0 + h*s_nb1 + (2*c + 1)*s_nb2 + b*s_nb3;

    input_complex[h * H2W + w].x = *((float *)point_re); //  -- re
    input_complex[h * H2W + w].y = *((float *)point_im); //  --- im
}

// Get input_complex from src
static void irfft2_f32_cuda_get_input(cufftComplex * input_complex, const int n,
    const float* src, const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
    const int b, const int c, const int H, const int W, cudaStream_t stream) {
    int num_blocks = (n + CUDA_RFFT2_BLOCK_SIZE - 1) / CUDA_RFFT2_BLOCK_SIZE;
    irfft2_f32_get_input<<<num_blocks, CUDA_RFFT2_BLOCK_SIZE, 0, stream>>>(input_complex, n, 
       src, s_nb0, s_nb1, s_nb2, s_nb3, b, c, H, W);
}

// Save output_real to dst
static __global__ void irfft2_f32_save_output(float *dst, const int n, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
    const int b, const int c, const int H, const int W, const float *output_real) {
    // float HxW = (float) H * W;
    // for (int i = 0; i < H * W; i++) {
    //     output_float[i] /= HxW;
    // }
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) { // n -- H * W
        return;
    }
    int h = index/W;
    int w = index%W;
    float HxW = (float) H * W;
    void *point = (char *) dst + w*d_nb0 + h*d_nb1 + c*d_nb2 + b*d_nb3;
    *((float *)point) = output_real[h * W + w]/HxW;
}

// Save output_real to dst
static void irfft2_f32_cuda_scale_output(float *dst, const int n, 
    const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
    const int b, const int c, const int H, const int W, const float *output_real, 
    cudaStream_t stream) {
    int num_blocks = (n + CUDA_RFFT2_BLOCK_SIZE - 1) / CUDA_RFFT2_BLOCK_SIZE;
    irfft2_f32_save_output<<<num_blocks, CUDA_RFFT2_BLOCK_SIZE, 0, stream>>>(dst, n, 
        d_nb0, d_nb1, d_nb2, d_nb3, b, c, H, W, output_real);
}

// dell_xxxx
void ggml_cuda_op_irfft2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor *src = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // const int ith = params->ith;
    // const int nth = params->nth;
    // GGML_TENSOR_UNARY_OP_LOCALS
    // Torch format:
    // RFFT2: Real: (B, C, H, W) --> Complex: (B, C, H, (W/2)+1) --> Real: (B, 2*C, H, (W/2) + 1)
    //IRFFT2: Real: (B, 2*C, H, (W/2) + 1) --> Complex: (B, C, H, (W/2)+1) --> Real: (B, C, H, W)

    int W = (int)dst->ne[0];
    int H = (int)dst->ne[1];
    int C = (int)dst->ne[2];
    int B = (int)dst->ne[3];
    int H2W = (W/2) + 1;
    // W = 64, H = 64, C = 192, B = 1, H2W = 33

    cufftHandle plan;
    cufftComplex *input_complex; // H, H2W
    cufftReal *output_real; // H, W

    cudaMalloc((void**)&input_complex, sizeof(cufftComplex)*H*H2W);
    cudaMalloc((void**)&output_real, sizeof(cufftReal)*H*W);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    /* Create a 2D FFT plan. */
    if (cufftPlan2d(&plan, H, W, CUFFT_C2R) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Error: Unable to create plan\n");
        return;
    }
    // cufftSetStream(plan, stream);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            // 1) Get input_complex from src (B, 2*C, H, (W/2) + 1)
            irfft2_f32_cuda_get_input(input_complex, H*H2W, 
                (const float *)src->data, src->nb[0], src->nb[1], src->nb[2], src->nb[3], 
                b, c, H, W, stream);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // 2) Do fft
            if (cufftExecC2R(plan, input_complex, output_real) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
                return;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // 3) Save output_real to dst
            irfft2_f32_cuda_scale_output((float *)dst->data, H * W,
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
                b, c, H, W, (float *)output_real, stream);

            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cufftDestroy(plan);
    cudaFree(output_real);
    cudaFree(input_complex);
}
