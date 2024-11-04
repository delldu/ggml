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

static __global__ void flip_f32(const float * x, float * dst,
        const int ne0, const int ne1, const int ne2, const int ne3,
        const int nb0, const int nb1, const int nb2, const int nb3,
        const int dim) {
    // xxxx_temp
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= ne0 * ne1 * ne2 * ne3) {
        return;
    }

    // dst index ...
    int d0 = index % ne0;
    int d1 = (index / ne0) % ne1;
    int d2 = (index / (ne0 * ne1)) % ne2;
    int d3 = (index / (ne0 * ne1 * ne2)) % ne3;

    // src index ...
    int s0 = (dim == 0)? ne0 - 1 - d0 : d0;
    int s1 = (dim == 1)? ne1 - 1 - d1 : d1;
    int s2 = (dim == 2)? ne2 - 1 - d2 : d2;
    int s3 = (dim == 3)? ne3 - 1 - d3 : d3;

    dst[index] = *(float *)((char *)x + s3 * nb3 + s2 * nb2 + s1 * nb1 + s0 * nb0);
}

static __global__ void scatter_copy_f32(const float *x, float * dst,
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
static __global__ void scatter_embed_f32(const float * x, float * dst,
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

    int d3 = (dim == 0)? start + x3 * step : x3;
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

static void flip_f32_cuda(const float * x, float * dst,
        const int ne0, const int ne1, const int ne2, const int ne3,
        const int nb0, const int nb1, const int nb2, const int nb3,
        const int dim,
        cudaStream_t stream) {
    int num_blocks = (ne0 * ne1 * ne2 * ne3 + CUDA_FLIP_BLOCK_SIZE - 1) / CUDA_FLIP_BLOCK_SIZE;

    flip_f32<<<num_blocks, CUDA_FLIP_BLOCK_SIZE, 0, stream>>>(x, dst, ne0, ne1, ne2, ne3, 
        nb0, nb1, nb2, nb3, dim);
}

static void scatter_f32_cuda(const float * x0, const float * x1, float * dst,
        const int x1_ne0, const int x1_ne1, const int x1_ne2, const int x1_ne3,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        const int dim, const int start, const int stop, const int step,
        cudaStream_t stream) {

    int dst_size = d_ne0 * d_ne1 * d_ne2 * d_ne3;
    int num_blocks = (dst_size + CUDA_SCATTER_BLOCK_SIZE - 1) / CUDA_SCATTER_BLOCK_SIZE;
    scatter_copy_f32<<<num_blocks, CUDA_SCATTER_BLOCK_SIZE, 0, stream>>>(x0, dst, 
        d_ne0, d_ne1, d_ne2, d_ne3);
    // CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    dst_size = x1_ne0 * x1_ne1 * x1_ne2 * x1_ne3;
    num_blocks = (dst_size + CUDA_SCATTER_BLOCK_SIZE - 1) / CUDA_SCATTER_BLOCK_SIZE;
    scatter_embed_f32<<<num_blocks, CUDA_SCATTER_BLOCK_SIZE, 0, stream>>>(x1, dst, 
        x1_ne0, x1_ne1, x1_ne2, x1_ne3,
        d_ne0, d_ne1, d_ne2, d_ne3,
        d_nb0, d_nb1, d_nb2, d_nb3,
        dim, start, stop, step);
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

    // torch convert x from src0: (B, C*r*2, H, W) to dst: (B, C, H*r, W*r)
    const int R = dst->ne[0]/src0->ne[0];
    shuffle_f32_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], R, stream);
}

void ggml_cuda_op_flip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    const int dim = dst->op_params[0];

    flip_f32_cuda(src0_d, dst_d, 
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        dim, stream);
}

// xxxx_debug
void ggml_cuda_op_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int start = dst->op_params[1];
    int stop = dst->op_params[2];
    const int step = dst->op_params[3];

    if (stop > dst->ne[dim])
        stop = dst->ne[dim];

    scatter_f32_cuda(src0_d, src1_d, dst_d, 
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], 
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        dim, start, (const int)stop, step, stream);
}


// xxxx_debug
#include <cufft.h>

static __global__ void rfft2_f32_save_output(ggml_tensor * dst, const int b, const int c, const int H, const int W,
    const cufftComplex *out_complex) {

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
    int H2W = (W/2) + 1;
    if (index >= H * H2W) {
        return;
    }
    int h = index/H2W;
    int w = index % H2W;

    float *point  = (float *)((char *) dst->data + w*dst->nb[0] + h*dst->nb[1] + (2*c + 0)*dst->nb[2] + b*dst->nb[3]);
    *point = out_complex[h * H2W + W].x; // complex-re
    point  = (float *)((char *) dst->data + w*dst->nb[0] + h*dst->nb[1] + (2*c + 1)*dst->nb[2] + b*dst->nb[3]);
    *point = out_complex[h * H2W + W].y; // complex-im
}


// save out_complex to dst
static void rfft2_f32_cuda_save_output(ggml_tensor * dst, const int b, const int c, const int H, const int W, 
    const cufftComplex * out_complex, cudaStream_t stream) {
    int H2W = (W/2) + 1;
    int num_blocks = (H * H2W + CUDA_RFFT2_BLOCK_SIZE - 1) / CUDA_RFFT2_BLOCK_SIZE;
    rfft2_f32_save_output<<<num_blocks, CUDA_RFFT2_BLOCK_SIZE, 0, stream>>>(dst, b, c, H, W, out_complex);
}


void ggml_cuda_op_rfft2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    // const float * src0_d = (const float *)src0->data;
    // float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS

    int W = (int)src0->ne[0];
    int H = (int)src0->ne[1];
    int C = (int)src0->ne[2];
    int B = (int)src0->ne[3];
    int H2W = (W/2) + 1;

    cufftHandle plan;
    cufftReal *input_real; // H, W
    cufftComplex *output_complex; // H, H2W
    void *point;

    cudaMalloc((void**)&input_real, sizeof(cufftReal)*H*W);
    CUDA_CHECK(cudaGetLastError());
    cudaMalloc((void**)&output_complex, sizeof(cufftComplex)*H*H2W);
    CUDA_CHECK(cudaGetLastError());

    /* Create a 2D FFT plan. */
    if (cufftPlan2d(&plan, H, W, CUFFT_R2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Error: Unable to create plan\n");
        return;
    }
    cufftSetStream(plan, stream);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            // 1) Get input_float from src0 (b, c, H, W)
            point = (char *) src0->data + c*src0->nb[2] + b*src0->nb[3];
            // memcpy(input_float, point, H*W*sizeof(float));
            CUDA_CHECK(cudaMemcpyAsync(input_real, point, H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            // CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // 2) Do fft
            if (cufftExecR2C(plan, input_real, output_complex) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
                return;
            }
            // CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // 3) Save output_complex to dst
            rfft2_f32_cuda_save_output(dst, b, c, H, W, output_complex, stream);
            // CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(output_complex));
    CUDA_CHECK(cudaFree(input_real));
}


static __global__ void irfft2_f32_get_input(cufftComplex *input_complex, 
    const ggml_tensor *src, const int b, const int c, const int H, const int W) {
    // for (int h = 0; h < H; h++) { // H
    //     for (int w = 0; w < H2W; w++) { // W
    //         point = (char *) src0->data + w*src0->nb[0] + h*src0->nb[1] + (2*c + 0)*src0->nb[2] + b*src0->nb[3];
    //         input_complex[h * H2W + w][0] = *(float *)point; // 0 -- re

    //         point = (char *) src0->data + w*src0->nb[0] + h*src0->nb[1] + (2*c + 1)*src0->nb[2] + b*src0->nb[3];
    //         input_complex[h * H2W + w][1] = *(float *)point; // 1 --- im
    //     }
    // }

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int H2W = (W/2) + 1;
    if (index >= H * H2W) {
        return;
    }
    int h = index/H2W;
    int w = index % H2W;

    void *point = (char *) src->data + w*src->nb[0] + h*src->nb[1] + (2*c + 0)*src->nb[2] + b*src->nb[3];
    input_complex[h * H2W + w].x = *(float *)point; //  -- re

    point = (char *) src->data + w*src->nb[0] + h*src->nb[1] + (2*c + 1)*src->nb[2] + b*src->nb[3];
    input_complex[h * H2W + w].y = *(float *)point; //  --- im
}


static void irfft2_f32_cuda_get_input(cufftComplex * input_complex,
    const ggml_tensor * src, const int b, const int c, const int H, const int W, cudaStream_t stream) {
    int H2W = (W/2) + 1;
    int num_blocks = (H * H2W + CUDA_RFFT2_BLOCK_SIZE - 1) / CUDA_RFFT2_BLOCK_SIZE;
    irfft2_f32_get_input<<<num_blocks, CUDA_RFFT2_BLOCK_SIZE, 0, stream>>>(input_complex, src, b, c, H, W);
}


static __global__ void irfft2_f32_scale_output(cufftReal *output_real, const int H, const int W) {
    // float HxW = (float) H * W;
    // for (int i = 0; i < H * W; i++) {
    //     output_float[i] /= HxW;
    // }
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= H * W) {
        return;
    }
    output_real[index] /= (float)(H * W);
}

static void irfft2_f32_cuda_scale_output(cufftReal * output_real, const int H, const int W, cudaStream_t stream) {
    int num_blocks = (H * W + CUDA_RFFT2_BLOCK_SIZE - 1) / CUDA_RFFT2_BLOCK_SIZE;
    irfft2_f32_scale_output<<<num_blocks, CUDA_RFFT2_BLOCK_SIZE, 0, stream>>>(output_real, H, W);
}


// xxxx_debug
void ggml_cuda_op_irfft2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    // const float * src0_d = (const float *)src0->data;
    // float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // const int ith = params->ith;
    // const int nth = params->nth;
    GGML_TENSOR_UNARY_OP_LOCALS
    // Torch format:
    // RFFT2: Real: (B, C, H, W) --> Complex: (B, C, H, (W/2)+1) --> Real: (B, 2*C, H, (W/2) + 1)
    //IRFFT2: Real: (B, 2*C, H, (W/2) + 1) --> Complex: (B, C, H, (W/2)+1) --> Real: (B, C, H, W)

    int W = (int)dst->ne[0];
    int H = (int)dst->ne[1];
    int C = (int)dst->ne[2];
    int B = (int)dst->ne[3];
    int H2W = (W/2) + 1;

    cufftHandle plan;
    cufftComplex *input_complex; // H, H2W
    cufftReal *output_real; // H, W

    cudaMalloc((void**)&input_complex, sizeof(cufftComplex)*H*H2W);
    CUDA_CHECK(cudaGetLastError());
    cudaMalloc((void**)&output_real, sizeof(cufftReal)*H*W);
    CUDA_CHECK(cudaGetLastError());

    /* Create a 2D FFT plan. */
    if (cufftPlan2d(&plan, H, W, CUFFT_C2R) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Error: Unable to create plan\n");
        return;
    }
    cufftSetStream(plan, stream);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            ///////////////////////////////
            // 1) Get input_complex from src (B, 2*C, H, (W/2) + 1)
            irfft2_f32_cuda_get_input(input_complex, src0, b, c, H, W, stream);
            // CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // 2) Do fft
            if (cufftExecC2R(plan, input_complex, output_real) != CUFFT_SUCCESS){
                fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
                return;
            }
            // CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // 3) Save output_real to dst
            irfft2_f32_cuda_scale_output(output_real, H, W, stream);
            // CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            void *point = (char *) dst->data + c*dst->nb[2] + b*dst->nb[3];
            // memcpy(point, output_float, H*W*sizeof(float));
            CUDA_CHECK(cudaMemcpyAsync(point, output_real, H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            // CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(output_real));
    CUDA_CHECK(cudaFree(input_complex));
}
