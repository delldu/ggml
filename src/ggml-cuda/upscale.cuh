#include "common.cuh"

#define CUDA_UPSCALE_BLOCK_SIZE 256
#define CUDA_SHUFFLE_BLOCK_SIZE 256
#define CUDA_FLIP_BLOCK_SIZE 256
#define CUDA_SCATTER_BLOCK_SIZE 256
#define CUDA_RFFT2_BLOCK_SIZE 256

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_interpolate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_grid_mesh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_grid_sample(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_soft_splat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_eluer_motion(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_shuffle(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_flip(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_slice_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_rfft2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_irfft2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_win_part(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_win_unpart(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
