#include "common.cuh"

#define CUDA_PAD_BLOCK_SIZE 256

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_replication_pad2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_deconv_pad2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
