/************************************************************************************
***
*** Copyright 2024 Dell(18588220928g@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 30 Jan 2024 11:52:34 PM CST
***
************************************************************************************/

#ifndef _GGML_NN_H_
#define _GGML_NN_H_

#include <ggml.h>


struct ggml_tensor* ggml_nn_identity(struct ggml_context* ctx, struct ggml_tensor* x)
{
    return ggml_dup_inplace(ctx, x);
}

struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w,
    struct ggml_tensor* b, int s0 = 1, int s1 = 1, int p0 = 0, int p1 = 0, int d0 = 1, int d1 = 1)
{
    x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);

    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }

    return x;
}

struct ggml_tensor* ggml_nn_layer_norm(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, float eps = 1e-06f)
{
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

struct ggml_tensor* ggml_nn_group_norm(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int num_groups = 32)
{
    if (ggml_n_dims(x) >= 3) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    x = ggml_group_norm(ctx, x, num_groups);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

struct ggml_tensor* ggml_nn_group_norm_32(struct ggml_context* ctx, struct ggml_tensor* a)
{
    return ggml_group_norm(ctx, a, 32);
}

struct ggml_tensor* ggml_nn_linear(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b)
{
    x = ggml_mul_mat(ctx, w, x);
    x = ggml_add(ctx, x, b);
    return x;
}

#endif // _GGML_NN_H_