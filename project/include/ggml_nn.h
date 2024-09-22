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

#pragma GCC diagnostic ignored "-Wformat-truncation"

typedef struct ggml_tensor ggml_tensor_t;
typedef struct ggml_context ggml_context_t;

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
ggml_tensor_t* ggml_nn_identity(ggml_context_t* ctx, ggml_tensor_t* x);

struct Identity {
    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_identity(ctx, x);
    }
};


// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
// class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_linear(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b);

struct Linear {
    int64_t in_features;
    int64_t out_features;
    bool has_bias = true;

    ggml_tensor_t* weight;
    ggml_tensor_t* bias = NULL;

    void create_weight_tensors(ggml_context_t* ctx, ggml_type wtype = GGML_TYPE_Q8_0)
    {
        weight = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (has_bias) {
            bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
        }
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (has_bias) {
            ggml_format_name(bias, "%s%s", prefix, "bias");
        }
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_linear(ctx, x, weight, bias);
    }
};


// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
//      padding_mode='zeros', device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_conv_2d(ggml_context_t* ctx, ggml_tensor_t * x, ggml_tensor_t * w,
    ggml_tensor_t * b, int s0 /*=1*/, int s1 /*=1*/, int p0 /*=0*/, int p1 /*=0*/, int d0 /*=1*/, int d1 /*=1*/);

struct Conv2d {
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride = { 1, 1 };
    std::pair<int, int> padding = { 0, 0 };
    std::pair<int, int> dilation = { 1, 1 };
    bool has_bias = true;

    ggml_tensor_t* weight;
    ggml_tensor_t* bias = NULL;

    void create_weight_tensors(ggml_context_t* ctx, ggml_type wtype=GGML_TYPE_F16)
    {
        weight = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (has_bias) {
            bias = ggml_new_tensor_1d(ctx, (wtype == GGML_TYPE_Q8_0)? GGML_TYPE_F16 : GGML_TYPE_F32, out_channels);
        }
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (has_bias) {
            ggml_format_name(bias, "%s%s", prefix, "bias");
        }
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_conv_2d(ctx, x, weight, bias, stride.second, stride.first, padding.second, padding.first,
            dilation.second, dilation.first);
    }
};


// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
// class torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_layer_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, float eps);

struct LayerNorm {
    int64_t normalized_shape;
    float eps = 1e-5;

    ggml_tensor_t* w;
    ggml_tensor_t* b;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_layer_norm(ctx, x, w, b, eps);
    }
};


// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
// class torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_group_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, int num_groups);

struct GroupNorm {
    int num_groups = 32;
    int64_t num_channels;
    float eps = 1e-6;

    ggml_tensor_t* weight;
    ggml_tensor_t* bias;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        // norm use GGML_TYPE_F32 !!!
        weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        ggml_format_name(bias, "%s%s", prefix, "bias");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_group_norm(ctx, x, weight, bias, num_groups); // hardcoded eps === 1e-6 now
    }
};

// ----------------------------------------------------------------------------------------------------------------------------------------
// class torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, 
//      add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_attention(ggml_context_t* ctx, ggml_tensor_t* q, ggml_tensor_t* k, ggml_tensor_t* v, bool mask);

struct MultiheadAttention {
    int64_t embed_dim;
    int64_t n_head;
    bool bias = true;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        q_proj.in_features = embed_dim;
        q_proj.out_features = embed_dim;
        q_proj.has_bias = bias;
        q_proj.create_weight_tensors(ctx, GGML_TYPE_Q8_0);

        k_proj.in_features = embed_dim;
        k_proj.out_features = embed_dim;
        k_proj.has_bias = bias;
        k_proj.create_weight_tensors(ctx, GGML_TYPE_Q8_0);

        v_proj.in_features = embed_dim;
        v_proj.out_features = embed_dim;
        v_proj.has_bias = bias;
        v_proj.create_weight_tensors(ctx, GGML_TYPE_Q8_0);

        out_proj.in_features = embed_dim;
        out_proj.out_features = embed_dim;
        out_proj.has_bias = bias;
        out_proj.create_weight_tensors(ctx, GGML_TYPE_Q8_0);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "q_proj.");
        q_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "k_proj.");
        k_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "v_proj.");
        v_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "out_proj.");
        out_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x, bool mask)
    {
        // embed_dim = 768, n_head = 12, bias = 1, mask = 1
        // x:    f32 [768, 77, 1, 1], 
        // embed_dim = 1280, n_head = 20, bias = 1, mask = 1
        // x:    f32 [1280, 77, 1, 1], 

        int64_t N = x->ne[2]; // ==> 1
        int64_t n_token = x->ne[1]; // ==> 77
        int64_t d_head = embed_dim / n_head; // ==> 64

        ggml_tensor_t* q = q_proj.forward(ctx, x);
        q = ggml_reshape_4d(ctx, q, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [N, n_head, n_token, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, n_token, n_head * N); // [N * n_head, n_token, d_head]

        ggml_tensor_t* k = k_proj.forward(ctx, x);
        k = ggml_reshape_4d(ctx, k, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [N, n_head, n_token, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, n_token, n_head * N); // [N * n_head, n_token, d_head]

        ggml_tensor_t* v = v_proj.forward(ctx, x);
        v = ggml_reshape_4d(ctx, v, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [N, n_head, d_head, n_token]
        v = ggml_reshape_3d(ctx, v, n_token, d_head, n_head * N); // [N * n_head, d_head, n_token]

        ggml_tensor_t* kqv = ggml_nn_attention(ctx, q, k, v, mask); // [N * n_head, n_token, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, n_token, n_head, N);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [N, n_token, n_head, d_head]
        x = ggml_reshape_3d(ctx, kqv, d_head * n_head, n_token, N); // [N, n_token, d_head * n_head]

        x = out_proj.forward(ctx, x); // [N, n_token, embed_dim]
        return x;
    }
};

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://paperswithcode.com/method/pixelshuffle

// class torch.nn.PixelShuffle(upscale_factor)[source] -- convert x from (∗,C*r*2, H, W) to (∗, C, H*r, W*r)


ggml_tensor_t* pixel_shuffle(ggml_context_t *ctx, ggml_tensor_t *x, int upscale_factor);

struct PixelShuffle {
    int upscale_factor;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return pixel_shuffle(ctx, x, upscale_factor);
    }
};


// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html
// class torch.nn.PixelUnshuffle(downscale_factor)[source] -- convert x from (∗,C,H×r,W×r) to (∗, C×r2, H, W)

ggml_tensor_t* pixel_unshuffle(ggml_context_t *ctx, ggml_tensor_t *x, int downscale_factor);

struct PixelUnshuffle {
    int downscale_factor;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return pixel_unshuffle(ctx, x, downscale_factor);
    }
};

#endif // _GGML_NN_H_

#ifdef GGML_NN_IMPLEMENTATION
ggml_tensor_t* ggml_nn_identity(ggml_context_t* ctx, ggml_tensor_t* x)
{
    return ggml_dup_inplace(ctx, x);
}

ggml_tensor_t* ggml_nn_conv_2d(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w,
    ggml_tensor_t* b, int s0 /*=1*/, int s1 /*=1*/, int p0 /*=0*/, int p1 /*=0*/, int d0 /*=1*/, int d1 /*=1*/)
{
    x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);

    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }

    return x;
}

ggml_tensor_t* ggml_nn_layer_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, float eps)
{
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}


// q: [N * n_head, n_token, d_head]
// k: [N * n_head, n_k, d_head]
// v: [N * n_head, d_head, n_k]
// return: [N * n_head, n_token, d_head]
ggml_tensor_t* ggml_nn_attention(ggml_context_t* ctx, ggml_tensor_t* q, ggml_tensor_t* k, ggml_tensor_t* v, bool mask /* = false*/)
{
#if defined(SD_USE_FLASH_ATTENTION) && !defined(SD_USE_CUBLAS) && !defined(SD_USE_METAL)
    ggml_tensor_t* kqv = ggml_flash_attn(ctx, q, k, v, false); // [N * n_head, n_token, d_head]
#else
    float d_head = (float)q->ne[0];

    ggml_tensor_t* kq = ggml_mul_mat(ctx, k, q); // [N * n_head, n_token, n_k]
    kq = ggml_scale_inplace(ctx, kq, 1.0f / sqrt(d_head));
    if (mask) {
        kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
    }
    kq = ggml_soft_max_inplace(ctx, kq);

    ggml_tensor_t* kqv = ggml_mul_mat(ctx, v, kq); // [N * n_head, n_token, d_head]
#endif
    return kqv;
}

ggml_tensor_t* ggml_nn_group_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, int num_groups)
{
    if (ggml_n_dims(x) >= 3) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    x = ggml_group_norm(ctx, x, num_groups, 1e-6); // TODO: eps is hardcoded to 1e-6 for now
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}


ggml_tensor_t* ggml_nn_linear(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b)
{
    x = ggml_mul_mat(ctx, w, x);
    if (b != NULL) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

ggml_tensor_t* pixel_shuffle(ggml_context_t *ctx, ggml_tensor_t *x, int upscale_factor)
{
    int C = x->ne[2]; // channel numbers
    int R = upscale_factor;

    ggml_tensor_t *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, R, R, C, C);
    x = ggml_im2col(ctx, a, x, R /*s0*/, R /*s1*/, 0 /*p0*/, 0 /*p1*/, 1 /*d0*/, 1 /*d1*/, true /*is_2D*/, GGML_TYPE_F32);
    x = ggml_permute(ctx, x, 2, 0, 1, 3); // from src index to dst: 0->2, 1->0, 2->1, 3->3
    x = ggml_cont(ctx, x); // !!! import !!!

    return x;
}

ggml_tensor_t* pixel_unshuffle(ggml_context_t *ctx, ggml_tensor_t *x, int downscale_factor)
{
    int C = x->ne[2]; // channel numbers
    int R = downscale_factor;

    ggml_tensor_t *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, R, R, C, C);
    x = ggml_im2col(ctx, a, x, R /*s0*/, R /*s1*/, 0 /*p0*/, 0 /*p1*/, 1 /*d0*/, 1 /*d1*/, true /*is_2D*/, GGML_TYPE_F32);
    x = ggml_permute(ctx, x, 2, 0, 1, 3); // from src index to dst: 0->2, 1->0, 2->1, 3->3
    x = ggml_cont(ctx, x); // !!! import !!!

    return x;
}

#endif // GGML_NN_IMPLEMENTATION