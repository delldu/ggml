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

// dell_xxxx
static __global__ void interpolate_f32(const float * x, float * dst,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // for src ...
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, // for src ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int dim, const float sf0, const float sf1, const float sf2, const float sf3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= d_ne0 * d_ne1 * d_ne2 * d_ne3) {
        return;
    }

    int d_0 = index % d_ne0;
    int d_1 = (index / d_ne0) % d_ne1;
    int d_2 = (index / (d_ne0 * d_ne1)) % d_ne2;
    int d_3 = (index / (d_ne0 * d_ne1 * d_ne2)) % d_ne3;

    float u = 1.0;
    int s1_0, s1_1, s1_2, s1_3;
    int s2_0, s2_1, s2_2, s2_3;
    
    s1_0 = s2_0 = d_0;
    s1_1 = s2_1 = d_1;
    s1_2 = s2_2 = d_2;
    s1_3 = s2_3 = d_3;

    if (dim == 0) {
        float d = (float)(d_0 + 0.5)/sf0 - 0.5;
        if (d < 0.0) d = 0.0;
        s1_0 = (int)d;
        s2_0 = (s1_0 + 1 < s_ne0)? s1_0 + 1 : s1_0;
        u = d - s1_0;
    }

    if (dim == 1) {
        float d = (float)(d_1 + 0.5)/sf1 - 0.5;
        if (d < 0.0) d = 0.0;
        s1_1 = (int)d;
        s2_1 = (s1_1 + 1 < s_ne1)? s1_1 + 1 : s1_1;
        u = d - s1_1;
    }

    if (dim == 2) {
        float d = (float)(d_2 + 0.5)/sf2 - 0.5;
        if (d < 0.0) d = 0.0;
        s1_2 = (int)d;
        s2_2 = (s1_2 + 1 < s_ne2)? s1_2 + 1 : s1_2;
        u = d - s1_2;
    }

    if (dim == 3) {
        float d = (float)(d_3 + 0.5)/sf3 - 0.5;
        if (d < 0.0) d = 0.0;
        s1_3 = (int)d;
        s2_3 = (s1_3 + 1 < s_ne3)? s1_3 + 1 : s1_3;
        u = d - s1_3;
    }

    int64_t x1_offset = tensor_full_offset(s1_0, s1_1, s1_2, s1_3, s_nb0, s_nb1, s_nb2, s_nb3);
    int64_t x2_offset = tensor_full_offset(s2_0, s2_1, s2_2, s2_3, s_nb0, s_nb1, s_nb2, s_nb3);
    float *x1 = (float *)((char *)x + x1_offset);
    float *x2 = (float *)((char *)x + x2_offset);

    dst[index] = (1.0 - u) * (*x1) + u*(*x2); // interpolate ...  more near x*, more weight !
}

// dell_xxxx
static __global__ void grid_sample_f32(const float * src, const float *grid, float * dst,
        const int n, const int H, const int W,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, // for src ...
        const int g_nb0, const int g_nb1, const int g_nb2, const int g_nb3, // for grid ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3) { // for dst ...
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) {
        return;
    }

    int d_0 = index % d_ne0; // W
    int d_1 = (index / d_ne0) % d_ne1; // H
    int d_2 = (index / (d_ne0 * d_ne1)) % d_ne2; // C
    int d_3 = (index / (d_ne0 * d_ne1 * d_ne2)) % d_ne3; // B

    int64_t g_offset, src_offset, dst_offset;

    // grid[b, h, w, 0]
    g_offset = tensor_full_offset(0/*for x*/, d_0/*Wout*/, d_1 /*Hout*/, d_3/*B*/, g_nb0, g_nb1, g_nb2, g_nb3);
    float x0 = *(float *)((char *)grid + g_offset);

    // grid[b, h, w, 1]
    g_offset = tensor_full_offset(1/*for y*/, d_0/*Wout*/, d_1 /*Hout*/, d_3/*B*/, g_nb0, g_nb1, g_nb2, g_nb3);
    float y0 = *(float *)((char *)grid + g_offset);

    // Going on tensor src 
    // because x0 in [-1.0, 1.0], y0 in [-1.0, 1.0], so we do (x0 + 1.0)/2.0 ...
    float fx = (x0 + 1.0f)/2.0f * (W - 1);
    float fy = (y0 + 1.0f)/2.0f * (H - 1);

    int x1 = (int)floor(fx);
    int y1 = (int)floor(fy);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // weight ...
    float w_x1y1 = (x2 - fx) * (y2 - fy);
    float w_x2y1 = (fx - x1) * (y2 - fy);
    float w_x1y2 = (x2 - fx) * (fy - y1);
    float w_x2y2 = (fx - x1) * (fy - y1);

    x1 = (x1 < 0)? 0 : x1; x1 = (x1 > W - 1)? W - 1: x1;
    x2 = (x2 < 0)? 0 : x2; x2 = (x2 > W - 1)? W - 1: x2;
    y1 = (y1 < 0)? 0 : y1; y1 = (y1 > H - 1)? H - 1: y1;
    y2 = (y2 < 0)? 0 : y2; y2 = (y2 > H - 1)? H - 1: y2;

    src_offset = tensor_full_offset(x1 /*w*/, y1 /*h*/, d_2 /*c*/, d_3 /*b*/, s_nb0, s_nb1, s_nb2, s_nb3);
    float v_x1y1 = *(float *)((char *)src + src_offset);

    src_offset = tensor_full_offset(x2 /*w*/, y1 /*h*/, d_2 /*c*/, d_3 /*b*/, s_nb0, s_nb1, s_nb2, s_nb3);
    float v_x2y1 = *(float *)((char *)src + src_offset);

    src_offset = tensor_full_offset(x1 /*w*/, y2 /*h*/, d_2 /*c*/, d_3 /*b*/, s_nb0, s_nb1, s_nb2, s_nb3);
    float v_x1y2 = *(float *)((char *)src + src_offset);

    src_offset = tensor_full_offset(x2 /*w*/, y2 /*h*/, d_2 /*c*/, d_3 /*b*/, s_nb0, s_nb1, s_nb2, s_nb3);
    float v_x2y2 = *(float *)((char *)src + src_offset);

    float v = w_x1y1 * v_x1y1 + w_x2y1 * v_x2y1 + w_x1y2 * v_x1y2 + w_x2y2 * v_x2y2;

    // ----------------------------------------------------------------------------
    dst_offset = tensor_full_offset(d_0, d_1, d_2, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
    *(float *)((char *)dst + dst_offset) = v;
}



// dell_xxxx
static __global__ void soft_splat_f32(const float * src, const float *flow, float * dst,
        const int n, const int H, const int W,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, // for src ...
        const int f_nb0, const int f_nb1, const int f_nb2, const int f_nb3, // for flow ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3) { // for dst ...
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) {
        return;
    }

    int d_0 = index % d_ne0; // W
    int d_1 = (index / d_ne0) % d_ne1; // H
    int d_2 = (index / (d_ne0 * d_ne1)) % d_ne2; // C
    int d_3 = (index / (d_ne0 * d_ne1 * d_ne2)) % d_ne3; // B

    int64_t f_offset, src_offset, dst_offset;

    // flow[b, 0, h, w]
    f_offset = tensor_full_offset(d_0 /*W*/, d_1 /*H*/,  0 /*for x*/, d_3 /*B*/, f_nb0, f_nb1, f_nb2, f_nb3);
    float x0 = *(float *)((char *)flow + f_offset);

    // flow[b, 1, h, w]
    f_offset = tensor_full_offset(d_0 /*W*/, d_1 /*H*/,  1 /*for y*/, d_3 /*B*/, f_nb0, f_nb1, f_nb2, f_nb3);
    float y0 = *(float *)((char *)flow + f_offset);

    // Going on tensor src 
    float fx = x0 + d_0;  // d_0 -- W, mesh_x + flow_x
    float fy = y0 + d_1;  // d_1 -- H, mesh_y + flow_y

    int x1 = (int)floor(fx);
    int y1 = (int)floor(fy);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // weight ...
    float w_x1y1 = (x2 - fx) * (y2 - fy);
    float w_x2y1 = (fx - x1) * (y2 - fy);
    float w_x1y2 = (x2 - fx) * (fy - y1);
    float w_x2y2 = (fx - x1) * (fy - y1);

    x1 = (x1 < 0)? 0 : x1; x1 = (x1 > W - 1)? W - 1: x1;
    x2 = (x2 < 0)? 0 : x2; x2 = (x2 > W - 1)? W - 1: x2;
    y1 = (y1 < 0)? 0 : y1; y1 = (y1 > H - 1)? H - 1: y1;
    y2 = (y2 < 0)? 0 : y2; y2 = (y2 > H - 1)? H - 1: y2;

    src_offset = tensor_full_offset(x1 /*w*/, y1 /*h*/, d_2 /*c*/, d_3 /*b*/, s_nb0, s_nb1, s_nb2, s_nb3);
    float v_x1y1 = *(float *)((char *)src + src_offset);

    src_offset = tensor_full_offset(x2 /*w*/, y1 /*h*/, d_2 /*c*/, d_3 /*b*/, s_nb0, s_nb1, s_nb2, s_nb3);
    float v_x2y1 = *(float *)((char *)src + src_offset);

    src_offset = tensor_full_offset(x1 /*w*/, y2 /*h*/, d_2 /*c*/, d_3 /*b*/, s_nb0, s_nb1, s_nb2, s_nb3);
    float v_x1y2 = *(float *)((char *)src + src_offset);

    src_offset = tensor_full_offset(x2 /*w*/, y2 /*h*/, d_2 /*c*/, d_3 /*b*/, s_nb0, s_nb1, s_nb2, s_nb3);
    float v_x2y2 = *(float *)((char *)src + src_offset);

    // ----------------------------------------------------------------------------
    dst_offset = tensor_full_offset(d_0, d_1, d_2, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
    *(float *)((char *)dst + dst_offset) = \
        w_x1y1 * v_x1y1 + w_x2y1 * v_x2y1 + w_x1y2 * v_x1y2 + w_x2y2 * v_x2y2;
}


static __global__ void euler_motion_f32(const float *flow, float * dst,
        const int n, const int H, const int W, const int ntimes,
        const int f_nb0, const int f_nb1, const int f_nb2, const int f_nb3, // for flow ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3) { // for dst ...
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) {
        return;
    }

    int d_0 = index % d_ne0; // W
    int d_1 = (index / d_ne0) % d_ne1; // H
    // int d_2 = (index / (d_ne0 * d_ne1)) % d_ne2; // C
    int d_3 = (index / (d_ne0 * d_ne1 * d_ne2)) % d_ne3; // B

    float d, f;
    int64_t f_offset, d_offset;

    // 1) Init
    d_offset = tensor_full_offset(d_0, d_1, 0 /*d_2 -- for x*/, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
    *(float *)((char *)dst + d_offset) = (float)d_0;
    d_offset = tensor_full_offset(d_0, d_1, 1 /*d_2 -- for y*/, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
    *(float *)((char *)dst + d_offset) = (float)d_1;

    // 2) Update
    for (int i = 0; i < ntimes; i++) {
        // update x
        d_offset = tensor_full_offset(d_0, d_1, 0 /*d_2 -- for x*/, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
        d = *(float *)((char *)dst + d_offset);
        f_offset = tensor_full_offset(d_0 /*W*/, d_1 /*H*/, 0 /*d_2 for x*/, d_3 /*B*/, f_nb0, f_nb1, f_nb2, f_nb3);
        f = *(float *)((char *)flow + f_offset);
        d += f;
        if (d < 0.0 || d >= W - 1) {
            d = (float)d_0;
        }
        *(float *)((char *)dst + d_offset) = d;

        // update y
        d_offset = tensor_full_offset(d_0, d_1, 1 /*d_2 -- for y*/, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
        d = *(float *)((char *)dst + d_offset);
        f_offset = tensor_full_offset(d_0 /*W*/, d_1 /*H*/, 1 /*d_2 for y*/, d_3 /*B*/, f_nb0, f_nb1, f_nb2, f_nb3);
        f = *(float *)((char *)flow + f_offset);
        d += f;
        if (d < 0.0 || d >= H - 1) {
            d = (float)d_1;
        }
        *(float *)((char *)dst + d_offset) = d;
    }

    // 3) Final
    d_offset = tensor_full_offset(d_0, d_1, 0 /*d_2 -- for x*/, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
    d = *(float *)((char *)dst + d_offset);
    *(float *)((char *)dst + d_offset) = d - (float)d_0;
    d_offset = tensor_full_offset(d_0, d_1, 1 /*d_2 -- for y*/, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
    d = *(float *)((char *)dst + d_offset);
    *(float *)((char *)dst + d_offset) = d - (float)d_1;
}

// torch convert x from (B, C*R^2, H, W) to (B, C, H*R, W*R)
static __global__ void shuffle_f32(const float * src, float * dst, const int n,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // for src ...
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3, // for src ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3, // for dst ...
        const int R) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) {
        return;
    }

    int d0 = index % d_ne0;
    int d1 = (index / d_ne0) % d_ne1;
    int d2 = (index / (d_ne0 * d_ne1)) % d_ne2;
    int d3 = (index / (d_ne0 * d_ne1 * d_ne2)) % d_ne3;

    // s_c = d_c*R*R + (d_h % R)*R + (d_w % R);    
    int s0 = d0/R; // W
    int s1 = d1/R; // H
    int s2 = d2 * R * R + (d1 % R) * R + (d0 % R); // C
    int s3 = d3; // B

    int64_t s_offset = tensor_full_offset(s0, s1, s2, s3, s_nb0, s_nb1, s_nb2, s_nb3);
    int64_t d_offset = tensor_full_offset(d0, d1, d2, d3, d_nb0, d_nb1, d_nb2, d_nb3);
    *(float *)((char *)dst + d_offset) = *(float *)((char *)src + s_offset);
}

static __global__ void win_part_f32(const float * src, float * dst, const int n,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // for src ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int npx, const int npy, const int w) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) {
        return;
    }

    for (int py = 0; py < npy; ++py) {
        for (int px = 0; px < npx; ++px) {
            const int64_t d3 = py*npx + px;
            for (int64_t d2 = 0; d2 < d_ne2; ++d2) {
                for (int64_t d1 = 0; d1 < d_ne1; ++d1) {
                    // for (int64_t d0 = 0; d0 < d_ne0; ++d0) {
                        const int64_t s2 = py*w + d2;
                        const int64_t s1 = px*w + d1;
                        // const int64_t s0 = d0;
                        // d0 == s0 == index ...    
                        const int64_t i = d3*d_ne2*d_ne1*d_ne0 + d2*d_ne1*d_ne0 + d1*d_ne0 + index; // d0;
                        const int64_t j =                        s2*s_ne1*s_ne0 + s1*s_ne0 + index; // s0;
                        if (py*w + d2 >= s_ne2 || px*w + d1 >= s_ne1) {
                            dst[i] = 0.0f;
                        } else {
                            dst[i] = src[j];
                        }
                    // }
                }
            }
        }
    }
}

static __global__ void win_unpart_f32(const float * src, float * dst, const int n,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // for src ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int npx, const int w) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) {
        return;
    }

    for (int64_t d2 = 0; d2 < d_ne2; ++d2) {
        for (int64_t d1 = 0; d1 < d_ne1; ++d1) {
            // for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int ip2 = d2/w;
                const int ip1 = d1/w;
                const int64_t s2 = d2%w;
                const int64_t s1 = d1%w;
                // const int64_t s0 = d0;

                // const int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                // const int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;
                const int64_t i = (ip2*npx + ip1)*s_ne2*s_ne1*s_ne0 + s2*s_ne1*s_ne0 + s1*s_ne0 + index; // s0;
                const int64_t j =                                     d2*d_ne1*d_ne0 + d1*d_ne0 + index; // d0;
                dst[j] = src[i];
            // }
        }
    }
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


static void interpolate_f32_cuda(const float * x, float * dst,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int dim, const float sf0, const float sf1, const float sf2, const float sf3,
        cudaStream_t stream) {
    int dst_size = d_ne0 * d_ne1 * d_ne2 * d_ne3;
    int num_blocks = (dst_size + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

    interpolate_f32<<<num_blocks, CUDA_UPSCALE_BLOCK_SIZE,0,stream>>>(x, dst, 
        s_ne0, s_ne1, s_ne2, s_ne3, s_nb0, s_nb1, s_nb2, s_nb3, 
        d_ne0, d_ne1, d_ne2, d_ne3, dim, sf0, sf1, sf2, sf3);
}

#define CUDA_GRID_SAMPLE_BLOCK_SIZE 256
static void grid_sample_f32_cuda(const float * src, const float *grid, float * dst,
        const int n, const int H, const int W,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
        const int g_nb0, const int g_nb1, const int g_nb2, const int g_nb3,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        cudaStream_t stream) {

    int num_blocks = (n + CUDA_GRID_SAMPLE_BLOCK_SIZE - 1) / CUDA_GRID_SAMPLE_BLOCK_SIZE;

    grid_sample_f32<<<num_blocks, CUDA_GRID_SAMPLE_BLOCK_SIZE, 0, stream>>>(src, grid, dst, 
        n, H, W,
        s_nb0, s_nb1, s_nb2, s_nb3, g_nb0, g_nb1, g_nb2, g_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3);
}

// dell_xxxx
#define CUDA_SOFT_SPLAT_BLOCK_SIZE 256
static void soft_splat_f32_cuda(const float * src, const float *flow, float * dst,
        const int n, const int H, const int W,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
        const int f_nb0, const int f_nb1, const int f_nb2, const int f_nb3,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        cudaStream_t stream) {

    int num_blocks = (n + CUDA_SOFT_SPLAT_BLOCK_SIZE - 1) / CUDA_SOFT_SPLAT_BLOCK_SIZE;

    soft_splat_f32<<<num_blocks, CUDA_SOFT_SPLAT_BLOCK_SIZE, 0, stream>>>(src, flow, dst, 
        n, H, W,
        s_nb0, s_nb1, s_nb2, s_nb3, f_nb0, f_nb1, f_nb2, f_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3);
}

#define CUDA_EULER_MOTION_BLOCK_SIZE 256
static void euler_motion_f32_cuda(const float *flow, float * dst,
        const int n, const int H, const int W, const int ntimes,
        const int f_nb0, const int f_nb1, const int f_nb2, const int f_nb3,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        cudaStream_t stream) {

    int num_blocks = (n + CUDA_EULER_MOTION_BLOCK_SIZE - 1) / CUDA_EULER_MOTION_BLOCK_SIZE;

    euler_motion_f32<<<num_blocks, CUDA_EULER_MOTION_BLOCK_SIZE, 0, stream>>>(flow, dst, 
        n, H, W, ntimes, f_nb0, f_nb1, f_nb2, f_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3);
}

static void shuffle_f32_cuda(const float * x, float * dst, const int n,
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3,
        const int s_nb0, const int s_nb1, const int s_nb2, const int s_nb3,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        const int R,
        cudaStream_t stream) {
    int num_blocks = (n + CUDA_SHUFFLE_BLOCK_SIZE - 1) / CUDA_SHUFFLE_BLOCK_SIZE;
    shuffle_f32<<<num_blocks, CUDA_SHUFFLE_BLOCK_SIZE,0,stream>>>(x, dst, n,
        s_ne0, s_ne1, s_ne2, s_ne3, s_nb0, s_nb1, s_nb2, s_nb3,
        d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3, R);
}

#define CUDA_WIN_PART_BLOCK_SIZE 256
static void win_part_f32_cuda(const float * src, float * dst, const int n, 
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // for src ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int npx, const int npy, const int w, cudaStream_t stream) {
    int num_blocks = (n + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;
    
    win_part_f32<<<num_blocks, CUDA_WIN_PART_BLOCK_SIZE,0,stream>>>(src, dst, n,
        s_ne0, s_ne1, s_ne2, s_ne3, d_ne0, d_ne1, d_ne2, d_ne3, npx, npy, w);
}

static void win_unpart_f32_cuda(const float * src, float * dst, const int n, 
        const int s_ne0, const int s_ne1, const int s_ne2, const int s_ne3, // for src ...
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int npx, const int w, cudaStream_t stream) {
    int num_blocks = (n + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;
    
    win_unpart_f32<<<num_blocks, CUDA_WIN_PART_BLOCK_SIZE,0,stream>>>(src, dst, n,
        s_ne0, s_ne1, s_ne2, s_ne3, d_ne0, d_ne1, d_ne2, d_ne3, npx, w);
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

void ggml_cuda_op_interpolate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];

    const float sf0 = (float)dst->ne[0]/src0->ne[0];
    const float sf1 = (float)dst->ne[1]/src0->ne[1];
    const float sf2 = (float)dst->ne[2]/src0->ne[2];
    const float sf3 = (float)dst->ne[3]/src0->ne[3];

    interpolate_f32_cuda(src0_d, dst_d, 
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], 
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dim, sf0, sf1, sf2, sf3, stream);
}

// dell_xxxx
void ggml_cuda_op_grid_sample(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * grid = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const float * grid_d = (const float *)grid->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(grid->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    grid_sample_f32_cuda(src0_d, grid_d, dst_d, 
        ggml_nelements(dst), src0->ne[1] /*H*/, src0->ne[0] /*W*/, 
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        grid->nb[0], grid->nb[1], grid->nb[2], grid->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        stream);
}

// dell_xxxx
static __global__ void grid_mesh_f32(float * dst, const int n, const int norm,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3, // for dst ...
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3) { // for dst ...
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) {
        return;
    }
    // dst shape: [2, W, H, B]

    // int d_0 = index % d_ne0; // 2
    int d_1 = (index / d_ne0) % d_ne1; // W
    int d_2 = (index / (d_ne0 * d_ne1)) % d_ne2; // H
    int d_3 = (index / (d_ne0 * d_ne1 * d_ne2)) % d_ne3; // B

    int64_t d_offset;
    d_offset = tensor_full_offset(0 /*d_0 -- C -- for x*/, d_1, d_2, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
    *(float *)((char *)dst + d_offset) = (norm)?(float)d_1/d_ne1 : (float)d_1; // w

    d_offset = tensor_full_offset(1 /*d_0 -- C -- for y*/, d_1, d_2, d_3, d_nb0, d_nb1, d_nb2, d_nb3);
    *(float *)((char *)dst + d_offset) = (norm)?(float)d_2/d_ne2 : (float)d_2; // h
}

// dell_xxxx
#define CUDA_GRID_MESH_BLOCK_SIZE 256
static void grid_mesh_f32_cuda(float * dst, const int n, const int norm,
        const int d_ne0, const int d_ne1, const int d_ne2, const int d_ne3,
        const int d_nb0, const int d_nb1, const int d_nb2, const int d_nb3,
        cudaStream_t stream) {

    int num_blocks = (n + CUDA_GRID_MESH_BLOCK_SIZE - 1) / CUDA_GRID_MESH_BLOCK_SIZE;

    grid_mesh_f32<<<num_blocks, CUDA_GRID_MESH_BLOCK_SIZE, 0, stream>>>(dst, 
        n, norm, d_ne0, d_ne1, d_ne2, d_ne3, d_nb0, d_nb1, d_nb2, d_nb3);
}

void ggml_cuda_op_grid_mesh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int norm = dst->op_params[0]; // 1 -- yes or no

    grid_mesh_f32_cuda(dst_d, ggml_nelements(dst), norm, 
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        stream);
}


void ggml_cuda_op_soft_splat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * flow = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const float * flow_d = (const float *)flow->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(flow->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    soft_splat_f32_cuda(src0_d, flow_d, dst_d, 
        ggml_nelements(dst), src0->ne[1] /*H*/, src0->ne[0] /*W*/, 
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        flow->nb[0], flow->nb[1], flow->nb[2], flow->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        stream);
}

void ggml_cuda_op_eluer_motion(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * flow = dst->src[0];
    const float * flow_d = (const float *)flow->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(flow->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    const int ntimes = dst->op_params[0];

    euler_motion_f32_cuda(flow_d, dst_d, 
        ggml_nelements(dst), dst->ne[1] /*H*/, dst->ne[0] /*W*/, ntimes,
        flow->nb[0], flow->nb[1], flow->nb[2], flow->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        stream);
}

void ggml_cuda_op_shuffle(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // torch convert from src0: (B, C*r*2, H, W) to dst: (B, C, H*r, W*r)
    // const int R = dst->ne[0]/src->ne[0];
    const int R = dst->op_params[0];
    GGML_ASSERT(R == (int)dst->ne[0]/src->ne[0]);

    shuffle_f32_cuda(src_d, dst_d, ggml_nelements(dst),
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], 
        R, stream);
}

// dell_xxxx
void ggml_cuda_op_win_part(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int npx = dst->op_params[0];
    const int npy = dst->op_params[1]; // cuda index ...
    const int w = dst->op_params[2];

    win_part_f32_cuda(src_d, dst_d, (int)dst->ne[0], // dst->ne[0] == src->ne[0]
        src->ne[0], src->ne[1], src->ne[2], src->ne[3], 
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], 
        npx, npy, w, stream);
}

// dell_xxxx
void ggml_cuda_op_win_unpart(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int w = dst->op_params[0];

    const int px = (w - dst->ne[1]%w)%w;
    const int npx = (px + dst->ne[1])/w;

    win_unpart_f32_cuda(src_d, dst_d, dst->ne[0], // dst->ne[0] == src->ne[0]
        src->ne[0], src->ne[1], src->ne[2], src->ne[3], 
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], npx, w, stream);
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
