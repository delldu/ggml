/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#include "unet.h"

TENSOR* unet_forward(UNetModel* unet, 
    TENSOR* image_latent, TENSOR* timesteps, TENSOR* cond_latent, TENSOR* cond_pooled,
    TENSOR* controls_0, TENSOR* controls_1, TENSOR* controls_2, TENSOR* controls_3)
{
    TENSOR* argv[8]
        = { image_latent, timesteps, cond_latent, cond_pooled, controls_0, controls_1, controls_2, controls_3 };

    TENSOR* cond_output = unet->engine_forward(ARRAY_SIZE(argv), argv);

    return cond_output;
}
