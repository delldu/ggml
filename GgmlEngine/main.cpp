#include "tensor.h"
#include "vae.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>


struct IdNetwork : GGMLNetwork {
    void create_weight_tensors(struct ggml_context* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    };
};

void test_id_network()
{
    syslog_info("Test id_network ...");

    struct IdNetwork net;
    net.start_engine();

    TENSOR* x = tensor_create(1, 4, 64, 64);
    for (int j = 0; j < 1 * 4 * 64 * 64; j++) {
        x->data[j] = (float)j / (64.0 * 64.0 * 4.0);
    }
    TENSOR* argv[] = { x };
    tensor_show((char*)"x", x);

    TENSOR* y = net.engine_forward(ARRAY_SIZE(argv), argv);
    if (tensor_valid(y)) {
        tensor_show((char*)"y", y);
        tensor_destroy(y);
    } else {
        syslog_error("y == null");
    }

    net.stop_engine();

    tensor_destroy(x);
}

struct AddNetwork : GGMLNetwork {
    void create_weight_tensors(struct ggml_context* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    };

    struct ggml_tensor* forward(struct ggml_context* ctx, int argc, struct ggml_tensor* argv[])
    {
        // GGML_UNUSED(ctx);
        GGML_UNUSED(argc);
        auto x = argv[0];
        auto y = argv[1];

        return ggml_add(ctx, x, y);
    }
};

void test_add_network()
{
    syslog_info("Test add_network ...");

    struct AddNetwork net;
    net.set_device(1); // CUDA 0
    net.start_engine();

    TENSOR* x1 = tensor_create(1, 3, 8, 8);
    for (int j = 0; j < 1 * 3 * 8 * 8; j++) {
        x1->data[j] = (float)j;
    }
    TENSOR* x2 = tensor_create(1, 3, 8, 8);
    for (int j = 0; j < 1 * 3 * 8 * 8; j++) {
        x2->data[j] = (float)(j * 2.0);
    }
    TENSOR* argv[] = { x1, x2 };
    tensor_show((char*)"x1", x1);
    tensor_show((char*)"x2", x2);

    TENSOR* y = net.engine_forward(ARRAY_SIZE(argv), argv);
    if (tensor_valid(y)) {
        tensor_show((char*)"y", y);
        tensor_destroy(y);
    } else {
        syslog_error("y == null");
    }

    // -----------------------------------------------
    for (int j = 0; j < 1 * 3 * 8 * 8; j++) {
        x1->data[j] = (float)10 * j;
    }

    y = net.engine_forward(ARRAY_SIZE(argv), argv);
    if (tensor_valid(y)) {
        // tensor_show((char*)"y", y);
        tensor_destroy(y);
    } else {
        syslog_error("y == null");
    }

    // net.dump();

    net.stop_engine();

    tensor_destroy(x1);
    tensor_destroy(x2);
}

void test_vae_encoder()
{
    syslog_info("Test vae encoder ...");

    struct Encoder* encoder = new Encoder();
    // encoder->dump();

    encoder->set_device(1); // CUDA 0
    encoder->load("sdxl_vae_fp16_fix.gguf", "encoder."); // decoder.mid.block_2.
    encoder->start_engine();

    // encoder->dump();
    TENSOR* x = tensor_load_image((char*)"0001.png", 0); // tensor_create(1, 3, 256, 256);
    // tensor_show((char*)"x", x);

    TENSOR* argv[] = { x };
    for (int i = 0; i < 10; i++) {
        TENSOR* y = encoder->engine_forward(ARRAY_SIZE(argv), argv);
        if (tensor_valid(y)) {
            // tensor_show((char*)"y", y);
            tensor_destroy(y);
        } else {
            syslog_error("y == null");
        }
    }

    // encoder->dump();
    encoder->stop_engine();
    tensor_destroy(x);

    delete encoder;
}

void test_vae_decoder()
{
    syslog_info("Test vae decoder ...");

    struct Decoder decoder;

    decoder.set_device(1); // CUDA 0
    decoder.load("sdxl_vae_fp16_fix.gguf", "decoder."); // decoder.mid.block_2.
    decoder.start_engine();
    // decoder.dump();

    TENSOR* x = tensor_create(1, 4, 64, 64);
    for (int j = 0; j < 1 * 4 * 64 * 64; j++) {
        x->data[j] = (float)j / (64.0 * 64.0 * 4.0);
    }
    TENSOR* argv[] = { x };

    for (int i = 0; i < 10; i++) {
        TENSOR* y = decoder.engine_forward(ARRAY_SIZE(argv), argv);

        if (tensor_valid(y)) {
            // tensor_show((char*)"y", y);
            tensor_destroy(y);
        } else {
            syslog_error("y == null");
        }
    }

    // decoder.dump();
    decoder.stop_engine();
    tensor_destroy(x);
}

int main(int argc, char** argv)
{
    test_id_network();
    test_add_network();

    test_vae_encoder();
    test_vae_decoder();

    return 0;
}
