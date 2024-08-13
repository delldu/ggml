# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Fri 19 Jan 2024 05:51:13 PM CST
# ***
# ************************************************************************************/
#
import jinja2
from collections import OrderedDict
import torch
import pdb

def create_makefile(model):
    print("Stay tuned ...")
    # t = listmodel_has_same_children(model)
    # print(t)


def create_network(model):
    define_file_head = model.__class__.__name__.upper()

    model_dict = get_named_user_models(model)
    # print(model_dict.keys()) # ['AttnBlock', 'Downsample', 'ResnetBlock', 'Encoder']

    print(f"#ifndef __{define_file_head}__H__")
    print(f"#define __{define_file_head}__H__")
    print("#include \"ggml_engine.h\"")
    print("#include \"ggml_nn.h\"")
    print("")
    print("#pragma GCC diagnostic ignored \"-Wformat-truncation\"")
    print("")

    for n, m in model_dict.items():
        create_single_model_source_code(m)

    print(f"#endif // __{define_file_head}__H__")


def is_system_model(model):
    standard_models = [
        "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
        "AdaptiveLogSoftmaxWithLoss",
        "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
        "AlphaDropout",
        "AvgPool1d", "AvgPool2d", "AvgPool3d",
        "BCELoss",
        "BCEWithLogitsLoss",
        "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
        "Bilinear",
        "CELU",
        "CTCLoss",
        "ChannelShuffle",
        "ConstantPad1d", "ConstantPad2d", "ConstantPad3d",
        "Container",
        "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
        "CosineEmbeddingLoss",
        "CosineSimilarity",
        "CrossEntropyLoss",
        "CrossMapLRN2d",
        # "DataParallel",
        "Dropout", "Dropout1d", "Dropout2d", "Dropout3d",
        "ELU",
        "Embedding",
        "EmbeddingBag",
        "FeatureAlphaDropout",
        "Flatten",
        "Fold",
        "FractionalMaxPool2d", "FractionalMaxPool3d", "GELU",
        "GLU",
        "GRU",
        "GRUCell",
        "GaussianNLLLoss",
        "GroupNorm",
        "Hardshrink", "Hardsigmoid", "Hardswish", "Hardtanh",
        "HingeEmbeddingLoss",
        "HuberLoss",
        "Identity",
        "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
        "KLDivLoss",
        "L1Loss",
        "LPPool1d", "LPPool2d",
        "LSTM",
        "LSTMCell",
        "LayerNorm", "LazyBatchNorm1d", "LazyBatchNorm2d", "LazyBatchNorm3d",
        "LazyConv1d", "LazyConv2d", "LazyConv3d",
        "LazyConvTranspose1d", "LazyConvTranspose2d", "LazyConvTranspose3d",
        "LazyInstanceNorm1d", "LazyInstanceNorm2d", "LazyInstanceNorm3d",
        "LazyLinear",
        "LeakyReLU",
        "Linear",
        "LocalResponseNorm",
        "LogSigmoid",
        "LogSoftmax",
        "MSELoss",
        "MarginRankingLoss",
        "MaxPool1d", "MaxPool2d", "MaxPool3d",
        "MaxUnpool1d", "MaxUnpool2d", "MaxUnpool3d",
        "Mish",
        # "Module",
        "MultiLabelMarginLoss",
        "MultiLabelSoftMarginLoss",
        "MultiMarginLoss",
        "MultiheadAttention",
        "NLLLoss", "NLLLoss2d",
        "PReLU",
        "PairwiseDistance",
        "Parameter",
        "PixelShuffle", "PixelUnshuffle",
        "PoissonNLLLoss",
        "RNN",
        "RNNBase",
        "RNNCell",
        "RNNCellBase",
        "RReLU",
        "ReLU",
        "ReLU6",
        "ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d",
        "ReplicationPad1d", "ReplicationPad2d", "ReplicationPad3d",
        "SELU",
        "SiLU",
        "Sigmoid",
        "SmoothL1Loss",
        "SoftMarginLoss",
        "Softmax",
        "Softmax2d",
        "Softmin",
        "Softplus",
        "Softshrink",
        "Softsign",
        "SyncBatchNorm",
        "Tanh",
        "Tanhshrink",
        "Threshold",
        "Transformer",
        "TransformerDecoder",
        "TransformerDecoderLayer",
        "TransformerEncoder",
        "TransformerEncoderLayer",
        "TripletMarginLoss",
        "TripletMarginWithDistanceLoss",
        "Unflatten",
        "Unfold",
        "UninitializedBuffer",
        "UninitializedParameter",
        "Upsample",
        "UpsamplingBilinear2d",
        "UpsamplingNearest2d",
        "ZeroPad2d",
        # "__builtins__",
        # "__cached__",
        # "__doc__",
        # "__file__",
        # "__loader__",
        # "__name__",
        # "__package__",
        # "__path__",
        # "__spec__",
        # "_reduction",
        # "common_types",
        # "factory_kwargs",
        # "functional",
        # "grad",
        # "init",
        # "intrinsic",
        # "modules",
        # "parallel",
        # "parameter",
        # "qat",
        # "quantizable",
        # "quantized",
        # "utils",
    ]
    # LeakyReLU(negative_slope=0.2, inplace=True)
    name = model.__class__.__name__
    return name in standard_models and "torch.nn" in str(model.__class__)


def is_list_model(model):
    name = model.__class__.__name__
    return name in ["Sequential", "ModuleList", "ModuleDict", "ParameterDict", "ParameterList", "Module"]


def is_user_model(model):
    return not (is_system_model(model) or is_list_model(model))


def var_safe_name(k):
    return k.replace(".", "_")


def var_shape_list(v):
    s = []
    for t in list(v.size()):
        s.insert(0, str(t)) # reversed sort
    return ", ".join(s)


def get_system_model_source(name, model):
    # name = nn.Conv2d(...)
    # ====>
    # name.weight = ...
    # name.bias = ...
    assert is_system_model(model) or is_list_model(model), "ONLY Support standard/list model"

    declare_source = []
    create_source = []
    setup_source = []

    for k, v in model.state_dict().items():
        if v.dim() < 1:
            continue # skip track_running_stats=True

        vname = var_safe_name(name + "." + k)

        # struct ggml_tensor* conv_w;
        # struct ggml_tensor* conv_b;
        line = f"    struct ggml_tensor* {vname};  // {v.dtype}, {list(v.size())} "
        declare_source.append(line)

        #     conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        #     conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        shape = var_shape_list(v)
        if v.dtype == torch.float32:
            line = f"        {vname} = ggml_new_tensor_{v.dim()}d(ctx, GGML_TYPE_F32, {shape});"
        else:
            if k.endswith(".bias"): # force set bias set f32
                line = f"        {vname} = ggml_new_tensor_{v.dim()}d(ctx, GGML_TYPE_F32, {shape});"
            else:
                line = f"        {vname} = ggml_new_tensor_{v.dim()}d(ctx, GGML_TYPE_F16, {shape});"
        create_source.append(line)

        #     ggml_format_name(conv_w, "%s%", prefix, "conv_w.weight");
        #     ggml_format_name(conv_b, "%s%", prefix, "conv_b.bias");
        line = f'        ggml_format_name({vname}, "%s%s", prefix, "{name}.{k}");'
        setup_source.append(line)

    return declare_source, create_source, setup_source


def get_user_model_source(name, model):
    assert is_user_model(model), "ONLY Support user define model"

    declare_source = []
    create_source = []
    setup_source = []

    dname = name # dot name
    vname = var_safe_name(name)

    # ---------------------------
    declare_source.append(f"    struct {model.__class__.__name__} {vname};")

    # ---------------------------
    create_source.append(f"        {vname}.create_weight_tensors(ctx);")

    # ---------------------------
    # snprintf(s, sizeof(s), "%sencoder.", prefix);
    # encoder.setup_weight_names(s);
    setup_source.append(f'        snprintf(s, sizeof(s), "%s%s", prefix, "{dname}.");')
    setup_source.append(f'        {vname}.setup_weight_names(s);')

    return declare_source, create_source, setup_source


def get_listmodel_source_code(name, model):
    declare_source = []
    create_source = []
    setup_source = []

    for n, m in model.named_children():
        if is_user_model(m):
            declare_i, create_i, setup_i = get_user_model_source(name + "." + n, m)
        else: #if is_system_model(m):
            declare_i, create_i, setup_i = get_system_model_source(name + "." + n, m)

        # save ...
        declare_source = declare_source + declare_i
        create_source = create_source + create_i
        setup_source = setup_source + setup_i   

    return declare_source, create_source, setup_source

# def listmodel_has_same_children(model):
#     if not is_list_model(model):
#         return False

#     declare_string = []
#     for n, m in model.named_children():
#         if is_user_model(m):
#             declare_string.append(f"struct {m.__class__.__name__}");
#         else:
#             s = ""
#             for k, v in m.state_dict().items():
#                 if v.dim() < 1:
#                     continue # skip track_running_stats=True
#                 s += f"{k}={v.dtype};"
#             declare_string.append(s)

#     if len(declare_string) == 1:
#         return True;
#     s0 = declare_string[0]
#     for i in range(1, len(declare_string)):
#         if s0 != declare_string[i]:
#             return False
#     return True


def create_single_model_source_code(model):
    temp = jinja2.Template(
"""
struct {{class_name}} {
    // network hparams
    {{hparams_source_code}}

    // network params
    {{declare_source_code}}


    void create_weight_tensors(struct ggml_context* ctx) {
        {{create_source_code}}
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        {{setup_source_code}}
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

"""
    )

    print("/*\n", str(model), "*/")

    declare_source_code = []
    create_source_code = []
    setup_source_code = []

    # network hparams
    # ------------------------------------------------------------------
    hparams_source_code = []
    for k, v in model.__dict__.items():
        if k == "training" or k.startswith("_"):
            continue
        if isinstance(v, int):
            hparams_source_code.append(f"    int {k} = {v};")
        elif isinstance(v, float):
            hparams_source_code.append(f"    float {k} = {v};")
        elif isinstance(v, str):
            hparams_source_code.append(f"    char* {k} = \"{v}\";")

    # # ------------------------------------------------------------------
    # for k, v in model.named_parameters():
    #     vname = var_safe_name(k)

    #     declare_source_code.append(f"    struct ggml_tensor* {vname};  // {v.dtype}, {list(v.size())} ")

    #     shape = var_shape_list(v)
    #     if v.dtype == torch.float32:
    #         line = f"        {vname} = ggml_new_tensor_{v.dim()}d(ctx, GGML_TYPE_F32, {shape});"
    #     else:
    #         line = f"        {vname} = ggml_new_tensor_{v.dim()}d(ctx, GGML_TYPE_F16, {shape});"
    #     create_source_code.append(line)

    #     line = f'        ggml_format_name({vname}, "%s%s", prefix, "{k}.");'
    #     setup_source_code.append(line)

    for child_name, child_model in model.named_children():
        if is_system_model(child_model):
            declare_i, create_i, setup_i = get_system_model_source(child_name, child_model)
        elif is_user_model(child_model):
            declare_i, create_i, setup_i = get_user_model_source(child_name, child_model)
        else: #if is_list_model(child_model):
            declare_i, create_i, setup_i = get_listmodel_source_code(child_name, child_model)

        # save ...
        declare_source_code = declare_source_code + declare_i
        create_source_code = create_source_code + create_i
        setup_source_code = setup_source_code + setup_i

    hparams_source_code = "\n".join(hparams_source_code).strip() # Remove first line space
    declare_source_code = "\n".join(declare_source_code).strip() # Remove first line space
    create_source_code = "\n".join(create_source_code).strip() # Remove first line space
    setup_source_code = "\n".join(setup_source_code).strip() # Remove first line space


    source_code = temp.render(
        class_name=model.__class__.__name__,
        hparams_source_code=hparams_source_code,
        declare_source_code=declare_source_code,
        create_source_code=create_source_code,
        setup_source_code=setup_source_code,
    )
    print(source_code)


def get_named_user_models(model):
    model_dict = OrderedDict()

    for m in model.modules():
        if is_user_model(m):
            model_dict[m.__class__.__name__] = m

    return OrderedDict(reversed(model_dict.items()))


if __name__ == '__main__':
    print("Please use this module in python program.")
