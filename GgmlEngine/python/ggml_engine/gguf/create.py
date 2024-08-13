#!/usr/bin/env python
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
import argparse
import sys
import gguf
import torch
import numpy as np
from safetensors.torch import load_file
import pdb

def create_gguf(torch_model, gguf_format, gguf_model, prefix):
    if torch_model.endswith(".safetensors"):
        state_dict = load_file(torch_model)
    else:
        state_dict = torch.load(torch_model, map_location="cpu")

    if len(prefix) > 0 and prefix in state_dict:
        state_dict = state_dict[prefix]

    if gguf_model.endswith(".gguf"):
        gguf_output_path = gguf_model
    else:
        gguf_output_path = f"{gguf_model}.gguf"

    gguf_writer = gguf.GGUFWriter(gguf_output_path, gguf_model)
    for k, v in state_dict.items():
        if v.dim() > 4:
            v = v.squeeze(4)
        if v.dim() > 4:
            v = v.squeeze(0)
            
        if gguf_format == "F32":
            gguf_writer.add_tensor(k, v.numpy().astype(np.float32))
        elif gguf_format == "F16":
            if k.endswith(".bias"): # bias must be float32 
                gguf_writer.add_tensor(k, v.numpy().astype(np.float32))
            else:
                gguf_writer.add_tensor(k, v.numpy().astype(np.float16))
        else:
            # gguf now only support F32/F16, NO F32/F16 data need to been converted
            if v.dtype != torch.float32 and v.dtype != torch.float16:
                gguf_writer.add_tensor(k, v.numpy().astype(np.float32))
            else:
                gguf_writer.add_tensor(k, v.numpy())

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"{gguf_format} model has been saved to '{gguf_output_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create gguf from torch model")
    parser.add_argument("model", type=str, help="torch model file (*.pth, *.safetensors)")
    parser.add_argument("--prefix", type=str, default="params_ema", help="model parameters prefix (default: params_ema)")
    parser.add_argument("--format", type=str, default="F16", help="ouput format (F32, F16, Raw, default: F16)")
    parser.add_argument("--output", type=str, default="/tmp/output.gguf",  help="output gguf file (default: /tmp/output.gguf)")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])

    torch_model = args.model
    model_prefix = args.prefix
    gguf_format = args.format
    gguf_model = args.output
    create_gguf(torch_model, gguf_format, gguf_model, model_prefix)
