## GGML Engine Package

### 1. Python API
#### 1.1 Create makefile for engine
`from ggml_engine import create_makefile`
`create_makefile()`

#### 1.2 Create network for engine
`from ggml_engine import create_network`
`create_network(model)`

### 2. Command Line Interface
#### 2.1 Create gguf from torch model
`python -m ggml_engine.gguf.create`

`python -m ggml_engine.gguf.create /tmp/sdxl_vae_f16_fix.safetensors`

#### 22.Dump gguf metadata

`python -m ggml_engine.gguf.dump`

`python -m ggml_engine.gguf.dump /tmp/output/ggf`
