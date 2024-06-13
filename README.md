# ONNX-TensorRT
Convert PyTorch models to TensorRT via ONNX

## Overview
This repository provides a tool to convert PyTorch models into TensorRT format using the ONNX intermediate representation. This conversion allows for optimized inference.

## Configuration
The conversion process is controlled by a configuration file (`config.yaml`). Below is an example configuration:

```yaml
net_model: AttentionUnet3D
torch_model_path: ./work/best.pth
onnx_model_path: ./work/best.onnx
trt_model_path: ./work/best.trt
task: onnx2trt # torch2onnx or onnx2trt or torch2trt
use_verify: True
use_fp16: False
in_ch: 1
num_classes: 1
input_shape: !!python/tuple [64, 288, 288].
```

ggg