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
use_fp16: False # TODO: Will be updated
in_ch: 1
num_classes: 1
input_shape: !!python/tuple [64, 288, 288].
```

## Main Dependencies
- Python 3.10.xx
- PyTorch: pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

- TensorRT==8.5.3.1
- ONNX==1.16.1
- ONNX Runtime==1.18.0
- PyCUDA==2022.1+cuda116
- NumPy==1.23.5


## Get Started
### Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/ONNX-TensorRT.git
cd ONNX-TensorRT
```
### Configuration
Prepare the config.yaml file with the necessary configurations. Below is an example configuration:
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
input_shape: !!python/tuple [64, 288, 288]
```
- **Task could be torch2onnx, onnx2trt, torch2trt**

### Conversion Process
#### Convert PyTorch to ONNX
To convert a PyTorch model to ONNX format, you need to use the converter.py script. Make sure your config.yaml is set up correctly.Ensureing your `task` iin `config.yaml` is set to `torch2onnx`.

Run the script:
```bash
python converter.py -c path/to/config.yaml
```
#### Convert ONNX to TensorRT
If you have already converted your model to ONNX and want to convert it to TensorRT, ensure your `task` in `config.yaml`` is set to `onnx2trt`. Then, run the converter.py script again:
```bash
python converter.py -c path/to/config.yaml
```

### Verification
If `use_verify` is set to `True` in your configuration file, the conversion process will include a verification step. This step ensures that the outputs from the ONNX and TensorRT models are within a specified tolerance of the original PyTorch model.

