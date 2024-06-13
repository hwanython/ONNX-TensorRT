import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from libs.models.ModelFactory import ModelFactory

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class ModelConverter:
    def __init__(self, torch_model, input_shape, use_verify=True,
                 onnx_model_path=None, trt_model_path=None):
        self.model = torch_model  # PyTorch model with eval mode
        self.onnx_model_path = onnx_model_path  # path to save the converted onnx model
        self.trt_model_path = trt_model_path
        self.input_shape = input_shape  # input shape of the model (ch, cls, H, W, D)
        self.use_verify = use_verify  # verify the converted model
        self.rtol = 1e-02  # fp16은 atol=1e-03 정도로 설정 차후 use_fp16=True로 설정할 경우 수정
        self.atol = 1e-08

    def to_onnx(self):
        dummy_input = torch.randn(*self.input_shape, requires_grad=False)
        onnx_path = os.path.join(self.onnx_model_path)

        # 모델 변환
        torch.onnx.export(self.model.cpu(),  # 실행될 모델
                          dummy_input,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                          self.onnx_model_path,  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                          export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                          opset_version=11,  # 모델을 변환할 때 사용할 ONNX 버전
                          do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                          input_names=['input'],  # 모델의 입력값을 가리키는 이름
                          output_names=['output'],  # 모델의 출력값을 가리키는 이름
                          )
        print(f"Model successfully converted to ONNX format and saved to {onnx_path}")

        if self.use_verify:
            print(f" Verfiy the onnx model in {onnx_path} ...")
            import onnx
            import onnxruntime
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            # get torch out
            torch_out = self.model(dummy_input)

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            ort_session = onnxruntime.InferenceSession(onnx_path)
            # ONNX 런타임에서 계산된 결과값
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
            ort_outs = ort_session.run(None, ort_inputs)

            # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
            np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=self.rtol, atol=self.atol)
            print("ONNX model output matches PyTorch model output within tolerance.")

    def to_tensorrt(self):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(self.onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError('Failed to parse the ONNX model')

        config = builder.create_builder_config()
        config.max_workspace_size = 4096 * (1 << 20)  # 4 GB

        # if use_fp16:
        #     config.set_flag(trt.BuilderFlag.FP16)

        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError('Failed to build the TensorRT engine')

        with open(self.trt_model_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"Model successfully converted to TensorRT format and saved to {self.trt_model_path}")

        if self.use_verify:
            print(f" Verfiy the TensorRT model in {self.trt_model_path} ...")
            import onnxruntime
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
            
            dummy_input = torch.randn(*self.input_shape, requires_grad=False).cuda()

            # ONNX 모델에서 더미 입력으로 추론
            onnx_session = onnxruntime.InferenceSession(self.onnx_model_path)
            onnx_input = {onnx_session.get_inputs()[0].name: to_numpy(dummy_input)}
            onnx_output = onnx_session.run(None, onnx_input)

            # TensorRT 엔진 로드 및 추론
            engine = self.load_engine(self.trt_model_path)
            trt_output = self.infer(engine, dummy_input.cpu().numpy())
            
            # 출력 비교
            np.testing.assert_allclose(onnx_output[0], trt_output, rtol=self.rtol, atol=self.atol)
            print("ONNX and TensorRT outputs are within tolerance.")

    def load_engine(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, engine, input_data):
        # Allocate buffers
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        # Transfer input data to device
        np.copyto(inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)

        # Run inference
        context = engine.create_execution_context()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from GPU
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
        stream.synchronize()

        return outputs[0].host.reshape(engine.get_binding_shape(1))

# Usage Example
if __name__ == '__main__':
    # model configurations
    model_name = 'AttentionUnet3D' 
    num_classes = 1
    in_ch = 1
    input_shape = (1, 1, 64, 288, 288)

    # Load your PyTorch model here
    model = ModelFactory(model_name, num_classes, in_ch).get().cuda()  
    model.eval()

    pth = r'/home/jhhan/02_dev/one3/nerve/src/experiments/CT_NerveSeg_v1.1.0-MICCAI-baseline_FD5BF6AB96/checkpoints/best.pth'
    output_onnx = pth.replace('.pth', '.onnx')
    output_trt = pth.replace('.pth', '.trt')
    
    state_dict = torch.load(pth)
    model.load_state_dict(state_dict['state_dict'], strict=True)
    
    converter = ModelConverter(model, input_shape, onnx_model_path=output_onnx, trt_model_path=output_trt)
    converter.to_onnx()
    converter.to_tensorrt()