import torch
import numpy as np
# import onnx
# import tensorrt as trt
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from libs.models.ModelFactory import ModelFactory

class ModelConverter:
    def __init__(self, model, input_shape, output_path='output', use_verify=True):
        self.model = model # PyTorch model with eval mode
        self.input_shape = input_shape # input shape of the model (ch, cls, H, W, D)
        self.output_path = output_path # output path of the converted model
        self.use_verify = use_verify # verify the converted model
        self.rtol = 1e-02  # fp16은 atol=1e-03 정도로 설정 차후 use_fp16=True로 설정할 경우 수정
        self.atol = 1e-08
    
    def to_onnx(self):
        dummy_input = torch.randn(*self.input_shape, requires_grad=True)
        onnx_path = os.path.join(self.output_path)

            # 모델 변환
        torch.onnx.export(self.model.cpu(),  # 실행될 모델
                      dummy_input,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      self.output_path,  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=11,  # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=['output'],  # 모델의 출력값을 가리키는 이름
                      )
        print(f"Model successfully converted to ONNX format and saved to {onnx_path}")

        if self.use_verify:
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

    # def to_tensorrt(self, onnx_path='model.onnx', trt_path='model.trt', use_fp16=False):
    #     TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    #     EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    #     builder = trt.Builder(TRT_LOGGER)
    #     network = builder.create_network(EXPLICIT_BATCH)
    #     parser = trt.OnnxParser(network, TRT_LOGGER)

    #     with open(onnx_path, 'rb') as model:
    #         if not parser.parse(model.read()):
    #             for error in range(parser.num_errors):
    #                 print(parser.get_error(error))
    #             raise RuntimeError('Failed to parse the ONNX model')

    #     config = builder.create_builder_config()
    #     # config.max_workspace_size = 4096 * (1 << 20)  # 4 GB

    #     if use_fp16:
    #         config.set_flag(trt.BuilderFlag.FP16)

    #     serialized_engine = builder.build_serialized_network(network, config)

    #     if serialized_engine is None:
    #         raise RuntimeError('Failed to build the TensorRT engine')

    #     with open(trt_path, 'wb') as f:
    #         f.write(serialized_engine)
    #     print(f"Model successfully converted to TensorRT format and saved to {trt_path}")

# Usage Example
if __name__ == '__main__':
    # model configurations
    model_name = 'AttentionUnet3D' 
    num_classes = 1
    in_ch = 1
    input_shape = (num_classes, in_ch, 64, 288, 288)

    # Load your PyTorch model here
    model = ModelFactory(model_name, num_classes, in_ch).get().cuda()  
    model.eval()


    pth = r'/home/jhhan/02_dev/one3/nerve/src/experiments/CT_NerveSeg_v1.1.0-MICCAI-baseline_FD5BF6AB96/checkpoints/best.pth'
    output_onnx = pth.replace('.pth', '.onnx')
    state_dict = torch.load(pth)
    model.load_state_dict(state_dict['state_dict'], strict=True)
    # model = model.cuda()
    
    if not os.path.exists(output_onnx):
        converter = ModelConverter(model, input_shape, output_path=output_onnx)
        converter.to_onnx()
    else:
        print(f" Verfiy the onnx model in {output_onnx} ...")
        import onnx
        import onnxruntime
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
        dummy_input = torch.randn(*input_shape, requires_grad=True).cuda()
        # get torch out
        torch_out = model(dummy_input)


        onnx_model = onnx.load(output_onnx)
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(output_onnx)
        # ONNX 런타임에서 계산된 결과값
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)

        # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-02, atol=1e-08) # fp16은 atol=1e-03 정도로 설정

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")