from libs.converter.ModelConverter import *

class TaskFactory:
    def __init__(self, config, debug=False):
        self.name = config.task
        self.config = config
        self.debug = debug
        self.model = config.net_model
        self.num_classes = config.num_classes
        self.in_ch = config.in_ch
        self._input = (self.num_classes, self.in_ch, *config.input_shape)

    def run(self):
        if self.name == 'torch2onnx':
            model = ModelFactory(self.model, self.num_classes, self.in_ch).setup()
            state_dict = torch.load(self.config.torch_model_path)
            model.load_state_dict(state_dict['state_dict'], strict=True)
            converter = ModelConverter(model = model, 
                                       input_shape = self._input, 
                                       output_path = self.config.onnx_model_path,
                                       use_verify = self.config.use_verify)
            converter.to_onnx()
        elif self.name == 'onnx2trt':
            model = ModelFactory(self.model, self.num_classes, self.in_ch).get()
            converter = ModelConverter(model, self.config.input_shape, self.config.output_path)
            converter.to_onnx()
            # converter.to_tensorrt()
        elif self.name == 'torch2trt':
            model = ModelFactory(self.config.model).get()
            converter = ModelConverter(model, self.config.input_shape, self.config.output_path)
            converter.to_onnx()
            # converter.to_tensorrt()
        else:
            raise ValueError(f'Experiment \'{self.name}\' not found')
