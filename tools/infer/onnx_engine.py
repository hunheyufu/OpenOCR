import os

import numpy as np
import onnxruntime


class ONNXEngine:

    def __init__(self, onnx_path, use_gpu):
        """
        :param onnx_path:
        """
        if not os.path.exists(onnx_path):
            raise Exception(f'{onnx_path} is not exists')

        available_providers = onnxruntime.get_available_providers()
        providers = ['CPUExecutionProvider']
        if use_gpu:
            providers = [
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]

        providers = [p for p in providers if p in available_providers]
        if 'CPUExecutionProvider' in available_providers and 'CPUExecutionProvider' not in providers:
            providers.append('CPUExecutionProvider')
        if not providers:
            raise RuntimeError(
                f'No valid ONNX providers found. Available providers: {available_providers}'
            )
        self.onnx_session = onnxruntime.InferenceSession(onnx_path,
                                                         providers=providers)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def run(self, image_numpy):
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        result = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return result

    def warmup(self, num_iters=3, input_shape=None):
        if num_iters <= 0:
            return

        input_meta = self.onnx_session.get_inputs()[0]
        if input_shape is None:
            shape = []
            for idx, dim in enumerate(input_meta.shape):
                if isinstance(dim, int) and dim > 0:
                    shape.append(dim)
                else:
                    # Typical OCR input fallback: NCHW = 1x3x48x320
                    default_shape = [1, 3, 48, 320]
                    shape.append(default_shape[idx] if idx < len(default_shape) else 1)
            input_shape = tuple(shape)

        input_dtype = np.float32
        if input_meta.type == 'tensor(float16)':
            input_dtype = np.float16
        elif input_meta.type == 'tensor(double)':
            input_dtype = np.float64

        dummy_input = np.random.rand(*input_shape).astype(input_dtype)
        for _ in range(num_iters):
            self.run(dummy_input)

