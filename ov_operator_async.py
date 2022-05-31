from array import array
from locale import ABDAY_1
import numpy as np
from datetime import datetime
from openvino.runtime import Core, get_version, AsyncInferQueue, InferRequest, Layout, Type, Tensor
from openvino.preprocess import PrePostProcessor
import copy

class OV_Operator(object):
    core = None
    model = None
    input_name = None
    out_name = None
    exec_net = None
    infer_queue = None

    def __init__(self, model, core =None):
        if core is None :
            self.core = Core()
        else :
            self.core = core
        self.model = self.core.read_model(model=model)
        self.input_name = self.model.input().get_any_name()
        self.output = [0]

    def setup_model(self, stream_num, shape) :
        if shape is not None :
            self.model.reshape({self.input_name: shape})
        config = self.prepare_for_cpu(stream_num)
        self.exec_net = self.core.compile_model(self.model, 'CPU', config)
        self.num_requests = self.exec_net.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        self.infer_queue = AsyncInferQueue(self.exec_net, self.num_requests)
        
    
    def prepare_for_cpu(self, stream_num) :
        device = "CPU"
        config = {}
        supported_properties = self.core.get_property(device, 'SUPPORTED_PROPERTIES')
        config['NUM_STREAMS'] = str(stream_num)
        config['AFFINITY'] = 'CORE'
        config['INFERENCE_NUM_THREADS'] = "8"
        config['PERF_COUNT'] = 'NO'
        config['INFERENCE_PRECISION_HINT'] = 'f32'
        config['PERFORMANCE_HINT'] = 'THROUGHPUT' # 'THROUGHPUT' #"LATENCY"
        config['PERFORMANCE_HINT_NUM_REQUESTS'] = "0"
        config['CPU_THREADS_NUM'] = "0"
        config['CPU_THROUGHPUT_STREAMS'] = 'CPU_THROUGHPUT_AUTO'
        config['CPU_BIND_THREAD'] = 'HYBRID_AWARE' 
        config['CPU_RUNTIME_CACHE_CAPACITY'] = '0'
        return config


class OV_Result :
    results = None
    outputs = None
    def __init__(self, outputs) :
        self.outputs = outputs
        self.results = {}
        for i in outputs:
            print('add results item {}'.format(i))
            self.results[i] = {}

    def completion_callback(self, infer_request: InferRequest, index: any) :
        for i in self.outputs:
            self.results[i][index] = copy.deepcopy(infer_request.get_output_tensor(i).data)
        return 

class TextDetectorResult(OV_Result) :
    def __init__(self, output) :
        super().__init__(output)


class TextDetector(OV_Operator):
    def __init__(self, model, core):
        super().__init__(model, core)


    def setup_model(self, stream_num = 1, shape=[1, -1,-1, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_layout(Layout('NHWC')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))


        mean = [123.675, 116.28, 103.53]
        scale = [58.395, 57.12, 57.375]
        ppp.input(self.input_name).preprocess() \
            .convert_element_type(Type.f32) \
            .mean(mean)  \
            .scale(scale) 

        self.model = ppp.build()
        
        super().setup_model(stream_num, None)
        self.det_res = TextDetectorResult(self.output)
        self.infer_queue.set_callback(self.det_res.completion_callback)


    def __call__(self, image) :   
        self.infer_queue.start_async({0: image}, 0)
        self.infer_queue.wait_all()
        return self.det_res.results[0][0]

class TextRecognizerResult(OV_Result):
    def __init__(self, output) :
        super().__init__(output)


class TextRecognizer(OV_Operator):
    def __init__(self, model, core):
        super().__init__(model, core) 

    def setup_model(self, stream_num = 2, shape=[-1,32,-1,3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_layout(Layout('NHWC')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))

        mean = [1.0, 1.0, 1.0]
        scale = [127.5, 127.5, 127.5]
        
        ppp.input(self.input_name).preprocess() \
            .convert_element_type(Type.f32) \
            .mean(scale) \
            .scale(scale)

        self.model = ppp.build()
        
        super().setup_model(stream_num, None)

        self.ocr_res = TextRecognizerResult(self.output)
        self.infer_queue.set_callback(self.ocr_res.completion_callback)

    def __call__(self, norm_img_batch_list) :
        for i, input_tensor in enumerate(norm_img_batch_list):
            self.infer_queue.start_async({0: input_tensor}, userdata=i)
            
        self.infer_queue.wait_all()

        res = []
        for i in range(len(norm_img_batch_list)) :
            res.append(self.ocr_res.results[0][i])
        return res


class TextClassfierResult(OV_Result):
    def __init__(self, output) :
        super().__init__(output)


class TextClassfier(OV_Operator):
    def __init__(self, model, core):
        super().__init__(model, core) 

    def setup_model(self, stream_num = 2, shape=[-1,32,-1,3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_layout(Layout('NHWC')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))

        scale = [127.5, 127.5, 127.5]

        ppp.input(self.input_name).preprocess() \
            .convert_element_type(Type.f32) \
            .mean(scale) \
            .scale(scale)

        self.model = ppp.build()
        
        super().setup_model(stream_num, None)
        self.cls_res = TextRecognizerResult(self.output)
        self.infer_queue.set_callback(self.cls_res.completion_callback)

    def __call__(self, norm_img_batch_list) :
        for i, input_tensor in enumerate(norm_img_batch_list):
            self.infer_queue.start_async({0: input_tensor}, userdata=i)

        self.infer_queue.wait_all()

        res = []
        for i in range(len(norm_img_batch_list)) :
            res.append(self.cls_res.results[i])
        return res
