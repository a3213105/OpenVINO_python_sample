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
    input_names = None
    input_shapes = None
    out_name = None
    exec_net = None
    infer_queue = None
    outputs= None

    def __init__(self, model, core=None, postprocess=None):
        self.postprocess = postprocess
        if core is None :
            self.core = Core()
        else :
            self.core = core
        self.model = self.core.read_model(model=model)
        output_size = self.model.get_output_size()
        self.outputs = []
        for i in range (0,output_size):
            self.outputs.append(i)
        print('output: {}'.format(len(self.outputs)))
        self.input_names = []
        self.input_shapes = []
        ops = self.model.get_ordered_ops()
        for it in ops:
            if it.get_type_name() == 'Parameter':
                self.input_names.append(it.get_friendly_name())
                self.input_shapes.append(it.partial_shape)
                print('input {}: {}'.format(it.get_friendly_name(),it.partial_shape))
        self.input_name = self.input_names[0]

    def setup_model(self, stream_num, bf16, shape) :
        if shape is not None :
            self.model.reshape({self.input_name: shape})
        config = self.prepare_for_cpu(stream_num, bf16)
        self.exec_net = self.core.compile_model(self.model, 'CPU', config)
        self.num_requests = self.exec_net.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        if self.num_requests > 1:
            self.infer_queue = AsyncInferQueue(self.exec_net, self.num_requests)
            self.request = None
        else :
            self.request = self.exec_net.create_infer_request()
            self.infer_queue = None
        print('Model ({})  using {} streams'.format(self.model.get_friendly_name(), self.num_requests))

    
    def prepare_for_cpu(self, stream_num, bf16=True) :
        device = "CPU"
        hint = 'THROUGHPUT' if stream_num>1 else 'LATENCY'
        data_type = 'bf16' if bf16 else 'f32'
        config = {}
        supported_properties = self.core.get_property(device, 'SUPPORTED_PROPERTIES')
        config['NUM_STREAMS'] = str(stream_num)
        config['AFFINITY'] = 'CORE'
        config['INFERENCE_NUM_THREADS'] = "0" #str(stream_num) #"0"
        config['PERF_COUNT'] = 'NO'
        config['INFERENCE_PRECISION_HINT'] = data_type #'bf16'#'f32'
        config['PERFORMANCE_HINT'] = hint # 'THROUGHPUT' #"LATENCY"
        #config['PERFORMANCE_HINT_NUM_REQUESTS'] = "0"
        config['CPU_THREADS_NUM'] = "0"
        #config['CPU_THROUGHPUT_STREAMS'] = 'CPU_THROUGHPUT_AUTO'
        config['CPU_BIND_THREAD'] = 'YES'  #'YES'#'NUMA' #'HYBRID_AWARE' 
        #config['DYN_BATCH_ENABLED'] = 'YES'
        #config['CPU_RUNTIME_CACHE_CAPACITY'] = '0'
        return config


class OV_Result :
    results = None
    outputs = None
    def __init__(self, outputs) :
        self.outputs = outputs
        self.results = {}
        #for i in outputs:
        #    #print('add results item {}'.format(i))
        #    self.results[i] = {}

    def completion_callback(self, infer_request: InferRequest, index: any) :
        #if index not in self.results :
        self.results[index] = []
        for i in self.outputs:
            self.results[index].append(copy.deepcopy(infer_request.get_output_tensor(i).data))
        return 

    def sync_parser(self, result, index: any) :
        # self.results = {}
        self.results[index] = []
        values = result.values()
        for i, value in enumerate(values):
            self.results[index].append(value)
        return 
    
    def sync_clean(self):
        self.results = {}
        

class FingerprintResult(OV_Result):
    def __init__(self, outputs) :
        super().__init__(outputs)
        
    # def completion_callback(self, infer_request: InferRequest, index: any) :
    #     self.results=[]
    #     for i in self.outputs:
    #         self.results.append(copy.deepcopy(infer_request.get_output_tensor(i).data))
    #     return

class Fingerprint(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 2, bf16=True, shape=None) :       
        super().setup_model(stream_num, bf16, shape)
        self.res = FingerprintResult(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensor) :
        if self.infer_queue:
            self.infer_queue.start_async({0: input_tensor}, userdata=0)
        self.infer_queue.wait_all()
       
        if self.postprocess is None:
               return self.res.results
        else :
           return self.postprocess(self.res.results)
        return res
    

class CTCSimpleOCRResult(OV_Result):
    def __init__(self, outputs) :
        super().__init__(outputs)

class CTCSimpleOCR(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 2, bf16=True, shape=None) :
        ppp = PrePostProcessor(self.model)
        if shape is None:
            ppp.input(self.input_name).tensor() \
                .set_element_type(Type.u8) \
                .set_layout(Layout('NHWC')) 
        else :
            ppp.input(self.input_name).tensor() \
                .set_element_type(Type.u8) \
                .set_shape(shape) \
                .set_layout(Layout('NHWC')) 
        #ppp.input(self.input_name).model().set_layout(Layout('NCHW'))

        scale = [127.5]
        
        ppp.input(self.input_name).preprocess() \
            .convert_element_type(Type.f32) \
            .mean(scale) \
            .scale(scale)

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)

        self.ocr_res = CTCSimpleOCRResult(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.ocr_res.completion_callback)

    def __call__(self, norm_img_batch_list) :
        if self.request :
            for i, input_tensor in enumerate(norm_img_batch_list):
                result = self.request.infer(input_tensor)
                self.ocr_res.sync_parser(result, 0)
            return [self.ocr_res.results[0]] 
        
        nsize=len(norm_img_batch_list)

        for i, input_tensor in enumerate(norm_img_batch_list):
            self.infer_queue.start_async({0: input_tensor}, userdata=i)
            
        self.infer_queue.wait_all()

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.ocr_res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.ocr_res.results[i]))
        return res


class SqlBertResult(OV_Result):
    def __init__(self, outputs) :
        super().__init__(outputs)

class SqlBertProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num = 2, bf16=True, shape=None) :
        super().setup_model(stream_num, bf16, None)
        self.res = SqlBertResult(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize=len(input_tensors)
        if self.request :
            self.res.sync_clean()
            for i, input_tensor in enumerate(input_tensors):
                result = self.request.infer(input_tensor)
                self.res.sync_parser(result, i)
        else :
            for i, input_tensor in enumerate(input_tensors):
                self.infer_queue.start_async(input_tensor, userdata=i)
            self.infer_queue.wait_all()
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
    
    def __async_call_(self, input_tensors):
        res = []
        for input_tensor in input_tensors:
            idle_id = self.infer_queue.get_idle_request_id()
            res.append(self.res.results[idle_id])
            self.infer_queue.start_async(input_tensor, userdata=idle_id)
        return res


class TextDetectorResult(OV_Result) :
    def __init__(self, output) :
        super().__init__(output)

class TextDetector(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)


    def setup_model(self, stream_num = 1, bf16=True, shape=[1, -1,-1, 3]) :
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
        
        super().setup_model(stream_num, bf16, None)
        self.det_res = TextDetectorResult(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.det_res.completion_callback)


    def __call__(self, images) :   
        nsize=len(images)
        if self.request :
            self.det_res.sync_clean()
            for i, image in enumerate(images):
                result = self.request.infer({0: image})
                self.det_res.sync_parser(result, i)
        else :
            for i, image in enumerate(images):
                self.infer_queue.start_async({0: image}, userdata=i)
            self.infer_queue.wait_all()
            
        if self.postprocess is None:
            return self.det_res.results[0][0]
        else :
            return self.postprocess(self.det_res.results[0][0])


class TextRecognizerResult(OV_Result):
    def __init__(self, output) :
        super().__init__(output)

class TextRecognizer(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 2, bf16=True, shape=[-1,32,-1,3]) :
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
        
        super().setup_model(stream_num, bf16, None)

        self.ocr_res = TextRecognizerResult(self.outputs)
        self.infer_queue.set_callback(self.ocr_res.completion_callback)

    def __call__(self, norm_img_batch_list) :
        nsize=len(norm_img_batch_list)
        #for i in range(nsize-1, -1, -1):
        #    self.infer_queue.start_async({0: norm_img_batch_list[i]}, userdata=i)

        for i, input_tensor in enumerate(norm_img_batch_list):
            self.infer_queue.start_async({0: input_tensor}, userdata=i)
            
        self.infer_queue.wait_all()

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.ocr_res.results[0][i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.ocr_res.results[0][i]))
        return res


class TextClassfierResult(OV_Result):
    def __init__(self, output) :
        super().__init__(output)

class TextClassfier(OV_Operator):
    def __init__(self, model, core, postprocess):
        super().__init__(model, core=None, postprocess=None) 

    def setup_model(self, stream_num=2, bf16=True, shape=[-1,32,-1,3]) :
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
        
        super().setup_model(stream_num, bf16, None)
        self.cls_res = TextRecognizerResult(self.outputs)
        self.infer_queue.set_callback(self.cls_res.completion_callback)

    def __call__(self, norm_img_batch_list) :
        for i, input_tensor in enumerate(norm_img_batch_list):
            self.infer_queue.start_async({0: input_tensor}, userdata=i)

        self.infer_queue.wait_all()

        res = []
        if self.postprocess is None:
            for i in range(len(norm_img_batch_list)) :
                res.append(self.cls_res.results[0][i])
        else :
            for i in range(len(norm_img_batch_list)) :
                res.append(self.postprocess(self.cls_res.results[0][i]))
        return res
