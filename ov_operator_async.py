from array import array
from locale import ABDAY_1
import numpy as np
from datetime import datetime
from openvino.runtime import Core, get_version, AsyncInferQueue, InferRequest, Layout, Type, Tensor
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm

import copy
    
class OV_Operator(object):
    core = None
    model = None
    input_names = None
    input_shapes = None
    out_name = None
    exec_net = None
    infer_queue = None
    request = None
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
        # print('output: {}'.format(len(self.outputs)))
        self.input_names = []
        self.input_shapes = []
        ops = self.model.get_ordered_ops()
        for it in ops:
            if it.get_type_name() == 'Parameter':
                self.input_names.append(it.get_friendly_name())
                self.input_shapes.append(it.partial_shape)
                # print('input {}: {}'.format(it.get_friendly_name(),it.partial_shape))
        self.input_name = self.input_names[0]

    def create_single_request(self, bf16) :
        config = self.prepare_for_cpu(1, bf16)
        self.exec_net_single = self.core.compile_model(self.model, 'CPU', config)
        self.request = self.exec_net_single.create_infer_request()

    def setup_model(self, stream_num, bf16, shape) :
        if shape is not None :
            self.model.reshape({self.input_name: shape})
        config = self.prepare_for_cpu(stream_num, bf16)
        self.exec_net = self.core.compile_model(self.model, 'CPU', config)
        self.num_requests = self.exec_net.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
 
        # if self.num_requests > 1:
        #     self.infer_queue = AsyncInferQueue(self.exec_net, self.num_requests)
        #     self.create_single_request(bf16)
        # else :
        #     self.request = self.exec_net.create_infer_request()
        #     self.infer_queue = None
        self.infer_queue = AsyncInferQueue(self.exec_net, self.num_requests)
        self.create_single_request(bf16)
        # print('Model ({})  using {} streams'.format(self.model.get_friendly_name(), self.num_requests))

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
        # config['CPU_THREADS_NUM'] = "0"
        #config['CPU_THROUGHPUT_STREAMS'] = 'CPU_THROUGHPUT_AUTO'
        # config['CPU_BIND_THREAD'] = 'YES'  #'YES'#'NUMA' #'HYBRID_AWARE' 
        #config['DYN_BATCH_ENABLED'] = 'YES'
        #config['CPU_RUNTIME_CACHE_CAPACITY'] = '0'
        return config

    def __call__(self, input_tensors):
        nsize=len(input_tensors)
        if self.request and nsize==1:
            self.res.sync_clean()
            for i, input_tensor in enumerate(input_tensors):
                result = self.request.infer(input_tensor, share_inputs=True)
                self.res.sync_parser(result, i)
        elif self.infer_queue :
            for i, input_tensor in enumerate(input_tensors):
                print(f"input_tensor={input_tensor}")
                self.infer_queue.start_async(input_tensor, userdata=i, share_inputs=True)
            self.infer_queue.wait_all()
        else :
            print("Can not enter here!!!")
        return nsize

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
        self.results[index] = []
        values = result.values()
        for i, value in enumerate(values):
            # print("output {} value shape {}".format(i, value.shape))
            self.results[index].append(value)
        return 
    
    def sync_clean(self):
        self.results = {}

class ClipSegProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 1, bf16=True, means=[0.485, 0.456, 0.406], scales=[0.229, 0.224, 0.225], shape=[1, 352, 352, 3]) :
        ppp = PrePostProcessor(self.model)
        print(f"self.input_names={self.input_names}")
        ppp.input(self.input_names[0]).tensor() \
            .set_element_type(Type.u8) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))

        ppp.input(self.input_names[0]).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_names[0]).preprocess() \
            .resize(ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, shape[1], shape[2]) \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 

        # ppp.input(self.input_names[1]).tensor() \
        #     .set_shape([shape[0], -1])    

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
    
class EfficientMMOENetProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 1, bf16=True, means=[0.485, 0.456, 0.406], scales=[0.229, 0.224, 0.225], shape=[1, 224, 224, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_names[0]).tensor() \
            .set_element_type(Type.u8) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))

        ppp.input(self.input_names[0]).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_names[0]).preprocess() \
            .resize(ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, shape[1], shape[2]) \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 

        ppp.input(self.input_names[1]).tensor() \
            .set_shape([shape[0], -1])    

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class AudioProjProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 1, bf16=True, means=[0.485, 0.456, 0.406], scales=[0.229, 0.224, 0.225], shape=[1, 1280, 1280, 3]) :
        # ppp = PrePostProcessor(self.model)
        # ppp.input(self.input_name).tensor() \
        #     .set_element_type(Type.u8) \
        #     .set_shape(shape) \
        #     .set_color_format(ColorFormat.BGR) \
        #     .set_layout(Layout('NHWC'))
        #     # 


        # ppp.input(self.input_name).model() \
        #     .set_layout(Layout('NCHW'))

        # ppp.input(self.input_name).preprocess() \
        #     .convert_color(ColorFormat.RGB) \
        #     .convert_element_type(Type.f32) \
        #     .mean([x*255.0 for x in means])  \
        #     .scale([x*255.0 for x in scales]) 


        # self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class FaceLocaterProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 1, bf16=True, means=[0.485, 0.456, 0.406], scales=[0.229, 0.224, 0.225], shape=[1, 1280, 1280, 3]) :
        # ppp = PrePostProcessor(self.model)
        # ppp.input(self.input_name).tensor() \
        #     .set_element_type(Type.u8) \
        #     .set_shape(shape) \
        #     .set_color_format(ColorFormat.BGR) \
        #     .set_layout(Layout('NHWC'))
        # ppp.input(self.input_name).model() \
        #     .set_layout(Layout('NCHW'))
        # ppp.input(self.input_name).preprocess() \
        #     .convert_color(ColorFormat.RGB) \
        #     .convert_element_type(Type.f32) \
        #     .mean([x*255.0 for x in means])  \
        #     .scale([x*255.0 for x in scales]) 
        # self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
    
class DenoiseUnetProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num = 1, bf16=True, shape=None) :
        super().setup_model(stream_num, bf16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize = super().__call__(input_tensors)

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class ReferenceUnetProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num = 1, bf16=True, scale = 0.18215, shape=[-1, 4, 48, 48]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_names[1]).tensor() \
            .set_shape(shape) \
            .set_layout(Layout('NCHW'))
        ppp.input(self.input_names[1]).model() \
            .set_layout(Layout('NCHW'))
        ppp.input(self.input_names[1]).preprocess() \
            .scale(1.0/scale) 
        # ppp.input(self.input_names[0]).tensor() \
        #     .set_shape([-1]) 
        self.model = ppp.build()
        super().setup_model(stream_num, bf16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize = super().__call__(input_tensors)

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class VaeEncProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 1, bf16=True, mean=127.5, scale= 127.5, shape=[1, 384, 384, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))

        ppp.input(self.input_name).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .resize(ResizeAlgorithm.RESIZE_LINEAR, shape[1], shape[2]) \
            .mean(127.5) \
            .scale(127.5)

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class VaeDecProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 4, bf16=True, scale=0.18215, shape=[1, 384, 384, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .scale(scale)

        self.model = ppp.build()
        super().setup_model(stream_num, bf16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i][0])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i][0]))
        return res

    
    def clear_requests(self) :
        if self.request:
            self.request.reset_state()
        if self.infer_queue:
            self.infer_queue.reset_state()

class DonutEncProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 2, bf16=True, means=[0.485, 0.456, 0.406], scales=[0.229, 0.224, 0.225], shape=[1, 1280, 1280, 3]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))
            # 


        ppp.input(self.input_name).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 


        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class DonutDecProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess) 

    def setup_model(self, stream_num = 2, bf16=True) :       
        super().setup_model(stream_num, bf16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue:
            self.infer_queue.set_callback(self.res.completion_callback)

    def __call__(self, input_tensors):
        nsize = super().__call__(input_tensors)
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
    
    def clear_requests(self) :
        if self.request:
            self.request.reset_state()
        if self.infer_queue:
            self.infer_queue.reset_state()
        
class LayoutLMv3Processor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num = 2, bf16=True, means=[0.5, 0.5, 0.5], scales=[0.5, 0.5, 0.5], shape=[1, 224, 224, 3]) :
        # self.patch_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=torch.tensor((0.5, 0.5, 0.5)),
        #         std=torch.tensor((0.5, 0.5, 0.5)))
        # ])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # size = (image.shape[1], image.shape[0])
        # image = Image.fromarray(image)
        # image = image.resize((224, 224), Image.LANCZOS)
        # patch = self.patch_transform(image)
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout('NHWC'))

        ppp.input(self.input_name).model() \
            .set_layout(Layout('NCHW'))

        ppp.input(self.input_name).preprocess() \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .mean([x*255.0 for x in means])  \
            .scale([x*255.0 for x in scales]) 
            # .resize(ResizeAlgorithm.RESIZE_LINEAR) \


        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)

        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize = super().__call__(input_tensors)
 
        res = []
        if self.postprocess is None:
            for j in range(len(self.res.results[0])):
                res_list = []
                for i in range(nsize) :
                    res_list.append(self.res.results[i][j][0])
                res.append(res_list)
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res

class RelationsProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num = 2, bf16=True, shape=None) :
        super().setup_model(stream_num, bf16, None)
        self.res = OV_Result(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensors) :
        nsize=len(input_tensors)
        if nsize>1 or self.request is None:
            for i, input_tensor in input_tensors:
                self.infer_queue.start_async(input_tensor, userdata=i, share_inputs=True)
            self.infer_queue.wait_all()
        else :
            self.res.sync_clean()
            for i, input_tensor in input_tensors:
                result = self.request.infer(input_tensor)
                self.res.sync_parser(result, i)

        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append(self.res.results[i])
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i]))
        return res
   
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
            self.infer_queue.start_async({0: input_tensor}, userdata=0, share_inputs=True)
        self.infer_queue.wait_all()
       
        if self.postprocess is None:
               return self.res.results
        else :
           return self.postprocess(self.res.results)
        return res
    
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

        self.ocr_res = OV_Result(self.outputs)
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

class SqlBertProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num = 2, bf16=True, shape=None) :
        super().setup_model(stream_num, bf16, None)
        self.res = OV_Result(self.outputs)
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

class ObjDetector(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)


    def setup_model(self, stream_num = 1, bf16=True, shape=[1, 3, 512, 512]) :
        ppp = PrePostProcessor(self.model)
        ppp.input(self.input_name).tensor() \
            .set_element_type(Type.u8) \
            .set_shape(shape) \
            .set_layout(Layout('NCHW')) 
        ppp.input(self.input_name).model().set_layout(Layout('NCHW'))


        # mean = [123.675, 116.28, 103.53]
        # scale = [58.395, 57.12, 57.375]
        # ppp.input(self.input_name).preprocess() \
        #     .convert_element_type(Type.f32) \
        #     .mean(mean)  \
        #     .scale(scale) 

        self.model = ppp.build()
        
        super().setup_model(stream_num, bf16, None)
        self.det_res = OV_Result(self.outputs)
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
        self.det_res = OV_Result(self.outputs)
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

        self.ocr_res = OV_Result(self.outputs)
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
        self.cls_res = OV_Result(self.outputs)
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