# -*- coding: UTF-8 -*-
import openvino as ov
import torch
import numpy as np

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        model.eval()
        self.model_wrapper = model

    def forward(self, example_inputs):
        # print("### ModelWrapper")
        with torch.no_grad():
            return self.model_wrapper(**example_inputs)
    
    def convert_model(self, xml_Path, example_inputs, compress_weights=False):
        with torch.no_grad():
            ov_model = ov.convert_model(self, example_input=example_inputs)
            if compress_weights:
                ov_model = nncf.compress_weights(ov_model)
            ov.save_model(ov_model, xml_Path, compress_to_fp16=False)
            print(f"### save ov model @ {xml_Path}")
        # return self.forward(**example_inputs)
    
    def convert_onnx(self, onnx_Path, example_inputs, input_names, dynamic_axes):
        trace_model = torch.jit.trace(self, example_inputs)
        torch.onnx.export(trace_model, (), onnx_Path, input_names=input_names, dynamic_axes=dynamic_axes)

    
class EfficientMMOENetWrapper(ModelWrapper) :
    def __init__(self, model):
        super().__init__(model)
        
    def forward(self, anno, img):
        with torch.no_grad():
            preds = self.model_wrapper(anno, img)
            complete_score = torch.nn.Softmax(dim=1)(torch.tensor(preds['obj0'].reshape((1, -1))))
            audits = torch.nn.Softmax(dim=1)(torch.tensor(preds['obj1'].reshape((1, -1))))
            category_of_dishes = preds['obj2'].reshape((1, -1)).argmax(axis=-1)
            main_obvious_score = torch.nn.Softmax(dim=1)(torch.tensor(preds['obj2'].reshape((1, -1))))
            # print(f"audits={audits}, map_score={map_score}, complete_score={complete_score}, category_of_dishes={category_of_dishes}, main_obvious_score={main_obvious_score}")
            return complete_score, audits, category_of_dishes, main_obvious_score

    def convert_model(self, inputs, xml_Path, compress_weights=False):
        example_inputs = {"x" : inputs}
        return super().convert_model(example_inputs, xml_Path, compress_weights)

    def convert_onnx(self, onnx_Path, inputs):
        for k,v in inputs.items() :
            print(f"{k}={v.shape}")
        input_names = [k for k in inputs.keys()]
        dynamic_axes = {'anno': { 1: 'length'},
                        'img': {  2: 'width', 3: 'height'},} 
        trace_model = torch.jit.trace(self, (inputs['anno'], inputs['img']))
        torch.onnx.export(trace_model, (inputs['anno'], inputs['img']), onnx_Path, 
                          input_names=input_names, dynamic_axes=dynamic_axes)

class ClipSegWrapper(ModelWrapper) :
    def __init__(self, model):
        super().__init__(model)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        print("### in ClipSegWrapper")
        with torch.no_grad():
            outputs = self.model_wrapper(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_dict=False)
            return outputs[0]
            cla = outputs.logits.argmax(axis=0)
            return cla
            pos = torch.mean(torch.argwhere(cla == 1).to(torch.float))
            pianyi = (torch.sum(((pos - torch.tensor([176, 176])) ** 2)) ** 0.5) / (torch.sum(((torch.tensor([176, 176])) ** 2)) ** 0.5) * 2
            auditsV2 = 1 / (1 + torch.exp(-torch.tensor([pianyi, audits]) @ torch.tensor([-2.10123607,  4.12227572]) + 2.33651427))

            # pos = np.argwhere(cla == 1).mean(axis=0)
            # pianyi = (np.sum(((pos - np.array([176, 176])) ** 2)) ** 0.5) / (np.sum(((np.array([176, 176])) ** 2)) ** 0.5) * 2
            # auditsV2 = 1 / (1 + np.exp(-np.array([pianyi, audits]) @ np.array([-2.10123607,  4.12227572]) + 2.33651427))
            # print(f"cla={cla.mean()}, pos={pos}, pianyi={pianyi}, auditsV2={auditsV2}, audits={audits}")
            return torch.tensor(auditsV2)
        
    def convert_model(self, xml_Path, inputs, compress_weights=False):
        example_inputs = {k:v for k,v in inputs.items()}
        for k,v in example_inputs.items() :
            print(f"{k}={v.shape}")
        return super().convert_model(xml_Path, example_inputs, compress_weights)
    
    def convert_onnx(self, onnx_Path, inputs):
        for k,v in inputs.items() :
            print(f"{k}={v.shape}")
        input_names = [k for k in inputs.keys()]
        dynamic_axes = {'input_ids': { 1: 'length',},
                        'attention_mask': { 1: 'length',},
                        'pixel_values': { 2: 'width', 3: 'height'}} 
        trace_model = torch.jit.trace(self, (inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values']))
        torch.onnx.export(trace_model,  (inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values']), onnx_Path, 
                          input_names=input_names, dynamic_axes=dynamic_axes)