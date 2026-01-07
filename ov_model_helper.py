# -*- coding: UTF-8 -*-
import gc
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import AutoConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput

import openvino as ov
from openvino import save_model, convert_model
try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13

def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0

def model_has_input_output_name(ov_model: ov.Model, name: str):
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])

def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    gather_dim: int,
):
    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("encoder_outputs").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()

def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    input_ids = ov_model.input("encoder_outputs")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()

def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    key_value_output_names: list[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)

def patch_stateful(ov_model):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )
    
def cleanup_torchscript_cache():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()
    gc.collect()

def patch_model_stateful(ov_model, input_names, output_names) :
    for input, input_name in zip(ov_model.inputs, input_names):
        input.get_tensor().set_names({input_name})
    for output, output_name in zip(ov_model.outputs, output_names):
        output.get_tensor().set_names({output_name})
    patch_stateful(ov_model)
    return ov_model

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        model.eval()
        self.model_wrapper = model

    def forward(self, example_inputs):
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

#simple wrapper for OpenVINO Model Convert
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
        with torch.no_grad():
            outputs = self.model_wrapper(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_dict=False)
            return outputs[0]
            cla = outputs.logits.argmax(axis=0)
            return cla
            pos = torch.mean(torch.argwhere(cla == 1).to(torch.float))
            pianyi = (torch.sum(((pos - torch.tensor([176, 176])) ** 2)) ** 0.5) / (torch.sum(((torch.tensor([176, 176])) ** 2)) ** 0.5) * 2
            auditsV2 = 1 / (1 + torch.exp(-torch.tensor([pianyi, audits]) @ torch.tensor([-2.10123607,  4.12227572]) + 2.33651427))
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

# Enc-Dec model for OpenVINO Model Convert
# Encoder is simple mode
# Decoder always has kv-cache system 
# So we need to stateful model convert    
class FireRedAsrAedWrapper() :
    def __init__(self, model):
        class ModelEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, feats, lengths):
                with torch.no_grad():
                    enc_outputs, _, enc_mask = self.model.encoder(feats, lengths)
                return enc_outputs, enc_mask

        class ModelDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, t_ys, encoder_outputs, src_mask, softmax_smoothing, eos_penalty,
                        is_finished, B, N, scores, scores_mask, caches):
                with torch.no_grad():
                    topB_row_number_in_ys, t_ys, scores, caches = self.model.decoder.infer_decoder0(t_ys, 
                                        encoder_outputs, src_mask, scores_mask, caches, scores,
                                        softmax_smoothing, eos_penalty, is_finished, B, N)
                    return topB_row_number_in_ys, t_ys, scores, caches
        self.enc_wrapper = ModelEncoderWrapper(model)
        self.enc_wrapper.eval()
        self.dec_wrapper = model.encoder
        self.dec_wrapper.eval()

    def convert_ov_model(self, feats, lengths, beam_size, nbest, decode_max_len,
                   softmax_smoothing, length_penalty, eos_penalty,
                   ov_config_path, ov_encoder_path, ov_decoder_path,
                   sos_id, eos_id, pad_id):
        if not os.path.exists(ov_config_path):
            folder_path = os.path.dirname(ov_config_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            with open(ov_config_path, "w") as file:
                data = {
                    "sos_id": sos_id,
                    "eos_id": eos_id,
                    "pad_id": selpad_id,
                }
                json.dump(data, file, indent=2)
                print(f"✅ Save model config to {ov_config_path}")

        
        if not ov_encoder_path.exists() :
            example_inputs = {"feats":feats, "lengths":lengths}
            ov_model = convert_model(self.enc_wrapper, example_input=example_inputs)
            save_model(ov_model, ov_encoder_path, compress_to_fp16=False)
            print(f"✅ ModelEncoder completed {ov_encoder_path}")
            del ov_model
            cleanup_torchscript_cache()

        enc_outputs, enc_mask = self.enc_wrapper(feats, lengths)

        if not ov_decoder_path.exists() :
            beam_size=3
            num = 2
            cache_size = 16
            
            B = beam_size
            N, Ti, H = enc_outputs.size()
            cache_shape = (B*N, num, 1280)

            encoder_outputs = enc_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
            src_mask = enc_mask.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
            t_ys = torch.ones(N*B, 1).fill_(self.sos_id).long()
            scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float()
            scores = scores.repeat(N).view(N*B, 1)
            scores_mask = scores
            is_finished = torch.zeros_like(scores)
            N = torch.tensor(N).long()
            B = torch.tensor(B).long()

            caches = []
            input_names = ["t_ys", "encoder_outputs", "src_mask", "softmax_smoothing",
                           "eos_penalty", "is_finished", "B", "N", "scores", "scores_mask"]
            output_names = ["topB_row_number_in_ys", "new_t_ys", "new_scores"]

            for i in range(cache_size):
                cache = torch.randn(cache_shape)
                caches.append(cache)
                input_names.extend([f"key_values.{i}"])
                output_names.extend([f"present.{i}"])

            example_input = {"t_ys":t_ys, "encoder_outputs": encoder_outputs, "src_mask": src_mask,
                             "softmax_smoothing": softmax_smoothing, "eos_penalty": eos_penalty,
                             "is_finished":is_finished, "B": B, "N": N, "scores": scores,
                             "scores_mask": scores_mask, "caches": caches}
                
            ov_model = ov.convert_model(self.dec_wrapper, example_input=example_input)
            
            ov_model = patch_model_stateful(ov_model, input_names, output_names)
            print("✅ ModelDecoder model successfully converted")

            ov.save_model(ov_model, ov_decoder_path, compress_to_fp16=False)
            del ov_model
            cleanup_torchscript_cache()
            print(f"✅ ModelDecoder completed {ov_decoder_path}")
 