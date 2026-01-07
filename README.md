# 1. Prepare OpenVINO models
## 1.1. Direct using model files
```python
import openvino as ov
core = ov.Core()
ov_model = core.read_model("/PATH/TO/INPUT_MODEL")
ov.save_model(ov_model, "/PATH/TO/OV_MODEL.xml", compress_to_fp16=False)
compiled_model = ov.compile_model(ov_model)
```
### 1.1.1 ONNX
ONNX model, which is a single .onnx file, can been read directly by OpenVINO read_model function

### 1.1.2 PaddlePaddle
PaddlePaddle models saved for inference, which has two files that names lik "inference.pdmodel" and "inference.pdiparams" in the same directory.
Then pass the "PATH/TO/inference.pdmodel" to OpenVINO read_model function

### 1.1.3 Tensorflow
TensorFlow models saved in frozen graph format can also be passed to OpenVINO read_model function

### 1.1.4 TFLite
TFLite models saved for inference with extension .tflite can be read directly by OpenVINO read_model function

After **read_model**, you can use **compile_model** to generate compiled_model for inference
## 1.2. Convert with OVC CLI tool
Using the **OVC CLI tool** provided by OpenVINO
#### Command line example:
```bash
ovc PATH/TO/INPUT/MODEL --input input_ids[1,128],attention_mask[-1,128] --output_model PATH/TO/OUTPUT/MODEL.xml
```
The **input** parameter is optional.
- By default, the input shapes will remain the same as the original model.
- Alternatively, you can set specific shapes to generate a fixed-shape model,
- or use *-1* to indicate that the shape of this dimension is dynamic.

## 1.3. Convert with Python API

### 1.3.1. Tensorflow 2 SavedModel / MetaGraph / Checkpoint
```python
import openvino as ov
ov_model_tf_SavedModel = ov.convert_model("PATH/TO/TF/SavedModel/DIR")
ov_model_tf_MetaGraph = ov.convert_model("PATH/TO/TF/meta_graph.meta")
ov_model_tf_Checkpoint = ov.convert_model(["PATH/TO/TF/inference_graph.pb", "PATH/TO/TF/checkpoint_file.ckpt"])
```
* **Save OpenVINO model files. *compress_to_fp16* = True to save f16 based model, otherwise save f32 based model**
```python
ov.save_model(ov_model_tf, "model/exported_tf_model.xml", compress_to_fp16=False)
```
* **Directly compile model for inference**
```python
compiled_model_tf = core.compile_model(ov_model_tf, device_name="CPU")
```

### 1.3.2. Simple Torch model
* **Load torch model with PyTorch functions**
```python
pt_model = LOADED_TORCH_MODEL_WITH_TORCH_FUNCTIONS
pt_model.eval()
```
* **Prepare *example_input***
```python
example_input = torch.zeros((1, 3, 224, 224))
```
* **Convert to openvino model.**
- The *input* parameter is optional.   
By default, the input shapes will remain the same as the original model.  
Alternatively, you can set specific shapes to generate a fixed-shape model,  
or use -1 to indicate that the shape of this dimension is dynamic.
```python
import openvino as ov
ov_model_pytorch = ov.convert_model(pt_model, example_input=example_input, input=[[1, 3, 224, 224]])
```
* **Save OpenVINO model files. *compress_to_fp16* = True to save f16 based model, otherwise save f32 based model**
```python
ov.save_model(ov_model_pytorch, "model/exported_pytorch_model.xml", compress_to_fp16=False)
```
* **Directly compile model for inference**
```python
compiled_model_pytorch = core.compile_model(ov_model_pytorch, device_name="CPU")
```

### 1.3.3. Convert from Torch Model with KV-Cache
The main process is the same as simple Torch model, but need a step to make model stateful **(store kv-cache in OpenVINO internal)** before save_model.  
In the ov_model_helper.py, We have provided the function *"patch_model_stateful"*.  
- First, named *INPUTs*/*OUTPUTs* which are KV-Cache tensors with *key_values.** and *present.**.
- Second, call patch_model_stateful before save_model
- [Refs to ov_model_helper.py#L300-L318](https://github.com/intel-sandbox/xDataPrep/blob/main/tools/OpenVINO_Wrapper/ov_model_helper.py#L300-L318) or [Ref to FireRedAsrAedWrapper::convert_ov_model](./ov_model_helper.py)


# 2. Using OpenVINO to inference
We have provided a base class *OV_Operator* in [ov_operator_async.py](./ov_operator_async.py).  
**Using UnimernetEncoderModel as example**. It inherits from the base class and implement the setup_model and __call__ methods.  
The setup_model method can integrate certain data preprocessing functions into the OpenVINO execution workflow.  
The __call__ method mainly defines the model’s input.
- First init class
```python
ov_model = UnimernetEncoderModel(model_path)
```

- Second setup model.
```python
# stream_num==1 then activate sync mode (LATENCY MODE) otherwise using async mode (THROUGHPUT MODE).  
# bf16==True then will use BF16 data type for inference.  
# f16==True then will use F16 data type for inference.  
# Priority level: BF16 > F16 > F32  
# TODO: AMX-F16 > AMX-BF16 > AVX512-F16 > AVX512-BF16 > F32
ov_model.setup_model(stream_num = 2, bf16=True, f16=True)
```

- Final call as Torch
```
res = ov_model(inputs)
```
