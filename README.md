# Covid-Detection-with-VitisAI-and-TensorRT

>Repository for the Embeeded Systems II.

---
**authors:**  
<div style = "fonr-size:15px">
Cristian Conchari  
</div>
<div style = "fonr-size:15px">
Franklin Ticona  
</div>
<div style = "fonr-size:15px">
Hamed Quenta  
</div>
<div style = "fonr-size:15px">
Sergio Fernandez  
</div>
<div style = "fonr-size:15px">
<b>date:</b> June 3th 2021
</div>
<br>
  

<h2 style="background-color:tomato;font-size: 20px;">Introduction</h2>
<p>To demonstrate the correct operation of the artificial intelligence model, two different datasets were used, which fulfill the task of solving two different problems. Initially, we have a fruit classifier which separates strawberries from blueberries, which can have an industrial application being that by recognizing and detecting these fruits, each fruit can be sent to its respective process. On the other hand, we implemented a dataset with zeros and ones, which fulfills the task of character recognition, an application widely used since the 70s, which today continues to improve and through this project we try to demonstrate the functionality of our model to fulfill this task satisfactorily.

Translated with www.DeepL.com/Translator (free version)</p>

<h2 style="background-color:tomato;font-size: 20px;">Dataset</h2>

[Dataset link](https://www.kaggle.com/azaemon/preprocessed-ct-scans-for-covid19)
<br>

<h2 style="background-color:tomato;font-size: 20px;">Tools</h2>

### **Vitis - AI**
![Vitis AI](assets/Vitis-AI.png)
<p>The Vitis™ AI development environment is Xilinx's development platform for AI inference on Xilinx hardware platforms, including both edge devices and Alveo™ cards. It consists of optimized IP, tools, libraries, models, and example designs.</p>

### **Cuda - RT**
![Cuda rt](assets/cudart.png)
<p>Ray tracing, which has long been used for non-real-time rendering, provides realistic lighting by simulating the physical behavior of light. Ray tracing calculates the color of pixels by tracing the path that light would take if it were to travel from the eye of the viewer through the virtual 3D scene.</p>

### **Tensorflow**
![Tensorflow](assets/tensorflow.png)
<p>TensorFlow is an open-source library developed by Google primarily for deep learning applications. It also supports traditional machine learning. TensorFlow was originally developed for large numerical computations without keeping deep learning in mind.</p>

### **Docker**
![Docker](assets/docker.png)
<p>Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications.</p>

## Numbers model
### **Training**
~~~bash
Epoch 00015: LearningRateScheduler reducing learning rate to 0.0001.
Epoch 15/15
291/291 [==============================] - ETA: 0s - loss: 0.1125 - accuracy: 0.9971 
Epoch 00015: val_accuracy did not improve from 0.99750
291/291 [==============================] - 54s 186ms/step - loss: 0.1125 - accuracy: 0.9971 - val_loss: 0.1056 - val_accuracy: 0.9969
~~~

### **Quantize**
~~~bash
------------------------------------
TensorFlow version :  2.3.0
3.7.9 (default, Aug 31 2020, 12:42:55) 
[GCC 7.3.0]
------------------------------------
Command line options:
 --float_model  :  float_model/numbers_model.h5
 --quant_model  :  quant_model/q_numbers_model.h5
 --batchsize    :  60
 --tfrec_dir    :  tfrecords
 --evaluate     :  False
------------------------------------

[INFO] Start CrossLayerEqualization...
10/10 [==============================] - 3s 324ms/step
[INFO] CrossLayerEqualization Done.
[INFO] Start Quantize Calibration...
27/27 [==============================] - 6s 232ms/step
[INFO] Quantize Calibration Done.
[INFO] Start Generating Quantized Model...
[INFO] Generating Quantized Model Done.
Saved quantized model to quant_model/q_numbers_model.h5
~~~

### **Compile**
~~~bash
-----------------------------------------
COMPILING MODEL FOR ZCU102..
-----------------------------------------
/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.7/site-packages/xnnc/translator/tensorflow_translator.py:1809: H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
  value = param.get(group).get(ds).value
[INFO] parse raw model     :100%|██████████| 71/71 [00:00<00:00, 17408.84it/s]               
[INFO] infer shape (NHWC)  :100%|██████████| 109/109 [00:00<00:00, 35572.61it/s]             
[INFO] generate xmodel     :100%|██████████| 109/109 [00:00<00:00, 4114.02it/s]              
[INFO] Namespace(inputs_shape=None, layout='NHWC', model_files=['quant_model/q_numbers_model.h5'], model_type='tensorflow2', out_filename='compiled_model/customcnn_numbers_org.xmodel', proto=None)
[INFO] tensorflow2 model: quant_model/q_numbers_model.h5
[OPT] No optimization method available for xir-level optimization.
[INFO] generate xmodel: /workspace/Vitis-Tutorials/Machine_Learning/Design_Tutorials/Numbers_project/vitis-numbers/files/compiled_model/customcnn_numbers_org.xmodel
[UNILOG][INFO] The compiler log will be dumped at "/tmp/vitis-ai-user/log/xcompiler-20210604-012735-99"
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: function
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Graph name: functional_1, with op num: 233
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/workspace/Vitis-Tutorials/Machine_Learning/Design_Tutorials/Numbers_project/vitis-numbers/files/compiled_model/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/workspace/Vitis-Tutorials/Machine_Learning/Design_Tutorials/Numbers_project/vitis-numbers/files/compiled_model/customcnn_numbers.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 504761973848a4c8729dae40947d78ab, and been saved to "/workspace/Vitis-Tutorials/Machine_Learning/Design_Tutorials/Numbers_project/vitis-numbers/files/compiled_model/md5sum.txt"
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
-----------------------------------------
MODEL COMPILED
-----------------------------------------
~~~


<h2 style="background-color:tomato;font-size: 15px;">Results</h2>
<!---TODO: Result-->
