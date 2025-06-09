# nnopt
A demo for optimizing neural network inference throughoutput and storage to use on embedded device.

## Project Summary

The objective of this project is to showcase the optimization of neural network inference by reducing the output, storage and computational requirements for deployment on embedded devices. The project includes a demo that utilizes two pretrained models, specifically designed for image and audio processing tasks, and demonstrates how to optimize their performance for embedded systems.

### Image Processing Model
* Task: Image classification
* Model: MobileNetV2
* Dataset: CIFAR-10

### Audio Processing Model
* Task: Audio classification
* Model: YAMNet
* Dataset: ESC-50

### Optimization pipeline
The optimization pipeline includes the following steps:
1. **Pruning**: Remove low-importance weights, either unstructured (i.e., low-scored weights set to 0 to induce sparsity) 
or structured (i.e., low-importance channels or filters removed entirely to keep the model dense).
2. **Quantization**: Convert model weights and activations to INT8 using PTQ or QAT.
3. **Export**: Export the optimized model to ONNX format for optimized deployment on CPU devices and TensorRT for NVIDIA GPUs.