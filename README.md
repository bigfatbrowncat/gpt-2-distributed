## Abstract
To score gpt2-medium on GPU exclusively it is needed to have 9.5GB VRAM. That means at least GTX 1080Ti.
This application allows you to modify a configuration file to split a DNN into parts that are running on different devices -- CPU and possibly multiple GPUs.

## Requirements

The currently supported software versions are:
* Python 3.10
* PyTorch 1.13 + CUDA 1.16

## Installing


1. Create a new `venv` with Python 3.10.
2. Activate it.
3. Install the basic dependencies:
```
(venv) $ pip install -r requirements.txt
```
4. Install torch for your computing architecture.<br/>

If CUDA is supported by your system, run:
```
(venv) $ pip install -r requirements_cuda.txt --extra-index-url https://download.pytorch.org/whl/cu116
```
For a CPU-only, run:
```
(venv) $ pip install -r requirements_nogpu.txt
```


## Inference

