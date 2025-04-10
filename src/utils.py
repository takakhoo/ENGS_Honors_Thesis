""" 
Often, CUDA and PyTorch version cause conflict. 
Use this script to output your's, but also check in your GPU
And on PyTorch's website for what version will work for you


"""
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA version from torch:", torch.version.cuda)
    print("GPU Name:", torch.cuda.get_device_name(0))

